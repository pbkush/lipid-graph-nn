"""Orchestrate one Martini 3 bilayer simulation: build → minimize → equilibrate → produce."""
from __future__ import annotations

import dataclasses
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from lipid_gnn.martini_pipeline import INSANE_CMD, MARTINI3_ITP_DIR
from lipid_gnn.martini_pipeline.composition import Composition
from lipid_gnn.martini_pipeline.manifest import (
    Manifest,
    detect_gmx_version,
    detect_insane_version,
    hash_file,
    host_info,
    write_manifest,
)
from lipid_gnn.martini_pipeline.mdp_writer import MDPParams, write_mdps
from lipid_gnn.martini_pipeline.system_builder import BoxParams, BuildResult, build_system

# Per-stage filename conventions matching legacy layout exactly.
_STAGE_DEFFNM = {
    "minimization": "martini_em",
    "equilibration": "martini_eq",
    "production": "prun",
}
_STAGE_HANDOFF = {
    "minimization": "minimized.gro",
    "equilibration": "equilibrated.gro",
    "production": "prun.gro",
}
_STAGE_MDP_NAME = {
    "minimization": "em.mdp",
    "equilibration": "eq.mdp",
    "production": "run.mdp",
}
_MDP_STAGE_KEY = {
    "minimization": "minimization",
    "equilibration": "equilibration",
    "production": "run",
}


@dataclass(frozen=True)
class StageResult:
    name: str                   # "minimization" | "equilibration" | "production"
    status: str                 # "ok" | "skipped" | "failed"
    walltime_s: float
    grompp_cmd: list
    mdrun_cmd: list
    tpr_path: str
    final_gro_path: str
    log_path: str
    error: Optional[str] = None


@dataclass(frozen=True)
class PipelineResult:
    composition: dict
    out_dir: str
    build: BuildResult
    stages: tuple
    manifest_path: str
    overall_status: str         # "ok" | "failed_at_<stage>"


def run(
    composition: dict,
    out_dir: str,
    *,
    box: BoxParams = BoxParams(),
    mdp_params: MDPParams = MDPParams(),
    seed: Optional[int] = None,
    gmx_executable: str = "gmx",
    mdrun_extra_args: tuple = (),
    force_rerun: bool = False,
    maxwarn: int = 2,
    insane_cmd: str = INSANE_CMD,
    itp_dir: str = MARTINI3_ITP_DIR,
) -> PipelineResult:
    """Build bilayer and run minimization → equilibration → production.

    Idempotent: stages whose handoff .gro already exists are skipped unless
    *force_rerun* is True.  Manifest is written after each stage transition so
    a killed run still leaves a useful manifest.
    """
    if not shutil.which(gmx_executable):
        raise FileNotFoundError(
            f"gmx executable not found on PATH: {gmx_executable!r}"
        )

    comp = Composition(composition)
    canonical_name = comp.name
    os.makedirs(out_dir, exist_ok=True)

    if seed is None:
        seed = _derive_seed(canonical_name)
    params_fixed = dataclasses.replace(mdp_params, gen_seed=seed)

    # --- Build initial bilayer ---
    build = build_system(
        composition,
        out_dir,
        box=box,
        insane_cmd=insane_cmd,
        itp_dir=itp_dir,
        gmx_executable=gmx_executable,
    )

    # --- Write MDPs to per-stage subdirs ---
    stage_dirs = {
        "minimization": os.path.join(out_dir, "minimization"),
        "equilibration": os.path.join(out_dir, "equilibration"),
        "production": os.path.join(out_dir, "run"),
    }
    for d in stage_dirs.values():
        os.makedirs(d, exist_ok=True)

    mdp_paths = _write_stage_mdps(out_dir, stage_dirs, params_fixed)

    # --- Run stages ---
    stage_order = ("minimization", "equilibration", "production")
    stage_inputs = {
        "minimization": build.gro_path,
    }

    stages: list[StageResult] = []
    overall_status = "ok"
    manifest_path = os.path.join(out_dir, "manifest.json")

    for stage_name in stage_order:
        stage_dir = stage_dirs[stage_name]
        deffnm = _STAGE_DEFFNM[stage_name]
        handoff = os.path.join(stage_dir, _STAGE_HANDOFF[stage_name])
        tpr_path = os.path.join(stage_dir, f"{deffnm}.tpr")
        mdp_path = mdp_paths[stage_name]
        gro_in = stage_inputs[stage_name]

        if not force_rerun and os.path.isfile(handoff):
            sr = StageResult(
                name=stage_name,
                status="skipped",
                walltime_s=0.0,
                grompp_cmd=[],
                mdrun_cmd=[],
                tpr_path=tpr_path,
                final_gro_path=handoff,
                log_path=os.path.join(stage_dir, f"{deffnm}.log"),
            )
        else:
            sr = _run_stage(
                stage_name=stage_name,
                stage_dir=stage_dir,
                deffnm=deffnm,
                mdp_path=mdp_path,
                gro_in=gro_in,
                top_path=build.top_path,
                ndx_path=build.ndx_path,
                gmx=gmx_executable,
                extra_args=mdrun_extra_args,
                maxwarn=maxwarn,
            )

        stages.append(sr)

        if sr.status == "failed":
            overall_status = f"failed_at_{stage_name}"
            _write_manifest(
                manifest_path, composition, canonical_name, out_dir,
                build, stages, overall_status, seed, box, params_fixed, mdp_paths,
            )
            raise RuntimeError(
                f"Pipeline failed at {stage_name}: {sr.error}"
            )

        # Feed handoff into next stage
        if stage_name == "minimization":
            stage_inputs["equilibration"] = sr.final_gro_path
        elif stage_name == "equilibration":
            stage_inputs["production"] = sr.final_gro_path

        _write_manifest(
            manifest_path, composition, canonical_name, out_dir,
            build, stages, overall_status, seed, box, params_fixed, mdp_paths,
        )

    return PipelineResult(
        composition=dict(composition),
        out_dir=out_dir,
        build=build,
        stages=tuple(stages),
        manifest_path=manifest_path,
        overall_status=overall_status,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _derive_seed(composition_name: str) -> int:
    return int(hashlib.sha256(composition_name.encode()).hexdigest()[:8], 16)


def _write_stage_mdps(out_dir: str, stage_dirs: dict, params: MDPParams) -> dict:
    """Write one MDP per stage to its stage subdir. Returns {stage_name: mdp_path}."""
    with tempfile.TemporaryDirectory() as staging:
        written = write_mdps(staging, params=params)
        # write_mdps returns {stage_key: path}; stage keys are "minimization", "equilibration", "run"
        paths = {}
        for stage_name, stage_dir in stage_dirs.items():
            mdp_key = _MDP_STAGE_KEY[stage_name]
            src = written[mdp_key]
            dst = os.path.join(stage_dir, _STAGE_MDP_NAME[stage_name])
            shutil.copy(src, dst)
            paths[stage_name] = dst
    return paths


def _run_stage(
    *,
    stage_name: str,
    stage_dir: str,
    deffnm: str,
    mdp_path: str,
    gro_in: str,
    top_path: str,
    ndx_path: Optional[str],
    gmx: str,
    extra_args: tuple,
    maxwarn: int,
) -> StageResult:
    tpr_path = os.path.join(stage_dir, f"{deffnm}.tpr")
    handoff = os.path.join(stage_dir, _STAGE_HANDOFF[stage_name])
    log_path = os.path.join(stage_dir, f"{deffnm}.log")

    t0 = time.monotonic()

    # grompp
    grompp_cmd, grompp_rc, grompp_err = _run_grompp(
        stage_dir=stage_dir,
        mdp=os.path.abspath(mdp_path),
        gro_in=os.path.abspath(gro_in),
        top=os.path.abspath(top_path),
        ndx=os.path.abspath(ndx_path) if ndx_path else None,
        tpr_out=os.path.abspath(tpr_path),
        gmx=gmx,
        maxwarn=maxwarn,
    )
    if grompp_rc != 0:
        return StageResult(
            name=stage_name,
            status="failed",
            walltime_s=time.monotonic() - t0,
            grompp_cmd=grompp_cmd,
            mdrun_cmd=[],
            tpr_path=tpr_path,
            final_gro_path="",
            log_path=log_path,
            error=f"grompp exit {grompp_rc}: {grompp_err[-500:]}",
        )

    # mdrun
    mdrun_cmd, mdrun_rc, mdrun_err = _run_mdrun(
        stage_dir=stage_dir,
        deffnm=deffnm,
        gmx=gmx,
        extra_args=extra_args,
    )
    if mdrun_rc != 0:
        return StageResult(
            name=stage_name,
            status="failed",
            walltime_s=time.monotonic() - t0,
            grompp_cmd=grompp_cmd,
            mdrun_cmd=mdrun_cmd,
            tpr_path=tpr_path,
            final_gro_path="",
            log_path=log_path,
            error=f"mdrun exit {mdrun_rc}: {mdrun_err[-500:]}",
        )

    gro_out = os.path.join(stage_dir, f"{deffnm}.gro")
    if os.path.abspath(gro_out) != os.path.abspath(handoff):
        shutil.copy(gro_out, handoff)

    return StageResult(
        name=stage_name,
        status="ok",
        walltime_s=time.monotonic() - t0,
        grompp_cmd=grompp_cmd,
        mdrun_cmd=mdrun_cmd,
        tpr_path=tpr_path,
        final_gro_path=handoff,
        log_path=log_path,
    )


def _run_grompp(*, stage_dir, mdp, gro_in, top, ndx, tpr_out, gmx, maxwarn):
    cmd = [gmx, "grompp",
           "-f", mdp,
           "-c", gro_in,
           "-p", top,
           "-o", tpr_out,
           "-maxwarn", str(maxwarn)]
    if ndx is not None:
        cmd += ["-n", ndx]
    result = subprocess.run(cmd, capture_output=True, text=True)
    log_path = os.path.join(stage_dir, "grompp.log")
    with open(log_path, "w") as fh:
        fh.write(result.stdout)
        if result.stderr:
            fh.write("\n--- stderr ---\n")
            fh.write(result.stderr)
    return cmd, result.returncode, result.stderr


def _run_mdrun(*, stage_dir, deffnm, gmx, extra_args):
    cmd = [gmx, "mdrun", "-deffnm", deffnm] + list(extra_args)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=stage_dir)
    log_path = os.path.join(stage_dir, "mdrun.log")
    with open(log_path, "w") as fh:
        fh.write(result.stdout)
        if result.stderr:
            fh.write("\n--- stderr ---\n")
            fh.write(result.stderr)
    return cmd, result.returncode, result.stderr


def _write_manifest(
    manifest_path: str,
    composition: dict,
    canonical_name: str,
    out_dir: str,
    build: BuildResult,
    stages: list,
    overall_status: str,
    seed: int,
    box: BoxParams,
    params: MDPParams,
    mdp_paths: dict,
) -> None:
    mdp_hashes = {}
    for stage_name, mdp_path in mdp_paths.items():
        if os.path.isfile(mdp_path):
            mdp_hashes[_STAGE_MDP_NAME[stage_name]] = hash_file(mdp_path)

    m = Manifest(
        schema_version="1.0",
        composition=dict(composition),
        canonical_name=canonical_name,
        out_dir=out_dir,
        created_utc=datetime.now(timezone.utc).isoformat(),
        gmx_version=detect_gmx_version(
            next(
                (s.grompp_cmd[0] for s in stages if s.grompp_cmd),
                "gmx",
            )
        ),
        insane_version=detect_insane_version(),
        insane_cmd=list(build.insane_cmd),
        seed=seed,
        box=dataclasses.asdict(box),
        mdp_params=dataclasses.asdict(params),
        mdp_hashes=mdp_hashes,
        stages=[_serialise_stage(s) for s in stages],
        build_stats={
            "molecule_counts": build.molecule_counts,
            "total_atoms": build.total_atoms,
            "n_membrane_beads": build.n_membrane_beads,
            "n_solvent_atoms": build.n_solvent_atoms,
        },
        overall_status=overall_status,
        host=host_info(),
    )
    write_manifest(manifest_path, m)


def _serialise_stage(s: StageResult) -> dict:
    return {
        "name": s.name,
        "status": s.status,
        "walltime_s": s.walltime_s,
        "grompp_cmd": list(s.grompp_cmd),
        "mdrun_cmd": list(s.mdrun_cmd),
        "tpr_path": s.tpr_path,
        "final_gro_path": s.final_gro_path,
        "log_path": s.log_path,
        "error": s.error,
    }
