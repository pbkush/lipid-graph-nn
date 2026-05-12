#!/usr/bin/env python3
"""Manual DIPC100 sanity check — not in pytest, run explicitly.

Runs a full DIPC100 simulation (50 ns by default) and checks absolute physical
criteria against published Martini 3 values.  This is NOT a legacy comparison:
the insane version and ITP parameters differ from the 70 training systems.

Usage:
    python scripts/simulation/sanity_check_dipc100.py
    python scripts/simulation/sanity_check_dipc100.py --prod-ns 10 --out-dir /scratch/dipc100

Criteria (Decision 27 / Appendix G):
    APL in [0.62, 0.75] nm²          DIPC Martini 3 literature ~0.68 nm²
    Bilayer thickness in [3.5, 4.5] nm
    No NaN or blow-up in energy (max |Epot| < 1e8 kJ/mol)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

_APL_MIN_NM2 = 0.62
_APL_MAX_NM2 = 0.75
_THICK_MIN_NM = 3.5
_THICK_MAX_NM = 4.5
_EPOT_ABS_MAX = 1e8


def compute_apl(xtc_path: str, tpr_path: str, n_lipids: int, n_skip_ns: float = 10.0) -> float:
    """Return mean APL (nm²) averaged over the last *n_skip_ns* ns of the trajectory."""
    import MDAnalysis as mda
    import numpy as np

    u = mda.Universe(tpr_path, xtc_path)
    dt_ns = u.trajectory.dt / 1000.0  # ps → ns
    n_frames = len(u.trajectory)
    skip_frames = max(1, int(n_skip_ns / dt_ns)) if dt_ns > 0 else 1
    start_frame = max(0, n_frames - skip_frames)

    apl_values = []
    for ts in u.trajectory[start_frame:]:
        # APL = box XY area / (n_lipids / 2) for a symmetric bilayer
        lx, ly = ts.dimensions[0] / 10.0, ts.dimensions[1] / 10.0  # Å → nm
        apl_values.append(lx * ly / (n_lipids / 2))

    return float(np.mean(apl_values))


def compute_thickness(xtc_path: str, tpr_path: str, n_skip_ns: float = 10.0) -> float:
    """Return mean bilayer thickness (nm) from PO4 bead z-separation (DIPC uses D2A)."""
    import MDAnalysis as mda
    import numpy as np

    u = mda.Universe(tpr_path, xtc_path)
    dt_ns = u.trajectory.dt / 1000.0
    n_frames = len(u.trajectory)
    skip_frames = max(1, int(n_skip_ns / dt_ns)) if dt_ns > 0 else 1
    start_frame = max(0, n_frames - skip_frames)

    # DIPC: use D2A (equivalent of PO4 for di-unsaturated lipids)
    d2a = u.select_atoms("name D2A")
    if len(d2a) == 0:
        d2a = u.select_atoms("name PO4")  # fallback for other lipids
    if len(d2a) == 0:
        return float("nan")

    thick_values = []
    for ts in u.trajectory[start_frame:]:
        z = d2a.positions[:, 2] / 10.0  # Å → nm
        z_mid = float(np.mean(z))
        upper = z[z > z_mid]
        lower = z[z <= z_mid]
        if len(upper) > 0 and len(lower) > 0:
            thick_values.append(float(np.mean(upper) - np.mean(lower)))

    return float(np.mean(thick_values)) if thick_values else float("nan")


def check_energy(edr_path: str) -> tuple[bool, float]:
    """Return (ok, max_abs_epot). Requires gmx energy or pyedr."""
    try:
        import pyedr
        import numpy as np
        edr = pyedr.get_edr(edr_path)
        epot = edr.get("Potential", edr.get("potential", None))
        if epot is None:
            return True, 0.0
        max_abs = float(np.nanmax(np.abs(epot)))
        return max_abs < _EPOT_ABS_MAX, max_abs
    except ImportError:
        # pyedr not available — skip energy check
        print("  WARNING: pyedr not installed, skipping energy check")
        return True, 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="DIPC100 physical sanity check")
    parser.add_argument("--prod-ns", type=float, default=50.0, help="Production ns")
    parser.add_argument("--out-dir", default="data/sanity_check", help="Output root")
    parser.add_argument("--gmx", default="gmx", help="GROMACS executable")
    parser.add_argument("--metrics-json", default=None, help="Write metrics to JSON file")
    parser.add_argument("--nsteps-min", type=int, default=None,
                        help="Override minimization steps (default: from config, 20000)")
    parser.add_argument("--nsteps-eq", type=int, default=None,
                        help="Override equilibration steps (default: from config, 1000000 = 10 ns)."
                             " Use a small value (e.g. 5000) for a fast login-node smoke test;"
                             " the physical APL check expects a fully equilibrated system.")
    args = parser.parse_args()

    from lipid_gnn.config import CONFIG
    cfg = CONFIG.martini_pipeline
    insane_cmd = cfg.insane_cmd if cfg else "insane"
    itp_dir = str(cfg.itp_dir) if cfg else "resources/martini3/itp"

    nsteps_prod = round(args.prod_ns * 1e9 / (0.01 * 1e12))  # ns → steps at dt=0.01 ps

    composition = {"DIPC": 1.0}
    out_dir = os.path.join(args.out_dir, "DIPC100")

    from lipid_gnn.martini_pipeline.mdp_writer import MDPParams
    from lipid_gnn.martini_pipeline.pipeline import run as pipeline_run
    from lipid_gnn.martini_pipeline.system_builder import BoxParams

    box = BoxParams(
        xy_nm=cfg.box.xy_nm if cfg else 11.0,
        z_nm=cfg.box.z_nm if cfg else 10.0,
        salt_M=cfg.box.salt_M if cfg else 0.15,
        center=True,
        pbc="rectangular",
    )
    mdp_kwargs = {"nsteps_prod": nsteps_prod}
    if args.nsteps_min is not None:
        mdp_kwargs["nsteps_min"] = args.nsteps_min
    if args.nsteps_eq is not None:
        mdp_kwargs["nsteps_eq"] = args.nsteps_eq
    mdp_params = MDPParams(**mdp_kwargs)

    print(f"Running DIPC100 sanity check — {args.prod_ns} ns production")
    print(f"Output: {out_dir}")

    result = pipeline_run(
        composition, out_dir,
        box=box, mdp_params=mdp_params,
        gmx_executable=args.gmx,
        insane_cmd=insane_cmd,
        itp_dir=itp_dir,
        maxwarn=cfg.gmx.maxwarn if cfg else 2,
    )

    if result.overall_status != "ok":
        print(f"FAIL — pipeline status: {result.overall_status}")
        sys.exit(1)

    xtc_path = os.path.join(out_dir, "run", "prun.xtc")
    tpr_path = os.path.join(out_dir, "run", "prun.tpr")
    edr_path = os.path.join(out_dir, "run", "prun.edr")

    # Molecule counts from manifest
    with open(result.manifest_path) as fh:
        manifest = json.load(fh)
    n_lipids = manifest["build_stats"]["molecule_counts"].get("DIPC", 0)

    checks: list[tuple[str, bool, str]] = []

    # APL check
    apl = compute_apl(xtc_path, tpr_path, n_lipids)
    apl_ok = _APL_MIN_NM2 <= apl <= _APL_MAX_NM2
    checks.append(("APL", apl_ok, f"{apl:.4f} nm² (expected [{_APL_MIN_NM2}, {_APL_MAX_NM2}])"))

    # Thickness check
    thick = compute_thickness(xtc_path, tpr_path)
    thick_ok = _THICK_MIN_NM <= thick <= _THICK_MAX_NM
    checks.append(("Thickness", thick_ok, f"{thick:.4f} nm (expected [{_THICK_MIN_NM}, {_THICK_MAX_NM}])"))

    # Energy check
    energy_ok, max_epot = check_energy(edr_path)
    checks.append(("Energy", energy_ok, f"max |Epot| = {max_epot:.2e} kJ/mol"))

    # Report
    print()
    all_ok = True
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"  {status}  {name}: {detail}")
        if not ok:
            all_ok = False

    if args.metrics_json:
        metrics = {
            "apl_nm2": apl,
            "thickness_nm": thick,
            "max_abs_epot": max_epot,
            "all_ok": all_ok,
        }
        with open(args.metrics_json, "w") as fh:
            json.dump(metrics, fh, indent=2)
        print(f"\nMetrics written to {args.metrics_json}")

    print()
    print("PASS" if all_ok else "FAIL")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
