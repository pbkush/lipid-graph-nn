"""Build an initial Martini 3 bilayer using the insane membrane builder."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

from lipid_gnn.martini_pipeline import INSANE_CMD, MARTINI3_ITP_DIR

_INSANE_EXTRA_LIPIDS_DAT = os.path.join(
    os.path.dirname(__file__), "templates", "insane_extra_lipids.dat"
)

_MARTINI3_ITPS = (
    # Core force field — must come first
    "martini_v3.0.0.itp",
    "martini_v3.0.0_ffbonded_v2.itp",
    # Standard phospholipids (v2, by headgroup)
    "martini_v3.0.0_phospholipids_PC_v2.itp",
    "martini_v3.0.0_phospholipids_PE_v2.itp",
    "martini_v3.0.0_phospholipids_PS_v2.itp",
    "martini_v3.0.0_phospholipids_PA_v2.itp",
    "martini_v3.0.0_phospholipids_PG_v2.itp",
    "martini_v3.0.0_phospholipids_PI_v2.itp",
    "martini_v3.0.0_phospholipids_CL_v2.itp",
    "martini_v3.0.0_phospholipids_SM_v2.itp",
    "martini_v3.0.0_phospholipids_2,2-BMP_v2.itp",
    "martini_v3.0.0_phospholipids_3,3-BMP_v2.itp",
    # Ether phospholipids (v2)
    "martini_v3.0.0_etherphospholipids_PC_v2.itp",
    "martini_v3.0.0_etherphospholipids_PE_v2.itp",
    "martini_v3.0.0_etherphospholipids_PS_v2.itp",
    "martini_v3.0.0_etherphospholipids_PA_v2.itp",
    "martini_v3.0.0_etherphospholipids_PG_v2.itp",
    # Plasmalogens (v2)
    "martini_v3.0.0_plasmalogens_PC_v2.itp",
    "martini_v3.0.0_plasmalogens_PE_v2.itp",
    "martini_v3.0.0_plasmalogens_PS_v2.itp",
    "martini_v3.0.0_plasmalogens_PA_v2.itp",
    "martini_v3.0.0_plasmalogens_PG_v2.itp",
    # Sterols, glycerolipids, other
    "martini_v3.0.0_sterols_v1.itp",
    "martini_v3.0.0_ceramides_v2.itp",
    "martini_v3.0.0_monoglycerides_v2.itp",
    "martini_v3.0.0_diglycerides_v2.itp",
    "martini_v3.0.0_triglycerides_v2.itp",
    "martini_v3.0.0_fattyacids_v2.itp",
    "martini_v3.0.0_hydrocarbons_v2.itp",
    "martini_v3.0.0_DOTAP_v2.itp",
    # Ions and solvents — last
    "martini_v3.0.0_ions_v1.itp",
    "martini_v3.0.0_solvents_v1.itp",
)

_MARTINI_INCLUDE_RE = re.compile(r'^\s*#include\s+"martini[^"]*\.itp"\s*$', re.IGNORECASE)


@dataclass(frozen=True)
class BoxParams:
    xy_nm: float = 11.0
    z_nm: float = 10.0
    salt_M: float = 0.15
    water_type: str = "W"
    charge_mode: str = "auto"
    center: bool = True
    pbc: str = "rectangular"


@dataclass(frozen=True)
class BuildResult:
    out_dir: str
    gro_path: str
    top_path: str
    ndx_path: str | None
    log_path: str
    molecule_counts: dict
    n_membrane_beads: int
    n_solvent_atoms: int
    total_atoms: int
    walltime_s: float
    insane_cmd: list


def build_command(
    composition: dict,
    box: BoxParams,
    out_gro: str,
    out_top: str,
    *,
    insane_cmd: str = INSANE_CMD,
) -> list:
    """Return the argv list for insane. Pure function, no I/O."""
    ratios = _fractions_to_ratios(composition)
    argv = [insane_cmd,
            "-o", out_gro,
            "-p", out_top,
            "-x", str(box.xy_nm),
            "-y", str(box.xy_nm),
            "-z", str(box.z_nm),
            "-pbc", box.pbc]   # always explicit — insane defaults to hexagonal
    # insane <1.2.0 ships a lipids.dat without M3.CHOL; supply our copy so
    # the M3 sterol resolves regardless of the installed insane version.
    argv += ["-dat", _INSANE_EXTRA_LIPIDS_DAT]
    for name, ratio in ratios.items():
        keyword = _insane_keyword(name)
        argv += ["-l", f"{keyword}:{ratio}"]
    argv += _alname_args(ratios)
    argv += ["-sol", box.water_type, "-salt", str(box.salt_M), "-charge", box.charge_mode]
    if box.center:
        argv.append("-center")
    return argv


def build_system(
    composition: dict,
    out_dir: str,
    *,
    box: BoxParams = BoxParams(),
    insane_cmd: str = INSANE_CMD,
    itp_dir: str = MARTINI3_ITP_DIR,
    gmx_executable: str = "gmx",
    make_ndx_script: str = "q\n",
) -> BuildResult:
    """Build bilayer, finalise topology, stage ITPs, generate index.ndx."""
    _preflight_check_itps(itp_dir)
    os.makedirs(out_dir, exist_ok=True)

    gro_path = os.path.join(out_dir, "run.gro")
    top_path = os.path.join(out_dir, "topol.top")
    log_path = os.path.join(out_dir, "insane.log")

    argv = build_command(composition, box, gro_path, top_path, insane_cmd=insane_cmd)
    t0 = time.monotonic()
    result = subprocess.run(argv, capture_output=True, text=True)
    walltime_s = time.monotonic() - t0

    with open(log_path, "w") as fh:
        fh.write(result.stdout)
        if result.stderr:
            fh.write("\n--- stderr ---\n")
            fh.write(result.stderr)

    if result.returncode != 0:
        raise RuntimeError(
            f"insane exited {result.returncode}:\n{result.stderr[-500:]}"
        )

    _finalise_topology(top_path)
    _normalise_ion_names(top_path, gro_path)
    _stage_itps(itp_dir, out_dir)

    ndx_path = _make_ndx(gro_path, out_dir, gmx_executable, make_ndx_script)

    molecule_counts = _parse_molecule_counts(top_path)
    total_atoms = _parse_gro_atom_count(gro_path)
    n_membrane_beads, n_solvent_atoms = _split_membrane_solvent(
        molecule_counts, total_atoms
    )

    return BuildResult(
        out_dir=out_dir,
        gro_path=gro_path,
        top_path=top_path,
        ndx_path=ndx_path,
        log_path=log_path,
        molecule_counts=molecule_counts,
        n_membrane_beads=n_membrane_beads,
        n_solvent_atoms=n_solvent_atoms,
        total_atoms=total_atoms,
        walltime_s=walltime_s,
        insane_cmd=argv,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fractions_to_ratios(composition: dict) -> dict:
    total = sum(composition.values())
    if total == 0:
        raise ValueError("composition must have non-zero total")
    scale = 100.0 / total
    ratios = {name: max(1, round(frac * scale)) for name, frac in composition.items()}
    return ratios


def _insane_keyword(name: str) -> str:
    try:
        from lipid_gnn.martini_pipeline.lipid_registry import get_lipid
        return get_lipid(name).insane_keyword
    except (KeyError, Exception):
        return name


def _alname_args(ratios: dict) -> list:
    """Build -alname/-alhead/-allink/-altail/-alcharge flag groups for any
    composition lipid whose registry entry has an inline insane spec.

    Lipids not in insane's packaged lipids.dat (e.g. DLPC, DOPE, DPPE in the
    M3.* namespace) must be defined this way so the bead layout matches the
    v2 ITP that grompp will read.  Order across the four lists is paired by
    index inside insane's add_from_def.
    """
    try:
        from lipid_gnn.martini_pipeline.lipid_registry import get_lipid
    except Exception:
        return []
    args: list = []
    for name in ratios:
        try:
            entry = get_lipid(name)
        except KeyError:
            continue
        if not entry.needs_alname:
            continue
        args += [
            "-alname",   entry.insane_keyword,
            "-alhead",   entry.insane_alhead,
            "-allink",   entry.insane_allink,
            "-altail",   entry.insane_altail,
            "-alcharge", str(entry.insane_alcharge),
        ]
    return args


def _preflight_check_itps(itp_dir: str) -> None:
    for itp in _MARTINI3_ITPS:
        path = os.path.join(itp_dir, itp)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Martini 3 ITP not found: {path}")


def _finalise_topology(top_path: str) -> None:
    with open(top_path) as fh:
        lines = fh.readlines()

    match_indices = [i for i, ln in enumerate(lines) if _MARTINI_INCLUDE_RE.match(ln)]
    if len(match_indices) != 1:
        raise ValueError(
            f"topology finalisation: expected exactly 1 martini include line in "
            f"{top_path}, found {len(match_indices)}"
        )

    replacement = "".join(
        f'#include "toppar/{itp}"\n' for itp in _MARTINI3_ITPS
    )
    idx = match_indices[0]
    lines[idx] = replacement

    with open(top_path, "w") as fh:
        fh.writelines(lines)


_ION_RENAME = {"NA+": "NA", "CL-": "CL"}


def _normalise_ion_names(top_path: str, gro_path: str) -> None:
    """Rewrite charged-suffix ion names (`NA+`, `CL-`) to `NA`, `CL`.

    Some `insane` builds emit `NA+`/`CL-` in topol.top and run.gro, which the
    v1 ions ITP (defining moleculetypes `NA` and `CL`) does not recognise.
    Normalising both spellings keeps grompp, make_ndx, and the legacy MDAnalysis
    selectors (`name NA`, `name CL`) consistent across new and legacy systems.
    """
    # ---- topol.top: rewrite [molecules] section only ----
    with open(top_path) as fh:
        lines = fh.readlines()
    in_molecules = False
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and "molecules" in stripped.lower():
            in_molecules = True
            continue
        if in_molecules and stripped.startswith("["):
            in_molecules = False
        if in_molecules and stripped and not stripped.startswith(";"):
            parts = line.split(None, 1)
            if len(parts) == 2 and parts[0] in _ION_RENAME:
                lines[i] = line.replace(parts[0], _ION_RENAME[parts[0]], 1)
    with open(top_path, "w") as fh:
        fh.writelines(lines)

    # ---- run.gro: rewrite residue and atom name fields ----
    if not os.path.isfile(gro_path):
        return
    with open(gro_path) as fh:
        gro_lines = fh.readlines()
    if len(gro_lines) < 3:
        return
    try:
        n_atoms = int(gro_lines[1].strip())
    except ValueError:
        return
    for i in range(2, min(2 + n_atoms, len(gro_lines))):
        ln = gro_lines[i]
        if len(ln) < 15:
            continue
        resname = ln[5:10].strip()
        atomname = ln[10:15].strip()
        new_res = _ION_RENAME.get(resname)
        new_atom = _ION_RENAME.get(atomname)
        if new_res is not None:
            ln = ln[:5] + new_res.ljust(5) + ln[10:]
        if new_atom is not None:
            ln = ln[:10] + new_atom.rjust(5) + ln[15:]
        gro_lines[i] = ln
    with open(gro_path, "w") as fh:
        fh.writelines(gro_lines)


def _stage_itps(itp_dir: str, out_dir: str) -> None:
    toppar_dir = os.path.join(out_dir, "toppar")
    os.makedirs(toppar_dir, exist_ok=True)
    for itp in _MARTINI3_ITPS:
        shutil.copy(os.path.join(itp_dir, itp), os.path.join(toppar_dir, itp))


def _make_ndx(gro_path: str, out_dir: str, gmx_executable: str, make_ndx_input: str = "q\n") -> str | None:
    if not shutil.which(gmx_executable):
        print(f"WARNING: {gmx_executable!r} not found — skipping index.ndx generation")
        return None
    ndx_path = os.path.join(out_dir, "index.ndx")
    result = subprocess.run(
        [gmx_executable, "make_ndx", "-f", gro_path, "-o", ndx_path],
        input=make_ndx_input,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"gmx make_ndx failed (exit {result.returncode}):\n{result.stderr[-500:]}"
        )
    return ndx_path


def _parse_molecule_counts(top_path: str) -> dict:
    counts: dict = {}
    in_molecules = False
    with open(top_path) as fh:
        for line in fh:
            stripped = line.strip()
            if stripped.startswith("[") and "molecules" in stripped.lower():
                in_molecules = True
                continue
            if in_molecules:
                if stripped.startswith("["):
                    break
                if stripped and not stripped.startswith(";"):
                    # Strip inline comments (insane appends `; Defined in '...'`
                    # for lipids loaded via -dat) before splitting NAME COUNT.
                    payload = stripped.split(";", 1)[0].strip()
                    parts = payload.split()
                    if len(parts) == 2:
                        counts[parts[0]] = counts.get(parts[0], 0) + int(parts[1])
    return counts


def _parse_gro_atom_count(gro_path: str) -> int:
    with open(gro_path) as fh:
        fh.readline()
        return int(fh.readline().strip())


_SOLVENT_RESIDUES = frozenset({"W", "WF", "NA", "CL", "NA+", "CL-", "ION"})


def _split_membrane_solvent(molecule_counts: dict, total_atoms: int) -> tuple:
    from lipid_gnn.martini_pipeline.lipid_registry import LIPID_REGISTRY
    membrane = 0
    for name, count in molecule_counts.items():
        if name in LIPID_REGISTRY:
            membrane += count * len(LIPID_REGISTRY[name].beads)
    return membrane, total_atoms - membrane
