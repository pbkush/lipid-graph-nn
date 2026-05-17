"""Build (resname, atom) → bead_type and per-lipid bond-parameter JSONs from M3 ITPs.

Reads:
- `resources/martini3/itp/*.itp` (M3 lipid library, vendored)
- `resources/martini3/itp/martini_v3.0.0_ffbonded_v2.itp` (named bond defines)
- `emil_extra/simulation_parameters/toppar/martini_v3.0_sterols_v1.0.itp`
- existing `resources/martini_ff_node_mapping.json` and `..._edge_params.json`
  (preserves legacy non-lipid entries — nucleobases, small molecules, ions, etc.)

Writes (in-place, with backup):
- `resources/martini_ff_node_mapping.json`
- `resources/martini_ff_edge_params.json`

Also audits `resources/martini_ff_params.json` against the union of bead types
used by the new node mapping; reports any missing bead types.

Usage:
    python scripts/python/build_ff_mappings_from_itps.py [--dry-run] [--scope {bilayer,full}]

Defaults to `--scope bilayer` which excludes nucleobases / sugars / small
molecules / ions / solvents — i.e. the bilayer-forming subset analysed in
`scripts/notebooks/analyze_m3_lipidome.py`.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ITP_DIR = REPO / "resources" / "martini3" / "itp"
STEROLS_ITP = REPO / "emil_extra" / "simulation_parameters" / "toppar" / "martini_v3.0_sterols_v1.0.itp"
FFBONDED = ITP_DIR / "martini_v3.0.0_ffbonded_v2.itp"

NODE_MAP_PATH = REPO / "resources" / "martini_ff_node_mapping.json"
EDGE_PARAMS_PATH = REPO / "resources" / "martini_ff_edge_params.json"
FF_PARAMS_PATH = REPO / "resources" / "martini_ff_params.json"

# Filename-substring filters for the bilayer scope. Anything matching these
# substrings is excluded.
NON_LIPID_SUBSTRINGS = (
    "ffbonded", "ions", "solvents", "fattyacids", "hydrocarbons",
    "small_molecules", "nucleobases", "sugars",
)


def parse_ffbonded(path: Path) -> dict[str, tuple[float, float]]:
    """Return mapping of bondname → (length_nm, force_constant)."""
    defs: dict[str, tuple[float, float]] = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*#define\s+(b_\S+)\s+\d+\s+([0-9.eE+\-]+)\s+([0-9.eE+\-]+)", line)
        if m:
            name, length, fc = m.group(1), float(m.group(2)), float(m.group(3))
            defs[name] = (length, fc)
    return defs


def parse_itp_moleculetypes(path: Path) -> list[dict]:
    """Parse every [moleculetype] in an ITP. Returns one dict per molecule."""
    text = path.read_text()
    lines = text.splitlines()
    out: list[dict] = []
    i = 0
    while i < len(lines):
        stripped = lines[i].strip().lower()
        if stripped.startswith("[") and "moleculetype" in stripped:
            j = i + 1
            molname = None
            while j < len(lines):
                s = lines[j].strip()
                if s and not s.startswith(";"):
                    molname = s.split()[0]
                    j += 1
                    break
                j += 1
            atoms: list[dict] = []
            bonds: list[tuple[int, int, str | None, float | None, float | None]] = []
            section: str | None = None
            k = j
            while k < len(lines):
                s = lines[k].strip()
                low = s.lower()
                if low.startswith("[") and "moleculetype" in low:
                    break
                if low.startswith("["):
                    if "atoms" in low:
                        section = "atoms"
                    elif low.startswith("[bonds") or low.startswith("[ bonds"):
                        section = "bonds"
                    else:
                        section = "other"
                    k += 1
                    continue
                if not s or s.startswith(";"):
                    k += 1
                    continue
                toks = s.split()
                if section == "atoms" and len(toks) >= 7:
                    try:
                        atoms.append({
                            "id":      int(toks[0]),
                            "type":    toks[1],
                            "residue": toks[3],
                            "atom":    toks[4],
                            "charge":  float(toks[6]),
                        })
                    except (ValueError, IndexError):
                        pass
                elif section == "bonds" and len(toks) >= 3:
                    try:
                        i_id = int(toks[0])
                        j_id = int(toks[1])
                    except ValueError:
                        k += 1
                        continue
                    rest = toks[2:]
                    # inline numeric form: funct length fc, with optional comment
                    if len(rest) >= 3:
                        try:
                            length = float(rest[1])
                            fc = float(rest[2])
                            bonds.append((i_id, j_id, None, length, fc))
                            k += 1
                            continue
                        except ValueError:
                            pass
                    # named form: bond name
                    bonds.append((i_id, j_id, rest[0], None, None))
                k += 1
            if molname and atoms:
                out.append({
                    "molname": molname,
                    "source":  path.name,
                    "atoms":   atoms,
                    "bonds":   bonds,
                })
            i = k
            continue
        i += 1
    return out


def build_mappings(
    bondtypes: dict[str, tuple[float, float]],
    molecules: list[dict],
) -> tuple[dict, dict, list[str]]:
    """Return (node_mapping, edge_params, unresolved_bond_warnings)."""
    node_mapping: dict[str, dict[str, str]] = {}
    edge_params: dict[str, dict[str, dict[str, float]]] = {}
    unresolved: list[str] = []

    for mol in molecules:
        molname = mol["molname"]
        atom_by_id = {a["id"]: a for a in mol["atoms"]}

        atom_to_bead = {a["atom"]: a["type"] for a in mol["atoms"]}
        node_mapping[molname] = atom_to_bead

        bonds_resolved: dict[str, dict[str, float]] = {}
        for i_id, j_id, name, length, fc in mol["bonds"]:
            if i_id not in atom_by_id or j_id not in atom_by_id:
                continue
            key = f"{atom_by_id[i_id]['atom']}-{atom_by_id[j_id]['atom']}"
            if length is not None and fc is not None:
                bonds_resolved[key] = {"length": length, "force_constant": fc}
            elif name is not None:
                resolved = bondtypes.get(name)
                if resolved is None:
                    unresolved.append(f"{molname}: bond {key} → unknown name '{name}'")
                    continue
                bonds_resolved[key] = {
                    "length":          resolved[0],
                    "force_constant":  resolved[1],
                }
        edge_params[molname] = bonds_resolved

    return node_mapping, edge_params, unresolved


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and report changes without writing JSONs.")
    ap.add_argument("--scope", choices=("bilayer", "full"), default="bilayer",
                    help="bilayer (default): exclude nucleobases/sugars/small_molecules. "
                         "full: include every ITP in resources/martini3/itp/.")
    ap.add_argument("--overwrite-existing", action="store_true",
                    help="Overwrite legacy entries that overlap with the M3 ITPs. "
                         "DEFAULT IS ADD-ONLY — legacy entries (including the 10 "
                         "training-pool lipids) are preserved so existing chunks "
                         "and trained weights stay valid. Set this flag to migrate "
                         "to the modern M3-Lipid-Parameters values, which will "
                         "invalidate existing data/processed/ chunks and require "
                         "re-preprocessing + retraining.")
    args = ap.parse_args()

    # 1. Collect ITP files
    itp_files = sorted(ITP_DIR.glob("martini_v3.0.0_*.itp"))
    if STEROLS_ITP.exists():
        itp_files.append(STEROLS_ITP)
    if args.scope == "bilayer":
        itp_files = [
            f for f in itp_files
            if not any(s in f.name.lower() for s in NON_LIPID_SUBSTRINGS)
        ]
    else:
        itp_files = [f for f in itp_files if "ffbonded" not in f.name.lower()]
    print(f"Scope: {args.scope} — parsing {len(itp_files)} ITP files")

    # 2. Parse ffbonded named bondtypes
    bondtypes = parse_ffbonded(FFBONDED)
    print(f"ffbonded bondtypes: {len(bondtypes)}")

    # 3. Parse every moleculetype
    molecules: list[dict] = []
    for f in itp_files:
        molecules.extend(parse_itp_moleculetypes(f))
    print(f"Moleculetypes parsed: {len(molecules)}")

    # 4. Build mappings
    new_nodes, new_edges, unresolved = build_mappings(bondtypes, molecules)
    print(f"New node_mapping entries: {len(new_nodes)}")
    print(f"New edge_params entries:  {len(new_edges)}")
    if unresolved:
        print(f"Unresolved bond names: {len(unresolved)} (showing first 5)")
        for u in unresolved[:5]:
            print(f"  {u}")

    # 5. Merge with legacy entries (preserve anything not in M3 scope)
    legacy_nodes = json.loads(NODE_MAP_PATH.read_text()) if NODE_MAP_PATH.exists() else {}
    legacy_edges = json.loads(EDGE_PARAMS_PATH.read_text()) if EDGE_PARAMS_PATH.exists() else {}

    added_nodes      = sorted(set(new_nodes) - set(legacy_nodes))
    overlapping      = sorted(set(new_nodes) & set(legacy_nodes))
    preserved_legacy = sorted(set(legacy_nodes) - set(new_nodes))

    if args.overwrite_existing:
        merged_nodes = {**legacy_nodes, **new_nodes}
        merged_edges = {**legacy_edges, **new_edges}
        overwritten = overlapping
    else:
        # Add-only: legacy entries win on overlap. New M3 entries are only added
        # for moleculetypes the legacy JSON has no entry for.
        merged_nodes = {**legacy_nodes}
        merged_edges = {**legacy_edges}
        for k in added_nodes:
            merged_nodes[k] = new_nodes[k]
            merged_edges[k] = new_edges[k]
        overwritten = []

    print()
    print(f"Diff vs existing JSONs ({'OVERWRITE' if args.overwrite_existing else 'ADD-ONLY'} mode):")
    print(f"  added (new in M3 mapping):           {len(added_nodes)}")
    print(f"  overlapping with legacy (handling):  {len(overlapping)} → "
          f"{'rebuilt from M3' if args.overwrite_existing else 'legacy kept'}")
    print(f"  legacy preserved (not in M3):        {len(preserved_legacy)}")
    if added_nodes:
        print(f"  first 10 added: {added_nodes[:10]}")
    if preserved_legacy:
        print(f"  first 10 legacy preserved: {preserved_legacy[:10]}")
    if overlapping and not args.overwrite_existing:
        # Audit: which overlapping entries actually differ between legacy and new?
        node_differs = [
            k for k in overlapping
            if dict(legacy_nodes[k]) != dict(new_nodes[k])
        ]
        print(f"  overlapping entries that DIFFER from M3 v2 (kept as legacy): "
              f"{len(node_differs)} / {len(overlapping)}")
        if node_differs:
            print(f"  first 10: {node_differs[:10]}")
            print(f"  → run with --overwrite-existing to migrate to modern M3 values "
                  f"(invalidates existing chunks).")

    # 6. Audit bead types vs ff_params.json
    ff_params = json.loads(FF_PARAMS_PATH.read_text())
    used_beads = Counter()
    for mol_map in new_nodes.values():
        used_beads.update(mol_map.values())
    missing_beads = sorted(b for b in used_beads if b not in ff_params)
    print()
    print(f"Bead-type coverage in ff_params.json:")
    print(f"  unique beads used by new mapping: {len(used_beads)}")
    print(f"  missing from ff_params.json:      {len(missing_beads)}")
    if missing_beads:
        print(f"  missing list: {missing_beads}")

    if args.dry_run:
        print()
        print("--dry-run set — no files written.")
        return

    # 7. Backup + write
    for path, data in [(NODE_MAP_PATH, merged_nodes), (EDGE_PARAMS_PATH, merged_edges)]:
        backup = path.with_suffix(path.suffix + ".bak")
        if path.exists():
            shutil.copy2(path, backup)
            print(f"Backed up {path.name} → {backup.name}")
        # Preserve insertion order: per-lipid atom dicts must keep head-to-tail
        # order from the ITP (a registry test elsewhere relies on this).
        path.write_text(json.dumps(data, indent=4, sort_keys=False))
        print(f"Wrote {path}  ({len(data)} entries)")


if __name__ == "__main__":
    main()
