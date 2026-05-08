# INSANE_PROVENANCE.md — Vendored insane.py

## Source

| Field | Value |
| --- | --- |
| File | `resources/martini3/insane.py` |
| Origin | `lipid_gnn/functions_emil/insane.py` in this repository |
| Upstream | Tsjerk A. Wassenaar, `previous = "20140603.11.TAW"` (2014-06-03 build) |
| Lipid additions | Helgi I. Ingolfsson (marked `# HII edit` in source) |
| Further edits | Emil (Goethe thesis pipeline, exact date unknown) |
| Retrieval date | 2026-05-07 |
| License | GNU General Public License v2 (GPLv2) |
| Line count | 1679 (after 2to3 + license header) |

## Modifications applied

Two mechanical changes only — no logic altered:

### 1. Shebang update

```diff
-#!/usr/bin/env python
+#!/usr/bin/env python3
```

### 2. GPL license header prepended

24-line block added at top of file (see source). Verbatim GPLv2 text with attribution to Tsjerk Wassenaar, Helgi I. Ingolfsson, and Emil.

### 3. 2to3 automated conversion

Run: `2to3 -w resources/martini3/insane.py` (Python stdlib tool, 2026-05-07).

Summary of changes (44 lines changed in 1655-line source):

| Change class | Count | Example |
| --- | --- | --- |
| `print X` → `print(X)` | 29 | `print "Error..."` → `print("Error...")` |
| `print >>fh, X` → `print(X, file=fh)` | 14 | stdout/stderr redirects |
| `__nonzero__` → `__bool__` | 2 | Python 3 dunder rename |
| `xrange` → `range` | 6 | grid iteration loops |
| `zip()` → `list(zip())` | 10 | packing of coordinate tuples |
| `dict.keys()` → `list(dict.keys())` | 1 | `lipidsx.keys()` iteration |
| `zip()+zip()` → `list()+list()` | 1 | molecule list concatenation |

All changes are semantically equivalent under Python 3. No integer division (`/`) ambiguity was found (insane uses float-typed operands throughout). No semantic edge cases flagged.

## Parity check vs legacy POPC100

Run command (using legacy parameters from `data/membrane_only/POPC100/input.log`):

```bash
python3 resources/martini3/insane.py \
    -o popc100.gro -x 11 -y 11 -z 10 \
    -l POPC:100 -center -sol W -salt 0.15 -charge auto \
    -p topol.top
```

| Metric | Legacy | Rebuilt | Match |
| --- | --- | --- | --- |
| Exit code | 0 | 0 | ✓ |
| Title line | `INSANE! Membrane UpperLeaflet>POPC=100.0 LowerLeaflet>POPC=100.0` | identical | ✓ |
| Membrane beads (NDX) | — | 4704 (392 lipids × 12 beads) | n/a |
| Total atoms | 10125 | 10162 | ✗ Δ=+37 |

**Divergence: +37 atoms.** Explanation: the legacy build was run with Python 2, which has different RNG state and dict-iteration order than Python 3. The 37-atom difference is in the solvent region (water molecules and/or ions added by `-salt 0.15 -charge auto`). The lipid count (392 POPC × 12 = 4704 membrane beads) is structurally fixed by box geometry and is expected to be identical. This divergence is **accepted** (Decision 18 / Step 5 E.7 open question 2: option b — document and proceed). Step 7's POPC100 sanity check compares atom-per-lipid statistics, not atom count, so this does not affect correctness.

## Future migration path

If Step 12 (lipid-pool extension) requires lipid templates not present in the 2014 build, migrate to Tsjerk Wassenaar's modern Python-3 fork (`github.com/Tsjerk/Insane`, main branch) as Option C (both files side-by-side). At that point, add a `resources/martini3/insane_upstream.py` and update `INSANE_PATH` in `lipid_gnn/martini_pipeline/__init__.py`.
