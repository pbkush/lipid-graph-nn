# ITP_PROVENANCE.md — Vendored Martini 3 ITP files

## Source

| Field | Value |
| --- | --- |
| Repository | `github.com/Martini-Force-Field-Initiative/M3-Lipid-Parameters` |
| Branch | `main` |
| Retrieval date | 2026-05-08 |
| Base URL | `https://raw.githubusercontent.com/Martini-Force-Field-Initiative/M3-Lipid-Parameters/main/ITPs/` |
| License | See individual ITP file headers (Marrink lab, CC or GPL as noted) |

## Scope

All 32 ITP files from the `ITPs/` directory of the M3-Lipid-Parameters repository retrieved at HEAD on 2026-05-08. Coverage:

- **Core force field**: `martini_v3.0.0.itp`, `martini_v3.0.0_ffbonded_v2.itp`
- **Standard phospholipids (v2)**: PC, PE, PS, PA, PG, PI, CL, SM, 2,2-BMP, 3,3-BMP headgroups
- **Ether phospholipids (v2)**: PC, PE, PS, PA, PG headgroups
- **Plasmalogens (v2)**: PC, PE, PS, PA, PG headgroups
- **Sterols**: `martini_v3.0.0_sterols_v1.itp` (CHOL, ergosterol, etc.)
- **Glycerolipids (v2)**: ceramides, mono/di/triglycerides, fatty acids, hydrocarbons, DOTAP
- **Ions**: `martini_v3.0.0_ions_v1.itp` (NA, CL)
- **Solvents**: `martini_v3.0.0_solvents_v1.itp` (W, WF)

## Modifications

None. Files are vendored verbatim from the upstream repository. No local edits.

## Inclusion order in `topol.top`

The order below is used by `system_builder._MARTINI3_ITPS` when finalising the topology. GROMACS requires atomtypes and ffbonded before molecule definitions.

1. `martini_v3.0.0.itp`
2. `martini_v3.0.0_ffbonded_v2.itp`
3. Standard phospholipids: PC → PE → PS → PA → PG → PI → CL → SM → 2,2-BMP → 3,3-BMP
4. Ether phospholipids: PC → PE → PS → PA → PG
5. Plasmalogens: PC → PE → PS → PA → PG
6. `martini_v3.0.0_sterols_v1.itp`
7. `martini_v3.0.0_ceramides_v2.itp`
8. `martini_v3.0.0_monoglycerides_v2.itp`
9. `martini_v3.0.0_diglycerides_v2.itp`
10. `martini_v3.0.0_triglycerides_v2.itp`
11. `martini_v3.0.0_fattyacids_v2.itp`
12. `martini_v3.0.0_hydrocarbons_v2.itp`
13. `martini_v3.0.0_DOTAP_v2.itp`
14. `martini_v3.0.0_ions_v1.itp`
15. `martini_v3.0.0_solvents_v1.itp`

## Note on ffbonded_v2.itp

The v2 lipid parameter files use *named* bond and angle types (e.g. `bond_CC_mid`) defined in `martini_v3.0.0_ffbonded_v2.itp`. This file must appear in `topol.top` before any v2 lipid ITP.

## Future updates

To refresh ITPs: re-download from the same base URL and replace files in this directory. Record the new retrieval date here and verify that the file list and ITP_PROVENANCE.md are still accurate.
