# Target Properties

Reference for the 8 membrane properties that can be predicted by the GNN. Properties are computed once per composition by [scripts/emil/general/calculate_properties.ipynb](../../scripts/emil/general/calculate_properties.ipynb) (wrapping `lipid_gnn.functions_emil.calculate_properties.compute_properties`) and stored as `<composition>.h5` files under `results/properties/`.

## Computation pipeline

- Trajectories loaded with `mdtraj` from `data/membrane_only/<comp>/run/prun.{xtc,gro}`
- Frames `[50:667]` used — the first 50 discard equilibration, the cut at 667 corresponds to ~1 µs of production (frame spacing `dt = 1.5 ns`)
- Box XY dimensions parsed from the last line of the `.gro` file
- `lag_persistence = 50` frames, `lag_diffusivity = 10` frames (hardcoded in the notebook)
- 70 compositions processed, each with ≤ 2 lipid types drawn from a 10-type pool
- Output for each composition: `(mean_dict, raw_dict)` pickled to `.h5`
  - `mean_dict[prop]` — scalar, time-averaged (this is what the GNN is trained against)
  - `raw_dict[prop]` — per-frame time series (used for validation plots only)

## The 8 properties

| Name | Description | Units |
| ---- | ----------- | ----- |
| `lipid_packing` | Lipid number density (inverse area per lipid) | lipids / nm² |
| `thickness` | Mean membrane thickness | Å |
| `thickness_std` | Frame-wise std. dev. of thickness | Å |
| `compressibility` | Area compressibility modulus | Å³ / kT |
| `bending_modulus` | Membrane bending stiffness (κ, from undulation spectrum fit) | kT / Å³ |
| `persistence` | Mean lipid-lipid contact persistence after `lag_persistence` frames | dimensionless |
| `diffusivity` | Mean lipid lateral diffusivity after `lag_diffusivity` frames | Å² |
| `variation` | Coefficient of variation of Voronoi cell areas (inverse homogeneity) | dimensionless |

### Typical scale (POPC100 reference values)

```
lipid_packing   : 3.015 lipids/nm²
thickness       : 38.88 Å
thickness_std   :  2.15 Å
compressibility : 4.886 Å³/kT
bending_modulus : 151.94 kT/Å³
persistence     : 0.078
diffusivity     : 757.4 Å²
variation       : 0.257
```

## Notes

- `bending_modulus` is **permanently dropped** from the target set. It is fit by regressing the undulation spectrum `⟨|h(q)|²⟩` against `q` and was already marked "not used in further analysis" in the original calculation notebook — too noisy/unreliable to serve as a trustworthy training signal. Still computed and stored in the `.h5` files (and exposed as a CLI choice for completeness), but excluded from `active_properties`. The 7-property Tier C set is final.
- `thickness_std` is intra-trajectory variability, not uncertainty across compositions
- Currently-trained property pairs: `lipid_packing` + `thickness` (PR #3 sweep default). The multi-target `out_dim` of `MembranePropertyGNN` equals `len(properties)`; any subset of the 8 can be selected via `prepare_colab_subset.py --properties ...`
