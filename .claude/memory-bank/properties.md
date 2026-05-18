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
| `bending_modulus` | Membrane bending stiffness (κ, from undulation spectrum fit) | kBT (legacy: dimensionless × 10³) |
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
- **`bending_modulus` units changed 2026-05-18 (legacy=False).** The historical pipeline returned `κ_fit * 1000` tagged "kT/Å³" — dimensionally wrong (κ is energy, not energy/volume) and ~10× the physical value due to a missing grid-area normalisation. The new (bug-fixed) path returns κ in kBT, after multiplying the raw fit by `1 / (Δx · Δy)` to undo the discrete-FFT spacing factor implicit in the Helfrich relation `⟨|H_k|²⟩_ortho = kBT / (κ q⁴ Δx Δy)`. For typical Martini POPC, the new value is ~15 kBT (was ~152 in legacy "kT/Å³" units). `legacy=True` preserves the historical scaling bit-for-bit for label re-derivation. Top-line numbers in the POPC100 table above were generated with the legacy convention; regenerate with `--legacy` to reproduce them.
- `thickness_std` is intra-trajectory variability, not uncertainty across compositions
- Currently-trained property pairs: `lipid_packing` + `thickness` (PR #3 sweep default). The multi-target `out_dim` of `MembranePropertyGNN` equals `len(properties)`; any subset of the 8 can be selected via `prepare_colab_subset.py --properties ...`

## Bug fixes layered on top of cleanup-plan §2 (added 2026-05-18)

Five additional issues found during a second-pass audit of `lipid_gnn/properties.py` after the initial rewrite. All landed in `legacy=False` mode; `legacy=True` is bit-for-bit unchanged.

1. **Bending-modulus normalisation.** Continuous Helfrich is `⟨|h(q)|²⟩ = kBT / (κ q⁴ A)` with `h(q) = (1/A) ∫ h(r) e^{−iq·r} d²r`. The discrete `np.fft.fft2(..., norm="ortho")` version is `⟨|H_k|²⟩_ortho = kBT / (κ q⁴ Δx Δy)`; the raw `curve_fit` therefore returns `κ_fit = κ · Δx Δy`. Function now divides by `Δx · Δy` and returns κ in kBT. The dimensionally-wrong `* 1000.0` "historical scaling" was removed from `legacy=False` (preserved in `legacy=True` by undoing the new normalisation, see orchestrator).
2. **FFT axis labelling.** `GridSpec.build()` now uses `indexing="ij"` so `Z.shape == (n_frames, nx, ny)` unambiguously. Previously `indexing="xy"` (the default) silently transposed axes; for non-square boxes the `fftfreq(nx, d=Lx/nx)` pair would mix x-extent with y-grid count. No effect on legacy data (all Martini training boxes are square), but the corrected code is now safe for arbitrary rectangular grids.
3. **FFT spacing off-by-one.** `fftfreq(n, d=...)` now uses the actual grid step `step_x = X[1,0] − X[0,0]` rather than `Lx/n` (which is `(n−1)/n · step` for half-open `arange` grids). Negligible numerical effect at large n but strictly correct.
4. **Frame-mask propagation to thickness series.** `_height_fields` drops frames whose interpolation NaN'd; the kept-frame series previously had a different length from the raw trajectory and from the per-frame series for `lipid_packing` / `variation` / `persistence` / `diffusivity` (cleanup-plan bug #7). `thickness_summary` now takes an optional `frame_mask` and pads dropped slots with NaN so all series are co-indexable. Means use `np.nanmean`.
5. **Voronoi CV NaN propagation.** `_voronoi_cv` returns `float("nan")` when no usable cells were extracted (empty leaflet / degenerate point set), instead of `0.0` which biased the frame-mean toward zero. `compute_variation` uses `np.nanmean` (warning suppressed for the all-NaN-slice case where NaN is the intended output).
6. **Inhomogeneity formula simplified.** `(flat − mean).std()² · 100` is mathematically `flat.std()² · 100`; rewritten as `kept_std ** 2` (Å²). No numerical change.
7. **`compute_diffusivity` legacy docstring** now credits bug #1 reproduction (always-lower-leaflet), matching the `compute_persistence` legacy docs.

Tests covering the new fixes: `test_physical_kappa_recovered_from_helfrich_field` (synthetic Gaussian random field with target κ; recovery within 25 %), `test_non_square_grid_uses_correct_axis_spacing`, `test_frame_mask_pads_series_with_nan`, `test_variation_nan_when_voronoi_fails_everywhere`.
