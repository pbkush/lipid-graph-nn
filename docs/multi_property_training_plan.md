# Multi-Property Training Plan

## Goal

Extend the property-regression track from the current 2-target baseline (`lipid_packing` + `thickness`) to all 8 available targets. The aim is not only better coverage but richer supervision signal for the shared backbone, improving the quality of the learned membrane embedding.

## Key fact: no re-preprocessing needed

Chunks were preprocessed with all 8 properties (uploaded to Google Drive 2026-04-22). Each graph stores `y` as a `[1, 8]` vector (`dataset.py:176`). Changing the training-time property set requires only column-slicing `y` — no chunk rebuild. The test `test_preprocess_and_save_all_8_properties` verifies this invariant.

Property column order in `y` (fixed at preprocessing time):

```
0: lipid_packing
1: thickness
2: thickness_std
3: compressibility
4: bending_modulus
5: persistence
6: diffusivity
7: variation
```

## Pitfalls

1. **Architecture-bound targets inflate the HP-tuning floor.** `bending_modulus` and `compressibility` require long-wavelength receptive fields that the current 11 Å spatial cutoff cannot provide (see [docs/efa_spatial_layer_future.md](efa_spatial_layer_future.md)). They will plateau at low R² regardless of HP changes, polluting the aggregate val-MSE signal used to compare sweep runs.

2. **`bending_modulus` carries label noise.** Fit from undulation-spectrum regression — noisier than the other 7 properties ([.claude/memory-bank/properties.md](.claude/../.claude/memory-bank/properties.md)). Equal-weight MSE lets its irreducible noise contribute a constant gradient floor.

3. **Negative transfer through the shared MLP trunk.** One final linear layer maps `hidden_dim/2 → out_dim`. Eight targets pulling one trunk can have conflicting gradients without explicit per-target weighting.

4. **HP-tuning signal becomes ambiguous.** A change to `hidden_dim` or `num_layers` can help 3 properties and hurt 5, netting to noise in the aggregate val-MSE. Per-property R² is already logged to W&B (`val/r2_{property}`) — use these, not the mean, to judge HP changes.

5. **Capacity assumption from the 2-target regime.** `FIXED` HPs (`hidden_dim=64`) were chosen for 2 targets. Jumping to 8 without adjusting capacity conflates "joint training hurts" with "model underfit for this many outputs".

## Recommended approach: tiered property adoption

**The key**: train in tiers by column-slicing `y` at runtime. The tier boundary is a one-line config change, not a preprocessing step.

### Tier A — local geometric, high SNR (start here, HP-tune here)

`lipid_packing`, `thickness`, `variation`, `thickness_std`

All reachable with the 11 Å spatial cutoff. Clean signal makes sweeps diagnostic. HP changes have visible, attributable effects.

Steps:

1. Confirm 2-property baseline (0.138 MSE) reproduces on the new all-8 chunks with `y[:, :2]`.
2. Add `variation` and `thickness_std` to reach 4 targets. Check per-property R² in W&B.
3. Run HP sweep (`hidden_dim`, `num_layers`, `lr`) with the 4-target config. This is the primary tuning phase.

### Tier B — local dynamical, lower SNR

Add `persistence`, `diffusivity`.

Check: do Tier A per-property R² values drop by more than ~5% relative? If yes → negative transfer. Remedy: adopt homoscedastic uncertainty weighting (Kendall & Gal 2017) — one learned `log_var` per property, training loss becomes:

```
Σᵢ  MSEᵢ / exp(log_varᵢ)  +  log_varᵢ
```

~10-line change to the `criterion` call in `run_sweep.py:151` and the notebook's `train_one_run`.

### Tier C — long-wavelength, architecture-constrained (add last, report only)

Add `compressibility`, `bending_modulus`.

Expect low R². **Do not optimize HP against their mean.** Report per-property R² to W&B but keep the primary HP metric on Tier A+B properties. Document as architecture-bound and flag for the EFA future work (see [docs/efa_spatial_layer_future.md](efa_spatial_layer_future.md)).

## Control run protocol

At each tier transition, keep one fixed-HP control run at the previous tier with the same seed/split. Compare per-property R² in W&B to isolate whether the new tier helps or hurts the earlier targets.

## Optional: uncertainty weighting implementation sketch

```python
# In MembranePropertyGNN or as standalone parameters
log_vars = nn.Parameter(torch.zeros(out_dim))  # one per property

# In train_one_run loss step (replaces criterion(out, target))
precision = torch.exp(-log_vars)
loss = (precision * (out - target) ** 2 + log_vars).mean()
```

The learned `log_vars` end up in W&B as a diagnostic: large values indicate the model down-weights noisy targets automatically.

## Verification checklist

- [ ] `pytest tests/test_multi_frame_loading.py::test_preprocess_and_save_all_8_properties` — `y.shape == [1, 8]` confirmed
- [ ] 2-property baseline reproduces on new chunks (overall MSE ≈ 0.138 ± noise)
- [ ] Tier A (4 props): all four `val/r2_*` metrics trending positive after 5 epochs
- [ ] Tier B transition: Tier A R² values do not degrade > 5% relative
- [ ] Tier C: `bending_modulus` and `compressibility` R² reported but excluded from HP selection criterion
