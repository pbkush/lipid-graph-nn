# Tier A — 4-Property Training Plan

**Properties**: `lipid_packing`, `thickness`, `thickness_std`, `variation`

**Locked HP baseline** (from single-property search):
- `hidden_dim=128`, `num_layers=2`, `lr=1e-4`, `wd=1e-3`

---

## Stage 1b — lr sanity check

Verify lr=1e-4 still optimal with 4 outputs. Targets are z-scored per property so
loss scale is stable, but `variation` (R²≤0.5 preliminary) introduces noisy gradients.

**Grid**: `lr ∈ {1e-5, 1e-4, 5e-4}` × `seed ∈ {0, 1}` = 6 runs  
**W&B group**: `stage_1b_tier_a_lr`  
**Decision rule**: pick lowest `val_min_last10` averaged over seeds.

- If `lr=1e-4` wins → lock and skip Stage 2b.  
- If different lr wins → run Stage 2b (same grid for `wd`).

**Watch**: `val/loss_variation` specifically. If it fails to decrease past ~epoch 20
the property is noise-limited, not lr-limited — this is expected.

---

## Stage 2b — wd check (only if Stage 1b changes lr)

**Grid**: `wd ∈ {1e-4, 1e-3, 1e-2}` × `seed ∈ {0, 1}` = 6 runs  
**W&B group**: `stage_2b_tier_a_wd`

---

## Stage 5b — 5-seed confirmation

Run 5 seeds at locked HP. Produces `test_artifacts.npz` for analysis.

**Grid**: `seed ∈ {0, 1, 2, 3, 4}` = 5 runs  
**W&B group**: `stage_5b_tier_a_confirm`

**Gate to pass** (normalized MSE per property, last-10-epoch val mean over seeds):
| Property       | Gate (norm. MSE) | Notes                          |
|----------------|-----------------|--------------------------------|
| lipid_packing  | < 0.056         | 3.6× improvement from baseline |
| thickness      | < 0.219         | from single-prop Stage 5       |
| thickness_std  | TBD             | set after Stage 1b             |
| variation      | TBD             | expect high; R² floor ~0.5     |

---

## Config change to activate Tier A

Edit `config.yaml`:
```yaml
vocab:
  active_properties: [lipid_packing, thickness, thickness_std, variation]
```

No chunk rebuild needed — all 8 properties are stored in `y` at preprocessing time;
`run_sweep.py` slices the relevant columns via `prop_cols`.

---

## Reporting

Use `analyze_stage_5.ipynb` with `GROUP = "stage_5b_tier_a_confirm"`. The per-property
test MSE is now logged as `test/mse_{prop}` in W&B summary, so `analyze_hp_search.ipynb`
will also show property-level breakdown when analyzing Stage 1b.
