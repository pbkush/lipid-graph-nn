# Tier A — 4-Property Training Plan

**Properties**: `lipid_packing`, `thickness`, `thickness_std`, `variation`

**Locked HP baseline** (from single-property search):
`hidden_dim=128`, `num_layers=2`, `lr=1e-4`, `wd=1e-3`

---

## Config change to activate Tier A

Edit `config.yaml` before any of the stages below:

```yaml
vocab:
  active_properties: [lipid_packing, thickness, thickness_std, variation]
```

No chunk rebuild needed — all 8 properties are stored in `y` at preprocessing time;
`run_sweep.py` slices the relevant columns via `prop_cols`.

---

## Stage 0b — GNN floor on 4 properties

Establishes the naive GNN baseline before any Tier A HP tuning. Required for the
paired t-test in `analyze_stage_5.ipynb` fig (i) — the 2-property Stage 5 result
cannot serve as a floor here because MSE is averaged over a different number of
outputs and `thickness_std`/`variation` have no prior floor.

**Grid**: `seed ∈ {0, 1, 2, 3, 4}` = 5 runs (locked HP, no tuning)
**W&B group**: `stage_0b_tier_a`

Also run the linear baseline for 4 properties:

```bash
python scripts/training/linear_baseline.py --stratified
```

Saves `results/training/linear_baseline_stratified.npz` in the same format as
`test_artifacts.npz`. Update `config.yaml` first so `active_properties` is already set.

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

**Stage 1b result (n=2 seeds)**: `lr=1e-5` won decisively. `variation` only learned at
1e-5 (val 0.082 vs 0.459/0.476 baseline-floor at higher lr); `thickness_std` and
`thickness` also best at 1e-5; `lipid_packing` lower mean at 1e-4 but with large
seed variance at 1e-5 (std=0.020) → instability uncertain at n=2.

---

## Stage 1b' — lr refinement around 1e-5

Two motivations: (i) the current grid is half-decade-spaced so the true optimum
could lie between tested points; (ii) `lipid_packing` showed std=0.020 across only
2 seeds at lr=1e-5 — need more seeds to separate genuine instability from a single
bad init. Peak GPU memory was ~58-63 GB (out of 64); keep `batch_size=2`.

**Grid**: `lr ∈ {3e-6, 1e-5, 3e-5}` × `seed ∈ {0, 1, 2, 3}` = 12 runs
**W&B group**: `stage_1b_refine_tier_a_lr`
**Decision rule**: pick lowest `val_min_last10` averaged over seeds; tie-break on
seed std then on `val/loss_variation` (the pivotal property).

---

## Stage 2b — wd check (only if Stage 1b/1b' changes lr)

**Grid**: `wd ∈ {1e-4, 1e-3, 1e-2}` × `seed ∈ {0, 1}` = 6 runs
**W&B group**: `stage_2b_tier_a_wd`

---

## Stage 5b — 5-seed confirmation

Run 5 seeds at locked HP. Produces `test_artifacts.npz` for analysis.

**Grid**: `seed ∈ {0, 1, 2, 3, 4}` = 5 runs
**W&B group**: `stage_5b_tier_a_confirm`

**Gate to pass** (normalized MSE per property, last-10-epoch val mean over seeds):

| Property      | Gate (norm. MSE) | Notes                                          |
|---------------|------------------|------------------------------------------------|
| lipid_packing | < 0.022          | Stage 0b 4-prop baseline (5-seed val mean)     |
| thickness     | < 0.074          | Stage 0b 4-prop baseline (5-seed val mean)     |
| thickness_std | < 0.359          | Stage 0b 4-prop baseline (5-seed val mean)     |
| variation     | < 0.462          | Stage 0b 4-prop baseline; expect noise-limited |

---

## Reporting

Use `analyze_stage_5.ipynb` with:

```python
GROUP          = "stage_5b_tier_a_confirm"
BASELINE_GROUP = "stage_0b_tier_a"
```

The per-property test MSE is logged as `test/mse_{prop}` in W&B summary, so
`analyze_hp_search.ipynb` will also show property-level breakdown when analyzing Stage 1b.

The paired t-test (fig i) compares Stage 0b vs Stage 5b on all 4 properties — this
is the primary statistical evidence that Stage 1b HP tuning was worthwhile for Tier A.
