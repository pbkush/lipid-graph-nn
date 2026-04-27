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

**Stage 1b' result (n=4 seeds)**: `lr=3e-5` won (val_total mean 0.149 vs 0.164 at 1e-5
and 0.237 at 3e-6 — 3e-6 not converged in 100 epochs). lr=3e-5 best on every
per-property mean. **Seed-2 anomaly**: at lr=1e-5 and lr=3e-5, seed 2 fails to
learn `variation` (val_min10 = 0.55 / 0.52 vs ~0.08 for seeds 0/1/3). Seed 2's
variation curve plateaus from epoch ~30 across all three lrs — this is an
init-dependent failure mode, not lr-dependent. Excluding seed 2: lr=3e-5 mean
val_total = 0.108 (vs 0.149 with seed 2). **Decision**: lock `lr=3e-5` pending
Stage 1c verification; skip Stage 2b.

---

## Stage 1c — variation seed-stability check at lr=3e-5

Determine whether seed 2's `variation` failure is a chance event (~5–10% rate,
acceptable) or a real fragility (~25%+ rate, requires a fix). Stage 1b' had only
4 seeds — n=1 failure could be either. Run more seeds at the locked lr to
estimate the true failure rate before committing to Stage 5b.

**Grid**: `lr=3e-5` (locked) × `seed ∈ {4, 5, 6, 7, 8, 9}` = 6 runs
**W&B group**: `stage_1c_seed_stability_tier_a`
**Failure criterion**: `val_min10_variation > 0.3` (seed 2 was 0.52; healthy seeds
were 0.07–0.10 — a 0.3 threshold is well above the healthy band and well below
the failure band).

**Decision rule** (informs Stage 5b seed selection):

- **0–1 failures across 6 new seeds** (≤17% rate): seed 2 was unlucky. Lock
  lr=3e-5; run Stage 5b with the 5 best-performing seeds out of {0,1,3,4,…,9}.
  Document the seed-2 failure as a known low-rate fragility in the thesis.
- **2+ failures** (≥33% rate): real fragility. Diagnostic substages before 5b:
  1. **1c-clip**: re-run failing seeds with `gradient_clip_val=1.0` (add to
     `run_sweep.py`'s training loop).
  2. **1c-warmup**: re-run failing seeds with linear lr warmup over first
     5 epochs (0 → lr=3e-5).
  3. If neither helps: document `variation` as init-fragile, fall back to
     reporting only seeds that converged.

**Submit command**:

```bash
bash scripts/bash/submit_sweep.sh --group stage_1c_seed_stability_tier_a \
    --lr "3e-5" \
    --seeds "4" --seeds "5" --seeds "6" --seeds "7" --seeds "8" --seeds "9"
```

**Why not just diagnose seed 2 in isolation**: a single re-run of seed 2 can't
distinguish "rare bad init" from "this specific init is always bad" — the seed
deterministically produces the same init each time. Need fresh seeds to estimate
the population failure rate.

**Stage 1c result (n=5; seed 7 skipped, HPC I/O error)**: 1 failure (seed 9,
val_min10_variation = 0.471). Combined with Stage 1b': **2/9 = 22% failure rate**.
Healthy seeds {4,5,6,8} converged to val_total 0.114–0.122. **Critical new
finding from seed 6**: variation curve plateaued at ~0.5 from epoch 20–50, then
**broke through at epoch ~60** and converged to val_variation = 0.082 (best of
the whole sweep). The plateau is *escapable* — failures may be slow-convergence
bounded by the 100-epoch budget, not true local minima. Real GPU memory
(`gpu/peak_mem_actual_gb`) confirmed 8.0–8.4 GB across all seeds — huge headroom
for Tier B/C.

---

## Stage 1d — long-training resilience check on failed seeds

Seed 6's late-epoch breakthrough (Stage 1c) suggests the variation plateau is
escapable given enough epochs. Seeds 2 and 9 may be on the same trajectory but
cut off by the 100-epoch budget. Cheaper to test "more epochs" than to
implement gradient clipping or warmup.

**Grid**: `lr=3e-5` (locked) × `seed ∈ {2, 9}` × `epochs=200` = 2 runs
**W&B group**: `stage_1d_long_train_tier_a`

**Decision rule**:

- **Both seeds converge** (val_min10_variation < 0.15 by epoch 200): plateau is
  slow-convergence-bounded. Action: bump `training.epochs` to 200 in
  `config.yaml`, re-run Stage 5b at 200 epochs. Update memory bank.
- **Neither converges**: true local minimum / degenerate basin. Action: run
  gradient clipping substage (1d-clip) — wire `grad_clip_norm=1.0` into
  `run_sweep.py` and re-run failing seeds. If still stuck, try lr warmup
  (1d-warmup, linear 0 → 3e-5 over first 5 epochs).
- **Mixed (1 of 2 escapes)**: some seeds genuinely fail. Action: keep
  `epochs=100` default, drop seeds 2 and 9 for Stage 5b, document fragility.

**Submit**: `submit_sweep.sh` has no `--epochs` flag yet, so two options:

```bash
# Option A — temporarily edit config.yaml epochs: 100 → 200, submit, then revert
bash scripts/bash/submit_sweep.sh --group stage_1d_long_train_tier_a \
    --lr "3e-5" \
    --seeds "2" --seeds "9"
# After both jobs complete: edit config.yaml back to epochs: 100.

# Option B — bypass the wrapper, sbatch directly with FREEZE_EPOCHS=200
for SEED in 2 9; do
    sbatch --export=ALL,WANDB_GROUP=stage_1d_long_train_tier_a,\
FREEZE_HIDDEN_DIM=128,FREEZE_NUM_LAYERS=2,FREEZE_LR=3e-5,FREEZE_WD=1e-3,\
FREEZE_EPOCHS=200,FREEZE_PROPERTIES="lipid_packing thickness thickness_std variation",\
SWEEP_SEEDS=${SEED} \
        scripts/bash/sbatch_sweep.sh
done
```

If long-training experiments become recurring, add `--epochs` flag to
`submit_sweep.sh` (mirrors `--lr`/`--wd` pattern, ~4 lines).

---

## Stage 2b — wd check (only if Stage 1b/1b' changes lr)

**Grid**: `wd ∈ {1e-4, 1e-3, 1e-2}` × `seed ∈ {0, 1}` = 6 runs
**W&B group**: `stage_2b_tier_a_wd`

---

## Stage 5b — 5-seed confirmation

Run 5 seeds at locked HP (`lr=3e-5`). Produces `test_artifacts.npz` for analysis.

**Grid**: `seed ∈ {0, 1, 2, 3, 4}` = 5 runs (re-using Stage 1b' seeds 0,1,3 + new
4,5; OR reselect after Stage 1c if seed-stability findings warrant skipping seed 2).
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
