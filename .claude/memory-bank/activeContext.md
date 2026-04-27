# Active Context

## Current Work Focus

**Tier A — 4-property HP search; lr=3e-5 winner pending seed-stability check (2026-04-27)**

`config.yaml` set to `active_properties: [lipid_packing, thickness, thickness_std, variation]`.
Locked HPs: `hidden_dim=128`, `num_layers=2`, `wd=1e-3`, **`lr=3e-5`** (provisional, pending Stage 1c).

### Tier A stage status

| Stage | W&B group | Status | Result |
|-------|-----------|--------|--------|
| 0b — GNN floor, 4-prop | `stage_0b_tier_a` | **done** | val_min_last10 means: lp=0.022, th=0.074, th_std=0.359, var=0.462 |
| 1b — lr sanity check | `stage_1b_tier_a_lr` | **done** | lr=1e-5 best of {1e-5, 1e-4, 5e-4}; only lr=1e-5 learned variation |
| 1b' — lr refinement | `stage_1b_refine_tier_a_lr` | **done** | lr=3e-5 wins (val_total 0.149 vs 0.164/0.237); seed-2 variation failure exposed |
| 1c — seed stability at lr=3e-5 | `stage_1c_seed_stability_tier_a` | **next to submit** | 6 new seeds {4..9} at lr=3e-5 to estimate variation-failure rate |
| 2b — wd check | `stage_2b_tier_a_wd` | **skipped** | 1b'/1c clearly favor lr=3e-5; wd revisit not needed |
| 5b — 5-seed confirmation | `stage_5b_tier_a_confirm` | pending 1c | Seeds reselected based on 1c findings |

**Submit command for Stage 1c:**
```bash
bash scripts/bash/submit_sweep.sh --group stage_1c_seed_stability_tier_a \
    --lr "3e-5" \
    --seeds "4" --seeds "5" --seeds "6" --seeds "7" --seeds "8" --seeds "9"
```

### Stage 1b' findings (4 seeds per lr)

| lr    | val_total | std    | val_var (with seed 2) | val_var (without seed 2) |
|-------|-----------|--------|------------------------|---------------------------|
| 3e-6  | 0.237     | 0.042  | 0.371                  | not converged in 100 ep   |
| 1e-5  | 0.164     | 0.069  | 0.200                  | ~0.085                    |
| 3e-5  | **0.149** | 0.070  | 0.188                  | **~0.076**                |

- **lr=3e-5 wins on every per-property mean**, including variation.
- **lr=3e-6 not converged** in 100 epochs (variation curve still descending) — practical reject.
- **Seed 2 variation failure is reproducible across all 3 lrs**: val_min10_variation = 0.60/0.55/0.52 at 3e-6/1e-5/3e-5 vs ~0.08 for seeds 0/1/3. Curve plateaus from ~epoch 30 onward at all lrs. Init-dependent, not lr-dependent.
- Other properties for seed 2 are healthy.

### Stage 1c plan (next)

- 6 new seeds {4..9} at locked lr=3e-5
- Failure criterion: `val_min10_variation > 0.3` (well above healthy band 0.07-0.10, well below failure band 0.5+)
- Decision: ≤1 failure → lock and pick best 5 seeds for Stage 5b. ≥2 failures → diagnostic substages (gradient clipping, lr warmup).

### GPU memory — interpretation revised (2026-04-27)

Earlier "97% peak = OOM danger" was a misread. W&B `memoryAllocated` reports PyTorch's *reserved* pool, not live tensors. Plot of one Stage 1b' run shows allocation rising to 98% then dropping to 13% mid-run without crashes — this is the caching allocator releasing back to ROCm. 12 Stage 1b' runs all completed without OOM.

- Real OOM proxy: `torch.cuda.max_memory_allocated()` (live-tensor high-water).
- Added to `run_sweep.py` line 244 as `gpu/peak_mem_actual_gb` in epoch metrics.
- Practical implications: keep `batch_size=2`; Tier B (more properties) likely fits.

### Gates (Stage 0b 4-prop baseline)
Set in `docs/tier_a_4prop_plan.md` and `analyze_hp_search.ipynb` Cell 1 `GATES`:
- `lipid_packing < 0.022`, `thickness < 0.074`, `thickness_std < 0.359`, `variation < 0.462`

## Latest Changes (2026-04-26, this session)

- `config.yaml`: `active_properties` switched from 2-prop to `[lipid_packing, thickness, thickness_std, variation]`.
- `docs/tier_a_4prop_plan.md`: gates updated from old 2-prop Stage 5 values → Stage 0b 4-prop baseline values. Stage 1b results and new Stage 1b' refinement sub-stage added.
- `scripts/notebooks/analyze_hp_search.ipynb` Cell 1 `GATES`: updated to all 4 properties with Stage 0b baseline values.

## Previous Latest Changes

**Tier A 4-property plan + per-property test logging (2026-04-26):**

- `scripts/training/run_sweep.py`: now logs `test/mse_{prop}` for every property alongside `test/mse_total`.
- `scripts/python/download_wandb_runs.py` `_download_run`: fetches `test_artifacts.npz` via `run.files()` with basename matching.
- `docs/tier_a_4prop_plan.md`: created — lightweight HP re-check plan for 4-property Tier A.
- `scripts/bash/submit_sweep.sh`: new script — freezes all HP values as `FREEZE_*` env vars at submission time to avoid queue drift. Repeatable `--seeds`, `--lr`, `--wd`, `--hidden-dim`, `--num-layers` flags expand as Cartesian product of SLURM jobs.
- `scripts/training/run_sweep.py` `_apply_submission_overrides()`: reads `FREEZE_*` / `SWEEP_SEEDS` env vars set by `submit_sweep.sh` at execution time, overrides module-level SWEEP/FIXED/PROPERTIES dicts.

**Bug fix — test_artifacts.npz was never actually saved (2026-04-26):**

- Fixed: test loop now collects compositions/system_idx per batch, saves `np.savez(artifacts_path, ...)` + `wandb.save()`. Old Stage 0 and Stage 5 runs re-run (now completed).

**Stage 5 analysis pipeline + publication notebook (2026-04-26):**

- `scripts/notebooks/analyze_stage_5.ipynb`: 20-cell publication notebook. 9 figures + `headline_numbers.json`.
- `scripts/training/linear_baseline.py`: `--stratified` mode.
- `scripts/notebooks/analyze_hp_search.ipynb`: 14-cell analysis notebook with GATES, multi-group comparison, 7 visualizations.

**Stratified system-level split (2026-04-25):**

- `prepare_colab_subset.py` now defaults to `--split-method stratified` (k-means in y-space). Fixes test-narrowness bug.

## Important Patterns and Preferences

- Test discrete components locally (dedicated test scripts, baseline metrics) before integrating into the heavy end-to-end training loop
- Results are uploaded to Weights & Biases for visualization
- Force field parameters are loaded from JSON files at graph build time
- `LIPID_TYPES` ordering must stay consistent across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py`
- `preprocess_and_save` is the single entry point for building and saving graph chunks
- All HP values are frozen at sbatch submission time via `submit_sweep.sh`; `run_sweep.py` reads them via `_apply_submission_overrides()`
