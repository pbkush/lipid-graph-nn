# Active Context

## Current Work Focus

**Tier A — 4-property HP search in progress (2026-04-26)**

`config.yaml` now set to `active_properties: [lipid_packing, thickness, thickness_std, variation]`.
Locked HPs from single-property search: `hidden_dim=128`, `num_layers=2`, `wd=1e-3`. lr is being refined.

### Tier A stage status

| Stage | W&B group | Status | Result |
|-------|-----------|--------|--------|
| 0b — GNN floor, 4-prop | `stage_0b_tier_a` | **done** | val_min_last10 means: lp=0.022, th=0.074, th_std=0.359, var=0.462 |
| 1b — lr sanity check | `stage_1b_tier_a_lr` | **done** | lr=1e-5 wins decisively (especially variation: 0.082 vs 0.459 at lr=1e-4) |
| 1b' — lr refinement | `stage_1b_refine_tier_a_lr` | **next to submit** | Grid: lr ∈ {3e-6, 1e-5, 3e-5} × seed ∈ {0,1,2,3} = 12 runs |
| 2b — wd check | `stage_2b_tier_a_wd` | pending 1b' result | Only if 1b' changes lr |
| 5b — 5-seed confirmation | `stage_5b_tier_a_confirm` | pending | 5 seeds, locked HP |

**Submit command for Stage 1b':**
```bash
bash scripts/bash/submit_sweep.sh --group stage_1b_refine_tier_a_lr \
    --lr "3e-6 1e-5 3e-5" \
    --seeds "0" --seeds "1" --seeds "2" --seeds "3"
```
Or 6 jobs (2 seeds each): `--seeds "0 1" --seeds "2 3"`.

### Stage 1b key findings (2 seeds per lr)

| lr   | lipid_packing (mean±std) | thickness | thickness_std | variation | total |
|------|--------------------------|-----------|---------------|-----------|-------|
| 1e-5 | 0.037 ± 0.020            | 0.081     | 0.334         | **0.082** | **0.136** |
| 1e-4 | **0.027** ± 0.002        | 0.082     | 0.384         | 0.459     | 0.240 |
| 5e-4 | 0.036 ± 0.008            | 0.119     | 0.361         | 0.476     | 0.251 |

- `variation` only learns at lr=1e-5 (0.082 vs baseline 0.462 — all other lrs stuck at baseline).
- `lipid_packing` at lr=1e-5: high variance (std=0.020, two seeds: 0.023 and 0.051) — need more seeds.
- Peak GPU memory 58–63 GB out of 64 GB — very tight. Keep batch_size=2.
- Refinement grid needed: half-decade spacing is too coarse to trust the optimum.

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
