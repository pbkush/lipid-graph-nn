# Progress

## What Works

- **Graph construction pipeline**: `MartiniHeteroGraphBuilder` successfully converts MD trajectories to `HeteroData` with continuous physics features, bonded + spatial edges, and composition vectors
- **Chunked dataset loading**: `MartiniDiskDataset` streams data from disk without OOM
- **Model forward pass**: `MembranePropertyGNN` runs in both GNN-only and GNN+composition modes
- **Force field parsing**: `ff_parser.py` extracts parameters from Martini 3 `.itp` files into JSON maps
- **Training infrastructure**: Local `run_sweep.py` (chunk-based + W&B + AMP, mirrors the Colab notebook), linear baseline, smoke tests, result summarization all functional
- **HP analysis tooling**: `scripts/python/download_wandb_runs.py` pulls W&B groups to `logs/training/`; `scripts/notebooks/analyze_hp_search.ipynb` aggregates over seeds, ranks HP cells, and produces 7 visualizations (loss curves, heatmap, training stats, system metrics).
- **HP search (single-property, 2-prop) complete**: Stages 0–5 done. Winner: `hidden_dim=128, num_layers=2`, `lr=1e-4`, `wd=1e-3`. Val MSE: **0.038** (2-prop, stratified chunks, 5 seeds).
- **Stratified system-level split**: `prepare_colab_subset.py` defaults to `--split-method stratified` (k-means in y-space). Fixes test-narrowness bug from random split.
- **Stage 5 analysis pipeline**: `dataset.py` tags graphs with `composition` + `system_idx`; `run_sweep.py` saves `test_artifacts.npz` per run and uploads via `wandb.save()`; `download_wandb_runs.py` fetches it via `run.files()` (basename matching); `linear_baseline.py` has `--stratified` mode. `scripts/notebooks/analyze_stage_5.ipynb` produces 9 publication-grade figures + `headline_numbers.json` with bootstrap CIs and paired t-test.
- **Per-property test MSE**: `run_sweep.py` logs `test/mse_{prop}` for each active property to W&B summary.
- **SLURM queue-drift fix**: `scripts/bash/submit_sweep.sh` freezes all HP values as `FREEZE_*` env vars at submission time. `run_sweep.py::_apply_submission_overrides()` reads them at execution time. Seed parallelization via repeatable `--seeds` flag.
- **Tier A plan**: `docs/tier_a_4prop_plan.md` — Stage 0b → 1b → 1b' (refinement) → optional 2b → 5b.
- **Test suite**: 9 test files, 42 tests.
- **Central config**: `config.yaml` + `lipid_gnn/config.py`. All runtime callers read defaults from `CONFIG`.

## Tier A Status (4 properties: lipid_packing, thickness, thickness_std, variation)

| Stage | Status | Key result |
|-------|--------|------------|
| Stage 0b — 4-prop GNN baseline | **done** | val_min_last10: lp=0.022, th=0.074, th_std=0.359, var=0.462 |
| Stage 1b — lr sweep {1e-5, 1e-4, 5e-4} × 2 seeds | **done** | lr=1e-5 wins; variation only learns at 1e-5 |
| Stage 1b' — lr refinement {3e-6, 1e-5, 3e-5} × 4 seeds | **next** | Need to submit |
| Stage 2b — wd sweep | pending | Only if 1b' changes lr |
| Stage 5b — 5-seed confirmation | pending | Locked HP to be determined |

**GATES** (Stage 0b 4-prop baseline, 5-seed val_min_last10 mean — in both plan doc and notebook):
`lipid_packing < 0.022`, `thickness < 0.074`, `thickness_std < 0.359`, `variation < 0.462`

## What's Left to Build

- **Stage 1b' (lr refinement)**: submit and analyze; decide locked lr for Tier A.
- **Stage 5b (5-seed confirmation)**: run at locked HP after lr is decided.
- **Tier A Stage 5 analysis**: run `analyze_stage_5.ipynb` with `GROUP="stage_5b_tier_a_confirm"`, `BASELINE_GROUP="stage_0b_tier_a"`.
- **Multi-property training (tiered)**: Tier A done after 5b. Tier B (+persistence, +diffusivity), Tier C (+compressibility, +bending_modulus).
- Switch `MartiniHeteroGraphBuilder` to require `.tpr` file for topology instead of `.gro`.
- Explore transfer to protein+membrane systems (long-term research goal).

## Current Status

### Phase: Tier A — Stage 1b' lr refinement next

`config.yaml` active_properties set to 4-property Tier A. Stage 0b and 1b complete. Stage 1b' refinement to be submitted:
```bash
bash scripts/bash/submit_sweep.sh --group stage_1b_refine_tier_a_lr \
    --lr "3e-6 1e-5 3e-5" \
    --seeds "0" --seeds "1" --seeds "2" --seeds "3"
```

## Known Issues

1. **Memory pressure**: Peak GPU memory 58–63 GB out of 64 GB (MI210) at `batch_size=2`. Keep at batch_size=2; fallback is batch_size=1 + gradient accumulation.
2. **LIPID_TYPES consistency**: The 10-element lipid list must be identical across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py` — maintained manually.
3. **Per-property test MSE missing in Stage 0b runs**: `test/mse_{prop}` logging was added after Stage 0b ran; only `test/mse_total` is in those summaries. Val-only analysis for Stage 0b.

## Deferred Ideas (Not Active Tasks)

- **Euclidean Fast Attention (EFA) block on the spatial channel.** Linear-cost, SE(3)-equivariant attention (Frank et al., Nat. Mach. Intell. 2026). Reconsider only after all 8 targets are implemented and simpler levers are exhausted. Full plan at [docs/efa_spatial_layer_future.md](../../docs/efa_spatial_layer_future.md).

## Evolution of Project Decisions

1. **Integer vocab → continuous physics features**: Switched to continuous `[mass, charge, sigma, epsilon]` from Martini 3 FF.
2. **Single graph type → heterogeneous graph**: Moved to `HeteroData` with bonded and spatial edge types.
3. **Full in-memory loading → chunked disk streaming**: Added `MartiniDiskDataset`.
4. **GNN-only → optional composition mode**: Added composition vector concatenation.
5. **Random split → stratified split**: Fixed test-narrowness bug (test std 4× narrower than train).
6. **Live config at execution → frozen env vars at submission**: `submit_sweep.sh` + `_apply_submission_overrides()` to prevent queue-drift corruption.
7. **Single lr=1e-4 → lr=1e-5 for 4-property Tier A**: `variation` property only learns at lower lr; grid spacing too coarse — refinement sweep needed.
