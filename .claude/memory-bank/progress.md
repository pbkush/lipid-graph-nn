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
| Stage 0b — 4-prop GNN baseline | done | val_min10: lp=0.022, th=0.074, th_std=0.359, var=0.462 |
| Stage 1b — lr sweep {1e-5,1e-4,5e-4} × 2 seeds | done | lr=1e-5 wins; variation only learns at 1e-5 |
| Stage 1b' — lr refinement {3e-6,1e-5,3e-5} × 4 seeds | done | lr=3e-5 wins (val_total 0.149); seed-2 variation failure exposed |
| Stage 1c — seed stability at lr=3e-5 | done | 1/5 fail (seed 9); 22% combined failure rate; seed-6 late-escape pattern |
| Stage 1d — long-training rescue at 200 ep | done | seed 9 rescued (val_var ~0.10); seed 2 stuck (~0.53) — drop permanently |
| Stage 2b — wd verification at lr=3e-5 | done | wd insensitive in [3e-4, 1e-3, 3e-3] (val_total 0.116-0.118) |
| Stage 5b — 5-seed confirmation | **next** | Seeds {0,1,3,4,5} at lr=3e-5, 200 epochs |

**Locked HPs (final, in config.yaml)**: `hidden_dim=128`, `num_layers=2`, `lr=3.0e-5`, `wd=1.0e-3`, `epochs=200`.

**Variation seed-fragility** — two failure modes identified:
- *Slow escapers* (seeds 6, 9): variation plateau at ~0.5 from epoch 20–50, then breakthrough to ~0.08–0.10. Rescued by 200-epoch training. Validates the epochs=200 default.
- *True dead-init* (seed 2): plateau forever, no movement after 200 epochs. Drop permanently.
- `thickness_std` and `variation` failures are correlated within a seed → single loss-landscape pathology, not two independent ones.
- ~20% population failure rate (2/9 seeds across Stages 1b' + 1c). Documented as a Tier A limitation for the thesis.

**HP saturation finding (from 2-prop Stage 5)**: paired t-test p=0.755 vs Stage 0 baseline — HP search produced no significant gain on 2 properties. Tier A's hope (and now confirmation) is that HP tuning matters more for harder properties: variation only learns at lr=3e-5/1e-5, not the original lr=1e-4. This is the main thesis story for Tier A.

**wd is small lever**: Stage 2b confirmed wd=1e-3 is roughly optimal in [3e-4, 3e-3] range. Per-property tradeoff exists (higher wd helps variation slightly, hurts thickness_std slightly), but val_total flat. Locked wd=1e-3.

**GATES** (Stage 0b 4-prop baseline, 5-seed val_min10 mean — in plan doc and notebook):
`lipid_packing < 0.022`, `thickness < 0.074`, `thickness_std < 0.359`, `variation < 0.462`

**GPU memory clarification**: earlier "97% peak = OOM danger" was a misread of W&B's `memoryAllocated` (reserved pool, not live tensors). Real proxy `torch.cuda.max_memory_allocated()` added to `run_sweep.py` as `gpu/peak_mem_actual_gb` (per-epoch reset). Live peak ~8 GB out of 64 GB. Tier B/C have huge memory headroom.

**Run-name encoding bug fixed**: original `gnn_only_h{h}_l{l}_lr{lr}_s{seed}` didn't include all varying HPs, causing Stage 2b download collisions. Future stages must include all varying HPs in run_name (e.g. `_wd{wd}` suffix).

**R² added to analyze_hp_search.ipynb**: complementary reporting metric (selection still MSE-driven). 4 cells modified: `cell-load-fn`, `cell-detect-hps`, `cell-aggregate`, `cell-ranking-table`, `cell-recommendation`. R² uses `_tail_mean()` (not `_tail_min()`) to avoid amplifying favourable noise spikes on the small val set.

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
