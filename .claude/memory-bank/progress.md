# Progress

## What Works

- **Graph construction pipeline**: `MartiniHeteroGraphBuilder` successfully converts MD trajectories to `HeteroData` with continuous physics features, bonded + spatial edges, and composition vectors
- **Chunked dataset loading**: `MartiniDiskDataset` streams data from disk without OOM
- **Model forward pass**: `MembranePropertyGNN` runs in both GNN-only and GNN+composition modes
- **Force field parsing**: `ff_parser.py` extracts parameters from Martini 3 `.itp` files into JSON maps
- **Training infrastructure**: Local `run_sweep.py` (chunk-based + W&B + AMP, mirrors the Colab notebook), linear baseline, smoke tests, result summarization all functional
- **HP analysis tooling**: `scripts/python/download_wandb_runs.py` pulls W&B groups to `logs/training/`; `scripts/notebooks/analyze_hp_search.ipynb` aggregates over seeds, ranks HP cells, and produces 7 visualizations (loss curves, heatmap, training stats, system metrics).
- **HP search (single-property, 2-prop) complete**: Stages 0–5 done. Winner: `hidden_dim=128, num_layers=2`, `lr=1e-4`, `wd=1e-3`. Val MSE: **0.038** (2-prop, stratified chunks, 5 seeds). Paired t-test p=0.755 — HP search produced no significant gain on the 2-property task.
- **HP search (4-property Tier A) complete**: Stages 0b–5b done. Winner: `hidden_dim=128, num_layers=2`, `lr=3e-5`, `wd=1e-3`, `epochs=200`. Paired t=−31.5, p=3.5e-5 vs Stage 0b — significant ~66 % test-MSE reduction. Per-property R² ≥ 0.87 on all four properties.
- **Stratified system-level split**: `prepare_colab_subset.py` defaults to `--split-method stratified` (k-means in y-space). Fixes test-narrowness bug from random split.
- **Stage 5 / 5b analysis pipeline**: `dataset.py` tags graphs with `composition` + `system_idx`; `run_sweep.py` saves `test_artifacts.npz` per run and uploads via `wandb.save()`; `download_wandb_runs.py` fetches it via `run.files()` (basename matching); `linear_baseline.py` has `--stratified` mode. `scripts/notebooks/analyze_stage_5.ipynb` produces 9 publication-grade figures + `headline_numbers.json` with bootstrap CIs and paired t-test. Re-pointed at `stage_5b_tier_a_confirm` for the Tier A run; outputs in `results/figures/stage_5b/`.
- **Per-property test MSE**: `run_sweep.py` logs `test/mse_{prop}` for each active property to W&B summary.
- **SLURM queue-drift fix + per-node GPU packing**: `scripts/bash/submit_sweep.sh` freezes all HP values at submission time as `RUN_<i>_*` env vars (one slot per Cartesian-product cell, including each individual seed). `sbatch_sweep.sh` then backgrounds N parallel `python run_sweep.py` processes on a single node — one per GPU, pinned via `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` — translating `RUN_<i>_*` → `FREEZE_*`/`SWEEP_SEEDS` per slot. Default packs up to 8 runs/node; excess runs spill into additional sbatch jobs. New CLI flags: `--partition`, `--time`, `--gpus-per-node`, `--cpus-per-gpu`, `--mem-per-gpu`. `gpu_test` partition has built-in guards (8h max, 2 jobs max). Per-process logs at `logs/sweeps/sweep-<jobid>-gpu<i>.{out,err}`. (Previous: 1 sbatch job = 1 GPU, 7 GPUs idle.)
- **Tier A plan**: `docs/tier_a_4prop_plan.md` — Stage 0b → 1b → 1b' → 1c → 1d → 2b → 5b. All complete.
- **Test suite**: 9 test files, 42 tests.
- **Central config**: `config.yaml` + `lipid_gnn/config.py`. All runtime callers read defaults from `CONFIG`.

## Tier A Status (4 properties: lipid_packing, thickness, thickness_std, variation)

| Stage | Status | Key result |
| --- | --- | --- |
| Stage 0b — 4-prop GNN baseline | done | val_min10: lp=0.022, th=0.074, th_std=0.359, var=0.462 |
| Stage 1b — lr sweep {1e-5,1e-4,5e-4} × 2 seeds | done | lr=1e-5 wins; variation only learns at 1e-5 |
| Stage 1b' — lr refinement {3e-6,1e-5,3e-5} × 4 seeds | done | lr=3e-5 wins (val_total 0.149); seed-2 variation failure exposed |
| Stage 1c — seed stability at lr=3e-5 | done | 1/5 fail (seed 9); 22% combined failure rate; seed-6 late-escape pattern |
| Stage 1d — long-training rescue at 200 ep | done | seed 9 rescued (val_var ~0.10); seed 2 stuck (~0.53) — drop permanently |
| Stage 2b — wd verification at lr=3e-5 | done | wd insensitive in [3e-4, 1e-3, 3e-3] (val_total 0.116-0.118) |
| Stage 5b — 5-seed confirmation | **done** | 6/7 seeds healthy; paired t=−31.5, p=3.5e-5 vs Stage 0b; per-prop R² ≥ 0.87; GNN beats Ridge by 56–84 % |

**Locked HPs (final, in config.yaml)**: `hidden_dim=128`, `num_layers=2`, `lr=3.0e-5`, `wd=1.0e-3`, `epochs=200`.

## Stage 5b headline numbers (test, pooled, normalised)

| Property | MSE mean ± std | R² (95 % CI) | Gate (val) |
| --- | --- | --- | --- |
| `lipid_packing` | 0.020 ± 0.003 | 0.975 [0.972, 0.978] | 0.0222 vs 0.022 — tied |
| `thickness` | 0.076 ± 0.007 | 0.908 [0.898, 0.917] | 0.0732 vs 0.074 — pass |
| `thickness_std` | 0.145 ± 0.024 | 0.873 [0.856, 0.888] | 0.299 vs 0.359 — pass (+17 %) |
| `variation` | 0.131 ± 0.171 | 0.872 [0.856, 0.887] | 0.151 vs 0.462 — pass (+67 %) |

The wide MSE std on `variation` is driven by seed 6 (failed to escape; widens std from ~0.02 to 0.171). For thesis numbers, prefer the planned 5-seed pool {0,1,3,4,5}.

Per-system errors concentrate on DPPC- and DOPC-rich mixtures (POPC30_DOPC70 worst, ~19 Å thickness MAE) — these sit at the boundary of the test cloud in PCA(composition) space where train density also drops. Documented as a Tier A scope limit.

Full report: [results/figures/stage_5b/stage_5b_analysis_report.md](../../results/figures/stage_5b/stage_5b_analysis_report.md).

## Variation seed-fragility — two failure modes

- *Slow escapers* (seeds 6, 9): variation plateau at ~0.5 from epoch 20–50, then breakthrough to ~0.08–0.10. Rescued by 200-epoch training. **Escape is non-deterministic** — seed 6 escaped in 1c, failed in 5b.
- *True dead-init* (seed 2): plateau forever, no movement after 200 epochs. Drop permanently.
- `thickness_std` and `variation` failures are correlated within a seed → single loss-landscape pathology, not two independent ones.
- ~20 % population failure rate (2/9 seeds across Stages 1b' + 1c, plus the recurring seed-6 jitter in 5b). Documented as a Tier A limitation for the thesis.

## Headline thesis story

**HP saturation finding**: 2-prop Stage 5 (lr=1e-4) had paired t-test p=0.755 — HP search produced no significant gain. Tier A reverses this: paired t=−31.5, p=3.5e-5. **HP tuning matters more for harder properties** is the main thesis story; lr was the dominant lever (variation only learns at lr=3e-5/1e-5, not lr=1e-4).

**GATES** (Stage 0b 4-prop baseline, 5-seed val_min10 mean — in plan doc and notebook):
`lipid_packing < 0.022`, `thickness < 0.074`, `thickness_std < 0.359`, `variation < 0.462`

**wd is small lever**: Stage 2b confirmed wd=1e-3 is roughly optimal in [3e-4, 3e-3] range. Per-property tradeoff exists, but val_total flat. Locked wd=1e-3.

**GPU memory clarification**: earlier "97 % peak = OOM danger" was a misread of W&B's `memoryAllocated` (reserved pool, not live tensors). Real proxy `torch.cuda.max_memory_allocated()` added to `run_sweep.py` as `gpu/peak_mem_actual_gb` (per-epoch reset). Live peak ~8 GB out of 64 GB. Tier B/C have huge memory headroom.

**Run-name schema (collision-proof)**: schema is `{comp_mode}_h{h}_l{l}_lr{lr:.0e}_wd{wd:.0e}_e{e}_s{seed}_{run_id}`. W&B `run.id` suffix (8 chars, globally unique) guarantees no collision even for same-(HPs, seed) retries. `download_wandb_runs.py` writes `.wandb_run_id` marker file and raises `RuntimeError` on mismatch (defence-in-depth). Preserve trailing `_{run.id}` in all future stages.

**R² added to analyze_hp_search.ipynb**: complementary reporting metric (selection still MSE-driven). 4 cells modified: `cell-load-fn`, `cell-detect-hps`, `cell-aggregate`, `cell-ranking-table`, `cell-recommendation`. R² uses `_tail_mean()` (not `_tail_min()`) to avoid amplifying favourable noise spikes on the small val set.

## Tier B Status (6 properties: + persistence, + diffusivity)

| Stage | Status | Key result |
| --- | --- | --- |
| Stage 0c — 6-prop GNN floor at locked Tier A HPs | done | val_min10: lp=0.019, th=0.067, th_std=0.302, var=0.151, persistence=0.362, diffusivity=0.059. No negative transfer; `diffusivity` learns cleanly (R² ≈ 0.96), `persistence` floor-like (R² ≈ 0.66). 4/5 seeds healthy (seed 3 stuck on `variation`). Decision matrix outcome **A** with caveat: `persistence` may need lr re-tune. |
| Stage 1e — lr sanity check (2 seeds) | done | Initial signal: lr=1e-5 wins (val_total 0.161); seed-0 variation failure at 3e-5 inflated 3e-5 mean. Triggered 1e'. |
| Stage 1e' — lr refinement (4 seeds × 3 lrs) | done | **lr=3e-5 wins (val_total 0.148)** — Tier A lock confirmed; signal from 1e was a single-seed bad-init artefact. 4/4 seeds healthy at 3e-5, ≈5× tighter seed-std than alternatives. Persistence floor (~0.35) flat across all lrs — architecture-limited. |
| Stage 1f — seed stability | skipped | Not triggered; 1e' kept the lock and 4/4 seeds at 3e-5 are implicit stability evidence. |
| Stage 5c — 5-seed confirmation | **done** | R² ≥ 0.88 on 5/6 props; persistence R²=0.578 (architecture floor); t-test p=0.286 vs Stage 0c (expected — same HPs). |

**Tier B GATES (Stage 0c 6-prop floor, 5-seed val_min10 mean — locked in `tier_b_6prop_plan.md` and `analyze_hp_search.ipynb` Cell 1)**:
`lipid_packing < 0.019`, `thickness < 0.067`, `thickness_std < 0.302`, `variation < 0.151`, `persistence < 0.362`, `diffusivity < 0.059`.

## What's Left to Build

- **Tier B remaining stages**: 1e (lr sanity → conditional 1e' refinement) → 5c (5-seed confirm). Plan in `docs/tier_b_6prop_plan.md`. Negative transfer was *not* observed at Stage 0c, so the homoscedastic uncertainty weighting remedy stays deferred.
- **Tier C (+compressibility, +bending_modulus)**: likely floor-bound until the spatial channel is extended (`docs/efa_spatial_layer_future.md`).
- **Train-coverage augmentation**: more DPPC- and DOPC-rich compositions to address the per-system MAE concentration on chemically extreme mixtures (Stage 5b finding).
- **Embedding evaluation, not just property prediction**: the long-term scientific question is the quality of the membrane embedding. Once Tier A/B/C land, probe the embedding directly (clustering, interpretability, transfer to held-out compositions or to protein+membrane systems).
- Explore transfer to protein+membrane systems (long-term research goal).

## Current Status

### Phase: Tier A complete; Tier B planning next

`config.yaml` `active_properties` is set to 4-property Tier A. All Stage 0b–5b runs done. Tier B (`+persistence`, `+diffusivity` → 6 active properties) is the next planning task.

## Known Issues

1. **LIPID_TYPES consistency**: The 10-element lipid list must be identical across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py` — maintained manually.
2. **Per-property test MSE missing in Stage 0b runs**: `test/mse_{prop}` logging was added after Stage 0b ran; only `test/mse_total` is in those summaries. Val-only analysis for Stage 0b.
3. **Seed-6 jitter in Stage 5b**: seed 6 escaped `variation` in Stage 1c but failed in 5b at the same config. Escape is non-deterministic per seed — running the same seed twice can produce different outcomes. For thesis reporting, prefer the planned 5-seed pool {0,1,3,4,5}.

## Deferred Ideas (Not Active Tasks)

- **Euclidean Fast Attention (EFA) block on the spatial channel.** Linear-cost, SE(3)-equivariant attention (Frank et al., Nat. Mach. Intell. 2026). Reconsider only after all 8 targets are implemented and simpler levers are exhausted. Full plan at [docs/efa_spatial_layer_future.md](../../docs/efa_spatial_layer_future.md).

## Evolution of Project Decisions

1. **Integer vocab → continuous physics features**: Switched to continuous `[mass, charge, sigma, epsilon]` from Martini 3 FF.
2. **Single graph type → heterogeneous graph**: Moved to `HeteroData` with bonded and spatial edge types.
3. **Full in-memory loading → chunked disk streaming**: Added `MartiniDiskDataset`.
4. **GNN-only → optional composition mode**: Added composition vector concatenation.
5. **Random split → stratified split**: Fixed test-narrowness bug (test std 4× narrower than train).
6. **Live config at execution → frozen env vars at submission**: `submit_sweep.sh` + `_apply_submission_overrides()` to prevent queue-drift corruption.
7. **2-prop lr=1e-4 → 4-prop lr=3e-5**: `variation` property only learns at lower lr; grid spacing too coarse — refinement sweep needed (Stage 1b').
8. **100 → 200 epochs (Tier A default)**: Stage 1d found slow-escaper seeds need >100 epochs to break through `variation` plateau.
