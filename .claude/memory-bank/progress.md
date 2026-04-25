# Progress

## What Works

- **Graph construction pipeline**: `MartiniHeteroGraphBuilder` successfully converts MD trajectories to `HeteroData` with continuous physics features, bonded + spatial edges, and composition vectors
- **Chunked dataset loading**: `MartiniDiskDataset` streams data from disk without OOM
- **Model forward pass**: `MembranePropertyGNN` runs in both GNN-only and GNN+composition modes
- **Force field parsing**: `ff_parser.py` extracts parameters from Martini 3 `.itp` files into JSON maps
- **Training infrastructure**: Local `run_sweep.py` (chunk-based + W&B + AMP, mirrors the Colab notebook), linear baseline, smoke tests, result summarization all functional
- **HP analysis tooling**: `scripts/python/download_wandb_runs.py` pulls W&B groups to `logs/training/`; `scripts/notebooks/analyze_hp_search.ipynb` aggregates over seeds, ranks HP cells, and produces 7 visualizations (loss curves, heatmap, training stats, system metrics). `docs/analyze_hp_search_notebook.md` documents each visualization.
- **HP search stages 0–3 complete**: Stage 0 baseline MSE ≈ 0.138 reproduced; Stage 1 locked `lr=1e-4`; Stage 2 locked `wd=1e-3`; Stage 3 winner: `hidden_dim=128, num_layers=2` (val_mean=0.03816, val_std=0.00036).
- **Stratified system-level split**: `prepare_colab_subset.py` now defaults to `--split-method stratified` (k-means in y-space). Fixes the bug where random `split_seed=0` made test 4× narrower than train on `lipid_packing`, causing test MSE to always appear lower than val. New CLI: `--split-method`, `--stratify-on`.
- **Test suite**: 9 test files, 35 tests (added `test_config.py` with 8 tests on 2026-04-24) covering graph construction, dataset loading, model modes, FF parsing, benchmarks, multi-frame preprocessing, interleaving invariant, train/val/test split disjointness, all-8-property y-shape invariant, and config loading/validation/env-override
- **Central config**: `config.yaml` + `lipid_gnn/config.py` landed 2026-04-24. All runtime callers in `lipid_gnn/` (ex-`functions_emil`), `scripts/training/`, `scripts/bash/`, and `tests/` read defaults from `CONFIG`. Bash shim at `scripts/python/print_config_var.py`.
- **Documentation**: `README.md` covers goal, architecture, install, training entry points, data layout, and evaluation story
- **GitHub workflow**: SSH auth via port 443, `gh` CLI authenticated, `.claude/settings.json` permissions, short-lived feature branches → PR → merge-commit-only; 3 PRs successfully cycled end-to-end

## What's Left to Build

- Chunks regenerated with all 8 properties (2026-04-22). Next: run first HPC sweep via `sbatch_sweep.sh` and confirm 0.138 MSE baseline reproduces on the new chunks.
- **Multi-property training (tiered)**: y-slicing is implemented — `ALL_PROPERTIES` + `prop_cols` in both `train_colab_rev.ipynb` and `run_sweep.py`. Change `PROPERTIES` in the config section to select a tier. Tier A (4 geometric props) → Tier B (+dynamical) → Tier C (+long-wavelength, report-only). Full plan: [docs/multi_property_training_plan.md](../../docs/multi_property_training_plan.md).
- Execute Goethe-HLR (AMD MI210 / ROCm) bootstrap end-to-end. Scaffolding is landed — `--no-zip`/`--sims-dir`/`--props-dir`/`--out-dir` flags on `prepare_colab_subset.py`, `CHUNKS_DIR` env on `run_sweep.py`, `scripts/bash/sbatch_preprocess.sh` + `sbatch_sweep.sh`, and [docs/hpc_goethe.md](docs/hpc_goethe.md). Remaining: rsync raw data to `/work`, install miniforge + ROCm 6.2 PyTorch inside a `gpu_test` allocation, and submit the first preprocess + smoke sweep
- Switch `MartiniHeteroGraphBuilder` to require `.tpr` file for topology instead of `.gro`
- Explore transfer to protein+membrane systems (long-term research goal)

## Current Status

### Phase: HP search stages 0–3 complete; re-preprocessing + Stage 5 pending

HP search complete through Stage 3. Winner: `hidden_dim=128, num_layers=2`, `lr=1e-4`, `wd=1e-3`. Val MSE at winner: **0.03816** (vs baseline 0.138, 3.6× improvement). Test MSE in old runs was artificially low due to narrow random test split — fixed by new stratified split. Next: re-preprocess chunks on HPC with `--split-method stratified --stratify-on lipid_packing thickness variation thickness_std`, then Stage 5 (5-seed confirmation on new chunks). Current best (old split, not directly comparable): test MSE ≈ 0.028 on a narrow test set.

## Known Issues

1. **Chunks must be regenerated** with the stratified split before Stage 5 and Tier A training. Current chunks (generated 2026-04-22 with `split_seed=0` random split) have a narrow test set — valid for HP selection only. Re-preprocess on HPC with `--split-method stratified --stratify-on lipid_packing thickness variation thickness_std`.
2. **Memory pressure**: Partially mitigated — removed `.pos` from graphs; spatial cutoff raised to 11.0 Å (doubles graph size vs 9.0 Å) but `num_frames` halved to 25 to compensate. Batch size still limited by VRAM.
3. **LIPID_TYPES consistency**: The 10-element lipid list must be identical across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py` — currently maintained manually.

## Deferred Ideas (Not Active Tasks)

- **Euclidean Fast Attention (EFA) block on the spatial channel.** Linear-cost, SE(3)-equivariant, globally-connected attention from Frank et al. (Nat. Mach. Intell. 2026, arXiv:2412.08541). Theoretical fit for the *spatial* edge type only — replaces the hard distance cutoff with a soft `sinc(ω·r)` kernel and reaches long-wavelength targets (`bending_modulus`, `compressibility`) that a cutoff-limited local MP cannot. Bonded channel should stay as `GATv2Conv` (chemistry needs explicit topology with force-constant edge features; EFA has no edge list). Requires the **PBC variant** of ERoPE, not the paper's default SO(3) integration, because Martini boxes are periodic and break the unit-cell SO(3) symmetry. Would reverse the 2026-04-18 `.pos` removal. Reconsider **only after** all 8 targets are implemented and simpler levers (deeper MP, HP sweeps, richer features) are exhausted. Suggested order when picked up: deeper-MP null hypothesis → readout-only EFA → per-layer parallel EFA. Full plan at [docs/efa_spatial_layer_future.md](../../docs/efa_spatial_layer_future.md).

## Evolution of Project Decisions

1. **Integer vocab → continuous physics features**: Originally used integer bead-type encodings with a learned embedding layer. Switched to continuous `[mass, charge, sigma, epsilon]` from Martini 3 FF for physics-informed input. `create_global_encoder` was deprecated.
2. **Single graph type → heterogeneous graph**: Moved to `HeteroData` with separate bonded and spatial edge types to distinguish chemical topology from physical proximity.
3. **Full in-memory loading → chunked disk streaming**: Added `MartiniDiskDataset` to handle memory constraints.
4. **GNN-only → optional composition mode**: Added composition vector concatenation as a model option to study whether bulk composition signals dominate or complement the GNN's per-bead features.
