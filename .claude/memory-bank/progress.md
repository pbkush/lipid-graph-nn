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
- **Stage 5 analysis pipeline**: `dataset.py` tags graphs with `composition` + `system_idx`; `run_sweep.py` saves `test_artifacts.npz` per run (bug-fixed 2026-04-26 — was missing despite being in commit message) and uploads via `wandb.save()`; `download_wandb_runs.py` fetches it via `run.files()`; `linear_baseline.py` has `--stratified` mode matching the same `.npz` format. `scripts/notebooks/analyze_stage_5.ipynb` produces 9 publication-grade figures + `headline_numbers.json` with bootstrap CIs and paired t-test.
- **Per-property test MSE**: `run_sweep.py` logs `test/mse_{prop}` for each active property to W&B summary, visible without downloading artifacts.
- **Tier A plan**: `docs/tier_a_4prop_plan.md` — lightweight Stage 1b lr check (6 runs) + Stage 5b (5 seeds) for 4-property training. No chunk rebuild needed.
- **Test suite**: 9 test files, 42 tests covering graph construction, dataset loading, model modes, FF parsing, benchmarks, multi-frame preprocessing, interleaving invariant, train/val/test split disjointness, all-8-property y-shape invariant, config loading/validation/env-override, composition label preservation, and stratified split coverage/disjointness.
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

### Phase: Stage 0 re-run + Stage 5 running on HPC; Tier A next

HP search complete through Stage 3. Winner: `hidden_dim=128, num_layers=2`, `lr=1e-4`, `wd=1e-3`. Val MSE at winner: **0.03816** (vs baseline 0.138, 3.6× improvement). New stratified chunks preprocessed. Stage 0 re-run and Stage 5 (5 seeds each) currently running on HPC with `WANDB_GROUP=stage_0_baseline` / `stage_5_confirm`. After those finish: download runs (including `test_artifacts.npz`), run `analyze_stage_5.ipynb`, then activate Tier A by setting `active_properties: [lipid_packing, thickness, thickness_std, variation]` in `config.yaml` and running Stage 1b.

## Known Issues

1. **Memory pressure**: Partially mitigated — removed `.pos` from graphs; spatial cutoff raised to 11.0 Å (doubles graph size vs 9.0 Å) but `num_frames` halved to 25 to compensate. Batch size still limited by VRAM.
2. **LIPID_TYPES consistency**: The 10-element lipid list must be identical across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py` — currently maintained manually.

## Deferred Ideas (Not Active Tasks)

- **Euclidean Fast Attention (EFA) block on the spatial channel.** Linear-cost, SE(3)-equivariant, globally-connected attention from Frank et al. (Nat. Mach. Intell. 2026, arXiv:2412.08541). Theoretical fit for the *spatial* edge type only — replaces the hard distance cutoff with a soft `sinc(ω·r)` kernel and reaches long-wavelength targets (`bending_modulus`, `compressibility`) that a cutoff-limited local MP cannot. Bonded channel should stay as `GATv2Conv` (chemistry needs explicit topology with force-constant edge features; EFA has no edge list). Requires the **PBC variant** of ERoPE, not the paper's default SO(3) integration, because Martini boxes are periodic and break the unit-cell SO(3) symmetry. Would reverse the 2026-04-18 `.pos` removal. Reconsider **only after** all 8 targets are implemented and simpler levers (deeper MP, HP sweeps, richer features) are exhausted. Suggested order when picked up: deeper-MP null hypothesis → readout-only EFA → per-layer parallel EFA. Full plan at [docs/efa_spatial_layer_future.md](../../docs/efa_spatial_layer_future.md).

## Evolution of Project Decisions

1. **Integer vocab → continuous physics features**: Originally used integer bead-type encodings with a learned embedding layer. Switched to continuous `[mass, charge, sigma, epsilon]` from Martini 3 FF for physics-informed input. `create_global_encoder` was deprecated.
2. **Single graph type → heterogeneous graph**: Moved to `HeteroData` with separate bonded and spatial edge types to distinguish chemical topology from physical proximity.
3. **Full in-memory loading → chunked disk streaming**: Added `MartiniDiskDataset` to handle memory constraints.
4. **GNN-only → optional composition mode**: Added composition vector concatenation as a model option to study whether bulk composition signals dominate or complement the GNN's per-bead features.
