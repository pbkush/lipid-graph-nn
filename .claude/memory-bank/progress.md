# Progress

## What Works

- **Graph construction pipeline**: `MartiniHeteroGraphBuilder` successfully converts MD trajectories to `HeteroData` with continuous physics features, bonded + spatial edges, and composition vectors
- **Chunked dataset loading**: `MartiniDiskDataset` streams data from disk without OOM
- **Model forward pass**: `MembranePropertyGNN` runs in both GNN-only and GNN+composition modes
- **Force field parsing**: `ff_parser.py` extracts parameters from Martini 3 `.itp` files into JSON maps
- **Training infrastructure**: Local `run_sweep.py` (chunk-based + W&B + AMP, mirrors the Colab notebook), linear baseline, smoke tests, result summarization all functional
- **HP analysis tooling**: `scripts/python/download_wandb_runs.py` pulls W&B groups to `logs/training/`; `scripts/notebooks/analyze_hp_search.ipynb` aggregates over seeds, ranks HP cells, and produces 7 visualizations (loss curves, heatmap, training stats, system metrics). `docs/analyze_hp_search_notebook.md` documents each visualization.
- **HP search stages 0–2 complete**: Stage 0 baseline MSE ≈ 0.138 reproduced; Stage 1 locked `lr=1e-4`; Stage 2 locked `wd=1e-3` (updated in `config.yaml`). Stage 3 architecture grid (`hidden_dim × num_layers`) is next.
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

### Phase: HP search in progress — Stage 3 (architecture grid) pending

Full pipeline implemented end-to-end. HP search stages 0–2 complete on Goethe HPC (AMD MI210 / ROCm 6.2): baseline MSE ≈ 0.138 reproduced, `lr=1e-4` and `wd=1e-3` locked. Stage 3 (`hidden_dim ∈ {32,64,128}` × `num_layers ∈ {2,3,4}`, 18 cells × 2 seeds) is next to submit. W&B offline analysis tooling is ready: download runs with `download_wandb_runs.py`, analyze with `analyze_hp_search.ipynb`. Current best results — Overall Test MSE: **0.1378** (lipid_packing: 0.0566, thickness: 0.2190).

## Known Issues

1. **Chunks**: Chunks (all 8 properties, `y.shape [1, 8]`, interleaved, 3-directory layout) generated 2026-04-22. Old chunks from before this date are incompatible. Baseline MSE ≈ 0.138 has been confirmed on HPC (Stage 0).
2. **Memory pressure**: Partially mitigated — removed `.pos` from graphs; spatial cutoff raised to 11.0 Å (doubles graph size vs 9.0 Å) but `num_frames` halved to 25 to compensate. Batch size still limited by VRAM.
3. **LIPID_TYPES consistency**: The 10-element lipid list must be identical across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py` — currently maintained manually.

## Deferred Ideas (Not Active Tasks)

- **Euclidean Fast Attention (EFA) block on the spatial channel.** Linear-cost, SE(3)-equivariant, globally-connected attention from Frank et al. (Nat. Mach. Intell. 2026, arXiv:2412.08541). Theoretical fit for the *spatial* edge type only — replaces the hard distance cutoff with a soft `sinc(ω·r)` kernel and reaches long-wavelength targets (`bending_modulus`, `compressibility`) that a cutoff-limited local MP cannot. Bonded channel should stay as `GATv2Conv` (chemistry needs explicit topology with force-constant edge features; EFA has no edge list). Requires the **PBC variant** of ERoPE, not the paper's default SO(3) integration, because Martini boxes are periodic and break the unit-cell SO(3) symmetry. Would reverse the 2026-04-18 `.pos` removal. Reconsider **only after** all 8 targets are implemented and simpler levers (deeper MP, HP sweeps, richer features) are exhausted. Suggested order when picked up: deeper-MP null hypothesis → readout-only EFA → per-layer parallel EFA. Full plan at [docs/efa_spatial_layer_future.md](../../docs/efa_spatial_layer_future.md).

## Evolution of Project Decisions

1. **Integer vocab → continuous physics features**: Originally used integer bead-type encodings with a learned embedding layer. Switched to continuous `[mass, charge, sigma, epsilon]` from Martini 3 FF for physics-informed input. `create_global_encoder` was deprecated.
2. **Single graph type → heterogeneous graph**: Moved to `HeteroData` with separate bonded and spatial edge types to distinguish chemical topology from physical proximity.
3. **Full in-memory loading → chunked disk streaming**: Added `MartiniDiskDataset` to handle memory constraints.
4. **GNN-only → optional composition mode**: Added composition vector concatenation as a model option to study whether bulk composition signals dominate or complement the GNN's per-bead features.
