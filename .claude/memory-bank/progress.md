# Progress

## What Works

- **Graph construction pipeline**: `MartiniHeteroGraphBuilder` successfully converts MD trajectories to `HeteroData` with continuous physics features, bonded + spatial edges, and composition vectors
- **Chunked dataset loading**: `MartiniDiskDataset` streams data from disk without OOM
- **Model forward pass**: `MembranePropertyGNN` runs in both GNN-only and GNN+composition modes
- **Force field parsing**: `ff_parser.py` extracts parameters from Martini 3 `.itp` files into JSON maps
- **Training infrastructure**: Local `run_sweep.py` (chunk-based + W&B + AMP, mirrors the Colab notebook), linear baseline, smoke tests, result summarization all functional
- **Test suite**: 8 test files, 26 tests covering graph construction, dataset loading, model modes, FF parsing, benchmarks, multi-frame preprocessing, interleaving invariant, train/val/test split disjointness, and all-8-property y-shape invariant
- **Documentation**: `README.md` covers goal, architecture, install, training entry points, data layout, and evaluation story
- **GitHub workflow**: SSH auth via port 443, `gh` CLI authenticated, `.claude/settings.json` permissions, short-lived feature branches → PR → merge-commit-only; 3 PRs successfully cycled end-to-end

## What's Left to Build

- Chunks regenerated with all 8 properties (2026-04-22, uploading to Google Drive). Next: run first full-scale Colab sweep and confirm 0.138 MSE baseline reproduces on the new chunks.
- **Multi-property training (tiered)**: chunks already store all 8 targets — adopt properties at training time by column-slicing `y`, no preprocessing needed. Tier A (4 geometric props) → Tier B (+dynamical) → Tier C (+long-wavelength, report-only). Full plan: [docs/multi_property_training_plan.md](../../docs/multi_property_training_plan.md).
- Execute Goethe-HLR (AMD MI210 / ROCm) bootstrap end-to-end. Scaffolding is landed — `--no-zip`/`--sims-dir`/`--props-dir`/`--out-dir` flags on `prepare_colab_subset.py`, `CHUNKS_DIR` env on `run_sweep.py`, `scripts/bash/sbatch_preprocess.sh` + `sbatch_sweep.sh`, and [docs/hpc_goethe.md](docs/hpc_goethe.md). Remaining: rsync raw data to `/work`, install miniforge + ROCm 6.2 PyTorch inside a `gpu_test` allocation, and submit the first preprocess + smoke sweep
- Switch `MartiniHeteroGraphBuilder` to require `.tpr` file for topology instead of `.gro`
- Explore transfer to protein+membrane systems (long-term research goal)

## Current Status

### Phase: Chunks ready, first full-scale Colab run pending

Full pipeline is implemented end-to-end and consistent between local and Colab: `prepare_colab_subset.py` bakes `NUM_FRAMES`/system into `.pt` chunks; both `scripts/training/run_sweep.py` (local) and `scripts/colab/train_colab_rev.ipynb` stream them via `MartiniDiskDataset` and run configurable sweeps logged to W&B. The zip contains only `processed/` and `lipid_gnn/` — no raw trajectory files. Current best results (from earlier smaller runs) — Overall Test MSE: **0.1378** (lipid_packing: 0.0566, thickness: 0.2190).

## Known Issues

1. **New chunks uploading**: Chunks rebuilt with all 8 properties, interleaved, 3-directory layout — upload in progress. Old chunks from before 2026-04-22 are incompatible (missing `val/`/`test/` dirs and stale `y` dimensions).
2. **Memory pressure**: Partially mitigated — removed `.pos` from graphs; spatial cutoff raised to 11.0 Å (doubles graph size vs 9.0 Å) but `num_frames` halved to 25 to compensate. Batch size still limited by VRAM.
3. **LIPID_TYPES consistency**: The 10-element lipid list must be identical across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py` — currently maintained manually.

## Deferred Ideas (Not Active Tasks)

- **Euclidean Fast Attention (EFA) block on the spatial channel.** Linear-cost, SE(3)-equivariant, globally-connected attention from Frank et al. (Nat. Mach. Intell. 2026, arXiv:2412.08541). Theoretical fit for the *spatial* edge type only — replaces the hard distance cutoff with a soft `sinc(ω·r)` kernel and reaches long-wavelength targets (`bending_modulus`, `compressibility`) that a cutoff-limited local MP cannot. Bonded channel should stay as `GATv2Conv` (chemistry needs explicit topology with force-constant edge features; EFA has no edge list). Requires the **PBC variant** of ERoPE, not the paper's default SO(3) integration, because Martini boxes are periodic and break the unit-cell SO(3) symmetry. Would reverse the 2026-04-18 `.pos` removal. Reconsider **only after** all 8 targets are implemented and simpler levers (deeper MP, HP sweeps, richer features) are exhausted. Suggested order when picked up: deeper-MP null hypothesis → readout-only EFA → per-layer parallel EFA. Full plan at [docs/efa_spatial_layer_future.md](../../docs/efa_spatial_layer_future.md).

## Evolution of Project Decisions

1. **Integer vocab → continuous physics features**: Originally used integer bead-type encodings with a learned embedding layer. Switched to continuous `[mass, charge, sigma, epsilon]` from Martini 3 FF for physics-informed input. `create_global_encoder` was deprecated.
2. **Single graph type → heterogeneous graph**: Moved to `HeteroData` with separate bonded and spatial edge types to distinguish chemical topology from physical proximity.
3. **Full in-memory loading → chunked disk streaming**: Added `MartiniDiskDataset` to handle memory constraints.
4. **GNN-only → optional composition mode**: Added composition vector concatenation as a model option to study whether bulk composition signals dominate or complement the GNN's per-bead features.
