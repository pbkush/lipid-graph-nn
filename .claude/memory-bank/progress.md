# Progress

## What Works

- **Graph construction pipeline**: `MartiniHeteroGraphBuilder` successfully converts MD trajectories to `HeteroData` with continuous physics features, bonded + spatial edges, and composition vectors
- **Chunked dataset loading**: `MartiniDiskDataset` streams data from disk without OOM
- **Model forward pass**: `MembranePropertyGNN` runs in both GNN-only and GNN+composition modes
- **Force field parsing**: `ff_parser.py` extracts parameters from Martini 3 `.itp` files into JSON maps
- **Training infrastructure**: Hyperparameter sweep (`run_sweep.py`), linear baseline, smoke tests, result summarization all functional
- **Test suite**: 7 test files covering graph construction, dataset loading, model modes, FF parsing, and benchmarks

## What's Left to Build

- Generate zip with `prepare_colab_subset.py` and run the first sweep on Colab with `train_colab_rev.ipynb`
- Explore transfer to protein+membrane systems (long-term research goal)
- Switch `MartiniHeteroGraphBuilder` to require `.tpr` file for topology instead of `.gro`
- Disable squash/rebase merging in GitHub repo settings (Settings → General → Pull Requests)

## Current Status

### Phase: Ready for first full-scale Colab run

Full pipeline is implemented end-to-end: `prepare_colab_subset.py` bakes 100 frames/system into `.pt` chunks; `train_colab_rev.ipynb` streams them via `MartiniDiskDataset` and runs configurable sweeps logged to W&B. The zip contains only `processed/` and `lipid_gnn/` — no raw trajectory files. Current best results (from earlier smaller runs) — Overall Test MSE: **0.1378** (lipid_packing: 0.0566, thickness: 0.2190).

## Known Issues

1. **Memory pressure**: Training is constrained by both RAM and VRAM. Batch size and number of frames are limited.
2. **LIPID_TYPES consistency**: The 10-element lipid list must be identical across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py` — currently maintained manually.

## Evolution of Project Decisions

1. **Integer vocab → continuous physics features**: Originally used integer bead-type encodings with a learned embedding layer. Switched to continuous `[mass, charge, sigma, epsilon]` from Martini 3 FF for physics-informed input. `create_global_encoder` was deprecated.
2. **Single graph type → heterogeneous graph**: Moved to `HeteroData` with separate bonded and spatial edge types to distinguish chemical topology from physical proximity.
3. **Full in-memory loading → chunked disk streaming**: Added `MartiniDiskDataset` to handle memory constraints.
4. **GNN-only → optional composition mode**: Added composition vector concatenation as a model option to study whether bulk composition signals dominate or complement the GNN's per-bead features.
