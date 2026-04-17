# Progress

## What Works

- **Graph construction pipeline**: `MartiniHeteroGraphBuilder` successfully converts MD trajectories to `HeteroData` with continuous physics features, bonded + spatial edges, and composition vectors
- **Chunked dataset loading**: `MartiniDiskDataset` streams data from disk without OOM
- **Model forward pass**: `MembranePropertyGNN` runs in both GNN-only and GNN+composition modes
- **Force field parsing**: `ff_parser.py` extracts parameters from Martini 3 `.itp` files into JSON maps
- **Training infrastructure**: Local `run_sweep.py` (now chunk-based + W&B, mirrors the Colab notebook), linear baseline, smoke tests, result summarization all functional
- **Test suite**: 7 test files covering graph construction, dataset loading, model modes, FF parsing, and benchmarks
- **Documentation**: `README.md` covers goal, architecture, install, training entry points, data layout, and evaluation story
- **GitHub workflow**: SSH auth via port 443, `gh` CLI authenticated, `.claude/settings.json` permissions, short-lived feature branches → PR → merge-commit-only; 3 PRs successfully cycled end-to-end

## What's Left to Build

- Run the first full-scale sweep on Colab with `train_colab_rev.ipynb` (chunks regenerated with NUM_FRAMES=50)
- Train on more properties than the current `lipid_packing`+`thickness` pair — all 8 available targets are documented in [properties.md](properties.md)
- Set up remote HPC cluster deployment for full-scale training on AMD Instinct MI210 GPUs. PyTorch's ROCm build exposes the same `torch.cuda` API so model/training code stays unchanged; the work is mostly environment + deployment plumbing:
  - A non-zipping preprocessing entry point that writes `.pt` chunks directly on the cluster filesystem (preprocessing runs remotely to avoid shipping processed chunks back and forth)
  - A SLURM `sbatch` script for a GPU training job
  - Deployment channel: `git pull` for code, `scp`/`rsync` over SSH for the raw `.tpr`/`.xtc` sim data
  - Install path on the cluster must match the AMD ROCm PyTorch wheel and compatible PyG extensions
- Switch `MartiniHeteroGraphBuilder` to require `.tpr` file for topology instead of `.gro`
- Explore transfer to protein+membrane systems (long-term research goal)

## Current Status

### Phase: Ready for first full-scale Colab run

Full pipeline is implemented end-to-end and consistent between local and Colab: `prepare_colab_subset.py` bakes `NUM_FRAMES`/system into `.pt` chunks; both `scripts/training/run_sweep.py` (local) and `scripts/colab/train_colab_rev.ipynb` stream them via `MartiniDiskDataset` and run configurable sweeps logged to W&B. The zip contains only `processed/` and `lipid_gnn/` — no raw trajectory files. Current best results (from earlier smaller runs) — Overall Test MSE: **0.1378** (lipid_packing: 0.0566, thickness: 0.2190).

## Known Issues

1. **Memory pressure**: Training is constrained by both RAM and VRAM. Batch size and number of frames are limited.
2. **LIPID_TYPES consistency**: The 10-element lipid list must be identical across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py` — currently maintained manually.

## Evolution of Project Decisions

1. **Integer vocab → continuous physics features**: Originally used integer bead-type encodings with a learned embedding layer. Switched to continuous `[mass, charge, sigma, epsilon]` from Martini 3 FF for physics-informed input. `create_global_encoder` was deprecated.
2. **Single graph type → heterogeneous graph**: Moved to `HeteroData` with separate bonded and spatial edge types to distinguish chemical topology from physical proximity.
3. **Full in-memory loading → chunked disk streaming**: Added `MartiniDiskDataset` to handle memory constraints.
4. **GNN-only → optional composition mode**: Added composition vector concatenation as a model option to study whether bulk composition signals dominate or complement the GNN's per-bead features.
