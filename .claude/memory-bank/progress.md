# Progress

## What Works

- **Graph construction pipeline**: `MartiniHeteroGraphBuilder` successfully converts MD trajectories to `HeteroData` with continuous physics features, bonded + spatial edges, and composition vectors
- **Chunked dataset loading**: `MartiniDiskDataset` streams data from disk without OOM
- **Model forward pass**: `MembranePropertyGNN` runs in both GNN-only and GNN+composition modes
- **Force field parsing**: `ff_parser.py` extracts parameters from Martini 3 `.itp` files into JSON maps
- **Training infrastructure**: Local `run_sweep.py` (chunk-based + W&B + AMP, mirrors the Colab notebook), linear baseline, smoke tests, result summarization all functional
- **Test suite**: 8 test files, 24 tests covering graph construction, dataset loading, model modes, FF parsing, benchmarks, and multi-frame preprocessing (including interleaving invariant)
- **Documentation**: `README.md` covers goal, architecture, install, training entry points, data layout, and evaluation story
- **GitHub workflow**: SSH auth via port 443, `gh` CLI authenticated, `.claude/settings.json` permissions, short-lived feature branches → PR → merge-commit-only; 3 PRs successfully cycled end-to-end

## What's Left to Build

- Regenerate chunks with interleaved layout, upload zip, and run first full-scale sweep on Colab with `train_colab_rev.ipynb`
- Train on more properties than the current `lipid_packing`+`thickness` pair — all 8 available targets are documented in [properties.md](properties.md)
- Execute Goethe-HLR (AMD MI210 / ROCm) bootstrap end-to-end. Scaffolding is landed — `--no-zip`/`--sims-dir`/`--props-dir`/`--out-dir` flags on `prepare_colab_subset.py`, `CHUNKS_DIR` env on `run_sweep.py`, `scripts/bash/sbatch_preprocess.sh` + `sbatch_sweep.sh`, and [docs/hpc_goethe.md](docs/hpc_goethe.md). Remaining: rsync raw data to `/work`, install miniforge + ROCm 6.2 PyTorch inside a `gpu_test` allocation, and submit the first preprocess + smoke sweep
- Switch `MartiniHeteroGraphBuilder` to require `.tpr` file for topology instead of `.gro`
- Explore transfer to protein+membrane systems (long-term research goal)

## Current Status

### Phase: Ready for first full-scale Colab run

Full pipeline is implemented end-to-end and consistent between local and Colab: `prepare_colab_subset.py` bakes `NUM_FRAMES`/system into `.pt` chunks; both `scripts/training/run_sweep.py` (local) and `scripts/colab/train_colab_rev.ipynb` stream them via `MartiniDiskDataset` and run configurable sweeps logged to W&B. The zip contains only `processed/` and `lipid_gnn/` — no raw trajectory files. Current best results (from earlier smaller runs) — Overall Test MSE: **0.1378** (lipid_packing: 0.0566, thickness: 0.2190).

## Known Issues

1. **Chunks must be regenerated**: interleaving fix requires a fresh preprocessing run before training. Old chunks produce system-homogeneous batches → model collapses to mean.
2. **Memory pressure**: Partially mitigated — removed `.pos` from graphs, spatial cutoff at 9.0 Å. Batch size still limited by VRAM.
3. **LIPID_TYPES consistency**: The 10-element lipid list must be identical across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py` — currently maintained manually.

## Evolution of Project Decisions

1. **Integer vocab → continuous physics features**: Originally used integer bead-type encodings with a learned embedding layer. Switched to continuous `[mass, charge, sigma, epsilon]` from Martini 3 FF for physics-informed input. `create_global_encoder` was deprecated.
2. **Single graph type → heterogeneous graph**: Moved to `HeteroData` with separate bonded and spatial edge types to distinguish chemical topology from physical proximity.
3. **Full in-memory loading → chunked disk streaming**: Added `MartiniDiskDataset` to handle memory constraints.
4. **GNN-only → optional composition mode**: Added composition vector concatenation as a model option to study whether bulk composition signals dominate or complement the GNN's per-bead features.
