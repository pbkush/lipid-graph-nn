# Active Context

## Current Work Focus

Training pipeline is complete with AMP and prefetch optimisations. Next focus is running the sweep on Colab with batch_size=4 and evaluating results.

## Previous Changes

**Data Pipeline:**

- Implemented disk-backed data loading (`MartiniDiskDataset` in `dataset.py`) to solve Colab RAM exhaustion from loading thousands of graphs into memory
- Custom `IterableDataset` lazily streams `.pt` chunk files: sequential for validation, shuffled (chunk-level + intra-chunk) for training
- Multi-worker prefetching via `get_worker_info()` to prevent GPU starvation during disk I/O
- Added `tests/test_dataset.py` to validate data integrity and sampling before long Colab runs

**Colab notebook (`train_colab_rev.ipynb`):**

- New notebook replaces `load_colab_data()` (raw trajectory graph building) with `MartiniDiskDataset` streaming from preprocessed `.pt` chunks
- Experiment configuration consolidated into two dicts: `FIXED` (shared across all runs) and `SWEEP` (grid of values to cross-product); `itertools.product` expands into a flat experiment list that is printed before any training starts
- PyG extension install now uses dynamic URL constructed from live torch/CUDA versions (no hardcoded version string)
- `sys.path.insert` on `colab_lipid_gnn_subset/` so all imports use `from lipid_gnn.xxx` directly
- `StandardScaler` fit via a y-only pre-pass over training chunks (no full graphs in RAM)
- `train_one_run(cfg, scaler, train_dataset, val_dataset)` is a flat, self-contained function: normalizes targets per-batch using scaler tensors on-device, logs per-property MSE + R² to W&B each epoch, logs final accuracy plot as W&B image
- MDAnalysis and mdtraj removed from install (not needed at training time)
- W&B run name encodes all sweep dimensions: `{comp_mode}_h{hidden_dim}_l{num_layers}_lr{lr}_s{seed}`

**Preprocessing pipeline refactor (`prepare_colab_subset.py` + `dataset.py`):**

- `preprocess_and_save` signature changed: now takes `sim_tuples` (list of `(tpr, xtc, h5)` paths) instead of flat `sims_dir`/`props_dir`; `target_properties` is now a required arg with no default
- Property validation added: checks all requested properties exist in the first `.h5` file before the expensive loop starts, raises `ValueError` listing available properties if any are missing
- Available properties (8 total): see [properties.md](properties.md) for the full list, physical meaning, units, and computation details
- `prepare_colab_subset.py` fully rewritten: iterates `data/membrane_only/{comp}/run/` directly, times one probe frame and prints a runtime estimate, then calls `preprocess_and_save`
- Zip now contains only `processed/` (`.pt` chunks, 100 frames/system, chunk_size=50) and `lipid_gnn/` — no raw `.tpr`/`.xtc`, no property `.h5` files, no `resources/` (all baked into graph features)
- Spatial cutoff raised from 9.0 Å to 10.0 Å
- `TARGET_PROPERTIES`, `NUM_FRAMES`, `CHUNK_SIZE`, `SPATIAL_CUTOFF` are top-level constants in `prepare_colab_subset.py` for easy future adjustment

**Model & Graph Builder:**

- Migrated from categorical vocabulary encodings to continuous Martini FF parameters (mass, charge, sigma, epsilon) as node features, bond force/length as edge features
- `MembranePropertyGNN` upgraded to dual edge-type processing (bonded + spatial) via `GATv2Conv` with `GraphNorm`
- Vectorized spatial cutoff masking in `MartiniHeteroGraphBuilder` using NumPy (replaced slow Python loops for bonded-pair and self-loop removal)

## Recent Changes

**AMP + DataLoader throughput (2026-04-18):**

- Enabled `torch.amp.autocast` (forward + val inference) and `torch.amp.GradScaler` (backward/step) in both `run_sweep.py` and `train_colab_rev.ipynb`. Auto-disabled on CPU (`use_amp = device.type == 'cuda'`). Halves activation memory on GPU, enabling larger batch sizes.
- `DataLoader` now uses `pin_memory=True`, `persistent_workers=True`, `prefetch_factor=2` (guarded against `num_workers=0`). Workers stay alive across epochs; graphs pre-staged in pinned RAM for faster GPU transfer.
- `FIXED["batch_size"]` raised from 2 → 4 in both files. If the largest sweep cells (`hidden_dim=64, num_layers=3`) still OOM, lower `spatial_cutoff` during chunk regeneration (8.0 Å) as the next lever.
- Added `.detach()` to per-property loss accumulator in `run_sweep.py` to avoid keeping `out` tensors live across batches.

**Graph memory optimizations (2026-04-18):**

- Removed `data['bead'].pos = current_pos` from `MartiniHeteroGraphBuilder.process_frame()` — positions are needed to compute spatial edges during preprocessing but serve no purpose in the saved graph (the GNN uses `node_x` features and pre-encoded RBF edge attributes, never raw coordinates). Eliminated N×3×float32 bytes per graph.
- Reduced default `spatial_cutoff` from 10–11 Å → **7.5 Å** across all three sites: `MartiniHeteroGraphBuilder.__init__`, `dataset.preprocess_and_save`, and `prepare_colab_subset.py`. Martini first-shell neighbors are at ~4.7 Å; the old default captured a dense second shell unnecessarily. **Requires regenerating `.pt` chunks** to take effect.

**Benchmark script enhancements (2026-04-18):**

- `load_real_data()` and `run_memory_scaling_test()` now accept `spatial_cutoff` as a parameter (was hardcoded to 11.0).
- New `print_graph_stats(data, label)`: prints node count, bonded/spatial edge counts, avg degree, min/max spatial degree per node.
- New `describe_graph_memory(data, label)`: per-tensor breakdown (name, shape, dtype, MB) + total.
- New `compare_built_vs_pt(args)` (`--compare-pt`): builds a live graph, saves/reloads via `.pt`, compares tensor keys and memory delta — detects stale tensors (e.g. old `.pos`) in existing chunks. Optionally shows a three-way comparison if `--processed-dir` points to existing chunks.
- New `compare_graph_memory(args)` (`--compare-mem`): builds at cutoff 11.0 vs 7.5, optionally adds a chunk graph, prints a delta summary table.
- New CLI flags: `--graph-stats`, `--compare-mem`, `--compare-pt`, `--processed-dir`, `--spatial-cutoff`.
- **Run benchmarks via `scripts/bash/run_benchmark.sh`**, not directly with python — it saves timestamped logs to `logs/benchmarks/benchmark_YYYYMMDD_HHMMSS.log`. All python args pass through via `"$@"`.
- Example full run (everything except stress test): `bash scripts/bash/run_benchmark.sh --use-real --real-system POPC100 --graph-stats --compare-mem --compare-pt --processed-dir data/processed --mem-test --skip-stress`

**Spatial cutoff corrected to 9.0 Å + benchmark three-way comparison (2026-04-18):**

- Benchmark at 7.5 Å revealed nodes with **zero spatial edges**. Root cause: after removing bonded pairs (~4.7 Å), the non-bonded search window is only 2.8 Å wide. Terminal tail beads (C4A, C4B) with one bonded neighbor can have no other bead within that window in disordered bilayer frames.
- This is physically significant: Martini's own non-bonded cutoff is 11–12 Å; beads with no spatial edges lose local packing density signal, which is critical for predicting thickness, compressibility, and diffusivity.
- **Default `spatial_cutoff` raised from 7.5 → 9.0 Å** across all three sites (`MartiniHeteroGraphBuilder.__init__`, `dataset.preprocess_and_save`, `prepare_colab_subset.py --spatial-cutoff`). 9.0 Å (~1.9σ) reliably covers the first non-bonded shell for all Martini bead types.
- Added runtime `warnings.warn` in `process_frame()` if any bead has zero spatial edges after building the graph — fires at any cutoff, useful diagnostically.
- `compare_graph_memory` (`--compare-mem`) now tests all three cutoffs **7.5 / 9.0 / 11.0 Å** in one run. 11.0 Å is the physics reference baseline. Summary table shows MB, ΔMB, Δ%, spatial edge count, and isolated bead count with a `!` flag on any row with isolated beads.
- `print_graph_stats` now reports isolated bead count (degree-0 in the spatial stream) alongside min/max degree.
- **Requires regenerating `.pt` chunks** after the cutoff change.

**Test suite fix (2026-04-18):**

- `tests/test_multi_frame_loading.py` was importing `load_data` from `run_sweep.py`, which was removed in PR #3 when the codebase moved to chunked preprocessing. Rewritten to test `preprocess_and_save()` in `dataset.py` instead — same multi-frame linspace sampling logic, now in its correct location.
- 3 new tests in `tests/benchmark_heterognn_test.py` for `print_graph_stats`, `describe_graph_memory`, and `_compare_graphs_roundtrip`. All 23 tests pass.

**GitHub workflow setup (2026-04-17):**

- Created `requirements.txt` with pinned minimum versions for all direct dependencies (`torch>=2.8.0`, `torch-geometric>=2.7.0`, `MDAnalysis>=2.10.0`, `numpy`, `h5py`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `pytest`, `wandb`)
- Updated `.gitignore` to exclude large simulation files (`*.xtc`, `*.gro`, `*.tpr`, etc.), `colab_lipid_gnn_subset.zip`, `tmp/`, `.vscode/`, `build/`, `lipid_gnn.egg-info/`, and agent scratch files
- Cleared outputs from `scripts/colab/train_colab.ipynb` (369 KB → 17 KB) using `jupyter nbconvert --clear-output`
- Fixed SSH auth: university network blocks port 22; added `ssh.github.com:443` to `~/.ssh/known_hosts` and configured `~/.ssh/config` to route all `github.com` connections through port 443
- Installed and authenticated `gh` CLI (HTTPS token in system keyring); Claude can now open/merge PRs directly from the terminal
- Added `.claude/settings.json` with permission rules allowing `git push`, `git commit`, `git add`, `gh pr create`, `gh pr merge` without prompts; denies `git push --force`
- Established workflow: short-lived feature branches (`feat/`, `fix/`, `exp/`, etc.) → PR → `gh pr merge --merge --delete-branch` (merge commits only, never squash/rebase)
- First end-to-end PR cycle completed: `fix/preprocess-stale-chunks` → PR #1 merged into `main`

**Bug fix — stale chunk files in preprocessing (PR #1, commit 5cbc094):**

- `preprocess_and_save()` in `dataset.py` now deletes existing `chunk_*.pt` files in `processed_dir` before writing new ones
- Previously, if a new run produced fewer chunks than the last, leftover chunks from the prior run silently mixed with the new ones in `MartiniDiskDataset`

**README (PR #2, commit 4e90a69):**

- Replaced the one-line stub `README.md` with a single-page overview: goal, architecture, install, training (smoke test / local sweep / Colab), expected data layout, evaluation story, repository layout
- `reinstall.sh` deleted — not referenced from the README; install path is `pip install --use-pep517 .` + `pip install -r requirements.txt`

**CLI args for prepare_colab_subset.py (PR #4, commit 95b9e61):**

- Replaced the module-level constants (`TARGET_PROPERTIES`, `NUM_FRAMES`, `CHUNK_SIZE`, `SPATIAL_CUTOFF`) with `argparse` flags so per-experiment tweaks no longer churn git history
- Flags: `--properties` (nargs='+' with `choices=AVAILABLE_PROPERTIES`), `--num-frames`, `--chunk-size`, `--spatial-cutoff`, `--subset-name`; `--help` lists all valid property names
- `AVAILABLE_PROPERTIES` expanded to 8 targets (added `bending_modulus`, `variation`)
- Established the "CLI for tunable runnable scripts" design pattern, recorded in `systemPatterns.md` — next candidate is the `FIXED` dict in `run_sweep.py`

**run_sweep.py aligned with notebook (PR #3, commit 5db84d2):**

- Rewrote `scripts/training/run_sweep.py` to mirror `scripts/colab/train_colab_rev.ipynb` exactly: reads `.pt` chunks from `colab_lipid_gnn_subset/processed/` via `MartiniDiskDataset`, uses `FIXED` + `SWEEP` dicts expanded with `itertools.product`, and calls the notebook's `train_one_run()` verbatim
- Dropped `load_data()` (raw-trajectory path) and local `results/training/` artifacts — all metrics/plots now go to W&B only
- `wandb login` required once before first local run; chunks must be generated by `prepare_colab_subset.py` first

## Next Steps

- Regenerate `.pt` chunks at 9.0 Å cutoff, upload zip to Google Drive, and run `train_colab_rev.ipynb` with `batch_size=4`
- If `batch_size=4` still OOMs on `hidden_dim=64, num_layers=3` cells, try regenerating chunks at `spatial_cutoff=8.0`
- Evaluate sweep results in W&B; tune `SWEEP` grid based on findings
- Train on more of the 8 available properties (currently only `lipid_packing` + `thickness`)
- Plan and implement remote HPC cluster deployment: non-zipping preprocessing entry point, SLURM sbatch for GPU training, `git pull` for code + `scp`/`rsync` for raw sim data, ROCm-compatible install on AMD MI210

Decisions recorded:

- GitHub squash/rebase merging disabled in repo settings (enforcement now automatic, not just convention)
- CLI-arg pattern will NOT be applied to `run_sweep.py` — the FIXED/SWEEP split + cartesian product grid doesn't translate cleanly to command-line flags

## Important Patterns and Preferences

- Test discrete components locally (dedicated test scripts, baseline metrics) before integrating into the heavy end-to-end `train_colab.ipynb` loop
- Results are uploaded to Weights & Biases for visualization. No model weights are saved currently, but this may change
- Force field parameters are loaded from JSON files at graph build time, not parsed from `.itp` at training time
- `LIPID_TYPES` ordering must stay consistent across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py`
- `preprocess_and_save` is the single entry point for building and saving graph chunks; callers are responsible for constructing `sim_tuples`
