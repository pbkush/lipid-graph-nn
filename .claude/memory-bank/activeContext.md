# Active Context

## Current Work Focus

Training pipeline is complete. Next focus is running the sweep on Colab and evaluating results.

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
- Available properties: `lipid_packing`, `thickness`, `thickness_std`, `compressibility`, `persistence`, `diffusivity`
- `prepare_colab_subset.py` fully rewritten: iterates `data/membrane_only/{comp}/run/` directly, times one probe frame and prints a runtime estimate, then calls `preprocess_and_save`
- Zip now contains only `processed/` (`.pt` chunks, 100 frames/system, chunk_size=50) and `lipid_gnn/` — no raw `.tpr`/`.xtc`, no property `.h5` files, no `resources/` (all baked into graph features)
- Spatial cutoff raised from 9.0 Å to 10.0 Å
- `TARGET_PROPERTIES`, `NUM_FRAMES`, `CHUNK_SIZE`, `SPATIAL_CUTOFF` are top-level constants in `prepare_colab_subset.py` for easy future adjustment

**Model & Graph Builder:**

- Migrated from categorical vocabulary encodings to continuous Martini FF parameters (mass, charge, sigma, epsilon) as node features, bond force/length as edge features
- `MembranePropertyGNN` upgraded to dual edge-type processing (bonded + spatial) via `GATv2Conv` with `GraphNorm`
- Vectorized spatial cutoff masking in `MartiniHeteroGraphBuilder` using NumPy (replaced slow Python loops for bonded-pair and self-loop removal)

## Recent Changes

**GitHub workflow setup (2026-04-17):**

- Created `requirements.txt` with pinned minimum versions for all direct dependencies (`torch>=2.8.0`, `torch-geometric>=2.7.0`, `MDAnalysis>=2.10.0`, `numpy`, `h5py`, `pandas`, `scikit-learn`, `matplotlib`, `tqdm`, `pytest`, `wandb`)
- Updated `.gitignore` to exclude large simulation files (`*.xtc`, `*.gro`, `*.tpr`, etc.), `colab_lipid_gnn_subset.zip`, `tmp/`, `.vscode/`, `build/`, `lipid_gnn.egg-info/`, and agent scratch files
- Cleared outputs from `scripts/colab/train_colab.ipynb` (369 KB → 17 KB) using `jupyter nbconvert --clear-output`
- Pushed all previously untracked files in 3 commits: packaging/config, core GNN modules + tests, training scripts + notebooks
- Fixed SSH auth: university network blocks port 22; added `ssh.github.com:443` to `~/.ssh/known_hosts` and configured `~/.ssh/config` to route all `github.com` connections through port 443
- Added `.claude/settings.json` with permission rules allowing `git push`, `git commit`, `git add`, `gh pr create`, `gh pr merge` without prompts; denies `git push --force`
- Repo is now fully up to date on GitHub with 6 commits on `main`

## Next Steps

- Run `prepare_colab_subset.py` locally to generate the new zip (100 frames/system, spatial cutoff 10.0 Å)
- Upload zip to Google Drive and run `train_colab_rev.ipynb` on Colab
- Evaluate sweep results in W&B; tune `SWEEP` grid based on findings
- Disable squash/rebase merging in GitHub repo settings (Settings → General → Pull Requests)

## Important Patterns and Preferences

- Test discrete components locally (dedicated test scripts, baseline metrics) before integrating into the heavy end-to-end `train_colab.ipynb` loop
- Results are uploaded to Weights & Biases for visualization. No model weights are saved currently, but this may change
- Force field parameters are loaded from JSON files at graph build time, not parsed from `.itp` at training time
- `LIPID_TYPES` ordering must stay consistent across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py`
- `preprocess_and_save` is the single entry point for building and saving graph chunks; callers are responsible for constructing `sim_tuples`
