# Active Context

## Current Work Focus

**Stage 0 re-run + Stage 5 in progress (2026-04-26)** — `gnn_only`, 2 properties (`lipid_packing` + `thickness`), on stratified chunks.

Locked HPs (from Stages 1–3): `hidden_dim=128`, `num_layers=2`, `lr=1e-4`, `wd=1e-3`

Stage 0 re-run (`WANDB_GROUP=stage_0_baseline`, 5 seeds) and Stage 5 (`WANDB_GROUP=stage_5_confirm`, 5 seeds) are **currently running** on the HPC on the new stratified chunks.

**Next after runs finish:**

1. `python scripts/python/download_wandb_runs.py --group stage_0_baseline stage_5_confirm` — now also downloads `test_artifacts.npz` via `run.files()`.
2. Open `analyze_stage_5.ipynb`, set `GROUP = "stage_5_confirm"`, run all 9 figures + `headline_numbers.json`.
3. Move to **Tier A** — 4 properties. See [docs/tier_a_4prop_plan.md](../../docs/tier_a_4prop_plan.md).

**Tier A activation** (no chunk rebuild needed):

```yaml
# config.yaml
vocab:
  active_properties: [lipid_packing, thickness, thickness_std, variation]
```

Then run Stage 1b (6 runs, lr sanity check) before Stage 5b.

## Latest Changes

**Tier A 4-property plan + per-property test logging (2026-04-26):**

- `scripts/training/run_sweep.py`: now logs `test/mse_{prop}` for every property alongside `test/mse_total` — visible directly in W&B without downloading artifacts. Replaces the single scalar with a dict `test_metrics` built in a loop over `properties`.
- `scripts/python/download_wandb_runs.py` `_download_run`: now fetches W&B file artifacts uploaded via `wandb.save()` (specifically `test_artifacts.npz`) using `run.files()`. Previously only downloaded history/system parquets.
- `docs/tier_a_4prop_plan.md`: lightweight HP re-check plan for 4-property Tier A. Stage 1b (6 runs, lr ∈ {1e-5, 1e-4, 5e-4} × 2 seeds) → optional Stage 2b → Stage 5b (5 seeds). No chunk rebuild needed.
- `config.yaml`: `hidden_dim` locked to 128 (was 64 — now matches Stage 3 winner).
- `sbatch_sweep.sh`: reset `--job-name` and `WANDB_GROUP` to `stage_0_baseline` for the re-run.

**Bug fix — test_artifacts.npz was never actually saved (2026-04-26):**

- The commit `5fead3c` claimed to add `test_artifacts.npz` saving to `run_sweep.py`, but the actual diff only updated the SWEEP grid — the saving code was missing. Fixed: test loop now collects `test_comps`/`test_sys_idx` per batch, and after the loop saves `np.savez(artifacts_path, test_preds, test_targets, test_compositions, test_system_idx, scaler_mean, scaler_scale, properties)` + `wandb.save(str(artifacts_path))`.
- **Consequence**: Stage 0 and Stage 5 runs that completed before this fix have no `test_artifacts.npz` in W&B. Those runs must be re-run (currently in progress on HPC).

**Stage 5 analysis pipeline + publication notebook (2026-04-26):**

- `lipid_gnn/dataset.py` `preprocess_and_save`: each graph now carries `hetero_data.composition` (composition directory name, e.g. `"POPC95_CHOL5"`) and `hetero_data.system_idx` (tensor). Derived from `Path(tpr).parents[1].name`. Required for per-system error analysis in the Stage 5 notebook.
- `scripts/training/run_sweep.py` `train_one_run`: after the test loop, saves `test_artifacts.npz` to `wandb.run.dir` and uploads via `wandb.save`. Contains: `test_preds`, `test_targets`, `test_compositions`, `test_system_idx`, `scaler_mean`, `scaler_scale`, `properties` — the same format read by the notebook.
- `scripts/training/linear_baseline.py`: new `run_stratified_baseline(chunks_dir, properties, out_npz)` — trains Ridge on train-split chunks, evaluates on test-split chunks, saves same-format `.npz` to `results/training/linear_baseline_stratified.npz`. Invoke with `python linear_baseline.py --stratified`.
- `tests/test_dataset.py`: 3 new tests — composition labels survive `torch.save`/load, survive DataLoader iteration, and train/test composition sets are disjoint. Total suite: **42 tests**.
- `scripts/notebooks/analyze_stage_5.ipynb`: 20-cell publication notebook. Produces 9 figures (PDF + PNG at 300 DPI) + `headline_numbers.json` with all annotated numbers. Figures: (a) loss curves with seed band, (b) pred-vs-true scatter coloured by composition, (c) per-system MAE bar chart, (d) residual histograms, (e) GNN vs baseline, (f) HP progression, (g) composition-space PCA with MAE overlay, (h) R² forest plot with bootstrap CI, (i) paired Stage-0 vs Stage-5 dot plot with t-test.

**Stratified system-level split (2026-04-25):**

- `prepare_colab_subset.py` now defaults to `--split-method stratified` instead of random shuffle. New function `_stratified_split_systems`: loads per-system y-means from `.h5` pickles, z-scores each property, k-means clusters in y-space (k = min(10, N//7)), uses cluster IDs as stratification labels for two-stage `sklearn.train_test_split`. Prints per-split y-stats so coverage is visible at preprocessing time.
- New CLI flags: `--split-method {stratified, random}` (default `stratified`), `--stratify-on PROP [PROP ...]` (default: `CONFIG.vocab.active_properties`; must be subset of `--properties`). Old random path kept as `--split-method random` for reproducibility.
- **Root cause fixed**: the original `split_seed=0` random split gave test `lipid_packing` std=0.059 vs train std=0.251 (4.2× narrower) — test was trivially easy. Stratified split now gives test std=0.146 (comparable to train). This was why test MSE was always lower than val across all HP search stages.
- New `tests/test_stratified_split.py` — 4 tests: `test_stratified_split_covers_y_range_2d` (adversarial mode-in-median distribution; asserts test/val std ≥ 0.5× train), `test_stratified_split_disjoint`, `test_stratified_split_deterministic`, `test_stratified_split_4d_tier_a`. Total test suite: **39 tests**.
- **For Tier A re-preprocessing**: use `--stratify-on lipid_packing thickness variation thickness_std` to guarantee range coverage on `variation` (R² ceiling ≈ 0.5 in early runs — hard to predict, so especially important to have proper range in holdouts).

**W&B offline analysis tooling (2026-04-25):**

- New [scripts/python/download_wandb_runs.py](../../scripts/python/download_wandb_runs.py) — pulls a W&B group's finished runs to `logs/training/<group>/`. Per run: `config.json`, `summary.json`, `history.parquet` (per-epoch loss + R²), `system.parquet` (GPU util %, memory, CPU RSS, sampled ~15 s). Group index at `logs/training/<group>/runs_index.json`. Skips already-cached runs unless `--force`; on re-download cleans the run dir via `shutil.rmtree` first (prevents stale file accumulation). CLI: `--group` (nargs+), `--project`, `--entity`, `--out-dir`, `--properties`, `--include-crashed`, `--force`. Defaults from `CONFIG` (project, entity, properties).
- New [scripts/notebooks/analyze_hp_search.ipynb](../../scripts/notebooks/analyze_hp_search.ipynb) — 14 code cells. Top-level `GROUP` variable (single edit point). Loads downloaded parquet/json, builds `runs_df` (one row per run) and `cells_df` (seeds collapsed, grouped by `VARYING_HPS`). Produces 7 visualizations: (a) val-loss curves per cell, (b) per-property val curves, (c) ranking table with background_gradient, (d) heatmap (2 dims only), (e) test vs val scatter with identity line, (f) 3-panel training stats (wall time, GPU memory, Pareto scatter), (g) system time-series twin-axis for top-3 cells. Recommendation cell applies Occam tie-break and per-property gate check (`lipid_packing < 0.056`, `thickness < 0.219`). Optional multi-group comparison cell.
- New [docs/analyze_hp_search_notebook.md](../../docs/analyze_hp_search_notebook.md) — visualization reference: what each panel shows, how to read it, what to look for; covers (a)–(g) + recommendation + multi-group cells.
- **pyarrow** added to `requirements.txt` (`pyarrow>=15.0.0`) — required for `run.history(pandas=True)` and `.parquet` I/O.
- **Bug fix — SLURM GPU column detection**: W&B logs all 8 visible GPUs (gpu.0–gpu.7) but SLURM only allocates one (e.g. gpu.3). Original code returned `system.gpu.0.gpu` (all zeros). Fixed in both the notebook's load cell and the visualization (g) cell: scan all `gpu.N.gpu` / `gpu.N.memoryAllocated` columns and select the one with the **highest mean/max** — robust to any SLURM GPU allocation. Confirmed: `gpu_util_col = system.gpu.3.gpu`, mean util ≈ 86.4%, peak mem ≈ 29.5 GB.

**HP search results locked (2026-04-25):**

- Stage 1 winner: `learning_rate = 1e-4` (already correct in `config.yaml`).
- Stage 2 winner: `weight_decay = 1e-3` → `config.yaml` line 64 updated from `5e-3` to `1e-3`.
- `run_sweep.py` SWEEP updated for Stage 3: `hidden_dim ∈ {32, 64, 128}` × `num_layers ∈ {2, 3, 4}`, lr/wd locked from config.

**Central config file (2026-04-24):**

- New [config.yaml](../../config.yaml) at repo root — sections: `paths`, `dataset`, `vocab`, `model`, `training`, `wandb`, `hpc`.
- New [lipid_gnn/config.py](../../lipid_gnn/config.py) — frozen `@dataclass` per section + top-level `Config`. Module-level `CONFIG = load_config()` singleton; `from lipid_gnn.config import CONFIG` is the import everywhere.
- Env overrides at the raw-dict layer inside `load_config()`: `CHUNKS_DIR` → `paths.chunks_dir`, `WANDB_MODE` → `wandb.mode`, `WANDB_GROUP` → `wandb.group`. Relative paths resolve against `REPO_ROOT`.
- Derived properties: `DatasetConfig.rbf_stop == spatial_cutoff` (single source); `VocabConfig.lipid_comp_dim == len(lipid_types)`. Validation rejects `spatial_edge_attr_dim != rbf_num_gaussians` and `active_properties ⊄ all_properties`.
- New [scripts/python/print_config_var.py](../../scripts/python/print_config_var.py) — bash-side shim for reading values from `config.yaml` (e.g. `python scripts/python/print_config_var.py dataset.spatial_cutoff`). Lists are space-separated so they word-split into CLI args.
- Migrated callers: all of `lipid_gnn/` except `functions_emil/` and `plotting.py`, all of `scripts/training/` except colab notebooks, `scripts/bash/sbatch_{preprocess,sweep}.sh`, `scripts/bash/gc_{copy,transfer}_files.sh`, and three test files. Callers use the `None`-sentinel pattern so explicit overrides still win.
- New [tests/test_config.py](../../tests/test_config.py) — 8 tests (load, abs-path resolution, `lipid_comp_dim`, `rbf_stop==cutoff`, `active⊂all`, env overrides for `CHUNKS_DIR` + `WANDB_MODE`, two validation-rejection paths). Total: 35 tests pass.
- PyYAML added to [requirements.txt](../../requirements.txt) (`PyYAML>=6.0`).
- **Intentional drift converged**: default `spatial_cutoff` for `dataset.py`, `lipid_graph.py`, and `sbatch_preprocess.sh` **9.0 → 11.0 Å**. Matches what `prepare_colab_subset.py` was already producing and the Martini non-bonded range.
- **Explicitly NOT migrated**: colab notebooks (legacy per memory bank), `lipid_gnn/functions_emil/`, `scripts/emil/`, `scripts/notebooks/`, `lipid_gnn/plotting.py` (styling only).
- **Design decisions** recorded when implementing: YAML source of truth + Python `@dataclass` loader (not pure Python constants) for easier per-experiment overrides and type safety. `FIXED` in `run_sweep.py` reads from `CONFIG.training.*`; `SWEEP` stays inline (grid is experiment-specific, not a project-wide default). argparse defaults source from `CONFIG`, CLI overrides still win — preserves the existing "CLI for tunable runnable scripts" pattern.

**Multi-property y-slicing fix (2026-04-23):**

- Chunks now store all 8 properties in `y` (`shape [1, 8]`). Training was broken because `normalize(batch.y)` produced an `[N, 8]` target while the model output was `[N, len(properties)]`.
- Fix: added `ALL_PROPERTIES` list (fixed column order matching `AVAILABLE_PROPERTIES` in `prepare_colab_subset.py`) and `prop_cols = [ALL_PROPERTIES.index(p) for p in properties]` in `train_one_run`. `s_mean`/`s_scale` are now sliced with `prop_cols`; all three `normalize(batch.y)` calls replaced with `normalize(batch.y[:, prop_cols])`.
- Applied identically to both `scripts/colab/train_colab_rev.ipynb` and `scripts/training/run_sweep.py`.
- To change the active property set, edit `PROPERTIES` in the config section — no chunk rebuild needed.

**Train/val/test system-level split (2026-04-20):**

- `prepare_colab_subset.py` now splits `sim_tuples` (70/15/15 default) by system before any preprocessing. New CLI flags: `--val-frac` (0.15), `--test-frac` (0.15), `--split-seed` (0, separate from `--shuffle-seed`). Calls `preprocess_and_save` three times into `processed/train/`, `processed/val/`, `processed/test/`. Zip packs all three.
- `train_colab_rev.ipynb` and `run_sweep.py`: load from the three subdirectories directly (no runtime chunk-level shuffle-split). `train_one_run` gains a `test_dataset` parameter; a held-out eval pass runs once after training, logging `test/accuracy_plot` and `test/mse_total` to W&B.
- `sbatch_sweep.sh`: staging changed from `cp chunk_*.pt` to `rsync -a` so the full subdirectory tree is staged.
- New test `test_train_val_test_splits_are_disjoint`: asserts pairwise disjoint y-value sets across train/val/test output directories. All 25 tests pass.
- **Chunks must be regenerated** to get the new three-directory layout.

**Training regression fix — interleaved preprocessing (2026-04-20):**

- Root cause: with `num_frames=50, chunk_size=50`, every `.pt` chunk held all 50 frames of a single system. All graphs in a chunk share the same `y` (per-system mean). PyTorch `DataLoader` with `IterableDataset` and `num_workers=2` builds each batch from one worker's stream = one chunk at a time → every batch of 8 had **identical targets**. MSE collapsed to dataset mean — same symptom as the old scaler-leakage bug.
- Fixed in [lipid_gnn/dataset.py](lipid_gnn/dataset.py) `preprocess_and_save`: new two-phase approach — (1) open all builders up front, pre-compute `(system_idx, frame_idx)` schedule, (2) shuffle schedule once with `shuffle_seed` (when `interleave=True`, default), stream in shuffled order. Each chunk now mixes many systems.
- New kwargs: `interleave: bool = True`, `shuffle_seed: int = 42` (deterministic). `interleave=False` restores old per-system sequential ordering for tests/debugging.
- [scripts/training/prepare_colab_subset.py](scripts/training/prepare_colab_subset.py): new `--shuffle-seed` CLI flag (default 42) wired through.
- [tests/test_multi_frame_loading.py](tests/test_multi_frame_loading.py): new `test_preprocess_and_save_interleaves_systems` asserts first chunk contains more than one distinct `y` value when systems have distinguishable targets. All 24 tests pass.
- **Chunks must be regenerated** for the fix to take effect. Run `python scripts/training/prepare_colab_subset.py` (or `--no-zip` for HPC).

**HPC deployment scaffolding (2026-04-18):**

- [scripts/training/prepare_colab_subset.py](scripts/training/prepare_colab_subset.py) gained `--sims-dir`, `--props-dir`, `--out-dir`, `--no-zip` flags. `--no-zip` skips the `lipid_gnn/` bundling + zip step and writes chunks directly to `--out-dir` — the HPC entry point. Colab behavior unchanged when flags are omitted.
- [scripts/training/run_sweep.py](scripts/training/run_sweep.py) now resolves its chunks directory from the `CHUNKS_DIR` env var (defaults to the previous `colab_lipid_gnn_subset/processed` path). Lets the sbatch script stage chunks to `/local/$SLURM_JOB_ID` without code churn.
- New [scripts/bash/sbatch_preprocess.sh](scripts/bash/sbatch_preprocess.sh) and [scripts/bash/sbatch_sweep.sh](scripts/bash/sbatch_sweep.sh): SLURM submit scripts for `gpu_test`. Both require `export GROUP=<goethe-group>`; sweep script stages chunks from `/work` to `/local/$SLURM_JOB_ID` for fast I/O and passes `CHUNKS_DIR` to the trainer. W&B defaults to `online` with `WANDB_MODE=offline` fallback.
- New [docs/hpc_goethe.md](docs/hpc_goethe.md): runbook covering filesystem layout, connectivity probe, rsync commands, miniforge + ROCm 6.2 PyTorch install, recurring git-pull → preprocess → train flow.

**Plan decisions recorded (not yet executed on cluster):**

- Install strategy: user-local miniforge in `$HOME`; PyTorch from `https://download.pytorch.org/whl/rocm6.2`; install from inside a `gpu_test` allocation so native extensions see the MI210.
- Storage layout: code in `$HOME` (30 GB), raw `.tpr`/`.xtc` + chunks + W&B offline runs in `/work/<grp>/<user>/lipid-data/` (5 TB group quota, 30-day TTL), per-job staging on `/local/$SLURM_JOB_ID` (1.4 TB).
- Preprocessing runs remotely (CPU-only sbatch) — avoids shipping ~74 GB of chunks over the network.
- Compute-node internet reachability unknown; A1 probe step determines online vs offline W&B path.
- ROCm-on-PyTorch keeps `torch.cuda.*` API intact, so model/training code is untouched.

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

**Default cutoff raised to 11.0 Å + num_frames halved to 25 (2026-04-20):**

- Even at 9.0 Å some trajectory frames produced beads with zero spatial edges. 11.0 Å matches Martini's own non-bonded cutoff and eliminates isolated beads entirely.
- `--spatial-cutoff` default raised **9.0 → 11.0 Å** in [scripts/training/prepare_colab_subset.py](scripts/training/prepare_colab_subset.py). Graph memory at 11.0 Å is roughly double that at 9.0 Å (benchmark reference).
- To compensate for the ~2× memory increase, `--num-frames` default lowered **50 → 25** frames per system. Net chunk count roughly unchanged; total dataset = 70 systems × 25 frames = 1,750 graphs.
- [lipid_gnn/benchmark_heterognn.py](lipid_gnn/benchmark_heterognn.py) `compare_graph_memory` label updated: row C now reads `(default, Martini range)` instead of the stale `(new default)` on row B.
- **Requires regenerating `.pt` chunks** with the new defaults.

**Spatial cutoff corrected to 9.0 Å + benchmark three-way comparison (2026-04-18):**

- Benchmark at 7.5 Å revealed nodes with **zero spatial edges**. Root cause: after removing bonded pairs (~4.7 Å), the non-bonded search window is only 2.8 Å wide. Terminal tail beads (C4A, C4B) with one bonded neighbor can have no other bead within that window in disordered bilayer frames.
- Added runtime `warnings.warn` in `process_frame()` if any bead has zero spatial edges after building the graph.
- `compare_graph_memory` (`--compare-mem`) tests all three cutoffs **7.5 / 9.0 / 11.0 Å** in one run. Summary table shows MB, ΔMB, Δ%, spatial edge count, and isolated bead count with a `!` flag.

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

- **Re-preprocess with stratified split before Stage 5 / Tier A**: run `prepare_colab_subset.py` with `--split-method stratified --stratify-on lipid_packing thickness variation thickness_std` on the HPC. Old chunks (random split) lack `composition`/`system_idx` attributes AND have the narrow test split — both issues fixed only by regenerating.
- **Stage 3 analysis**: winner is `hidden_dim=128, num_layers=2` (val_mean=0.03816, val_std=0.00036 — most stable cell). Optionally re-tune lr/wd at h=128 before Stage 5 (val_std is tiny → likely already well-tuned).
- **Stage 5 (5-seed confirmation)**: run the Stage-3 winner with 5 seeds on the new stratified chunks; must pass per-property gates (`lipid_packing < 0.056`, `thickness < 0.219`).
- **Multi-property training (tiered)**: change `PROPERTIES` in the config cell — no chunk rebuild needed. Full plan: [docs/multi_property_training_plan.md](../../docs/multi_property_training_plan.md).
  - **Tier A** (HP-tune here): `['lipid_packing', 'thickness', 'variation', 'thickness_std']`
  - **Tier B** (check negative transfer): add `'persistence'`, `'diffusivity'`
  - **Tier C** (report-only): add `'compressibility'`, `'bending_modulus'`
- Execute the Goethe-HLR bootstrap: rsync raw data to `/work`, install miniforge + ROCm PyTorch inside a `gpu_test` allocation, `pytest -q` on the cluster, then submit `sbatch_preprocess.sh` + 1-seed `sbatch_sweep.sh` smoke run. This is the primary training path going forward.

## Long-term parallel track: representation learning (2026-04-21)

Full plan: [docs/representation_learning_plan.md](../../docs/representation_learning_plan.md). Summary:

- **Motivation**: the thesis north-star is a transferable CG-membrane embedding that eventually extends to protein+membrane. Supervised regression on two scalars is not a sufficient training signal. Build a second, parallel track — SSL pretraining with a linear-probe evaluation on the same held-out splits as the property-regression track.
- **Primary objective**: force regression with per-bead local-frame projection. Per-bead forces are regenerated via `gmx mdrun -rerun` on existing `.xtc` frames (production runs had `nstfout=0`). Forces are projected onto a bonded-neighbor local frame, giving 3 invariant scalars per bead — predictable by the current GATv2 backbone, no equivariance required.
- **Auxiliary objective**: per-lipid order-parameter regression (S_CD, tilt) at small weight; free from existing trajectories via MDAnalysis.
- **Deprecated alternatives**: frame-pair contrastive (composition-identity shortcut even without `comp_vec`, encodes "which membrane" not "what the membrane is doing"), masked bead-type prediction (trivially solvable from bonded neighbors), RBF reconstruction (angle-blind).
- **Parallel track, not joint loss**: new files `representation_gnn.py`, `run_representation.py`, `sbatch_rerun.sh`; separate W&B project (`lipid-gnn-repr`). `MembranePropertyGNN`, `run_sweep.py`, `train_colab_rev.ipynb`, and the 25-test suite must keep passing unchanged. All shared-code extensions (`lipid_graph.py`, `dataset.py`) are additive opt-in flags with defaults preserved; SSL chunks live in a separate `processed_ssl/` directory.
- **Hard requirement**: `gnn_only` / `comp_dim=0` for every repr-learning run. The encoder must be composition-blind — required for transfer to unseen lipid types and to protein+membrane.
- **Headline evaluation**: leave-one-lipid-out linear probe on thickness / lipid_packing, compared head-to-head against the property-regression track on the identical split.
- **Fallback** if `mdrun -rerun` is blocked: promote per-lipid order-parameter regression to primary + add time-lagged predictive (JEPA-style) auxiliary; both work from existing `.xtc`s only.

Decisions recorded:

- GitHub squash/rebase merging disabled in repo settings (enforcement now automatic, not just convention)
- CLI-arg pattern will NOT be applied to `run_sweep.py` — the FIXED/SWEEP split + cartesian product grid doesn't translate cleanly to command-line flags

## Important Patterns and Preferences

- Test discrete components locally (dedicated test scripts, baseline metrics) before integrating into the heavy end-to-end `train_colab.ipynb` loop
- Results are uploaded to Weights & Biases for visualization. No model weights are saved currently, but this may change
- Force field parameters are loaded from JSON files at graph build time, not parsed from `.itp` at training time
- `LIPID_TYPES` ordering must stay consistent across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py`
- `preprocess_and_save` is the single entry point for building and saving graph chunks; callers are responsible for constructing `sim_tuples`
