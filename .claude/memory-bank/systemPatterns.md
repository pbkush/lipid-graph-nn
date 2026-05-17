# System Patterns

## System Architecture

```text
MD Trajectory (.tpr + .xtc/.trr)
        │
        ▼
┌──────────────────────────┐
│  MartiniHeteroGraphBuilder│  (lipid_graph.py)
│  - Builds HeteroData      │
│  - Caches static topology  │
│  - RBF encodes spatial dist│
│  - Vectorized sparsification│
└──────────┬───────────────┘
           │ .pt chunk files
           ▼
┌──────────────────────────┐
│  MartiniDiskDataset       │  (dataset.py)
│  - Streams chunks from disk│
│  - Multi-worker prefetch   │
│  - Shuffle: chunk + intra  │
│  - Sequential for validation│
└──────────┬───────────────┘
           │ batched HeteroData
           ▼
┌──────────────────────────┐
│  MembranePropertyGNN      │  (membrane_prop_gnn.py)
│  - HeteroConv + GATv2Conv  │
│  - GraphNorm per layer     │
│  - mean+max pool readout   │
│  - Optional comp vector    │
└──────────┬───────────────┘
           │ predicted properties
           │ (lipid_packing, thickness)
           ▼
   Training Loop (scripts/training/run_sweep.py — HPC via sbatch_sweep.sh)
        │
        ▼
   Weights & Biases (logging)
```

## Key Technical Decisions

1. **Heterogeneous graph with two edge types**: bonded (topology, static) and spatial (distance-based, dynamic per frame). Separates chemical bonding from physical proximity.
2. **Continuous physics features over learned embeddings**: Node features are `[mass, charge, sigma, epsilon]` from Martini 3 FF — physically meaningful rather than arbitrary representations.
3. **GATv2Conv over SAGEConv**: Attention mechanism weights edges differently based on edge attributes (bond params for bonded, RBF distances for spatial).
4. **Gaussian RBF encoding for distances**: 16 Gaussian basis functions expand scalar distances into a smooth, differentiable feature space for edge attributes.
5. **Chunked disk streaming**: `MartiniDiskDataset` avoids loading all graphs into RAM. Critical for Colab's memory constraints.
6. **GraphNorm over LayerNorm**: Better for graph-level prediction tasks as it considers graph size/structure.
7. **Vectorized graph sparsification**: `MartiniHeteroGraphBuilder` uses NumPy vectorized routines to mask bonded pairs and self-loops from spatial cutoff matrices, replacing slow Python loops.
8. **Multi-property prediction**: Model predicts multiple targets simultaneously (lipid packing + thickness) with a single forward pass.

## Design Patterns in Use

- **Stateful builder pattern**: `MartiniHeteroGraphBuilder` caches static topology on init, then efficiently generates per-frame graphs via `build_frame()`
- **Force field indirection**: Raw `.itp` files → `ff_parser.py` → JSON maps → loaded at graph build time. Separates parsing from training.
- **Modal architecture**: `comp_dim=0` for GNN-only, `comp_dim=10` for GNN+composition — single model class, toggled by constructor arg
- **Test-before-integrate**: Discrete components are validated locally with dedicated tests (e.g., `test_dataset.py`) before integration into the heavy `train_colab.ipynb` loop
- **Preprocessing separation**: Graph construction and chunk saving happen in preprocessing scripts; training only loads pre-built `.pt` files
- **Docstrings**: Write docstrings for all functions and classes. Write small descriptions for parameters if deemed necessary.
- **CLI for tunable runnable scripts**: Any script that is executed directly (`python3 scripts/...`) and has parameters that change per experiment/environment should expose them as `argparse` flags rather than module-level constants. Rule of thumb: if editing the value would produce a non-substantive git diff (just a number or list change), it belongs in the CLI.
  - Keep the current hardcoded values as argparse `default=` so existing invocations with no flags preserve behavior
  - Use `nargs="+"` for list-valued params, `choices=` when the valid set is enumerable, `type=int`/`type=float` for numerics
  - Enumerate valid choices in the `--help` string (e.g., `Available: a, b, c`) so users don't have to read source
  - Parse in a `_parse_args()` helper and pass values as function kwargs — keep the core function callable from other modules without the CLI
  - Applied in [scripts/training/prepare_colab_subset.py](scripts/training/prepare_colab_subset.py) (PR #4)
  - NOT a candidate: [scripts/training/run_sweep.py](scripts/training/run_sweep.py) — mixes FIXED hyperparams with a SWEEP grid (cartesian product) that is clumsy on the command line and belongs in code. Decision recorded after considering CLI-ifying `FIXED` and rejecting it

- **Central config (YAML + dataclass loader)**: Project-wide paths, vocabulary, and experiment defaults live in [config.yaml](config.yaml); [lipid_gnn/config.py](lipid_gnn/config.py) parses it into frozen `@dataclass` sections and exposes a module-level `CONFIG` singleton. Rule of thumb: if a value is referenced by more than one file (paths, `LIPID_TYPES`, `ALL_PROPERTIES`, `spatial_cutoff`, `rbf_num_gaussians`, model defaults, training defaults), it belongs in `config.yaml`; single-use locals stay where they are.
  - Callers consume via `from lipid_gnn.config import CONFIG` and the `None`-sentinel pattern for function defaults (`def fn(x=None): if x is None: x = CONFIG.foo.bar`) so explicit callers still override.
  - Env-var overrides are applied at the raw-dict layer inside `load_config()` — today: `CHUNKS_DIR`, `WANDB_MODE`, `WANDB_GROUP`.
  - Derived values are `@property` methods, not duplicated keys (`DatasetConfig.rbf_stop == spatial_cutoff`, `VocabConfig.lipid_comp_dim == len(lipid_types)`).
  - Validation in `load_config()` catches cross-section invariants (e.g. `spatial_edge_attr_dim == rbf_num_gaussians`, `active_properties ⊆ all_properties`).
  - Bash consumes the config through [scripts/python/print_config_var.py](scripts/python/print_config_var.py) — a tiny stdlib shim; lists are space-separated for word-splitting into CLI args. Bash scripts do NOT hardcode Python-derived values.
  - CLI arg defaults in scripts source from `CONFIG` rather than being hardcoded literals. This composes with the "CLI for tunable runnable scripts" pattern above: config is the default, CLI is the override, hardcoded literals in script bodies are a smell.
  - The experiment-grid in `run_sweep.py` `SWEEP` dict stays inline (cartesian product is experiment-specific, not a project default). `FIXED` reads from `CONFIG.training.*`.
  - Out of scope by convention: stylistic choices (plot DPI, colors) and Colab notebooks (legacy). See intentional exclusions in `.claude/memory-bank/activeContext.md` under the config landing entry.

## Component Relationships

- `lipid_graph.py` depends on: MDAnalysis, PyTorch Geometric, FF JSON maps from `resources/`
- `membrane_prop_gnn.py` is standalone (pure PyTorch + PyG)
- `dataset.py` loads `.pt` chunk files (optional import of `lipid_graph.py` for graph generation)
- `ff_parser.py` is standalone utility (parses raw Martini `.itp` text, no MDAnalysis dependency)
- `scripts/training/` — preprocessing, sweeps, baselines
- `scripts/colab/train_colab_rev.ipynb` — legacy Colab notebook (no longer the active training path; kept for reference)
- **W&B offline analysis chain**: `scripts/bash/sbatch_sweep.sh` submits sweep → W&B receives run data → `scripts/python/download_wandb_runs.py` pulls to `logs/training/<group>/` as parquet/json → `scripts/notebooks/analyze_hp_search.ipynb` aggregates and visualizes. No live W&B API needed after download. See `docs/analyze_hp_search_notebook.md` for visualization reference.
  - **SLURM GPU column pitfall**: W&B logs all 8 visible GPUs (gpu.0–gpu.7); only the allocated GPU is non-zero. Never hardcode `system.gpu.0.*` — always scan all `gpu.N.*` columns and select the one with the highest mean/max. (After the 2026-05-05 packing refactor, packed jobs do use multiple GPUs per node; each backgrounded process has `HIP_VISIBLE_DEVICES=$i` so it sees only its own GPU as device 0, but W&B's system metrics still report all visible cards — keep the scan-all logic.)
- **Martini simulation submission pattern (GPU + CPU)**: `scripts/bash/submit_simulations.sh` is the single orchestrator for production sims. Routes by `--partition`:
  - `gpu` / `gpu_test` / `test` → `sbatch_simulations.sh` (ROCm-built gmx 2025.4 module, HIP pinning, 8 sims/node default).
  - `general1` (CPU) → `sbatch_simulations_general1.sh` (spack openmpi + GROMACS-2022 via `_gmx_mpi_wrapper.sh` shim, OpenMP thread pinning, calibrated `hpc_defaults_cpu`: 2 sims/node, 1 MPI rank, 20 OMP threads, 16G mem).
  - Unknown partition → fail-fast.

  CLI flags worth knowing — work-source flags are mutually exclusive:
  - `--compositions COMP1 COMP2 ...` — explicit list.
  - `--missing-from-grid NAME` — grid-driven (e.g. `popc_interpolation`).
  - `--queue-file PATH` — read from a plain text file.
  - `--from-csv PATH` — read `canonical_name` column from a CSV.  Inverse of `--completed-csv`: the CSV IS the work list.  Use case: resimulate the legacy 70-system corpus with the modern M3 ITPs (`--from-csv resources/done.csv`) so all data shares one set of itp definitions.

  Plus the orthogonal filters / sizing flags: `--completed-csv PATH` (skip systems whose `canonical_name` is in the CSV — used to avoid re-running legacy data without uploading it to HPC), `--prod-ns NS`, `--time HH:MM:SS`, `--partition`, `--mpi-ranks-per-sim`, `--pin {on,off,auto}` (gmx mdrun thread pinning; default from `hpc_defaults.pin`, "on" if absent). Per-partition QOS caps enforced (general1=40, gpu_test=2).

  Env propagation uses **env-file-via-positional-arg**: orchestrator writes `logs/simulations/submit_env.<tmp>.sh` with `export VAR=$'...'` lines and passes the path as `$1` to the sbatch worker; SLURM `--export=ALL,VAR=...` silently drops entries on Goethe-HLR. Workers source `$1` on entry, then re-source defensively after `module load`. `--mdrun-args` is always placed LAST in `SIM_ARGS` (pipeline CLI used to take it as `argparse.REMAINDER`, which greedily absorbed any following flag; CLI now takes a single string, regression test enforced).

  Typical usage:

  ```bash
  # GPU sweep
  bash scripts/bash/submit_simulations.sh --missing-from-grid popc_interpolation \
       --prod-ns 1000 --partition gpu --time 24:00:00

  # general1 CPU sweep with skip-CSV
  bash scripts/bash/submit_simulations.sh --missing-from-grid popc_interpolation \
       --completed-csv resources/done.csv --prod-ns 1000 \
       --partition general1 --time 48:00:00

  # Mid-run ETA check
  python scripts/simulation/projected_finish.py /work/.../popc_interpolation \
       --walltime 48 --only over
  ```

- **Benchmark submission pattern**: `scripts/simulation/benchmark_hpc.sh` is a two-phase orchestrator. Phase 1 builds a shared `prun.tpr` reference once via the production worker; Phase 2 fans `sbatch_benchmark_hpc.sh` jobs across a sweep of `(sims_per_node, mpi_ranks, cpus)` points (read from `benchmark_points.tsv`). Each point runs `gmx mdrun -nsteps N -resethway` on the shared TPR, writes ns/day per slot, and `analyze_benchmark.py --recommend [--cpu]` picks the winning `hpc_defaults_*` block and prints YAML ready to paste into `config.yaml`.

  Typical usage:

  ```bash
  # GPU benchmark (setup on cheap partition, sweep on production)
  bash scripts/simulation/benchmark_hpc.sh --partition gpu --setup-partition gpu_test

  # CPU benchmark on general1 (uses general1.tsv + general1 sbatch)
  bash scripts/simulation/benchmark_hpc_general1.sh

  # After all points land, generate recommendation
  python scripts/python/analyze_benchmark.py --recommend         # GPU
  python scripts/python/analyze_benchmark.py --recommend --cpu   # CPU
  ```

  Same env-file-via-positional-arg and gmx `-ntmpi 1` patterns as production. Benchmark winner currently locked in `config.yaml::martini_pipeline.hpc_defaults_cpu`.

- **SLURM submission pattern (per-node GPU packing) for training**: `submit_sweep.sh` is the single orchestrator. It (1) reads HP defaults from `config.yaml` via `print_config_var.py`, (2) expands the Cartesian product of `--lr`, `--wd`, `--hidden-dim`, `--num-layers`, `--seeds` into a flat list of single-cell runs, (3) packs runs onto nodes at up to `--gpus-per-node` (default 8) and submits one sbatch per batch with `--gres=gpu:N`, scaled `--cpus-per-task=N×CPUS_PER_GPU` and `--mem=N×MEM_PER_GPU`. `sbatch_sweep.sh` then stages chunks once per node and backgrounds N `python run_sweep.py` processes, each pinned via `HIP_VISIBLE_DEVICES=$i`/`CUDA_VISIBLE_DEVICES=$i` and fed slot-specific `RUN_<i>_*` env vars that sbatch_sweep.sh translates into `FREEZE_HIDDEN_DIM/NUM_LAYERS/LR/WD` and `SWEEP_SEEDS`. `run_sweep.py::_apply_submission_overrides()` is the single read site for these vars and is unchanged by the refactor. Resource sizing convention: `--cpus-per-gpu 8 --mem-per-gpu 64G` (the original 1-GPU job's footprint, kept as defaults). Partition default comes from `hpc.partition_train`; `gpu_test` adds two static guards (8h max, 2 sbatch jobs max).

## Critical Implementation Paths

- **Graph construction**: `MartiniHeteroGraphBuilder.__init__()` → loads universe, caches topology, maps FF params → `build_frame(ts)` → returns `HeteroData`
- **Data loading**: Preprocessing scripts save `.pt` chunks → `MartiniDiskDataset` streams them → `DataLoader` with multi-worker prefetching
- **Training loop**: `scripts/training/run_sweep.py` (submitted via `scripts/bash/sbatch_sweep.sh` on HPC) → loads chunked `.pt` data via `MartiniDiskDataset` → trains `MembranePropertyGNN` → logs to W&B
- **LIPID_TYPES ordering**: The 10-element lipid vocabulary list must stay consistent across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py`
