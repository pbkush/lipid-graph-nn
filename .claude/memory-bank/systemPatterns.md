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

## Critical Implementation Paths

- **Graph construction**: `MartiniHeteroGraphBuilder.__init__()` → loads universe, caches topology, maps FF params → `build_frame(ts)` → returns `HeteroData`
- **Data loading**: Preprocessing scripts save `.pt` chunks → `MartiniDiskDataset` streams them → `DataLoader` with multi-worker prefetching
- **Training loop**: `scripts/training/run_sweep.py` (submitted via `scripts/bash/sbatch_sweep.sh` on HPC) → loads chunked `.pt` data via `MartiniDiskDataset` → trains `MembranePropertyGNN` → logs to W&B
- **LIPID_TYPES ordering**: The 10-element lipid vocabulary list must stay consistent across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py`
