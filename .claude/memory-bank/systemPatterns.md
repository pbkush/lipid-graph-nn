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
   Training Loop (scripts/colab/train_colab.ipynb)
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

## Component Relationships

- `lipid_graph.py` depends on: MDAnalysis, PyTorch Geometric, FF JSON maps from `resources/`
- `membrane_prop_gnn.py` is standalone (pure PyTorch + PyG)
- `dataset.py` loads `.pt` chunk files (optional import of `lipid_graph.py` for graph generation)
- `ff_parser.py` is standalone utility (parses raw Martini `.itp` text, no MDAnalysis dependency)
- `scripts/training/` — preprocessing, sweeps, baselines
- `scripts/colab/train_colab.ipynb` — main training notebook (runs on Colab with W&B logging)

## Critical Implementation Paths

- **Graph construction**: `MartiniHeteroGraphBuilder.__init__()` → loads universe, caches topology, maps FF params → `build_frame(ts)` → returns `HeteroData`
- **Data loading**: Preprocessing scripts save `.pt` chunks → `MartiniDiskDataset` streams them → `DataLoader` with multi-worker prefetching
- **Training loop**: `train_colab.ipynb` → loads chunked `.pt` data via `MartiniDiskDataset` → trains `MembranePropertyGNN` → logs to W&B
- **LIPID_TYPES ordering**: The 10-element lipid vocabulary list must stay consistent across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py`
