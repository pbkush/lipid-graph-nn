# lipid-graph-nn

A physics-informed heterogeneous Graph Neural Network for learning transferable embeddings of lipid membranes from Martini 3 coarse-grained molecular dynamics snapshots.

## Goal

The scientific aim is to find a **meaningful embedding of membrane systems** that captures their physical behavior in a transferable way. Property prediction (area per lipid, membrane thickness) serves as the training signal — the long-term target is an embedding that generalizes to unseen compositions and, eventually, to protein+membrane systems. Martini 3 force field parameters are encoded directly as node features, so the model works with physically meaningful inputs rather than arbitrary learned vocabularies.

## Model architecture

Input graphs are built by `MartiniHeteroGraphBuilder` ([lipid_gnn/lipid_graph.py](lipid_gnn/lipid_graph.py)) from a `(.tpr, .xtc)` pair. Each frame becomes a PyG `HeteroData` object with:

- **Nodes** (CG beads): 4-dim features `[mass, charge, sigma, epsilon]` from Martini 3 FF
- **Bonded edges** (chemical topology): 2-dim `[force_constant, bond_length]`
- **Spatial edges** (distance-based, 10.0 Å cutoff): 16-dim Gaussian RBF of pairwise distances

`MembranePropertyGNN` ([lipid_gnn/membrane_prop_gnn.py](lipid_gnn/membrane_prop_gnn.py)) processes the graph with:

- Input `Linear(4 → hidden_dim)` bead embedding
- `N × HeteroConv({bonded: GATv2Conv, spatial: GATv2Conv})` with `GraphNorm` per layer
- Readout: `global_mean_pool` + `global_max_pool` concatenated, with optional 10-dim composition vector (per-lipid fractions) appended
- MLP head → multi-property output

Three composition modes, toggled via `comp_dim`: `gnn_only`, `gnn_plus_comp`, `comp_only`.

## Installation

Python 3.13 conda environment `lipid_gnn` is assumed.

```bash
pip install --use-pep517 .
pip install -r requirements.txt
```

Before training with W&B logging, run `wandb login` once. Raw trajectory data lives under `data/membrane_only/<COMPOSITION>/run/` and is not tracked in git.

## Training

### Smoke test (start here)

```bash
python3 scripts/training/smoke_test_sweep.py
```

Runs 1 epoch × 1 seed across the 3 composition modes. Use it to verify the install and data paths before committing to long runs.

### Local sweep

```bash
python3 scripts/training/run_sweep.py
```

Main training entry point. Hyperparameters and target properties are hardcoded at the top of the script — edit the config there (no CLI flags). Results (loss curves, accuracy plots, metrics JSON) land under `results/training/`.

### HPC preprocessing

Preprocess Martini trajectories into chunked `.pt` graph datasets for training:

```bash
python3 scripts/training/preprocess_graphs.py --props-set prop_legacy_bugfixed_s0
```

Output: `data/preprocessed_graphs/<run-name>/{train,val,test}/` with a zip
mirror at `data/preprocessed_graphs/archives/<run-name>.zip`. `<run-name>`
defaults to `--props-set` so successive property sets do not overwrite each
other. Pass `--no-zip` on the HPC.

### Expected data layout

```text
data/membrane_only/
└── <COMPOSITION>/
    └── run/
        ├── prun.tpr
        └── prun.xtc
results/properties/
└── <COMPOSITION>.h5
```

Available target properties: `lipid_packing`, `thickness`, `thickness_std`, `compressibility`, `persistence`, `diffusivity`.

## Evaluating runs

Training scripts evaluate models inside the training loop — there is no standalone inference utility yet.

- Per-property MSE and R² are logged to Weights & Biases every epoch.
- A final accuracy scatter plot is logged as a W&B image at the end of each run.
- **Model checkpoints are not saved.** Trained weights do not persist across runs.

## Repository layout

```text
lipid_gnn/          core package: graph builder, dataset, model, FF parser
scripts/training/   preprocessing + training entry points
scripts/colab/      Colab training notebooks
scripts/emil/       collaborator analysis notebooks
tests/              pytest suite
resources/          Martini 3 force-field JSON maps
data/               raw MD trajectories (gitignored)
```
