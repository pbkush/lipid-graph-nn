# Tech Context

## Technologies Used

| Component | Technology |
|-----------|-----------|
| ML Framework | PyTorch + PyTorch Geometric |
| MD Analysis | MDAnalysis |
| Graph Type | PyG `HeteroData` |
| Convolution | `GATv2Conv` via `HeteroConv` |
| Normalization | `GraphNorm` |
| Force Field | Martini 3 (coarse-grained) |
| Baseline | scikit-learn Ridge Regression |
| Visualization | Matplotlib |
| Testing | pytest |

## Development Setup

- **Conda environment**: `lipid_gnn` (Python 3.13)
- **Installation**: `pip install --use-pep517 .` (minimal `setup.py`, no pinned dependencies)
- **Tests**: `pytest tests/` — no special config needed
- **Training**: `python3 scripts/training/run_sweep.py`
- **Smoke test**: `python3 scripts/training/smoke_test_sweep.py`

## Technical Constraints

- **RAM**: ~7.6 GB available before training → chunked dataset loading is mandatory
- **GPU VRAM**: ~10 GB → batch sizes limited to 1–4 with current graph sizes (~thousands of nodes per membrane)
- **Data volume**: 70 membrane compositions in `data/membrane_only/`, each with multiple frames
- **Google Colab compatibility**: project is designed to also run on Colab (see `scripts/colab/` and `colab_lipid_gnn_subset/`)

## Dependencies

Core (inferred from imports, not pinned in setup.py):

- `torch`, `torch_geometric` (PyTorch Geometric including `GATv2Conv`, `HeteroConv`, `GraphNorm`)
- `MDAnalysis` (trajectory loading, atom selection, distance calculations)
- `numpy`, `matplotlib`, `tqdm`
- `scikit-learn` (for linear baseline)

## Tool Usage Patterns

- **Force field data**: Stored as JSON files in `resources/` — `ff_params.json` (bead type → physics params), `ff_node_mapping.json` (molecule+atom → bead type), `ff_edge_params.json` (bond parameters)
- **Experiment results**: Saved to `results/training/lipid_packing/<timestamp_config>/` with metrics JSON and plots
- **Data flow**: Raw MD trajectories → `MartiniHeteroGraphBuilder` → `.pt` chunk files → `MartiniDiskDataset` → DataLoader
