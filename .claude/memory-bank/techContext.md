# Tech Context

## Technologies Used

| Component | Technology |
| --------- | --------- |
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
- **Installation**: `pip install --use-pep517 .` then `pip install -r requirements.txt`
- **Tests**: `pytest tests/` — no special config needed
- **Training**: `python3 scripts/training/run_sweep.py`
- **Smoke test**: `python3 scripts/training/smoke_test_sweep.py`

## Technical Constraints

- **RAM**: ~7.6 GB available before training → chunked dataset loading is mandatory
- **GPU VRAM**: ~10 GB → batch sizes limited to 1–4 with current graph sizes (~thousands of nodes per membrane)
- **Data volume**: 70 membrane compositions in `data/membrane_only/`, each with multiple frames
- **Google Colab compatibility**: project is designed to also run on Colab (see `scripts/colab/` and `colab_lipid_gnn_subset/`)

## Dependencies

Pinned in `requirements.txt` (added 2026-04-17):

- `torch>=2.8.0`, `torch-geometric>=2.7.0`
- `MDAnalysis>=2.10.0`
- `numpy>=2.4.0`, `h5py>=3.15.0`, `pandas>=2.0.0`, `scikit-learn>=1.8.0`
- `matplotlib>=3.10.0`, `tqdm>=4.67.0`, `pytest>=8.0.0`
- `wandb` (unpinned; Colab/remote runs only)

## GitHub / SSH

- Remote: `git@github.com:pbkush/lipid-graph-nn.git` (SSH)
- University network blocks port 22; `~/.ssh/config` routes `github.com` → `ssh.github.com:443`
- Claude Code permissions in `.claude/settings.json`: allows `git push/commit/add`, `gh pr create/merge`; denies `git push --force`
- `gh` CLI installed and authenticated (HTTPS token in system keyring); Claude opens and merges PRs with `gh pr create` / `gh pr merge <n> --merge --delete-branch`
- Branch strategy: short-lived feature branches (`feat/`, `exp/`, `fix/`, `refactor/`, `test/`, `data/`, `docs/`), merge commits only (no squash/rebase)

## Tool Usage Patterns

- **Force field data**: Stored as JSON files in `resources/` — `ff_params.json` (bead type → physics params), `ff_node_mapping.json` (molecule+atom → bead type), `ff_edge_params.json` (bond parameters)
- **Experiment results**: Saved to `results/training/lipid_packing/<timestamp_config>/` with metrics JSON and plots
- **Data flow**: Raw MD trajectories → `MartiniHeteroGraphBuilder` → `.pt` chunk files → `MartiniDiskDataset` → DataLoader
