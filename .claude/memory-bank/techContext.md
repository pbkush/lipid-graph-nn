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
- **Installation**: `pip install --use-pep517 .` then `pip install -r requirements.txt` (no `reinstall.sh` — removed as unused)
- **W&B**: `wandb login` once before running any sweep (local or Colab); all metrics now go to W&B only
- **Tests**: `pytest tests/` — no special config needed
- **Training**: `python3 scripts/training/run_sweep.py`
- **Smoke test**: `python3 scripts/training/smoke_test_sweep.py`

## Technical Constraints

- **Training target**: Goethe HPC cluster (AMD MI210 / ROCm 6.2). All training runs go through SLURM sbatch scripts. Colab notebooks in `scripts/colab/` are no longer the active training path.
- **GPU VRAM**: MI210 has 64 GB HBM2e; batch sizes can be raised well beyond the old Colab limit of 1–4.
- **Data volume**: 70 membrane compositions in `data/membrane_only/`, each with multiple frames; raw data lives in `/work/<grp>/<user>/lipid-data/` on the cluster (5 TB quota, 30-day TTL).

## Dependencies

Pinned in `requirements.txt` (added 2026-04-17):

- `torch>=2.8.0`, `torch-geometric>=2.7.0`
- `MDAnalysis>=2.10.0`
- `numpy>=2.4.0`, `h5py>=3.15.0`, `pandas>=2.0.0`, `scikit-learn>=1.8.0`
- `matplotlib>=3.10.0`, `tqdm>=4.67.0`, `pytest>=8.0.0`
- `PyYAML>=6.0` (added 2026-04-24 for `lipid_gnn/config.py`)
- `pyarrow>=15.0.0` (added 2026-04-25 for W&B parquet I/O in `download_wandb_runs.py` and `analyze_hp_search.ipynb`)
- `wandb` (unpinned; Colab/remote runs only)

## GitHub / SSH

- Remote: `git@github.com:pbkush/lipid-graph-nn.git` (SSH)
- University network blocks port 22; `~/.ssh/config` routes `github.com` → `ssh.github.com:443`
- Claude Code permissions in `.claude/settings.json`: allows `git push/commit/add`, `gh pr create/merge`; denies `git push --force`
- `gh` CLI installed and authenticated (HTTPS token in system keyring); Claude opens and merges PRs with `gh pr create` / `gh pr merge <n> --merge --delete-branch`
- Branch strategy: short-lived feature branches (`feat/`, `exp/`, `fix/`, `refactor/`, `test/`, `data/`, `docs/`), merge commits only (no squash/rebase)

## Tool Usage Patterns

- **Central config**: Project-wide paths and defaults live in `config.yaml`; loaded via `from lipid_gnn.config import CONFIG` (typed `@dataclass` tree). Bash reads values through `python scripts/python/print_config_var.py <dotted.key>`. See `systemPatterns.md` for the full pattern.
- **Force field data**: Stored as JSON files in `resources/` — `ff_params.json` (bead type → physics params), `ff_node_mapping.json` (molecule+atom → bead type), `ff_edge_params.json` (bond parameters). Paths sourced from `CONFIG.paths.ff_*_file`.
- **Experiment results**: Saved to `results/training/lipid_packing/<timestamp_config>/` with metrics JSON and plots
- **Data flow**: Raw MD trajectories → `MartiniHeteroGraphBuilder` → `.pt` chunk files → `MartiniDiskDataset` → DataLoader
- **W&B offline analysis**: After a sweep completes, download runs with `python scripts/python/download_wandb_runs.py --group <stage_name>` → `logs/training/<group>/`. Open `scripts/notebooks/analyze_hp_search.ipynb`, set `GROUP`, run all cells for rankings and visualizations. Re-download with `--force` to refresh; the script cleans stale files automatically.
