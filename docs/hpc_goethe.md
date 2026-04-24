# Goethe-HLR deployment runbook

Short checklist for running the training pipeline on Goethe-HLR (AMD MI210 / ROCm).
Longer rationale lives in the plan at `.claude/plans/` and in the memory bank.

Wiki: https://wiki.csc.uni-frankfurt.de/wiki/doku.php?id=public:usage:goethe

## Filesystem layout

| Path | Purpose |
| ---- | ------- |
| `$HOME/lipid-graph-nn/` | git clone (code only; `/home` is 30 GB) |
| `$HOME/miniforge3/`     | user-local conda |
| `/work/$GROUP/$USER/lipid-data/data/membrane_only/` | raw `.tpr`/`.xtc` |
| `/work/$GROUP/$USER/lipid-data/results/properties/` | `<COMP>.h5` property files |
| `/work/$GROUP/$USER/lipid-data/chunks/`             | preprocessed `.pt` chunks |
| `/work/$GROUP/$USER/lipid-data/wandb/`              | W&B offline runs |
| `/local/$SLURM_JOB_ID/chunks/`                      | per-job staged chunks (fast I/O) |

Set `export GROUP=<your-group>` in `~/.bashrc`. The sbatch scripts refuse to run without it.

## One-time bootstrap

### Probe connectivity (login node, then inside `gpu_test`)

```bash
quota
module avail 2>&1 | grep -iE 'rocm|python|miniforge'
curl -I -m 5 https://pypi.org
curl -I -m 5 https://api.wandb.ai
srun -p gpu_test --gres=gpu:1 --time=02:00:00 --pty bash   # repeat curl inside
```

### Transfer data from the laptop

```bash
rsync -avh --partial --progress \
  data/membrane_only/ \
  $USER@goethe.hhlr-gu.de:/work/$GROUP/$USER/lipid-data/data/membrane_only/

rsync -avh --partial --progress \
  results/properties/ \
  $USER@goethe.hhlr-gu.de:/work/$GROUP/$USER/lipid-data/results/properties/
```

### Install the env (inside a `gpu_test` allocation)

```bash
cd $HOME
curl -LO https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh -b -p $HOME/miniforge3
source $HOME/miniforge3/etc/profile.d/conda.sh
conda create -y -n lipid_gnn python=3.13
conda activate lipid_gnn

module load rocm/6.2.4
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
pip install torch-geometric
pip install -r $HOME/lipid-graph-nn/requirements.txt
pip install --use-pep517 $HOME/lipid-graph-nn

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
pytest -q $HOME/lipid-graph-nn/tests
```

If the ROCm 6.2 wheel is older than 2.8.0 and the import succeeds anyway, relax
the `torch>=2.8.0` pin in `requirements.txt` — it's a minimum-version guard,
not a correctness requirement.

If compute nodes turn out to be air-gapped: `pip download -d $HOME/wheels ...`
from the login node, then `pip install --no-index --find-links $HOME/wheels ...`
inside the allocation. Also `export WANDB_MODE=offline` before submitting jobs.

## Recurring flow

```bash
# Sync code (from laptop):  git push
# On cluster:
cd $HOME/lipid-graph-nn && git pull
sbatch scripts/bash/sbatch_preprocess.sh        # once per preprocessing-config change
sbatch scripts/bash/sbatch_sweep.sh             # per sweep
squeue -u $USER
tail -f logs/sweep-<jobid>.out
```

Offline W&B reconciliation:

```bash
wandb sync /work/$GROUP/$USER/lipid-data/wandb/offline-run-*
```

## Gotchas

- `torch.cuda.is_available()` returns `True` under ROCm — HIP shims the CUDA API, model code is unchanged.
- GPU software must be compiled on GPU nodes, not the login node. Always install from inside a `gpu_test` allocation.
- `/work` files older than 30 days get deleted — keep the raw data rsync target outside of temp directories, and rerun preprocessing if chunks age out.
- Full-node `gpu` partition is allocated per-node (8× MI210). Use `gpu_test --gres=gpu:1` until the sweep is validated on one GPU.
