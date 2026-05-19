# HP search plan — 2 properties, Goethe-HLR

## Context

The 2-property (`lipid_packing` + `thickness`) baseline reproduces around **MSE ≈ 0.138** with the Colab-era config, but hyperparameters were never swept systematically on the new chunk layout (11 Å cutoff, 25 frames/system, train/val/test split by system, 8-column `y` sliced at training time). With the Goethe-HLR pipeline now landed (`sbatch_sweep.sh`, `CHUNKS_DIR` env, ROCm 7.2.0 PyTorch), we can run real sweeps.

This plan lays out a **stage-based HP search** on 2 properties. Goal: find a config that beats the 0.138 baseline (and ideally pushes per-property MSE below **0.056** for `lipid_packing` and **0.219** for `thickness` — the current best individual-target numbers) before moving on to Tier A (4 properties) as described in [multi_property_training_plan.md](multi_property_training_plan.md).

## Baseline config

| Param | Value |
| ----- | ----- |
| epochs | 100 |
| batch_size | 2 |
| num_workers | 10 (→ 6 on HPC) |
| hidden_dim | 64 |
| num_layers | 2 |
| learning_rate | 5e-4 |
| weight_decay | 5e-3 |

Deviations from current [scripts/training/run_sweep.py](../scripts/training/run_sweep.py) (lines 44–61):

- `FIXED["batch_size"]` is `4` in code, baseline is `2` → change to `2`.
- `FIXED["num_workers"]` is `2` in code, baseline is `10` → cap at **6** on HPC (sbatch has `--cpus-per-task=8`; 10 would over-subscribe).
- Other 5 params match the current SWEEP grid at the baseline point.

## HPC target

- **Partition**: `gpu` (not `gpu_test`).
- **Allocation**: 1 MI210 GPU + 8 CPUs per job.
- **Wall-time limit**: 24 h per sbatch job.
- **Throughput**: ~23 runs per sbatch job (1 h/run, with 1 h slack for staging + eval + W&B upload).
- **ROCm**: 7.2.0.

### One-time sbatch / config changes (before Stage 0)

Edit [scripts/bash/sbatch_sweep.sh](../scripts/bash/sbatch_sweep.sh):

- `--partition=gpu_test` → `--partition=gpu`
- `--time=08:00:00` → `--time=24:00:00`
- Keep `--gres=gpu:1`, `--cpus-per-task=8`, `--mem=64G` (1-GPU fraction of a full node).
- Export `WANDB_GROUP="stage_N_<name>"` before the `python run_sweep.py` call so each stage's runs cluster in W&B.

Edit [docs/hpc_goethe.md](hpc_goethe.md) lines 56–57: ROCm wheel/module `6.2`/`6.2.4` → `7.2`/`7.2.0`.

Edit [scripts/training/run_sweep.py](../scripts/training/run_sweep.py) around line 90: pass `group=os.environ.get("WANDB_GROUP")` to `wandb.init(...)`.

## Hyperparameters to test

Sorted by expected impact. First four are primary; rest are secondary/optional.

### Primary (must sweep)

1. **`learning_rate`** — highest-variance knob on small GNNs; Adam is sensitive. Range: `{1e-4, 3e-4, 5e-4, 1e-3, 3e-3}`.
2. **`weight_decay`** — 5e-3 is already high; worth checking whether we're over-regularizing. Range: `{0, 1e-4, 1e-3, 5e-3, 1e-2}`.
3. **`hidden_dim`** — capacity. Range: `{32, 64, 128}`. (128 may hit VRAM ceiling at batch_size=2 on MI210; AMP may or may not rescue it.)
4. **`num_layers`** — receptive field for the GATv2 stack. Range: `{2, 3, 4}`. (4 may exceed Martini bilayer diameter — useful to document the empty result.)

### Secondary

1. **`batch_size`** — `{2, 4, 8}`. Trades gradient noise vs. memory.
2. **`dropout`** — currently hardcoded `p=0.3` at [lipid_gnn/membrane_prop_gnn.py:47](../lipid_gnn/membrane_prop_gnn.py#L47). Not sweep-able today; if needed, add a `dropout` kwarg to `MembranePropertyGNN.__init__` and plumb it through `FIXED`/`SWEEP`.
3. **`heads`** (GATv2) — hardcoded `4` at [lipid_gnn/membrane_prop_gnn.py:34-36](../lipid_gnn/membrane_prop_gnn.py#L34-L36). Out of scope.

### Not tuned here

- `spatial_cutoff`, `num_frames`, `chunk_size` — baked into chunks; would require regenerating all chunks.
- Optimizer (`Adam` vs `AdamW`), scheduler (`ReduceLROnPlateau` patience/factor) — hold fixed; revisit only if the LR sweep plateaus.

## Staged search plan

Each stage edits `FIXED`/`SWEEP` in `run_sweep.py`, commits the change, and submits one `sbatch_sweep.sh` job. `properties=['lipid_packing', 'thickness']` throughout.

### Stage 0 — Baseline reproduction (1 sbatch, ~3 h)

Confirm the cited baseline reproduces overall MSE ≈ 0.138 on the new 3-directory chunks.

```python
FIXED = {"epochs": 100, "batch_size": 2, "num_workers": 6}
SWEEP = {"hidden_dim": [64], "num_layers": [2],
         "learning_rate": [5e-4], "weight_decay": [5e-3], "seed": [0, 1, 2]}
```

3 runs. Gate: mean test MSE within ±0.02 of 0.138. Abort and investigate if not.

### Stage 1 — Learning-rate sweep (1 sbatch, ~10 h)

```python
SWEEP["learning_rate"] = [1e-4, 3e-4, 5e-4, 1e-3, 3e-3]
SWEEP["seed"] = [0, 1]
```

10 runs. Select top-2 LRs by mean val MSE for Stage 2.

### Stage 2 — Weight-decay sweep (1 sbatch, ~10 h)

At top LR from Stage 1:

```python
SWEEP["learning_rate"] = [<best_lr>]
SWEEP["weight_decay"] = [0, 1e-4, 1e-3, 5e-3, 1e-2]
SWEEP["seed"] = [0, 1]
```

10 runs.

### Stage 3 — Architecture grid (1 sbatch, ~18 h)

At best (LR, WD):

```python
SWEEP["hidden_dim"] = [32, 64, 128]
SWEEP["num_layers"] = [2, 3, 4]
SWEEP["seed"] = [0, 1]
```

18 runs. Fits in one 24 h job. If `hidden_dim=128, num_layers=4, batch_size=2` OOMs, fall back to `batch_size=1` for that cell only (AMP is already enabled).

### Stage 4 (optional) — Batch-size sweep (1 sbatch, ~9 h)

At best (LR, WD, arch):

```python
SWEEP["batch_size"] = [2, 4, 8]
SWEEP["seed"] = [0, 1, 2]
```

9 runs. Run only if Stage 3 leaves headroom or loss curves look gradient-noise-limited.

### Stage 5 — Confirmation (1 sbatch, ~5 h)

Single best config with **5 seeds** (0–4) for a reportable mean ± std. Logs under `WANDB_GROUP=stage_5_best`.

5 runs.

## Total budget

| Stage | Runs | Wall time (serial, 1 GPU) |
| ----- | ---- | ------------------------- |
| 0 | 3 | ~3 h |
| 1 | 10 | ~10 h |
| 2 | 10 | ~10 h |
| 3 | 18 | ~18 h |
| 4 (opt) | 9 | ~9 h |
| 5 | 5 | ~5 h |
| **Total** | **46–55** | **~46–55 h across 6 sbatch jobs** |

Each stage fits inside a single 24 h sbatch (Stage 3 is the tightest at ~18 h — submit fresh, not at the end of a busy day).

## Critical files to modify

- [scripts/bash/sbatch_sweep.sh](../scripts/bash/sbatch_sweep.sh) — one-time: partition `gpu`, time `24:00:00`, export `WANDB_GROUP=stage_N_<name>` per submission.
- [scripts/training/run_sweep.py](../scripts/training/run_sweep.py) lines 44–61 — per-stage: edit `FIXED` + `SWEEP`; commit each stage's values so the SHA logged to W&B is reproducible.
- [scripts/training/run_sweep.py](../scripts/training/run_sweep.py) around line 90 — one-time: add `group=os.environ.get("WANDB_GROUP")` to `wandb.init(...)`.
- [lipid_gnn/membrane_prop_gnn.py:47](../lipid_gnn/membrane_prop_gnn.py#L47) — only if a dropout sweep is added; otherwise untouched.
- [docs/hpc_goethe.md](hpc_goethe.md) — one-time: ROCm `6.2.4` → `7.2.0`, wheel URL `rocm6.2` → `rocm7.2`.

## Verification / review procedure

After each stage:

1. In W&B project `lipid_gnn_lipid_packing_thickness`, filter runs by `group=stage_N_*`.
2. Plot `val/mse_total` and per-property `val/mse_<prop>` vs. epoch for all seeds.
3. Rank by **min val MSE over the last 10 epochs** (not final epoch — too noisy).
4. Check test MSE isn't drifting away from val MSE (overfit guard).
5. Pick top-N for the next stage; record the decision in [.claude/memory-bank/activeContext.md](../.claude/memory-bank/activeContext.md).

End-to-end verification after Stage 5:

- Paired t-test (α=0.05) on per-seed test MSE: Stage-5 best < Stage-0 baseline.
- Per-property test MSE for `lipid_packing` < 0.056 **and** `thickness` < 0.219 on at least the best seed.
- Spot-check one Stage-5 run by rerunning locally (Colab or workstation) for 10 epochs and confirming the same loss trajectory to within seed noise — protects against HPC-specific artifacts (ROCm numerics, AMP quirks).

## Decisions locked in

- `num_workers=6` on HPC.
- Stage 4 (batch size) is **optional**.
- W&B grouping: single project, `group=stage_N_<name>` via `WANDB_GROUP` env.
- Partition: `gpu` (24 h limit), 1 MI210 + 8 CPUs per job. No 8-GPU fanout.
- ROCm 7.2.0.
