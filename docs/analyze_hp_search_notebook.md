# `analyze_hp_search.ipynb` — Visualisation Reference

Companion doc for [scripts/notebooks/analyze_hp_search.ipynb](../scripts/notebooks/analyze_hp_search.ipynb).
Explains what each visualisation shows, how to read it, and what to look for.

---

## Prerequisites

Download the target group first, then open the notebook and set `GROUP` in Cell 1:

```bash
python scripts/python/download_wandb_runs.py --group stage_3_arch
# then in Cell 1: GROUP = 'stage_3_arch'
```

---

## Data preparation (Cells 1–5)

Before any plot is drawn the notebook builds two key tables:

| Variable | Contents |
|---|---|
| `runs_df` | One row per individual run (seed resolved). Config keys flattened as columns alongside `val_min_last10`, `test_mse_total`, runtime, GPU stats. |
| `cells_df` | One row per **HP cell** (seeds collapsed). Columns: `val_mean`, `val_std`, `test_mean`, `test_std`, `gap`, `runtime_mean`, `gpu_mem_mb`, per-property `val_{prop}`. Sorted ascending by `val_mean`. |

`VARYING_HPS` is auto-detected: any config key that takes more than one value in the group (excluding `seed`). The groupby for all subsequent plots uses this list.

---

## Visualisations

### (a) Val-loss curves per HP cell

**What it shows** — a small-multiples grid of validation MSE vs. epoch. Each panel is one HP cell (unique combination of varying HPs). Within each panel, individual seed traces are plotted transparently, and a thick black mean curve is overlaid.

**How to read it** — the y-axis is shared across all panels (95th-percentile capped) so panels are directly comparable. Label in the panel title encodes the HP value(s).

**What to look for**
- Cells where the mean curve is still descending at epoch 100 → undertrained; may benefit from more epochs.
- High seed-to-seed spread (faint traces far from the mean) → unstable config; deprioritise even if mean looks good.
- A curve that dips early then climbs back → learning rate too high or weight decay too low (overfitting on training noise).

---

### (b) Per-property val curves

**What it shows** — same small-multiples layout as (a), but split into one **row per property** (`lipid_packing`, `thickness`, etc.). Each panel shows only that property's `val/loss_{prop}` over epochs. Seeds are overlaid with a mean curve.

**How to read it** — rows share a y-scale per property but not across properties (scales differ by orders of magnitude). Column layout matches (a).

**What to look for**
- A cell that wins on total val MSE but is weak on one property (its row in (b) is visibly higher than others) — that cell may fail the Stage-5 per-property gate even though the total looks fine.
- Asymmetric convergence: one property converges fast while the other is still noisy → multi-task balance issue; useful context for Tier B/C property additions.

---

### (c) Ranking table

**What it shows** — a colour-coded summary table with one row per HP cell, sorted best-to-worst by `val_mean`. Columns:

| Column | Meaning |
|---|---|
| varying HP(s) | The swept parameter values for this cell |
| `n_seeds` | Number of seeds that ran |
| `val_mean` | Primary ranking metric: mean over seeds of min(val/loss_total, last 10 epochs) |
| `val_std` | Seed-to-seed spread; lower = more stable |
| `test_mean` | Mean test MSE over seeds (overfit guard, not selection signal) |
| `gap` | `|test_mean − val_mean|`; large gap = possible overfit |
| `val_{prop}` | Per-property val MSE (mean over seeds, last-10-epoch min) |

The `val_mean` column is background-gradient coloured green (low) → red (high).

**What to look for**
- The top row is the primary candidate. Before committing, check that `val_std` is small (reproducible) and `gap` is not an outlier relative to other cells.
- If multiple rows share nearly the same `val_mean` (within ~1%), apply the Occam tie-break: prefer the smaller architecture (`hidden_dim × num_layers` product).

---

### (d) HP heatmap

**What it shows** — a 2-D grid where rows = one HP dimension, columns = another, and the cell colour encodes `val_mean`. Only rendered when **exactly 2 HPs vary** (e.g., Stage 3: `hidden_dim × num_layers`). Skipped otherwise with an explanatory message.

Colour map: green = low MSE (better), red = high MSE (worse). Cell values are annotated as text.

**How to read it** — scan for the green corner. In the Stage-3 architecture grid, this immediately shows whether performance scales with capacity (`hidden_dim`, `num_layers`) or plateaus early.

**What to look for**
- A clear green cell that is not in the largest-capacity corner → Occam win: you get the same quality at lower compute cost.
- A row or column that is uniformly red → one HP dimension is clearly harmful at a certain value (useful to rule out for Stage 5).
- Little variation across the whole grid → the architecture is not the bottleneck; LR or WD from earlier stages dominates.

---

### (e) Test vs. val scatter

**What it shows** — a scatter plot with one point per HP cell. x-axis = `val_mean`, y-axis = `test_mean`. Error bars show ±`val_std` / ±`test_std`. A dashed identity line (`test = val`) is drawn for reference.

**How to read it** — points close to the identity line are well-calibrated: the model generalises to the held-out test set roughly as well as it fits the validation set. The best cell is in the bottom-left corner.

**What to look for**
- Points **above** the identity line: test MSE > val MSE → the model may be overfitting to the val distribution. Check if the gap is consistent across seeds.
- Points **below** the identity line by a large margin: suspicious. Test set may be easier than val for this architecture, or there is a data-split artefact.
- The overall cluster position vs. the Stage-0 baseline: has the best cell moved meaningfully toward the lower-left compared to the baseline run?

---

### (f) Training-stats panel

Three side-by-side subplots covering compute cost.

#### f1 — Wall-time bar chart

**What it shows** — mean total training time (hours) per HP cell, with ±1 std error bars across seeds.

**What to look for**
- Large cells (`hidden_dim=128, num_layers=4`) that take disproportionately long for marginal quality gain → use (e) to check whether the extra runtime is justified.
- Unexpectedly long runs at small architectures → data-loader bottleneck (cross-check with (g)).

#### f2 — Peak GPU memory bar chart

**What it shows** — maximum GPU memory consumed (GB) across seeds per HP cell. A dashed red line marks the MI210's 64 GB ceiling.

**What to look for**
- Cells approaching 64 GB → risk of OOM at `batch_size > 2`. If Stage 4 will sweep batch size, this is the headroom check.
- Memory scales faster than linearly with `hidden_dim` due to attention weight matrices in `GATv2Conv`. A jump here is expected from `hidden_dim=64` to `128`.

#### f3 — Pareto scatter

**What it shows** — x-axis = total runtime (h), y-axis = `val_mean`. Each point is one HP cell. Point size scales with `n_seeds`.

**How to read it** — cells in the **bottom-left** give the best quality at the lowest cost. The ideal choice is not necessarily the cell with the lowest `val_mean`, but the one where further compute investment stops returning quality improvement (the "knee" of the curve).

**What to look for**
- A clear knee: one cell achieves near-minimum `val_mean` at much less runtime than the next step up — strong Occam argument.
- A nearly flat curve where all cells take similar time but quality varies — capacity, not compute, is the limiting factor.

---

### (g) System metrics time-series

**What it shows** — one panel per top-3 HP cell (by `val_mean`). Left y-axis (blue): GPU utilisation %; right y-axis (red): CPU process memory MB. x-axis: elapsed runtime in hours. All seeds for each cell are overlaid.

**How to read it** — this plot uses W&B's auto-sampled system metrics (~15 s intervals), so curves are much sparser than the per-epoch loss history. Shown only when `system.parquet` is non-empty and the relevant columns are detected.

**What to look for**
- **Low GPU utilisation (<50%)** despite high `sec_per_epoch` → data-loading bottleneck. Increasing `num_workers` or pre-staging chunks to `/local/$SLURM_JOB_ID` will help more than changing the architecture.
- **Memory growing over time** (upward trend in red) → potential memory leak; investigate `persistent_workers` or large graph cache.
- **Spiky GPU utilisation** (alternating high/zero) → small batch size causing idle time between batches. Consider raising `batch_size` or `prefetch_factor`.

---

## Recommendation cell

After all plots, the recommendation cell applies the selection rules from
[docs/hp_search_plan.md](hp_search_plan.md):

1. **Primary**: lowest `val_mean`.
2. **Tie-break** (within `OCCAM_TOL = 1%` of leader): smallest `val_std` → smallest `gap` → smaller model.
3. **Gate check**: each property's `val_{prop}` is compared to the Stage-5 acceptance thresholds (`lipid_packing < 0.056`, `thickness < 0.219`). A FAIL here means the recommended config needs further tuning before it can be reported.

The printed output names the winning HP combination and shows whether it passes or fails each gate.

---

## Multi-group comparison (optional)

Set `GROUPS = ['stage_0_baseline', 'stage_1_lr', 'stage_2_wd', 'stage_3_arch']`
in Cell 1 to activate. Produces a grouped bar chart of **best val MSE and best test MSE per stage**, making it easy to see whether each search stage produced a genuine improvement over the previous one.
