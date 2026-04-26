# Stage 5 analysis notebook — plan

Companion plan for the **publication-grade** analysis notebook of the Stage-5
confirmation runs (5 seeds at the Stage-3 winner config, on the new stratified
chunks). Output: thesis-quality figures + reportable mean ± std numbers
suitable for showing to other researchers (committee, paper readers, future
students).

Distinguished from [analyze_hp_search_notebook.md](analyze_hp_search_notebook.md):
that notebook is **exploratory / decision-making** (rank HP cells, pick a
winner). This one is **confirmatory / publication** (one config, five seeds,
honest test numbers, generalization story).

---

## Context

After completing HP stages 0–3, the Stage-3 winner is `hidden_dim=128,
num_layers=2, lr=1e-4, wd=1e-3` (val_mean ≈ 0.038, val_std ≈ 4e-4 — the
most stable cell in the grid). Stage 5 runs this config with **5 seeds (0–4)**
on the **stratified-split chunks** ([prepare_colab_subset.py](../scripts/training/prepare_colab_subset.py)
with `--split-method stratified`) so the test MSE is a defensible number.

This notebook turns those 5 runs into a **single self-contained analysis
artifact** — every plot stands alone with a caption, every number has a CI,
every claim is paired with the data that supports it.

---

## Prerequisites — pipeline changes before Stage 5

The current training loop ([run_sweep.py:220–243](../scripts/training/run_sweep.py#L220-L243))
logs only scalars + a rendered accuracy plot to W&B. For thorough analysis we
need raw arrays and per-graph composition labels. Two small changes:

### P1 — Tag graphs with composition label at preprocessing time

In [lipid_gnn/dataset.py](../lipid_gnn/dataset.py) `preprocess_and_save`,
extract the composition string from the sim_tuple's path (e.g.,
`data/membrane_only/POPC95_CHOL5/run/prun.tpr` → `"POPC95_CHOL5"`) and attach
it to each `HeteroData` object:

```python
data.composition = composition_str  # e.g. "POPC95_CHOL5"
data.system_idx  = system_idx       # int, stable per-system ID
```

This is one line per graph at preprocessing time, no model-side change. PyG
preserves arbitrary attributes through the DataLoader's batching as a list of
strings/ints in the batch.

**Test**: extend [tests/test_dataset.py](../tests/test_dataset.py) — assert
that loaded graphs carry `composition` and `system_idx`, and that
`set(test_compositions) ∩ set(train_compositions) == ∅`.

### P2 — Persist test predictions + metadata per run

In [run_sweep.py](../scripts/training/run_sweep.py) `train_one_run`, after the
held-out evaluation loop (around line 234), save raw arrays to the run directory
*before* `wandb.finish()`:

```python
out_path = Path(wandb.run.dir) / "test_artifacts.npz"
np.savez(
    out_path,
    test_preds=test_preds,                      # (N, P) normalized
    test_targets=test_targets,                  # (N, P) normalized
    test_compositions=np.array(test_comps),     # (N,) string
    test_system_idx=np.array(test_sys_idx),     # (N,) int
    scaler_mean=scaler.mean_[prop_cols],        # (P,) for de-normalization
    scaler_scale=scaler.scale_[prop_cols],      # (P,)
    properties=np.array(properties),            # (P,) string
)
wandb.save(str(out_path))
```

`wandb.save` uploads the file to W&B's run storage. The download script
[download_wandb_runs.py](../scripts/python/download_wandb_runs.py) already
copies the entire `run.dir`, so no change needed there. Per-run cost: ~50 KB.

**Test**: smoke test in [tests/test_dataset.py](../tests/test_dataset.py) or
new file — load `.npz`, check shapes, check that de-normalized predictions are
in physically plausible ranges.

### P3 — One-time linear-baseline rerun on stratified chunks

Re-run [linear_baseline.py](../scripts/training/linear_baseline.py) on the new
stratified split so it reports test MSE on the same held-out compositions as
the GNN. Save its predictions in the same `.npz` format under
`results/training/linear_baseline_stratified.npz` for the comparison panels.

---

## Notebook artifacts

- **File**: `scripts/notebooks/analyze_stage_5.ipynb`
- **Companion doc**: `docs/analyze_stage_5_notebook.md` (written after the
  notebook lands, mirroring `analyze_hp_search_notebook.md`)
- **Figure outputs**: `results/figures/stage_5/` — one PDF + one PNG per
  figure (PDF for print/thesis, PNG for slides). All saved at 300 DPI.
- **Numerical outputs**: `results/figures/stage_5/headline_numbers.json` —
  every annotated number in the figures (mean ± std, R², CI bounds, p-values)
  in one machine-readable file so the thesis text can pull from it.

---

## Style conventions

Set once in Cell 1, applied to every figure:

```python
plt.rcParams.update({
    "font.family":     "sans-serif",
    "font.size":       11,
    "axes.labelsize":  11,
    "axes.titlesize":  11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.dpi":      120,        # screen
    "savefig.dpi":     300,        # file
    "savefig.bbox":    "tight",
    "axes.spines.top":   False,
    "axes.spines.right": False,
})
PROP_LABELS = {                    # axis labels with units
    "lipid_packing":  "Lipid packing (a.u.)",
    "thickness":      "Bilayer thickness (Å)",
    "thickness_std":  r"$\sigma_{\mathrm{thick}}$ (Å)",
    "variation":      "Compositional variation",
}
PALETTE = {                        # colorblind-safe (Wong 2011)
    "gnn":      "#0072B2",
    "baseline": "#D55E00",
    "identity": "#999999",
    "seed":     "#56B4E9",
}
```

Every plot saves via a one-line `save_fig(fig, "name")` helper that writes
both PDF and PNG and records the figure name + caption + headline numbers
into the `headline_numbers.json` running dict.

---

## Cell layout

| # | Cell ID | Purpose |
|---|---------|---------|
| 1 | `setup` | Imports, rcParams, `LOGS_DIR`, `GROUP = "stage_5_best"`, `OUT_DIR` |
| 2 | `load-runs` | Load 5 runs' `config.json`, `summary.json`, `history.parquet`, `test_artifacts.npz` |
| 3 | `load-baseline` | Load `linear_baseline_stratified.npz` for comparison plots |
| 4 | `aggregate` | Build `runs_df` (per-run scalars), `histories` dict, `preds_per_seed` array (5, N, P), `mean_preds` (N, P) — ensemble |
| 5 | `denormalize` | Convert all preds/targets to physical units using `scaler_mean / scaler_scale` |
| 6 | `headline-table` | Print the reportable table: per-property test MSE / MAE / R² mean ± std + bootstrap 95 % CI |
| 7–15 | one cell per figure (a)–(i) below | Each cell: build figure, save via `save_fig`, display inline |
| 16 | `stat-tests` | Paired t-test (Stage-5 vs Stage-0 baseline, matched seeds), bootstrap CIs, save to `headline_numbers.json` |
| 17 | `gate-check` | Apply HP-plan gates (`lipid_packing < 0.056`, `thickness < 0.219`); report PASS/FAIL with margin |
| 18 | `export` | Write `headline_numbers.json`; print one-line summary suitable for the thesis abstract |

---

## Visualizations

Each entry below is one figure (one cell). For each: **what**, **how**,
**why for the audience**, **caption template**. Every figure is captioned
with one sentence stating the takeaway, then a method note.

### (a) Training dynamics — loss curves with seed band

**What** — 1×3 grid: total loss, `lipid_packing` loss, `thickness` loss
(extend to 1×5 for Tier A). Per panel: x = epoch, y = MSE (normalized).
Faint individual seed traces + thick mean curve + ±1 SD shaded band.
Include both train (lighter) and val (darker) curves with a legend.

**Why** — proves training is stable across seeds (bands stay narrow late in
training); shows when convergence is reached (mean curve flattens); reveals
whether 100 epochs is enough or undertrained.

**Caption template** — *"Validation loss converges by epoch ~X across all
five seeds (mean ± 1 SD shaded). The narrow band confirms reproducibility
of the chosen architecture."*

### (b) Predicted vs true scatter — per property, ensemble pooled

**What** — one panel per active property. x = true (physical units), y =
predicted (physical units, de-normalized via `scaler_mean / scale`). Pool
all 5 seeds × N_test_graphs points; identity line dashed; annotate R² and
MAE in the panel corner. Color points by *system* (one color per held-out
composition, ~10 colors) so reviewers can see whether systematic per-system
bias exists.

**Why** — the canonical generalization plot. Per-system coloring lets
readers spot "this composition is consistently over-/under-predicted" without
needing a separate plot.

**Method note in caption** — points pooled over 5 seeds and 250 test graphs;
R² computed on the pooled set; identity line shows perfect prediction.

### (c) Per-system error bar chart

**What** — bar chart, x = held-out composition (10 systems, sorted by total
MAE descending), y = MAE per property (grouped bars or stacked). Error bars
= ±1 SD across the 5 seeds. Color bars by lipid-class membership (e.g.,
single-component, binary PC, binary PE, ternary, cholesterol-containing).

**Why** — concrete answer to "which membranes does the model fail on?"
Useful for the thesis discussion: are failures correlated with a specific
lipid type, or with extreme compositions? Doubles as a defensibility plot
for committee questions.

**Caption template** — *"Per-system test MAE for the 10 held-out
compositions, sorted by total error. Error bars: ±1 SD across 5 seeds."*

### (d) Residual distribution

**What** — one histogram per property of `(pred − true)` in physical units,
pooled over seeds × test graphs. Overlay a Gaussian fit and annotate
`mean` (bias) and `std` (random error). Vertical line at 0.

**Why** — a model with non-zero residual mean is biased. The Gaussian fit
quality also indicates whether errors are heavy-tailed (a few catastrophic
predictions) or well-behaved.

**Audience cue** — researchers will look for `mean ≈ 0` (unbiased) and a
narrow, symmetric distribution (calibrated).

### (e) GNN vs linear-composition baseline

**What** — two-panel figure. Left: GNN predicted-vs-true overlaid on linear
baseline predicted-vs-true (different markers/colors), one panel per
property. Right: bar chart of test MSE comparing GNN (mean ± std over seeds)
vs linear baseline (single deterministic value). Annotate the relative
improvement (`(MSE_baseline − MSE_GNN) / MSE_baseline`).

**Why** — this is the most-asked question by reviewers: *"how much does
graph structure actually buy you over just the composition vector?"*
Quantifies the GNN's contribution above a no-structure baseline.

**Caption template** — *"GNN reduces test MSE on `lipid_packing` by X %
relative to a Ridge regression on composition fractions, demonstrating that
graph-level structure captures information beyond bulk composition."*

### (f) HP search progression

**What** — bar chart of best **stratified test MSE** per stage (0, 1, 2, 3, 5)
for total + per property. Stage 0 / 1 / 2 / 3 numbers come from re-evaluating
the corresponding winner configs on the new stratified test set (the bar
chart in the existing analysis notebook uses the *random-split* numbers,
which aren't comparable). Stage 5 = mean ± std over 5 seeds.

**Why** — the "did HP search work?" plot. One figure shows the cumulative
improvement from baseline to confirmed winner. Skipping this means the
thesis can't claim "we improved over baseline by X" with a defensible number.

**Note** — requires re-running each prior stage's winner config on the new
chunks (one short sbatch each, ~3–5 h total). Worth the cost for the cleanest
narrative figure.

### (g) Generalization map — composition-space embedding

**What** — 2-D PCA (or UMAP if compositions are too low-dimensional) of all
70 systems' composition vectors. Color points by split (train / val / test)
with shape encoding test-set MAE (small = good, large = bad).

**Why** — does the model fail on compositions far from training? Or is
error roughly uniform? Answers a fundamental generalization question that
*per-system MAE* alone can't.

**Caveat** — with only ~10 lipid types, PCA captures most variance in 2 PCs.
If most points cluster on a manifold, switch to a triangle plot of the most
common 3 lipids (POPC, DOPC, CHOL) with the others annotated.

### (h) Per-property R² with bootstrap CI

**What** — forest plot. Per property: point estimate of R² (ensemble mean)
with 95 % bootstrap CI as a horizontal error bar. Reference dashed line at
R² = 0 (predicting the mean). Annotate the per-property HP-plan acceptance
gate threshold as a colored span for context.

**Why** — gives the reportable per-property number with proper uncertainty.
Forest plots are conventional in ML papers and translate cleanly to thesis
text.

### (i) Stage-5 vs Stage-0 paired comparison

**What** — paired dot plot. Per seed (0–4): one line connecting Stage-0
test MSE to Stage-5 test MSE. Annotate paired t-test p-value.

**Why** — the matched-seed comparison is the cleanest evidence that HP
search caused the improvement (rules out seed-luck). One small figure with
high inferential value.

**Note** — requires Stage 0 to have been re-run with the *same 5 seeds* on
the new stratified chunks. Schedule this alongside the Stage-5 sbatch.

---

## Statistical procedures

All implemented in cell 16 (`stat-tests`):

1. **Paired t-test** (`scipy.stats.ttest_rel`) on per-seed test MSE: Stage 5
   vs Stage 0, both on stratified chunks. Report t-statistic, df, p-value,
   one-sided alternative ("Stage 5 < Stage 0").
2. **Bootstrap 95 % CI on R²**: 10 000 resamples of test predictions per
   property. Report CI bounds in the headline table.
3. **Bootstrap 95 % CI on test MSE**: same procedure for the headline number.
4. **Per-property gate check**: `lipid_packing < 0.056`, `thickness < 0.219`
   (val MSE thresholds from [gnn_only_hp_search_plan.md:171](gnn_only_hp_search_plan.md#L171)).
   Report PASS / FAIL with margin in σ.

All numerical outputs persisted to `results/figures/stage_5/headline_numbers.json`
so the thesis text doesn't have to re-run anything to cite a number.

---

## Output / file artifacts

```
results/figures/stage_5/
├── headline_numbers.json
├── fig_a_loss_curves.{pdf,png}
├── fig_b_pred_vs_true.{pdf,png}
├── fig_c_per_system_mae.{pdf,png}
├── fig_d_residuals.{pdf,png}
├── fig_e_vs_baseline.{pdf,png}
├── fig_f_hp_progression.{pdf,png}
├── fig_g_generalization_map.{pdf,png}
├── fig_h_r2_forest.{pdf,png}
└── fig_i_paired_stages.{pdf,png}
```

PDF for thesis insertion (vector, scales without artifacts), PNG for slides
and the auto-generated companion doc.

---

## Reuses / dependencies

- Data loading mirrors [analyze_hp_search.ipynb](../scripts/notebooks/analyze_hp_search.ipynb)
  cell 2 — no need to re-implement the parquet/JSON load.
- [lipid_gnn/plotting.py:plot_property_accuracies](../lipid_gnn/plotting.py)
  is the basis for figure (b) but needs reworking for the seeded / pooled /
  per-system-colored variant. Reuse as a starting template, not as-is.
- [linear_baseline.py](../scripts/training/linear_baseline.py) provides the
  baseline for figure (e); needs the prerequisite re-run on stratified
  chunks first.
- New dependency: `scipy.stats` for `ttest_rel`. Already in `sklearn` deps tree.
- Optional new dependency: `umap-learn` if PCA isn't enough for figure (g);
  decide after first inspection of the 2-D PCA.

---

## Files touched

- **New**: `scripts/notebooks/analyze_stage_5.ipynb`
- **New**: `docs/analyze_stage_5_notebook.md` (after notebook lands)
- **Edited**: `lipid_gnn/dataset.py` — attach `composition` and `system_idx`
  per graph in `preprocess_and_save` (P1)
- **Edited**: `scripts/training/run_sweep.py` — save `test_artifacts.npz`,
  upload via `wandb.save` (P2)
- **Edited**: `tests/test_dataset.py` — assert composition labels survive
  loading; assert no leakage between splits
- **Re-run**: `scripts/training/linear_baseline.py` on stratified chunks (P3)
- **No edits** to the existing `analyze_hp_search.ipynb` — it stays as the
  exploratory tool.

---

## Verification

1. **Prerequisite tests pass**:
   - `pytest tests/test_dataset.py -k composition` — graphs carry the label.
   - Smoke-load a single Stage-5 run's `test_artifacts.npz` and confirm
     shapes match `(N_test_graphs, n_properties)`.

2. **Notebook runs end-to-end** with all 5 Stage-5 runs downloaded:
   `python scripts/python/download_wandb_runs.py --group stage_5_best`,
   then `jupyter nbconvert --execute analyze_stage_5.ipynb`.

3. **Sanity checks on outputs**:
   - All 9 figures saved to `results/figures/stage_5/`.
   - `headline_numbers.json` parses; every figure's annotated numbers are
     present.
   - Headline test MSE is **higher** than the old random-split number
     (~0.028) — a *lower* number on the new chunks would be suspicious,
     given the wider test distribution.

4. **Caption coherence**: every figure's caption text matches the
   corresponding entry in `headline_numbers.json` (a fixture test could
   automate this if it becomes a maintenance burden — out of scope today).

---

## Decisions to lock in before starting

- **Tier scope**: notebook supports both 2-property (`lipid_packing`,
  `thickness`) and Tier A 4-property runs out of the box — the per-property
  loop is data-driven from `properties` field in the saved `.npz`. No
  config flag needed.
- **Skip figure (g)** if the 2-D PCA captures < 60 % variance — switch to
  a triangle plot per the caveat above; record the decision in the notebook.
- **Stage 0 re-run with matched seeds**: required for figure (i) and the
  paired t-test. Schedule the sbatch *concurrently* with Stage 5 to avoid
  serial wall-time blow-up.
