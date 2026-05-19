# Three-way comparison: bugged-legacy vs bugfixed-legacy vs bugfixed-M3

Plan for [scripts/notebooks/compare_bugfix_three_way.py](../scripts/notebooks/compare_bugfix_three_way.py),
a marimo notebook that disentangles three confounded effects on the
70-composition corpus:

- **Bug-fix effect** — `lipid_gnn/functions_emil/calculate_properties.py`
  (12 logical bugs, [docs/functions_emil_cleanup_plan.md](functions_emil_cleanup_plan.md) §2)
  replaced by `lipid_gnn/properties.py` (post-cleanup).
- **Force-field / trajectory effect** — legacy GMX-2 trajectories under
  vendored ITPs replaced by re-simulations under upstream M3-Lipid-Parameters
  ITPs (`submit_simulations.sh --from-csv resources/done.csv --rename-lipid DIPC=DLPC`).
- **RNG-seed effect** (bonus contrast) — the legacy pipeline left `numpy`'s
  global RNG unseeded (bug #8 in the cleanup plan), so the historical
  shipped labels are one of an ensemble of possible draws. Seeding fixes
  that, but it also means *the shipped labels are a different sample*
  than `legacy=True, seed=0` would now produce. Quantifying this is a
  cheap, important pre-test for the bug-fix contrast.

A pairwise legacy-vs-new comparison ([scripts/notebooks/compare_legacy_vs_new_m3.py](../scripts/notebooks/compare_legacy_vs_new_m3.py))
mixes the first two effects and ignores the third. The notebook here
isolates all three.

Built per the `marimo-data-analysis` skill — reactive layout, all imports
in cell 1, prose-first, figures returned as `_fig,` rather than `plt.show()`,
`mo.stop` guards after data loads, `mo.callout` for headline findings.

---

## 1. Label sets and directory layout

All property-pickle directories live under `results/properties/` so the
parent dir tracks all property sets the project ever produces. Origin
(trajectory provenance) and method (property-pipeline version) are
encoded in the directory name:

```text
results/properties/
├── prop_legacy_bugged_random/   ← shipped labels, used by every training run to date
├── prop_legacy_bugged_s0/       ← legacy pipeline rerun on legacy trajectories, seed=0
├── prop_legacy_bugfixed_s0/     ← bugfixed pipeline on legacy trajectories, seed=0
└── prop_m3_bugfixed_s0/         ← bugfixed pipeline on M3-rerun trajectories, seed=0
```

Naming key: `prop_<traj>_<method>_<seed>` where `<traj> ∈ {legacy, m3}`,
`<method> ∈ {bugged, bugfixed}`, `<seed> ∈ {random, sN}`.
`random` denotes an unseeded `numpy` global RNG (bug #8); `sN` denotes
a `seed=N` kwarg into `lipid_gnn.properties.compute_all`. The legacy
pipeline supports the seed kwarg via the `legacy=True` runtime switch
in the rewritten `compute_all`. Status as of the plan: these dirs are
not yet created — the user is generating the labels.

Four label sets total:

| Tag in notebook | Directory | Property pipeline | Trajectories | RNG |
| --- | --- | --- | --- | --- |
| `legacy_bugged_random` | `prop_legacy_bugged_random/` | `functions_emil` legacy (12 bugs) | legacy GMX-2, vendored ITPs | unseeded |
| `legacy_bugged_s0` | `prop_legacy_bugged_s0/` | `lipid_gnn.properties.compute_all(legacy=True, seed=0)` | same legacy trajectories | seed=0 |
| `legacy_bugfixed_s0` | `prop_legacy_bugfixed_s0/` | `lipid_gnn.properties.compute_all(legacy=False, seed=0)` | same legacy trajectories | seed=0 |
| `m3_bugfixed_s0` | `prop_m3_bugfixed_s0/` | `lipid_gnn.properties.compute_all(legacy=False, seed=0)` | M3-ITP resimulated | seed=0 |

Three primary contrasts and one bonus contrast, all paired on
`canonical_name`:

- **Seed contrast** (bonus) — `legacy_bugged_s0` − `legacy_bugged_random`.
  Same code, same trajectories, RNG only differs. Measures how much of
  the historical labels' value at any single composition is "true label"
  vs. "RNG draw on a noisy estimator". Sets a floor on the bug-fix
  contrast: an effect smaller than the seed contrast is not actionable.
- **Bug-fix contrast** — `legacy_bugfixed_s0` − `legacy_bugged_s0`.
  Same trajectories, same seed; code differs. Isolated.
- **FF contrast** — `m3_bugfixed_s0` − `legacy_bugfixed_s0`. Same code,
  same seed; trajectories differ. Isolated.
- **Total** — `m3_bugfixed_s0` − `legacy_bugged_random` (the "current
  shipped → final product" path the thesis actually needs).

Identity that drives §4:

```
total Δ = seed_Δ + bugfix_Δ + ff_Δ + interaction
```

If the seed Δ dominates any property, that property's bug-fix contrast
needs more than one seed before it is interpretable — flagged in §7.

## 2. Three trained models

Models train on the three *seeded* property sets (the bonus `_random`
set is for the property-side seed contrast only; no model is trained on
it — there is no value in burning a GNN run on an
unreproducible-by-construction label set). Parallel W&B groups:

| Tag | Labels | W&B group | Source |
| --- | --- | --- | --- |
| `model_legacy_bugged_s0` | `legacy_bugged_s0` | `stage_5d_tier_c_legacy_bugged_s0` (new) | new runs, 6 seeds {0,1,4,5,6,8}, locked HPs |
| `model_legacy_bugfixed_s0` | `legacy_bugfixed_s0` | `stage_5d_tier_c_legacy_bugfixed_s0` (new) | new runs, same 6 seeds, locked HPs |
| `model_m3_bugfixed_s0` | `m3_bugfixed_s0` | `stage_5d_tier_c_m3_bugfixed_s0` (new) | new runs, same 6 seeds, locked HPs |

The "GNN-training seed" (`{0,1,4,5,6,8}`) and the "property-computation
seed" (`s0` in the directory name) are different RNGs and should not be
confused — the property seed fixes the label, the GNN seed fixes the
init.

Locked HPs unchanged from Tier C 5d: `lr=3e-5, wd=1e-3, h=128, l=2, e=200`.

The existing Tier C 5d runs in `stage_5d_tier_c_confirm` were trained
on `prop_legacy_bugged_random` and pre-date the `model_final.pt` change
(2026-05-16), so they have `test_artifacts.npz` but no reload-ready
checkpoint. They are **not** one of the three models above — they
correspond to a label set the three-way design doesn't train on (the
unseeded one). Two options for using them:

- **(a) Re-train on `legacy_bugged_s0`** with the new checkpoint plumbing.
  Clean three-way model comparison; the property-seed contrast can also
  be tested on the model side by comparing
  `model_legacy_bugged_s0` to the historical `stage_5d_tier_c_confirm`
  numbers (the same code-vs-code contrast on the model side as the seed
  contrast on the label side).
- **(b) Use `stage_5d_tier_c_confirm` as a fourth model** (`model_legacy_bugged_random`)
  via `test_artifacts.npz` only, skipping any analysis that needs live
  model reload.

Decision deferred to execution time. The plan assumes (a) — symmetric
with the label-side design.

## 3. Notebook structure

Section breakdown mirrors [compare_legacy_vs_new_m3.py](../scripts/notebooks/compare_legacy_vs_new_m3.py)
where possible (reuse loaders, scatter helpers, mover tables). New
content where the three-way design demands it.

### §0 — Paths and rename rules (cell-level widgets)

The M3 resimulation is submitted with
`submit_simulations.sh --rename-lipid DIPC=DLPC` (M3-Lipid-Parameters
renamed di-C18:2 PC), so the M3 output dirs contain `DLPC*` stems while
every legacy set is `DIPC*`. Without normalisation, the four-way
intersection silently drops every DIPC/DLPC composition. A
`mo.ui.text` widget for **lipid rename rules** (default `DLPC=DIPC`,
comma-separated `OLD=NEW`) is applied to every loaded stem; lipid-token
order is re-canonicalised alphabetically so
`POPC50_DLPC50` → `DIPC50_POPC50` matches the legacy stem.

Six `mo.ui.text` widgets:

- `legacy_bugged_random_dir` — default `results/properties/prop_legacy_bugged_random/`
- `legacy_bugged_s0_dir` — default `results/properties/prop_legacy_bugged_s0/`
- `legacy_bugfixed_s0_dir` — default `results/properties/prop_legacy_bugfixed_s0/`
- `m3_bugfixed_s0_dir` — default `results/properties/prop_m3_bugfixed_s0/`
- `legacy_runs_root` — default `data/membrane_only/` (for EDR access)
- `m3_runs_root` — default `data/membrane_only_m3_rerun/`
- `wandb_logs_root` — default `logs/training/`

Empty-state safe — `mo.stop` if a directory is absent, render skeleton.
The `_random` and `_s0` legacy dirs are independent; the notebook
should render the seed-only section if both are present, the bug-fix
section if `legacy_bugged_s0` and `legacy_bugfixed_s0` are both
present, etc.

### §1 — Coverage and provenance

- Three-way coverage table (rows = compositions, cols = label set, cell = ✓/✗).
- Number of fully-paired compositions (intersect on `canonical_name`).
- Computation timestamps per `.h5` (sanity that bug-fixed labels were
  produced from the post-cleanup `properties.py`).
- One-liner reminder of which bugs were fixed (link to cleanup plan §2),
  rendered as a collapsible cell.

### §2 — ITP and code provenance

- ITP SHA-1 diff between `data/membrane_only/<sys>/run/itp/` and
  `resources/martini3/itp/` (already exists in the pairwise notebook —
  reuse). Confirms `bugfixed_m3_traj` runs the new ITPs.
- Code provenance: print the `lipid_gnn.properties` module version
  (git hash via `subprocess` if reachable, else file mtime). Confirms
  `bugfixed_legacy` and `bugfixed_m3` came from the same code.

### §3 — Property-level comparison (the headline)

Per active property (7 properties; `bending_modulus` excluded — permanently
dropped, `properties.md`):

**3a. Seed-only sanity panel** (renders only when both `_random` and
`_s0` legacy dirs are present). For each property:

- mean ± SD of `legacy_bugged_s0 − legacy_bugged_random`
- `frac |Δ| > SD_legacy` against `SD_legacy_bugged_random` across the 70
  systems (the within-corpus dispersion of the shipped labels).
- A `[seed]` row in §3b's summary table whose magnitude sets the
  "actionable threshold" for the bug-fix contrast — i.e. a bug-fix
  effect smaller than the seed effect should be flagged "not
  distinguishable from RNG variance".

This is cheap and lives at the top of §3 because the rest of the
section's interpretation depends on it.

**3b. Paired summary table** (one row per property, one column per
contrast):

| Property | Δ_seed (mean ± SD) | Δ_bugfix (mean ± SD) | Δ_ff (mean ± SD) | Δ_total | t_seed | t_bugfix | t_ff | frac \|d\|>SD (seed) | frac \|d\|>SD (bugfix) | frac \|d\|>SD (ff) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

Three `t`-statistics test the three effects independently. `Δ_total`
is `m3_bugfixed_s0 − legacy_bugged_random` (the path that matters for
the thesis). "frac" columns use `SD_legacy_bugged_random` as the
reference dispersion across all three contrasts so the magnitudes are
directly comparable (mechanical retraining trigger when > 0.5 on
bug-fix or ff for any active property; the seed column is diagnostic
not actionable).

**3c. Per-property quadruple-scatter (4-panel row per property)**:

- Panel A — `legacy_bugged_s0` vs `legacy_bugged_random` (seed contrast).
- Panel B — `legacy_bugfixed_s0` vs `legacy_bugged_s0` (bug-fix
  contrast, same trajectories, same seed).
- Panel C — `m3_bugfixed_s0` vs `legacy_bugfixed_s0` (FF contrast, same
  pipeline, same seed).
- Panel D — `m3_bugfixed_s0` vs `legacy_bugged_random` (total, for
  reference).

All four on a shared axis range per property; identity line + per-point
coloring by composition family (POPC-rich, DPPC-rich, DOPC-rich,
CHOL-containing). Re-uses the scatter helper from the pairwise notebook.

**3d. Bland–Altman per contrast** — four panels per property (seed,
bug-fix, FF, total). Surfaces magnitude-dependent bias that pure
scatter hides. The seed panel anchors what a "no real effect" B–A
looks like for this estimator.

**3e. KDE overlay** — four densities per property
(`legacy_bugged_random`, `legacy_bugged_s0`, `legacy_bugfixed_s0`,
`m3_bugfixed_s0`) on one axis. The two `legacy_bugged_*` curves should
nearly coincide; visible separation = RNG-induced label drift across
the corpus.

**3f. Top-N movers** per property per contrast. Four tables of 5 rows
(one per contrast). Compositions that move most under seed are
candidates for "noisy estimator at this composition" — useful to
exclude or down-weight in any qualitative claim. Compositions that
move most under bug-fix are likely candidates for a bug-2/3/4 (per-residue
persistence; midplane-bending) signature. Compositions that move most
under FF are likely CHOL-containing (M2 8-bead → M3 9-bead change).

### §4 — Variance decomposition (new in this notebook)

For each property across the 70-system pool, partition

```text
Var(m3_bugfixed_s0 − legacy_bugged_random)
    = Var(seed_Δ) + Var(bugfix_Δ) + Var(ff_Δ)
    + 2·Cov(seed_Δ, bugfix_Δ) + 2·Cov(seed_Δ, ff_Δ) + 2·Cov(bugfix_Δ, ff_Δ)
```

where each `_Δ` is a 70-vector of per-composition differences. Reported
per property as a stacked bar (seed / bug-fix / FF / pairwise
covariances, sign-coded), normalised to the total Δ variance.
**The headline diagnostic of the notebook**: tells the reader at a
glance which of the three effects drove the regression-target shift,
and whether the seed-only floor is non-negligible.

Hypothesis going in:

- `persistence` — large bug-fix variance (bugs #2/#3 land directly here);
  small seed variance; small FF variance.
- `diffusivity` — moderate bug-fix variance (bug #9); the seed
  contrast partially captures bug #8 already, so the bug-fix bar here
  should be smaller than `persistence`. Small FF variance.
- `variation` (Voronoi CV) — moderate bug-fix variance (bug #10 PBC),
  small seed variance, small FF variance.
- `thickness_inhomogeneity` (renamed from `compressibility`) — small
  bug-fix variance (label rename only; the underlying observable hasn't
  changed); FF effect dominates if anything.
- `lipid_packing` / `thickness` — both should be FF-dominated (geometry
  changes with the ITP rev); seed and bug-fix bars near zero.

Numerical pre-test against the hypothesis is the science output of this
section; either confirmation or surprise is informative.

### §5 — Composition-space view (PCA + Δ overlay)

PCA on the 70 × 10 lipid-fraction matrix → first two principal axes
explain a known ~90 % of composition variance (reuse the pairwise
notebook's PCA). Three scatter panels per property:

- Color = sign(Δ_bugfix), size = \|Δ_bugfix\|.
- Color = sign(Δ_ff), size = \|Δ_ff\|.
- Color = composition family, size = \|Δ_total\|.

Tells us *where in composition space* each effect concentrates. A clean
result would be: bug-fix Δ uniform across the simplex; FF Δ concentrated
on CHOL-containing or DPPC-rich corners.

### §6 — Model-level comparison

Loaders for `test_artifacts.npz` from each W&B group, keyed on
`run_id`. Three primary models; optionally a fourth from
`stage_5d_tier_c_confirm` for the seed contrast on the model side
(decision documented in §2). Columns of comparison:

**6a. Per-property test MSE / pooled R² table**:

| Property | model_legacy_bugged_s0 | model_legacy_bugfixed_s0 | model_m3_bugfixed_s0 | (optional) stage_5d_tier_c_confirm |
| --- | --- | --- | --- | --- |

Side-by-side with 95 % bootstrap CI (reuse the Stage 5d CI helper).
**The primary thesis-grade output of the notebook.** Question answered:
do the labels with bugs fixed produce a *better-behaved* regression
target, a *worse-behaved* one, or the same? And does swapping the FF
shift the achievable error band?

**6b. Paired t-test across split-system membership**. Same train/test
split-system seeds across all models (verify by hashing the split
file). Paired comparisons:

- `model_legacy_bugfixed_s0` vs `model_legacy_bugged_s0` — bug-fix
  effect on the model side.
- `model_m3_bugfixed_s0` vs `model_legacy_bugfixed_s0` — FF effect.
- `model_m3_bugfixed_s0` vs `model_legacy_bugged_s0` — total.
- (optional) `model_legacy_bugged_s0` vs `stage_5d_tier_c_confirm` —
  property-seed effect on the model side (this is *not* the GNN-init
  seed; it tests whether unseeded vs seeded property labels lead to
  different achievable test MSE).

**6c. Per-system error scatter**. One panel per (model, property):
residual vs system index, colored by composition family. Surfaces
*where* the new labels help vs hurt. If the bug fixes are pure noise
removal, post-fix residuals should be a uniform-magnitude shrinkage; if
they unmask a structural problem, residuals will redistribute toward a
specific composition region.

**6d. Cross-model prediction agreement**. For each test point,
multiple predictions; compute pairwise mean absolute disagreement. If
the models *agree* on predictions but disagree on residuals, the
targets moved, not the function the GNN learned — a useful sanity
check.

### §7 — Headline callout and retraining verdict

`mo.callout` at the top of the notebook, regenerated reactively:

- Properties where the seed contrast alone moves labels by > 1 σ_legacy
  (= "noisy estimator at single-seed resolution; multi-seed
  property-side averaging recommended before drawing bug-fix conclusions").
- Properties where the bug fix changes labels by > 1 σ_legacy
  *above* the seed floor.
- Properties where the FF change changes labels by > 1 σ_legacy.
- Properties where pooled test R² moves by > 0.05 across the seeded
  models.
- Verdict, four-way: "seed noise dominates" / "bug-fix changes labels
  but not model performance" / "labels and model both moved —
  retraining required" / "force-field swap is the dominant change".

Mechanical trigger from the pairwise notebook (`|paired_t|>3` or
`frac_|d|>sd_legacy>0.5`) carries over per contrast.

## 4. Empty-state behaviour (the "draft now" question)

`mo.stop` after each path-widget cell, so the notebook **renders structure
even with zero data present**:

- Sections render as headings + prose.
- Tables render as empty 0-row DataFrames.
- Figures render as "data not yet available" placeholders (one-line
  `mo.md` instead of `_fig,`).
- The callout in §7 renders with all bullets as "pending".

This is the same pattern [compare_legacy_vs_new_m3.py](../scripts/notebooks/compare_legacy_vs_new_m3.py)
already uses for the M3-rerun side.

Draftable now (no new data needed beyond what's already on disk —
`prop_legacy_bugged_random/` exists, the other three are pending):

- §0 path widgets
- §1 coverage scaffold (the loader functions; missing dirs → 0 rows)
- §2 ITP SHA-1 diff (works the moment `m3_runs_root` is filled)
- §3 table + scatter scaffold (Δ-arrays empty until each seeded dir
  lands)
- §4 variance-decomposition function (testable on a synthetic
  four-array tuple — write a small mock test)
- §5 PCA cell (works against `legacy_bugged_random` immediately; the
  Δ overlay sub-cells empty-state until each seeded dir lands)
- §6 model loader scaffold (`test_artifacts.npz` only — agnostic to
  whether the new W&B groups exist yet; the existing
  `stage_5d_tier_c_confirm` artefacts can be wired up immediately for
  scaffolding)
- §7 callout

Risks of writing now:

- Axis ranges and KDE bandwidths are unknown — fix on first real run
  (cosmetic).
- The variance-decomposition partitioning needs a one-property smoke
  test against real numbers; safest to dry-run on synthetic arrays
  before the regeneration lands.
- If the bug-fix Δs all line up in one direction, the signed-Δ panels
  may need a redesign (e.g. log-magnitude); cheap to swap later.

## 5. Execution order

1. Write notebook skeleton with empty-state guards (this task; no
   new-data dependency).
2. Block 4a — execute property regeneration on legacy trajectories,
   both legacy and bugfixed pipelines at `seed=0` → fills
   `prop_legacy_bugged_s0/` and `prop_legacy_bugfixed_s0/`. The two
   share trajectory I/O so they should be driven by one HPC submission.
3. Block 4b — execute property regeneration on M3 trajectories, bugfixed
   pipeline at `seed=0` → fills `prop_m3_bugfixed_s0/`.
4. Block 4c — submit the three new training W&B groups
   (`stage_5d_tier_c_legacy_bugged_s0`,
   `stage_5d_tier_c_legacy_bugfixed_s0`,
   `stage_5d_tier_c_m3_bugfixed_s0`), 6 GNN seeds each, locked HPs.
5. Download W&B groups via `download_wandb_runs.py`; the notebook
   reacts as each group lands.
6. Adjust axis ranges, validate variance-decomposition signs, fill in
   §7 callout with final numbers, write up findings.

## 6. Out-of-scope

- Candidate new properties (`S_CC` etc., cleanup plan §3). v1 ships with
  the existing 7 active properties. If §3 of the cleanup plan adds any
  before this notebook executes, plumb them through; otherwise defer.
- Comparison of `legacy=True` vs `legacy=False` on the *same* property
  pipeline (i.e. testing the runtime switch within `properties.py`).
  Distinct task — sanity-checks the legacy-mode shim, not the
  end-to-end pipeline.
- EFA / protein-extension implications. Mention in §7 narrative only if
  `compressibility` (renamed `thickness_inhomogeneity`) moves materially
  under either contrast.
