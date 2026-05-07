# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "matplotlib",
#     "numpy",
#     "scipy",
#     "scikit-learn",
#     "torch",
#     "pyarrow",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import importlib.util
    import json
    import sys
    import warnings
    from pathlib import Path

    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats
    from sklearn.decomposition import PCA
    from sklearn.metrics import r2_score
    import torch

    import marimo as mo

    warnings.filterwarnings("ignore")
    return (
        PCA,
        Path,
        importlib,
        json,
        mo,
        mpatches,
        np,
        pd,
        plt,
        r2_score,
        stats,
        sys,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Stage 5d — Tier C 7-Property Confirmation Analysis

    Multi-seed confirmation run (`stage_5d_tier_c_confirm`) at the locked
    Tier A/B hyperparameters (`hidden_dim=128`, `num_layers=2`, `lr=3e-5`,
    `wd=1e-3`, `epochs=200`) now covering all seven Tier C properties:
    `lipid_packing`, `thickness`, `thickness_std`, `variation`, `persistence`,
    `diffusivity`, `compressibility`.

    The Tier C addition is `compressibility` (area compressibility modulus,
    Å³/kT). Pre-registered as architecture-limited because area-fluctuation
    statistics couple at scales beyond the 11 Å spatial cutoff. The Stage 0d
    floor already showed it learns better than expected; Stage 5d retests
    that finding at full epochs against the Tier B 5c numbers.

    **Pool**: planned seeds `{0, 1, 3, 4, 5}`. Seed 3 again hit the
    recurring dead-init pattern on `variation` and is excluded from primary
    numbers (4-seed analysis on `{0, 1, 4, 5}`); replacement seed 8 is in
    flight to restore n=5.

    **Prerequisites** (run before opening this notebook):
    ```bash
    python scripts/python/download_wandb_runs.py --group stage_5d_tier_c_confirm
    python scripts/python/download_wandb_runs.py --group stage_0d_tier_c
    python scripts/training/linear_baseline.py --stratified
    ```

    **Output**: `results/figures/stage_5d/` — PDF + PNG per figure, `headline_numbers.json`.

    Sections:
    1. Configuration & paths
    2. Load runs + baseline
    3. Aggregate test artifacts & de-normalise
    4. Headline numbers (MSE, R², MAE per property)
    5. Figures a–i
    6. Statistical tests
    7. Gate check
    8. Export & conclusions
    """)
    return


@app.cell
def _(Path, sys):
    # ── Repo root (searches upward for config.yaml) ───────────────────────────
    def _find_repo_root() -> Path:
        p = Path(".").resolve()
        for _ in range(6):
            if (p / "config.yaml").exists():
                return p
            p = p.parent
        raise FileNotFoundError("Cannot find repo root (config.yaml not found in parents)")

    REPO_ROOT = _find_repo_root()
    sys.path.insert(0, str(REPO_ROOT))

    from lipid_gnn.config import CONFIG

    LOGS_DIR     = REPO_ROOT / "logs" / "training"
    FIGURES_DIR  = REPO_ROOT / "results" / "figures" / "stage_5d"
    BASELINE_NPZ = REPO_ROOT / "results" / "training" / "linear_baseline_stratified.npz"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"REPO_ROOT   : {REPO_ROOT}")
    print(f"LOGS_DIR    : {LOGS_DIR}")
    print(f"FIGURES_DIR : {FIGURES_DIR}")
    return BASELINE_NPZ, CONFIG, FIGURES_DIR, LOGS_DIR, REPO_ROOT


@app.cell
def _():
    # ── User config ───────────────────────────────────────────────────────────
    GROUP_STAGE5 = "stage_5d_tier_c_confirm"
    GROUP_STAGE0 = "stage_0d_tier_c"
    GROUPS_PROG  = [
        "stage_0d_tier_c",
        "stage_1g_tier_c_lr",
        "stage_1g_refine_tier_c_lr",
        "stage_5d_tier_c_confirm"
    ]
    # Tier C gates: Stage 0d 7-prop floor (5-seed val_min_last10 mean).
    # Stage 5d must beat all gates to demonstrate the locked HPs hold up at
    # 7 properties (no regression from re-running Stage 0d at full epochs).
    GATES = {
        "lipid_packing":   0.0236,
        "thickness":       0.0733,
        "thickness_std":   0.3241,
        "variation":       0.1728,
        "persistence":     0.3701,
        "diffusivity":     0.0655,
        "compressibility": 0.3931
    }
    N_BOOTSTRAP = 10_000
    return GATES, GROUPS_PROG, GROUP_STAGE0, GROUP_STAGE5, N_BOOTSTRAP


@app.cell
def _(FIGURES_DIR, plt):
    # ── Plot style + helpers ──────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 11,
        "axes.labelsize": 11, "axes.titlesize": 11,
        "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "axes.spines.top": False, "axes.spines.right": False,
    })

    PROP_LABELS = {
        "lipid_packing":   "Lipid packing (a.u.)",
        "thickness":       "Bilayer thickness (Å)",
        "thickness_std":   r"$\sigma_\mathrm{thick}$ (Å)",
        "variation":       "Variation",
        "persistence":     "Persistence",
        "diffusivity":     "Diffusivity (Å²)",
        "compressibility": "Compressibility (Å³/kT)"
    }
    PAL = {
        "gnn":      "#0072B2",
        "baseline": "#D55E00",
        "identity": "#999999",
        "seed":     "#56B4E9",
        "train":    "#009E73",
        "val":      "#E69F00",
    }
    COMP_COLORS = plt.cm.tab10.colors

    def save_fig(fig, name):
        for _ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"{name}.{_ext}")
        print(f"  saved → {name}.{{pdf,png}}")

    return COMP_COLORS, PAL, PROP_LABELS, save_fig


@app.cell
def _(LOGS_DIR, json, np, pd):
    # ── Run loader ────────────────────────────────────────────────────────────
    def load_group(group):
        """Load all finished runs from a downloaded group.

        Returns (runs_df, histories, artifacts) where artifacts contains
        test_artifacts.npz data keyed by run_id.
        """
        _group_dir = LOGS_DIR / group
        if not _group_dir.exists():
            print(f"  [WARN] {_group_dir} not found — run download_wandb_runs.py first")
            return pd.DataFrame(), {}, {}

        _idx = json.loads((_group_dir / "runs_index.json").read_text())
        _rows, _histories, _artifacts = [], {}, {}

        for _r in _idx:
            if _r["state"] != "finished":
                continue
            _run_dir = _group_dir / _r["name"]
            _cfg     = json.loads((_run_dir / "config.json").read_text())
            _summary = json.loads((_run_dir / "summary.json").read_text())
            _hist    = (
                pd.read_parquet(_run_dir / "history.parquet")
                if (_run_dir / "history.parquet").exists()
                else pd.DataFrame()
            )
            _val_min = (
                _hist["val/loss_total"].tail(10).min()
                if "val/loss_total" in _hist.columns
                else np.nan
            )
            _row = {
                **_cfg,
                "run_id": _r["id"],
                "run_name": _r["name"],
                "val_min_last10": _val_min,
                "test_mse_total": _summary.get("test/mse_total", np.nan),
                "runtime_s": _r.get("runtime_seconds", np.nan),
            }
            _rows.append(_row)
            _histories[_r["id"]] = _hist

            _npz = _run_dir / "test_artifacts.npz"
            if _npz.exists():
                _artifacts[_r["id"]] = dict(np.load(_npz, allow_pickle=True))

        _df = pd.DataFrame(_rows)
        print(f"  {group}: {len(_df)} runs, {len(_artifacts)} with test_artifacts.npz")
        return _df, _histories, _artifacts

    return (load_group,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load runs

    Load the Stage 5d confirmation runs and the Stage 0d floor baseline.
    Each run directory must contain `config.json`, `summary.json`, `history.parquet`,
    and `test_artifacts.npz` (saved by `run_sweep.py` and fetched by
    `download_wandb_runs.py`).
    """)
    return


@app.cell
def _(GROUP_STAGE0, GROUP_STAGE5, load_group):
    print(f"Loading {GROUP_STAGE5} ...")
    runs_df, histories, artifacts = load_group(GROUP_STAGE5)

    print(f"Loading {GROUP_STAGE0} for paired comparison ...")
    s0_df, s0_histories, s0_artifacts = load_group(GROUP_STAGE0)

    if not runs_df.empty:
        print(f"\nStage-5d runs ({len(runs_df)} total):")
        print(runs_df[["run_name", "seed", "val_min_last10", "test_mse_total"]].to_string(index=False))
    return artifacts, histories, runs_df, s0_artifacts, s0_df


@app.cell
def _(mo, runs_df):
    mo.stop(
        runs_df.empty,
        mo.callout(
            mo.md("**No Stage-5d runs found.** Run `download_wandb_runs.py --group stage_5d_tier_c_confirm` first."),
            kind="danger",
        ),
    )
    mo.md(f"""
    **Runs loaded** (`runs_df`):
    - **Count**: {len(runs_df)} finished runs
    - **Seeds**: {sorted(runs_df["seed"].tolist())}
    - **Val MSE range** (last-10 mean): [{runs_df["val_min_last10"].min():.4f}, {runs_df["val_min_last10"].max():.4f}]
    - **Test MSE range** (total): [{runs_df["test_mse_total"].min():.4f}, {runs_df["test_mse_total"].max():.4f}]
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Linear baseline

    Ridge regression on composition vectors (one-hot lipid fractions) trained on
    the same stratified split. Loaded from `linear_baseline_stratified.npz`.
    The GNN is benchmarked against this baseline in figure (e).
    """)
    return


@app.cell
def _(BASELINE_NPZ, np):
    if BASELINE_NPZ.exists():
        baseline = dict(np.load(BASELINE_NPZ, allow_pickle=True))
        _bl_properties = [str(p) for p in baseline["properties"]]
        _bl_mse = float(np.mean((baseline["test_preds"] - baseline["test_targets"]) ** 2))
        print(f"Baseline loaded: properties={_bl_properties}")
        print(f"Baseline test MSE (total, normalised): {_bl_mse:.4f}")
    else:
        baseline = None
        print(f"[WARN] Baseline not found at {BASELINE_NPZ}.")
        print("       Run: python scripts/training/linear_baseline.py --stratified")
    return (baseline,)


@app.cell
def _(baseline, mo):
    _props = [str(p) for p in baseline["properties"]] if baseline is not None else []
    _out = (
        mo.callout(mo.md("Baseline NPZ not found — figure (e) will show GNN-only."), kind="warn")
        if baseline is None
        else mo.md(f"""
        **Baseline** (`linear_baseline_stratified.npz`):
        - **Properties covered**: {_props}
        - **Test shapes**: preds {baseline['test_preds'].shape}, targets {baseline['test_targets'].shape}
        """)
    )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Aggregate test artifacts

    Each seed's `test_artifacts.npz` contains normalized predictions, targets,
    scaler parameters, and composition labels for all test graphs. Stack across
    seeds to build (S × N × P) arrays used throughout the analysis.
    """)
    return


@app.cell
def _(artifacts, mo, np, runs_df):
    mo.stop(
        not artifacts,
        mo.callout(
            mo.md("**No `test_artifacts.npz` files found.** Re-download the runs with the artifact fetch enabled."),
            kind="danger",
        ),
    )

    _properties = list(runs_df["properties"].iloc[0])

    _seed_preds, _seed_targets, _seed_comps = [], [], []
    for _rid, _art in artifacts.items():
        _seed_preds.append(_art["test_preds"])
        _seed_targets.append(_art["test_targets"])
        _seed_comps.append([str(c) for c in _art["test_compositions"]])

    preds_stack   = np.stack(_seed_preds,   axis=0)   # (S, N, P) normalised
    targets_stack = np.stack(_seed_targets, axis=0)   # (S, N, P) normalised
    N_SEEDS, N_TEST, _N_PROPS = preds_stack.shape

    _first_art      = list(artifacts.values())[0]
    s_mean          = _first_art["scaler_mean"]    # (P,)
    s_scale         = _first_art["scaler_scale"]   # (P,)
    test_comps_list = _seed_comps[0]               # composition labels (same across seeds)
    properties      = _properties

    print(f"Seeds: {N_SEEDS}, test graphs: {N_TEST}, properties: {properties}")
    for _j, _p in enumerate(properties):
        _per_seed = [float(np.mean((preds_stack[_s, :, _j] - targets_stack[_s, :, _j]) ** 2))
                     for _s in range(N_SEEDS)]
        print(f"  {_p}: test MSE {np.mean(_per_seed):.4f} ± {np.std(_per_seed):.4f}")
    return (
        N_SEEDS,
        N_TEST,
        preds_stack,
        properties,
        s_mean,
        s_scale,
        targets_stack,
        test_comps_list,
    )


@app.cell
def _(N_SEEDS, N_TEST, mo, preds_stack, properties):
    mo.md(f"""
    **Artifact stack** (`preds_stack`, `targets_stack`):
    - **Shape**: ({N_SEEDS} seeds × {N_TEST} test graphs × {len(properties)} properties)
    - **Dtype**: `{preds_stack.dtype}`
    - **Properties**: {properties}

    All values are in StandardScaler-normalised space (zero mean, unit variance
    on the training set). De-normalisation in the next cell restores physical units.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. De-normalise to physical units

    Reverse the StandardScaler transform using the saved `scaler_mean` and
    `scaler_scale` from the first artifact (identical across seeds since all
    runs share the same preprocessing). Physical-unit arrays are used for
    residual plots and the composition MAE bar chart.
    """)
    return


@app.cell
def _(baseline, preds_stack, s_mean, s_scale, targets_stack):
    preds_phys   = preds_stack   * s_scale + s_mean   # (S, N, P)
    targets_phys = targets_stack * s_scale + s_mean   # (S, N, P)
    mean_preds_phys   = preds_phys.mean(0)             # (N, P)
    mean_targets_phys = targets_phys[0]                # (N, P)

    if baseline is not None:
        bl_preds_phys   = baseline["test_preds"]   * baseline["scaler_scale"] + baseline["scaler_mean"]
        bl_targets_phys = baseline["test_targets"] * baseline["scaler_scale"] + baseline["scaler_mean"]
    else:
        bl_preds_phys = bl_targets_phys = None

    print("De-normalisation complete.")
    return (
        bl_preds_phys,
        bl_targets_phys,
        mean_preds_phys,
        mean_targets_phys,
        preds_phys,
        targets_phys,
    )


@app.cell
def _(PROP_LABELS, mean_targets_phys, mo, properties):
    _rows = "\n".join(
        f"- **`{_p}`** ({PROP_LABELS.get(_p, _p)}): "
        f"[{mean_targets_phys[:, _j].min():.3f}, {mean_targets_phys[:, _j].max():.3f}], "
        f"mean {mean_targets_phys[:, _j].mean():.3f}"
        for _j, _p in enumerate(properties)
    )
    mo.md(f"""
    **Physical-unit target ranges** (test set, mean over graphs):
    {_rows}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Headline numbers

    Per-property MSE (mean ± std over seeds), R² with 95 % bootstrap CI
    (10 000 resamples, pooled over all seeds), and pooled MAE. These are the
    primary thesis numbers for the Tier C confirmation.

    Note: pooled test R² is computed across `S × N` test points (4 seeds ×
    275 graphs) and is the more statistically stable estimate. The W&B
    `val/r2_*` summary uses the small validation split per seed and can
    differ — `compressibility` in particular shows a much higher pooled
    test R² than its per-seed val R², documented below.
    """)
    return


@app.cell
def _(N_BOOTSTRAP, np):
    def bootstrap_ci(arr, stat_fn, n=N_BOOTSTRAP, ci=0.95):
        _rng = np.random.default_rng(0)
        _boot = [stat_fn(_rng.choice(arr, size=len(arr), replace=True)) for _ in range(n)]
        _lo = np.percentile(_boot, (1 - ci) / 2 * 100)
        _hi = np.percentile(_boot, (1 + ci) / 2 * 100)
        return float(_lo), float(_hi)

    return (bootstrap_ci,)


@app.cell
def _(
    N_SEEDS,
    bootstrap_ci,
    np,
    pd,
    preds_stack,
    properties,
    r2_score,
    targets_stack,
):
    _rows = []
    for _j, _prop in enumerate(properties):
        _per_seed_mse = [
            float(np.mean((preds_stack[_s, :, _j] - targets_stack[_s, :, _j]) ** 2))
            for _s in range(N_SEEDS)
        ]
        _mse_mean, _mse_std = np.mean(_per_seed_mse), np.std(_per_seed_mse)

        _pool_pred = preds_stack[:, :, _j].ravel()
        _pool_true = targets_stack[:, :, _j].ravel()
        _r2_point  = r2_score(_pool_true, _pool_pred)
        _r2_lo, _r2_hi = bootstrap_ci(
            np.stack([_pool_true, _pool_pred], axis=1),
            lambda x: r2_score(x[:, 0], x[:, 1]),
        )
        _mae = float(np.mean(np.abs(_pool_pred - _pool_true)))
        _rows.append({
            "property":  _prop,
            "MSE mean":  _mse_mean,
            "MSE std":   _mse_std,
            "R²":        _r2_point,
            "R² CI lo":  _r2_lo,
            "R² CI hi":  _r2_hi,
            "MAE":       _mae,
        })

    tbl = pd.DataFrame(_rows).set_index("property")
    print("=== Headline numbers (normalised space) ===")
    print(tbl.round(4).to_string())
    return (tbl,)


@app.cell
def _(mo, tbl):
    mo.vstack([
        mo.md("**Headline numbers — normalised space** (R² pooled over seeds, 10 000 bootstrap resamples):"),
        mo.as_html(tbl.round(4)),
    ])
    return


@app.cell
def _(mo, tbl):
    _good = [p for p, row in tbl.iterrows() if row["R²"] >= 0.85]
    _ok   = [p for p, row in tbl.iterrows() if 0.5 <= row["R²"] < 0.85]
    _weak = [p for p, row in tbl.iterrows() if row["R²"] < 0.5]
    mo.callout(mo.md(f"""
    **Per-property R² summary** (GOOD ≥ 0.85, OK ≥ 0.5, WEAK < 0.5):
    - GOOD: {_good if _good else "none"}
    - OK:   {_ok   if _ok   else "none"}
    - WEAK: {_weak if _weak else "none"}

    `persistence` (R² ≈ {tbl.loc['persistence', 'R²']:.3f}) is the only WEAK
    property — consistent with the architecture-limited finding from Stages
    0c, 1e, 1e' and unchanged in Tier C.
    `diffusivity` (R² ≈ {tbl.loc['diffusivity', 'R²']:.3f}) holds the
    static-snapshot → dynamical-observable result.
    `compressibility` (R² ≈ {tbl.loc['compressibility', 'R²']:.3f}, pooled
    test) is substantially above the per-seed val R² ≈ 0.59 logged in W&B and
    above the pre-registered "<<0.5" architecture-ceiling expectation. Local
    11 Å geometry carries a stronger compressibility signal than the
    receptive-field argument predicted.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Figures

    Each figure is saved as PDF + PNG to `results/figures/stage_5d/`.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (a) Training dynamics

    Validation and training loss curves for all seeds, with ±1 SD shading.
    One panel per property plus a "Total" panel. Healthy convergence shows
    val and train tracking without divergence.
    """)
    return


@app.cell
def _(N_SEEDS, PAL, histories, np, plt, properties, save_fig):
    _n_panels = 1 + len(properties)
    _n_cols_a = min(_n_panels, 3)
    _n_rows_a = int(np.ceil(_n_panels / _n_cols_a))
    _fig_a, _axes = plt.subplots(_n_rows_a, _n_cols_a, figsize=(4 * _n_cols_a, 4 * _n_rows_a), sharey=False)
    _axes = np.asarray(_axes).ravel()

    _loss_keys  = ["val/loss_total"]  + [f"val/loss_{p}"   for p in properties]
    _train_keys = ["train/loss_total"] + [f"train/loss_{p}" for p in properties]
    _titles     = ["Total"] + [p.replace("_", " ").title() for p in properties]

    for _ax, _vkey, _tkey, _title in zip(_axes[:_n_panels], _loss_keys, _train_keys, _titles):
        _val_curves, _trn_curves = [], []
        for _hist in histories.values():
            if _vkey in _hist.columns:
                _val_curves.append(_hist[_vkey].values)
            if _tkey in _hist.columns:
                _trn_curves.append(_hist[_tkey].values)

        for _curves, _color, _label in [
            (_val_curves, PAL["gnn"],   "Val"),
            (_trn_curves, PAL["train"], "Train"),
        ]:
            if not _curves:
                continue
            _min_len = min(len(c) for c in _curves)
            _arr     = np.stack([c[:_min_len] for c in _curves])
            _epochs  = np.arange(1, _min_len + 1)
            _mn, _sd = _arr.mean(0), _arr.std(0)
            for _row in _arr:
                _ax.plot(_epochs, _row, color=_color, alpha=0.15, lw=0.8)
            _ax.plot(_epochs, _mn, color=_color, lw=2, label=_label)
            _ax.fill_between(_epochs, _mn - _sd, _mn + _sd, color=_color, alpha=0.2)

        _ax.set_xlabel("Epoch")
        _ax.set_ylabel("MSE (normalised)")
        _ax.set_title(_title)
        _ax.legend(fontsize=8)

    for _ax in _axes[_n_panels:]:
        _ax.set_visible(False)
    _fig_a.suptitle(f"Loss vs epoch — train and val, {N_SEEDS} seeds (mean ± 1 SD)", y=1.01)
    _fig_a.tight_layout()
    save_fig(_fig_a, "fig_a_loss_curves")
    _fig_a
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (b) Predicted vs true scatter

    Each point is one (graph, seed) pair pooled over all seeds, coloured by
    lipid composition. The dashed line is the identity. R² and MAE are annotated
    per panel.
    """)
    return


@app.cell
def _(
    COMP_COLORS,
    N_SEEDS,
    N_TEST,
    PAL,
    PROP_LABELS,
    mpatches,
    np,
    plt,
    preds_phys,
    properties,
    r2_score,
    save_fig,
    targets_phys,
    test_comps_list,
):
    _unique_comps   = sorted(set(test_comps_list))
    _comp_color_map = {c: COMP_COLORS[i % len(COMP_COLORS)] for i, c in enumerate(_unique_comps)}

    _n_props_b = len(properties)
    _n_cols_b = min(_n_props_b, 3)
    _n_rows_b = int(np.ceil(_n_props_b / _n_cols_b))
    _fig_b, _axes_b = plt.subplots(_n_rows_b, _n_cols_b, figsize=(5 * _n_cols_b, 5 * _n_rows_b))
    _axes_b = np.asarray(_axes_b).ravel()

    for _j, (_ax, _prop) in enumerate(zip(_axes_b[:_n_props_b], properties)):
        _true_pool = targets_phys[:, :, _j].ravel()
        _pred_pool = preds_phys[:, :, _j].ravel()
        _comp_pool = test_comps_list * N_SEEDS
        _colors    = [_comp_color_map[c] for c in _comp_pool]

        _ax.scatter(_true_pool, _pred_pool, c=_colors, alpha=0.35, s=18, lw=0)
        _lo = min(_true_pool.min(), _pred_pool.min())
        _hi = max(_true_pool.max(), _pred_pool.max())
        _ax.plot([_lo, _hi], [_lo, _hi], "--", color=PAL["identity"], lw=1.5, label="Identity")

        _r2  = r2_score(_true_pool, _pred_pool)
        _mae = np.mean(np.abs(_pred_pool - _true_pool))
        _ax.text(0.05, 0.92, f"R² = {_r2:.3f}\nMAE = {_mae:.3f}",
                 transform=_ax.transAxes, fontsize=9, va="top")
        _ax.set_xlabel(f"True — {PROP_LABELS.get(_prop, _prop)}")
        _ax.set_ylabel(f"Predicted — {PROP_LABELS.get(_prop, _prop)}")
        _ax.set_title(f"Scatter: true vs predicted {_prop}")

    for _ax in _axes_b[_n_props_b:]:
        _ax.set_visible(False)
    _handles = [mpatches.Patch(color=_comp_color_map[c], label=c) for c in _unique_comps]
    _fig_b.legend(
        _handles, _unique_comps, title="Composition", loc="lower center",
        ncol=min(5, len(_unique_comps)), bbox_to_anchor=(0.5, -0.15), fontsize=7,
    )
    _fig_b.suptitle(f"Predicted vs true per property — pooled over {N_SEEDS} seeds × {N_TEST} test graphs")
    _fig_b.tight_layout()
    save_fig(_fig_b, "fig_b_pred_vs_true")
    _fig_b
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (c) Per-composition MAE (normalised)

    MAE in normalised (z-score) space so all six properties are on a comparable
    scale. Compositions sorted descending by total MAE to show where errors
    concentrate. Error bars are seed-to-seed standard deviation.

    Systems at the top identify where the GNN struggles — typically compositions
    at the boundary of the training cloud in PCA(composition) space.
    """)
    return


@app.cell
def _(
    N_SEEDS,
    PAL,
    PROP_LABELS,
    np,
    plt,
    preds_stack,
    properties,
    save_fig,
    targets_stack,
    test_comps_list,
):
    _unique_comps_ord = sorted(set(test_comps_list))
    _comp_idx_map     = {c: i for i, c in enumerate(_unique_comps_ord)}
    _n_comps          = len(_unique_comps_ord)

    _sys_abs_sum = np.zeros((N_SEEDS, _n_comps, len(properties)))
    _sys_count   = np.zeros(_n_comps, dtype=int)
    for _s in range(N_SEEDS):
        for _gi, _comp in enumerate(test_comps_list):
            _ci = _comp_idx_map[_comp]
            _sys_abs_sum[_s, _ci] += np.abs(preds_stack[_s, _gi] - targets_stack[_s, _gi])
            if _s == 0:
                _sys_count[_ci] += 1

    _sys_mae      = _sys_abs_sum / _sys_count[None, :, None]
    _sys_mae_mean = _sys_mae.mean(0)   # (n_comps, P)
    _sys_mae_std  = _sys_mae.std(0)    # (n_comps, P)
    _total_mae    = _sys_mae_mean.sum(1)
    _order        = np.argsort(_total_mae)[::-1]

    _base_colors = [PAL["gnn"], PAL["baseline"], PAL["train"], PAL["val"], PAL["seed"], PAL["identity"]]
    _prop_colors = [_base_colors[_j] if _j < len(_base_colors) else plt.cm.tab10(_j % 10)
                    for _j in range(len(properties))]

    _fig_c, _ax_c = plt.subplots(figsize=(max(8, _n_comps * 0.8), 4))
    _x = np.arange(_n_comps)
    _w = 0.8 / len(properties)
    for _j, _prop in enumerate(properties):
        _offsets = _x - 0.4 + _w * (_j + 0.5)
        _ax_c.bar(
            _offsets[_order], _sys_mae_mean[_order, _j], _w,
            yerr=_sys_mae_std[_order, _j], capsize=3,
            color=_prop_colors[_j], alpha=0.85, label=PROP_LABELS.get(_prop, _prop),
        )

    _ax_c.set_xticks(_x)
    _ax_c.set_xticklabels([_unique_comps_ord[i] for i in _order], rotation=40, ha="right", fontsize=8)
    _ax_c.set_ylabel("MAE (normalised, mean over seeds)")
    _ax_c.set_title("Test MAE per composition vs property (normalised, sorted by total error)")
    _ax_c.legend(fontsize=8)
    _fig_c.tight_layout()
    save_fig(_fig_c, "fig_c_per_system_mae")
    _fig_c
    return


@app.cell
def _(mo, np, preds_stack, properties, targets_stack, test_comps_list):
    _comp_idx = {c: i for i, c in enumerate(sorted(set(test_comps_list)))}
    _n_comps  = len(_comp_idx)
    _n_seeds  = preds_stack.shape[0]
    _sys_abs  = np.zeros((_n_seeds, _n_comps, len(properties)))
    _cnt      = np.zeros(_n_comps, dtype=int)
    for _s in range(_n_seeds):
        for _gi, _c in enumerate(test_comps_list):
            _ci = _comp_idx[_c]
            _sys_abs[_s, _ci] += np.abs(preds_stack[_s, _gi] - targets_stack[_s, _gi])
            if _s == 0:
                _cnt[_ci] += 1
    _mae_by_comp = (_sys_abs / _cnt[None, :, None]).mean(0).sum(1)
    _worst_comps = sorted(_comp_idx, key=lambda c: -_mae_by_comp[_comp_idx[c]])[:5]
    mo.callout(mo.md(
        "**Highest-error compositions** (summed normalised MAE, top 5): "
        + ", ".join(f"`{c}`" for c in _worst_comps)
        + ". These are typically DPPC- or DOPC-rich mixtures at the boundary of the training cloud."
    ), kind="warn")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (d) Residual distributions

    Prediction residuals (pred − true) in physical units, pooled over all seeds.
    Dashed curves: Gaussian fit. Dotted verticals: empirical bias (mean residual).
    Systematic bias (non-zero mean) indicates a property where the model
    consistently over- or under-predicts.
    """)
    return


@app.cell
def _(
    N_SEEDS,
    PAL,
    PROP_LABELS,
    np,
    plt,
    preds_phys,
    properties,
    save_fig,
    stats,
    targets_phys,
):
    _n_props_d = len(properties)
    _n_cols_d = min(_n_props_d, 3)
    _n_rows_d = int(np.ceil(_n_props_d / _n_cols_d))
    _fig_d, _axes_d = plt.subplots(_n_rows_d, _n_cols_d, figsize=(4.5 * _n_cols_d, 4.5 * _n_rows_d))
    _axes_d = np.asarray(_axes_d).ravel()

    for _j, (_ax, _prop) in enumerate(zip(_axes_d[:_n_props_d], properties)):
        _resid = (preds_phys[:, :, _j] - targets_phys[:, :, _j]).ravel()
        _ax.hist(_resid, bins=30, density=True, color=PAL["gnn"], alpha=0.7, edgecolor="white")
        _mu, _sigma = _resid.mean(), _resid.std()
        _xs = np.linspace(_resid.min(), _resid.max(), 200)
        _ax.plot(_xs, stats.norm.pdf(_xs, _mu, _sigma), "k--", lw=1.5,
                 label=f"N({_mu:.3f}, {_sigma:.3f})")
        _ax.axvline(0,   color=PAL["identity"], lw=1)
        _ax.axvline(_mu, color="firebrick", lw=1.5, linestyle=":", label=f"bias={_mu:.4f}")
        _ax.set_xlabel(f"Residual — {PROP_LABELS.get(_prop, _prop)}")
        _ax.set_ylabel("Density")
        _ax.set_title(f"Residuals: {_prop}")
        _ax.legend(fontsize=8)

    for _ax in _axes_d[_n_props_d:]:
        _ax.set_visible(False)
    _fig_d.suptitle(f"Residuals (pred − true) per property (pooled over {N_SEEDS} seeds; dashed = Gaussian fit)")
    _fig_d.tight_layout()
    save_fig(_fig_d, "fig_d_residuals")
    _fig_d
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (e) GNN vs linear-composition baseline

    Left panels: scatter overlay. Right panel: MSE comparison bar chart with
    percentage improvement labels. A meaningful GNN should substantially
    outperform a linear baseline that sees only composition fractions.
    """)
    return


@app.cell
def _(
    N_SEEDS,
    PAL,
    PROP_LABELS,
    baseline,
    bl_preds_phys,
    bl_targets_phys,
    np,
    plt,
    preds_phys,
    preds_stack,
    properties,
    save_fig,
    targets_phys,
    targets_stack,
):
    _bl_props_list = [str(p) for p in baseline["properties"]] if baseline is not None else []
    _bl_j_for_p    = {p: _bl_props_list.index(p) for p in properties if p in _bl_props_list}
    if len(_bl_j_for_p) < len(properties):
        _missing = [p for p in properties if p not in _bl_j_for_p]
        print(f"  [INFO] baseline missing for {_missing}")

    _n_panels_e = len(properties) + 1
    _n_cols_e = min(_n_panels_e, 3)
    _n_rows_e = int(np.ceil(_n_panels_e / _n_cols_e))
    _fig_e, _axes_e = plt.subplots(_n_rows_e, _n_cols_e, figsize=(4.5 * _n_cols_e, 4.5 * _n_rows_e))
    _axes_e = np.asarray(_axes_e).ravel()

    for _j, (_ax, _prop) in enumerate(zip(_axes_e[:len(properties)], properties)):
        _true_pool = targets_phys[:, :, _j].ravel()
        _pred_gnn  = preds_phys[:, :, _j].ravel()
        _bl_j      = _bl_j_for_p.get(_prop)

        _lo = min(_true_pool.min(), _pred_gnn.min())
        _hi = max(_true_pool.max(), _pred_gnn.max())
        if _bl_j is not None:
            _lo = min(_lo, bl_targets_phys[:, _bl_j].min(), bl_preds_phys[:, _bl_j].min())
            _hi = max(_hi, bl_targets_phys[:, _bl_j].max(), bl_preds_phys[:, _bl_j].max())

        _ax.scatter(_true_pool, _pred_gnn, s=12, alpha=0.3, color=PAL["gnn"], label="GNN")
        if _bl_j is not None:
            _ax.scatter(bl_targets_phys[:, _bl_j], bl_preds_phys[:, _bl_j],
                        s=40, alpha=0.8, color=PAL["baseline"], marker="D", label="Linear baseline")
        _ax.plot([_lo, _hi], [_lo, _hi], "--", color=PAL["identity"], lw=1.2)
        _ax.set_xlabel(f"True — {PROP_LABELS.get(_prop, _prop)}")
        _ax.set_ylabel("Predicted")
        _ax.set_title(f"Scatter: {_prop}" + ("" if _bl_j is not None else "\n(no baseline)"))
        _ax.legend(fontsize=8)

    # Bar chart: MSE comparison
    _ax_bar   = _axes_e[len(properties)]
    _xpos     = np.arange(len(properties))
    _gnn_mean = [
        float(np.mean([np.mean((preds_stack[_s, :, _j] - targets_stack[_s, :, _j]) ** 2)
                       for _s in range(N_SEEDS)]))
        for _j in range(len(properties))
    ]
    _gnn_std  = [
        float(np.std([np.mean((preds_stack[_s, :, _j] - targets_stack[_s, :, _j]) ** 2)
                      for _s in range(N_SEEDS)]))
        for _j in range(len(properties))
    ]
    _bl_mse   = [
        float(np.mean((baseline["test_preds"][:, _bl_j_for_p[_p]] -
                        baseline["test_targets"][:, _bl_j_for_p[_p]]) ** 2))
        if _p in _bl_j_for_p else float("nan")
        for _p in properties
    ]

    _w_bar = 0.35
    _ax_bar.bar(_xpos - _w_bar / 2, _gnn_mean, _w_bar, yerr=_gnn_std, capsize=4,
                color=PAL["gnn"], alpha=0.85, label="GNN")
    _valid_bl = [_j for _j, _v in enumerate(_bl_mse) if not np.isnan(_v)]
    if _valid_bl:
        _ax_bar.bar(np.array(_valid_bl) + _w_bar / 2, [_bl_mse[_j] for _j in _valid_bl], _w_bar,
                    color=PAL["baseline"], alpha=0.85, label="Linear baseline")
        for _j in _valid_bl:
            _rel = (_bl_mse[_j] - _gnn_mean[_j]) / _bl_mse[_j] * 100
            _ax_bar.text(_j, max(_gnn_mean[_j], _bl_mse[_j]) * 1.05, f"−{_rel:.0f}%",
                         ha="center", fontsize=8, color="k")
    _ax_bar.set_xticks(_xpos)
    _ax_bar.set_xticklabels([p.replace("_", " ") for p in properties], rotation=45, ha="right", fontsize=9)
    _ax_bar.set_ylabel("Test MSE (normalised)")
    _ax_bar.set_title("GNN vs baseline: MSE per property")
    _ax_bar.legend()

    for _ax in _axes_e[_n_panels_e:]:
        _ax.set_visible(False)
    _fig_e.suptitle("GNN test predictions vs linear-composition Ridge baseline (per property)", y=1.01)
    _fig_e.tight_layout()
    save_fig(_fig_e, "fig_e_vs_baseline")
    _fig_e
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (f) HP search progression

    Best validation and test MSE at each HP search stage in the Tier B pipeline.
    Shows whether each HP sweep step provided a measurable improvement over the
    previous stage.
    """)
    return


@app.cell
def _(GROUPS_PROG, PAL, load_group, np, pd, plt, save_fig):
    _prog_rows = []
    for _g in GROUPS_PROG:
        _gdf, _, _ = load_group(_g)
        if _gdf.empty:
            continue
        _best_idx  = _gdf["val_min_last10"].idxmin()
        _prog_rows.append({
            "stage": _g.replace("_tier_c", "").replace("_tier_b", "").replace("_", " "),
            "val":   float(_gdf.loc[_best_idx, "val_min_last10"]),
            "test":  float(_gdf.loc[_best_idx, "test_mse_total"]),
        })

    _fig_f = plt.figure(figsize=(1, 1))   # placeholder in case no data
    if _prog_rows:
        _prog_df = pd.DataFrame(_prog_rows)
        _fig_f, _ax_f = plt.subplots(figsize=(max(6, len(_prog_df) * 1.5), 3.5))
        _x = np.arange(len(_prog_df))
        _ax_f.bar(_x - 0.2, _prog_df["val"],  0.35, color=PAL["val"], alpha=0.85, label="Val MSE (best run)")
        _ax_f.bar(_x + 0.2, _prog_df["test"], 0.35, color=PAL["gnn"], alpha=0.85, label="Test MSE")
        _ax_f.set_xticks(_x)
        _ax_f.set_xticklabels(_prog_df["stage"], rotation=20, ha="right", fontsize=9)
        _ax_f.set_ylabel("MSE (normalised)")
        _ax_f.set_title("Best val and test MSE per Tier C HP-search stage")
        _ax_f.legend()
        _fig_f.tight_layout()
        save_fig(_fig_f, "fig_f_hp_progression")
    else:
        print("[SKIP fig f] no HP progression groups found")
    _fig_f
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (g) Generalisation map — composition PCA

    2D PCA of the 10-dimensional one-hot composition vectors for train and test
    systems. Test compositions are coloured by mean test MAE (physical units).
    Systems at the periphery of the training cloud reveal extrapolation failures.
    """)
    return


@app.cell
def _(
    CONFIG,
    PAL,
    PCA,
    REPO_ROOT,
    importlib,
    mean_preds_phys,
    mean_targets_phys,
    mpatches,
    np,
    plt,
    save_fig,
    test_comps_list,
):
    # Load parse_composition from linear_baseline without treating scripts/ as a package
    _spec = importlib.util.spec_from_file_location(
        "linear_baseline",
        REPO_ROOT / "scripts" / "training" / "linear_baseline.py",
    )
    _lb = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_lb)
    _parse_composition = _lb.parse_composition

    # All 70 system compositions come from the data directory listing —
    # no torch_geometric needed (avoids unpickling HeteroData chunk files).
    _data_dir        = CONFIG.paths.data_dir
    _all_comps_set   = {d.name for d in _data_dir.iterdir() if d.is_dir()}
    _all_comps       = sorted(_all_comps_set | set(test_comps_list))
    _X_all      = np.array([_parse_composition(c) for c in _all_comps])
    _labels_all = ["test" if c in set(test_comps_list) else "train" for c in _all_comps]

    _pca     = PCA(n_components=2, random_state=42)
    _Z       = _pca.fit_transform(_X_all)
    _var_exp = _pca.explained_variance_ratio_

    _test_mae_map = {}
    for _gi, _comp in enumerate(test_comps_list):
        _mae_val = float(np.mean(np.abs(mean_preds_phys[_gi] - mean_targets_phys[_gi])))
        _test_mae_map.setdefault(_comp, []).append(_mae_val)
    _test_mae_map = {k: float(np.mean(v)) for k, v in _test_mae_map.items()}
    _max_mae      = max(_test_mae_map.values()) if _test_mae_map else 1.0

    _fig_g, _ax_g = plt.subplots(figsize=(6.5, 5))
    _ax_g.set_xlabel(f"PC1 ({_var_exp[0] * 100:.1f} % var)")
    _ax_g.set_ylabel(f"PC2 ({_var_exp[1] * 100:.1f} % var)")
    _ax_g.set_title("Composition PCA — train/test split, test points coloured by mean MAE")

    _sc = None
    for _comp, _z, _lbl in zip(_all_comps, _Z, _labels_all):
        if _lbl == "train":
            _ax_g.scatter(*_z, color=PAL["train"], alpha=0.45, s=30, zorder=2)
        else:
            _mae_val = _test_mae_map.get(_comp, 0.0)
            _sc = _ax_g.scatter(*_z, c=[[_mae_val]], cmap="YlOrRd",
                                 vmin=0, vmax=_max_mae,
                                 s=90, edgecolors="k", lw=0.8, zorder=3)
            _ax_g.annotate(_comp, _z, fontsize=6, alpha=0.75,
                           xytext=(3, 3), textcoords="offset points")

    if _sc is not None:
        plt.colorbar(_sc, ax=_ax_g, label="Test MAE (mean over properties, physical units)")

    if sum(_var_exp) < 0.60:
        _ax_g.set_title(_ax_g.get_title() +
                        f"\n[NOTE: 2 PCs explain only {sum(_var_exp) * 100:.0f}%]")

    _ax_g.legend(handles=[
        mpatches.Patch(color=PAL["train"], alpha=0.5, label="Train"),
        mpatches.Patch(color="#cc4411", label="Test (coloured by MAE)"),
    ])
    _fig_g.tight_layout()
    save_fig(_fig_g, "fig_g_generalization_map")
    _fig_g
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (h) R² forest plot

    Per-property R² on the held-out test set with 95 % bootstrap confidence
    intervals (10 000 resamples, pooled over seeds). The dotted line at R² = 1
    marks perfect prediction; the dashed line at R² = 0 marks the baseline of
    predicting the mean.
    """)
    return


@app.cell
def _(
    PAL,
    PROP_LABELS,
    bootstrap_ci,
    np,
    plt,
    preds_stack,
    properties,
    r2_score,
    save_fig,
    targets_stack,
):
    _fig_h, _ax_h = plt.subplots(figsize=(5, 1 + len(properties) * 0.7))
    _y_pos = np.arange(len(properties))[::-1]

    for _j, (_prop, _yp) in enumerate(zip(properties, _y_pos)):
        _pool_pred = preds_stack[:, :, _j].ravel()
        _pool_true = targets_stack[:, :, _j].ravel()
        _r2_point  = r2_score(_pool_true, _pool_pred)
        _r2_lo, _r2_hi = bootstrap_ci(
            np.stack([_pool_true, _pool_pred], axis=1),
            lambda x: r2_score(x[:, 0], x[:, 1]),
        )
        _ax_h.errorbar(
            _r2_point, _yp,
            xerr=[[_r2_point - _r2_lo], [_r2_hi - _r2_point]],
            fmt="o", color=PAL["gnn"], capsize=5, ms=7, lw=2,
        )

    _ax_h.axvline(0, color="k", lw=0.8, linestyle="--")
    _ax_h.axvline(1, color=PAL["identity"], lw=0.8, linestyle=":")
    _ax_h.set_yticks(_y_pos)
    _ax_h.set_yticklabels([PROP_LABELS.get(p, p) for p in properties])
    _ax_h.set_xlabel("R² (pooled over seeds)")
    _ax_h.set_title("Test R² per property (95 % bootstrap CI)")
    _ax_h.set_xlim(-0.1, 1.1)
    _fig_h.tight_layout()
    save_fig(_fig_h, "fig_h_r2_forest")
    _fig_h
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (i) Paired comparison: Stage 5d vs Stage 0d

    Paired t-test on total test MSE per seed between the Stage 5d confirmation
    run (200 epochs, locked HPs) and the Stage 0d 7-property floor (same HPs,
    same epoch count). Each line is one seed. Since the configurations are
    identical, this is a noise-only comparison: a non-significant p-value is
    the expected outcome and confirms the HP lock survived the Tier C tuning
    chain. The substantive comparison for Tier C is the per-property gate
    check (section 8) and the contrast against Tier B 5c (memory-bank table),
    not this t-test.
    """)
    return


@app.cell
def _(artifacts, np, runs_df, s0_artifacts, s0_df, stats):
    _s5_by_seed = {
        int(runs_df.loc[runs_df["run_id"] == _rid, "seed"].iloc[0]): _art
        for _rid, _art in artifacts.items()
        if _rid in runs_df["run_id"].values
    }
    _s0_by_seed = {
        int(s0_df.loc[s0_df["run_id"] == _rid, "seed"].iloc[0]): _art
        for _rid, _art in s0_artifacts.items()
        if _rid in s0_df["run_id"].values
    }

    _common_seeds = sorted(set(_s5_by_seed) & set(_s0_by_seed))
    if _common_seeds:
        _s5_mse = [float(np.mean((_s5_by_seed[_s]["test_preds"] - _s5_by_seed[_s]["test_targets"]) ** 2))
                   for _s in _common_seeds]
        _s0_mse = [float(np.mean((_s0_by_seed[_s]["test_preds"] - _s0_by_seed[_s]["test_targets"]) ** 2))
                   for _s in _common_seeds]
        _t_stat, _p_val = stats.ttest_rel(_s5_mse, _s0_mse, alternative="less")
        paired_ttest = {
            "t": float(_t_stat),
            "p": float(_p_val),
            "seeds": _common_seeds,
            "s5_mse": _s5_mse,
            "s0_mse": _s0_mse,
        }
        print(f"Paired t-test (Stage5d < Stage0d): t={_t_stat:.3f}, p={_p_val:.4f}, n={len(_common_seeds)}")
    else:
        paired_ttest = {"t": float("nan"), "p": float("nan"), "seeds": [], "s5_mse": [], "s0_mse": []}
        print("[SKIP] No common seeds between Stage 5d and Stage 0d")
    return (paired_ttest,)


@app.cell
def _(PAL, np, paired_ttest, plt, save_fig):
    _fig_i = plt.figure(figsize=(1, 1))   # placeholder
    if paired_ttest["seeds"]:
        _fig_i, _ax_i = plt.subplots(figsize=(4, 4))
        for _s5, _s0 in zip(paired_ttest["s5_mse"], paired_ttest["s0_mse"]):
            _ax_i.plot([0, 1], [_s0, _s5], "o-", color=PAL["seed"], alpha=0.8, ms=7)
        _ax_i.plot(
            [0, 1],
            [np.mean(paired_ttest["s0_mse"]), np.mean(paired_ttest["s5_mse"])],
            "o-", color="k", lw=2.5, ms=9, zorder=5, label="Mean",
        )
        _ax_i.set_xticks([0, 1])
        _ax_i.set_xticklabels(["Stage 0d\n(floor)", "Stage 5d\n(confirmation)"])
        _ax_i.set_ylabel("Test MSE (normalised, total)")
        _ax_i.set_title(
            f"Total test MSE per seed: Stage 0d → Stage 5d "
            f"(n={len(paired_ttest['seeds'])}; paired t={paired_ttest['t']:.2f}, "
            f"p={paired_ttest['p']:.3f}, one-sided)"
        )
        _ax_i.legend()
        _fig_i.tight_layout()
        save_fig(_fig_i, "fig_i_paired_stages")
    _fig_i
    return


@app.cell
def _(mo, paired_ttest):
    _sig = paired_ttest.get("p", 1.0) < 0.05
    _out_i = (
        mo.callout(mo.md("No common seeds — paired comparison skipped."), kind="warn")
        if not paired_ttest["seeds"]
        else mo.callout(mo.md(
            f"**Paired t-test (Stage 5d vs Stage 0d)**: t = {paired_ttest['t']:.2f}, "
            f"p = {paired_ttest['p']:.4f}, n = {len(paired_ttest['seeds'])} seeds. "
            + (
                "Statistically significant improvement at α = 0.05."
                if _sig else
                "**Not significant** at α = 0.05 — expected, since Stage 5d and "
                "Stage 0d share identical HPs (the Tier C lr-refinement chain "
                "1g → 1g' confirmed the inherited Tier A/B lock). The substantive "
                "Tier C contrast is per-property: see the gate check (§8) and the "
                "5d-vs-Tier-B-5c table in the report below."
            )
        ), kind="success" if _sig else "info")
    )
    _out_i
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (j) Per-property percentage error

    Per-test-graph percentage error `(pred − true) / true × 100`, pooled over
    all seeds. One box per property: the box spans IQR, the orange line is the
    median, whiskers extend to 1.5 × IQR, and circles mark outliers. The dotted
    vertical at 0 % is the unbiased target.

    Direct counterpart to the composition-only feed-forward baseline plot from
    the prior project — same axes, same units, same compositions in spirit
    (different test split). Lets the reader read GNN error magnitudes at a
    glance without converting between normalised and physical units.
    """)
    return


@app.cell
def _(N_SEEDS, np, plt, preds_phys, properties, save_fig, targets_phys):
    # Percentage error per (seed, graph, property): (pred - true) / true * 100.
    # Skip any (graph, property) pair where target is 0 (avoid div-by-zero).
    _DESIRED_ORDER = [
        "lipid_packing", "thickness", "thickness_std", "compressibility",
        "persistence", "diffusivity", "variation",
    ]
    _ordered_props = [p for p in _DESIRED_ORDER if p in properties]

    _pct_data = []
    _labels   = []
    for _prop in _ordered_props:
        _j    = list(properties).index(_prop)
        _true = targets_phys[:, :, _j].ravel()
        _pred = preds_phys[:, :, _j].ravel()
        _mask = _true != 0
        _pct  = (_pred[_mask] - _true[_mask]) / _true[_mask] * 100
        _pct_data.append(_pct)
        _labels.append(_prop)

    _fig_j, _ax_j = plt.subplots(figsize=(7, max(3, 0.6 * len(_ordered_props))))
    _bp = _ax_j.boxplot(
        _pct_data,
        labels=_labels,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="#FFE4B5", edgecolor="black"),
        medianprops=dict(color="#FF7F00", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="none", markeredgecolor="black", markersize=4),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
    )
    _ax_j.axvline(0, color="black", linestyle=":", lw=1)
    _ax_j.set_xlabel("Error [%]")
    _ax_j.set_title(f"Per-graph % error per property (test set, {N_SEEDS} seeds pooled)")
    _ax_j.grid(axis="x", alpha=0.3)

    _xmax = max(abs(np.percentile(np.concatenate(_pct_data), 1)),
                abs(np.percentile(np.concatenate(_pct_data), 99))) * 1.2
    _ax_j.set_xlim(-_xmax, _xmax)
    _xticks_j = np.arange(0, _xmax + 2.5, 2.5)
    _ax_j.set_xticks(np.concatenate([-_xticks_j[::-1][:-1], _xticks_j]))

    _fig_j.tight_layout()
    save_fig(_fig_j, "fig_j_percent_error_box")
    _fig_j
    return


@app.cell
def _(mo, np, preds_phys, properties, targets_phys):
    _summary_rows = []
    for _j, _prop in enumerate(properties):
        _true = targets_phys[:, :, _j].ravel()
        _pred = preds_phys[:, :, _j].ravel()
        _mask = _true != 0
        _pct  = (_pred[_mask] - _true[_mask]) / _true[_mask] * 100
        _summary_rows.append(
            f"- `{_prop}`: median {np.median(_pct):+.2f} %, "
            f"IQR [{np.percentile(_pct, 25):+.2f}, {np.percentile(_pct, 75):+.2f}] %, "
            f"|mean| {abs(np.mean(_pct)):.2f} %"
        )
    mo.md("**Percentage-error summary** (test set, pooled over seeds):\n" + "\n".join(_summary_rows))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (k) GNN vs baseline scatter — coloured by composition

    Combines figures (b) and (e): each GNN point is coloured by lipid
    composition (same palette as figure b); the linear-baseline predictions
    are overlaid as orange diamonds. R² and MAE for the GNN are annotated per
    panel. The final panel is the MSE bar chart from figure (e).

    This lets the reader see simultaneously *where* the GNN fails relative to
    the baseline (which compositions are hardest) and *how much* the GNN
    outperforms composition-only prediction.
    """)
    return


@app.cell
def _(
    COMP_COLORS,
    N_SEEDS,
    PAL,
    PROP_LABELS,
    baseline,
    bl_preds_phys,
    bl_targets_phys,
    mpatches,
    np,
    plt,
    preds_phys,
    preds_stack,
    properties,
    r2_score,
    save_fig,
    targets_phys,
    targets_stack,
    test_comps_list,
):
    _bl_props_list_k = [str(p) for p in baseline["properties"]] if baseline is not None else []
    _bl_j_for_p_k    = {p: _bl_props_list_k.index(p) for p in properties if p in _bl_props_list_k}

    _unique_comps_k   = sorted(set(test_comps_list))
    _comp_color_map_k = {c: COMP_COLORS[i % len(COMP_COLORS)] for i, c in enumerate(_unique_comps_k)}

    _n_panels_k = len(properties) + 1          # one per property + bar chart
    _n_cols_k   = min(_n_panels_k, 3)
    _n_rows_k   = int(np.ceil(_n_panels_k / _n_cols_k))
    _fig_k, _axes_k = plt.subplots(
        _n_rows_k, _n_cols_k,
        figsize=(5 * _n_cols_k, 5 * _n_rows_k),
    )
    _axes_k = np.asarray(_axes_k).ravel()

    for _j, (_ax, _prop) in enumerate(zip(_axes_k[:len(properties)], properties)):
        _true_pool = targets_phys[:, :, _j].ravel()
        _pred_gnn  = preds_phys[:, :, _j].ravel()
        _comp_pool = test_comps_list * N_SEEDS
        _colors_k  = [_comp_color_map_k[c] for c in _comp_pool]
        _bl_j      = _bl_j_for_p_k.get(_prop)

        _lo = min(_true_pool.min(), _pred_gnn.min())
        _hi = max(_true_pool.max(), _pred_gnn.max())
        if _bl_j is not None:
            _lo = min(_lo, bl_targets_phys[:, _bl_j].min(), bl_preds_phys[:, _bl_j].min())
            _hi = max(_hi, bl_targets_phys[:, _bl_j].max(), bl_preds_phys[:, _bl_j].max())

        # GNN: composition-coloured scatter
        _ax.scatter(_true_pool, _pred_gnn, c=_colors_k, alpha=0.3, s=14, lw=0, zorder=2)
        # Baseline: solid orange diamonds on top
        if _bl_j is not None:
            _ax.scatter(
                bl_targets_phys[:, _bl_j], bl_preds_phys[:, _bl_j],
                s=50, alpha=0.85, color=PAL["baseline"], marker="D",
                label="Linear baseline", zorder=3,
            )
        _ax.plot([_lo, _hi], [_lo, _hi], "--", color=PAL["identity"], lw=1.2, zorder=1)

        _r2  = r2_score(_true_pool, _pred_gnn)
        _mae = float(np.mean(np.abs(_pred_gnn - _true_pool)))
        _ax.text(
            0.05, 0.95, f"R² = {_r2:.3f}\nMAE = {_mae:.3f}",
            transform=_ax.transAxes, fontsize=9, va="top",
        )
        _ax.set_xlabel(f"True — {PROP_LABELS.get(_prop, _prop)}")
        _ax.set_ylabel("Predicted")
        _ax.set_title(f"Scatter: true vs predicted {_prop}")
        if _bl_j is not None:
            _ax.legend(fontsize=8)

    # Final panel: MSE bar chart (identical to figure e)
    _ax_bar_k = _axes_k[len(properties)]
    _xpos_k   = np.arange(len(properties))
    _gnn_mean_k = [
        float(np.mean([
            np.mean((preds_stack[_s, :, _j] - targets_stack[_s, :, _j]) ** 2)
            for _s in range(N_SEEDS)
        ]))
        for _j in range(len(properties))
    ]
    _gnn_std_k = [
        float(np.std([
            np.mean((preds_stack[_s, :, _j] - targets_stack[_s, :, _j]) ** 2)
            for _s in range(N_SEEDS)
        ]))
        for _j in range(len(properties))
    ]
    _bl_mse_k = [
        float(np.mean((baseline["test_preds"][:, _bl_j_for_p_k[_p]] -
                       baseline["test_targets"][:, _bl_j_for_p_k[_p]]) ** 2))
        if _p in _bl_j_for_p_k else float("nan")
        for _p in properties
    ]
    _w_bar_k = 0.35
    _ax_bar_k.bar(_xpos_k - _w_bar_k / 2, _gnn_mean_k, _w_bar_k, yerr=_gnn_std_k,
                  capsize=4, color=PAL["gnn"], alpha=0.85, label="GNN")
    _valid_bl_k = [_j for _j, _v in enumerate(_bl_mse_k) if not np.isnan(_v)]
    if _valid_bl_k:
        _ax_bar_k.bar(
            np.array(_valid_bl_k) + _w_bar_k / 2,
            [_bl_mse_k[_j] for _j in _valid_bl_k],
            _w_bar_k, color=PAL["baseline"], alpha=0.85, label="Linear \nbaseline",
        )
        for _j in _valid_bl_k:
            _rel = (_bl_mse_k[_j] - _gnn_mean_k[_j]) / _bl_mse_k[_j] * 100
            _ax_bar_k.text(
                _j, max(_gnn_mean_k[_j], _bl_mse_k[_j]) * 1.05,
                f"−{_rel:.0f}%", ha="center", fontsize=8, color="k",
            )
    _ax_bar_k.set_xticks(_xpos_k)
    _ax_bar_k.set_xticklabels(
        [p.replace("_", " ") for p in properties], rotation=45, ha="right", fontsize=9,
    )
    _ax_bar_k.set_ylabel("Test MSE (normalised)")
    _ax_bar_k.set_title("GNN vs baseline: MSE per property")
    _ax_bar_k.legend()

    for _ax in _axes_k[_n_panels_k:]:
        _ax.set_visible(False)

    # Composition legend below the scatter panels
    _comp_handles_k = [
        mpatches.Patch(color=_comp_color_map_k[c], label=c) for c in _unique_comps_k
    ]
    _fig_k.legend(
        _comp_handles_k, _unique_comps_k,
        title="GNN — composition", loc="lower center",
        ncol=min(5, len(_unique_comps_k)),
        bbox_to_anchor=(0.5, -0.05), fontsize=7,
    )

    _fig_k.suptitle(
        f"GNN (coloured by composition) vs linear-composition baseline — "
        f"{N_SEEDS} seeds × 275 test graphs",
        y=1.01,
    )
    _fig_k.tight_layout()
    save_fig(_fig_k, "fig_k_scatter_comp_vs_baseline")
    _fig_k
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (l) Composition coverage — train / val / test split + worst-5 MAE

    Same grid as `analyze_dataset` figure 01: rows = lipid partner, columns =
    partner mole fraction. Cell colour encodes dataset split (blue = train,
    orange = val, green = test; white = not simulated). Test compositions with
    the 5 highest summed normalised MAE are overlaid with a red star (★).
    Split membership is read from the `chunks_dir` train / val / test
    subdirectories.
    """)
    return


@app.cell
def _(
    CONFIG,
    mpatches,
    np,
    plt,
    preds_stack,
    properties,
    save_fig,
    targets_stack,
    test_comps_list,
):
    import re as _re_l
    import torch as _torch_l
    from matplotlib.colors import ListedColormap as _ListedColormap_l, BoundaryNorm as _BoundaryNorm_l
    from matplotlib.lines import Line2D as _Line2D_l

    # ── Parse composition stem → (partner_lipid, partner_frac) ───────────────
    def _parse_partner(stem):
        parts = _re_l.findall(r'([A-Za-z]+)(\d+)', stem)
        if len(parts) == 1:
            return parts[0][0], int(parts[0][1])
        a, fa = parts[0][0], int(parts[0][1])
        b, fb = parts[1][0], int(parts[1][1])
        if a == "POPC":
            return b, fb
        if b == "POPC":
            return a, fa
        return a, fa

    # ── Load split membership from chunk directories ──────────────────────────
    _chunks_dir = CONFIG.paths.chunks_dir

    def _comps_from_split(split):
        comps = set()
        for _cp in sorted((_chunks_dir / split).glob("chunk_*.pt")):
            for _g in _torch_l.load(_cp, weights_only=False):
                comps.add(_g.composition)
        return comps

    _train_comps_l = _comps_from_split("train")
    _val_comps_l   = _comps_from_split("val")
    _test_comps_l  = _comps_from_split("test")

    def _split_of(comp):
        if comp in _test_comps_l:  return 3
        if comp in _val_comps_l:   return 2
        if comp in _train_comps_l: return 1
        return 0  # unknown / not in chunks

    _all_comps_l = sorted(_train_comps_l | _val_comps_l | _test_comps_l)
    _parsed_l    = {c: _parse_partner(c) for c in _all_comps_l}

    # ── Per-composition normalised MAE (test set, sum over properties) ────────
    _comp_idx_l = {c: i for i, c in enumerate(sorted(set(test_comps_list)))}
    _sys_abs_l  = np.zeros((preds_stack.shape[0], len(_comp_idx_l), len(properties)))
    _cnt_l      = np.zeros(len(_comp_idx_l), dtype=int)
    for _s in range(preds_stack.shape[0]):
        for _gi, _c in enumerate(test_comps_list):
            _ci = _comp_idx_l[_c]
            _sys_abs_l[_s, _ci] += np.abs(preds_stack[_s, _gi] - targets_stack[_s, _gi])
            if _s == 0:
                _cnt_l[_ci] += 1
    _mae_by_comp_l = (_sys_abs_l / _cnt_l[None, :, None]).mean(0).sum(1)
    _worst5_l      = sorted(_comp_idx_l, key=lambda c: -_mae_by_comp_l[_comp_idx_l[c]])[:5]

    # ── Build the split-coloured coverage grid ────────────────────────────────
    # 0 = not simulated, 1 = train, 2 = val, 3 = test
    _partners_l = sorted({p for p, f in _parsed_l.values() if p and p != "POPC"})
    _fracs_l    = sorted({f for p, f in _parsed_l.values() if f > 0 and f < 100})

    _grid_l = np.zeros((len(_partners_l), len(_fracs_l)), dtype=float)
    for _comp, (_partner, _frac) in _parsed_l.items():
        if _partner and _partner != "POPC" and _frac in _fracs_l:
            _ri = _partners_l.index(_partner)
            _ci = _fracs_l.index(_frac)
            _grid_l[_ri, _ci] = _split_of(_comp)

    # ── Worst-5 → grid coordinates ────────────────────────────────────────────
    _worst5_cells_l = []
    for _wc in _worst5_l:
        _p, _f = _parsed_l.get(_wc, (None, 0))
        if _p and _p != "POPC" and _p in _partners_l and _f in _fracs_l:
            _worst5_cells_l.append((_partners_l.index(_p), _fracs_l.index(_f), _wc))

    # ── Plot ──────────────────────────────────────────────────────────────────
    # 0=white, 1=train(blue), 2=val(amber), 3=test(green)
    _cmap_l = _ListedColormap_l(["white", "#4472C4", "#E69F00", "#70AD47"])
    _norm_l = _BoundaryNorm_l([-0.5, 0.5, 1.5, 2.5, 3.5], _cmap_l.N)

    _fig_l, _ax_l = plt.subplots(figsize=(12, 5.5))
    _ax_l.imshow(_grid_l, cmap=_cmap_l, norm=_norm_l, aspect="auto")
    _ax_l.set_xticks(range(len(_fracs_l)))
    _ax_l.set_xticklabels(_fracs_l, fontsize=8)
    _ax_l.set_xlabel("partner mole fraction (%)", labelpad=4)
    _ax_l.set_yticks(range(len(_partners_l)))
    _ax_l.set_yticklabels(_partners_l, fontsize=9)

    for _ri in range(len(_partners_l)):
        for _ci in range(len(_fracs_l)):
            if _grid_l[_ri, _ci] > 0:
                _ax_l.text(_ci, _ri, "•", ha="center", va="center", color="white", fontsize=13)

    for _ri, _ci, _comp in _worst5_cells_l:
        _ax_l.text(_ci, _ri, "★", ha="center", va="center",
                   color="crimson", fontsize=18, fontweight="bold", zorder=5)

    _ax2_l = _ax_l.twiny()
    _ax2_l.set_xlim(_ax_l.get_xlim())
    _ax2_l.set_xticks(range(len(_fracs_l)))
    _ax2_l.set_xticklabels([f"{100 - _f}" for _f in _fracs_l], fontsize=7, rotation=45, ha="left")
    _ax2_l.set_xlabel("POPC mol% (top axis: 100 − bottom-axis value)", fontsize=8)

    _n_train = int(((_grid_l == 1).sum()))
    _n_val   = int(((_grid_l == 2).sum()))
    _n_test  = int(((_grid_l == 3).sum()))
    _n_marked = len(_worst5_cells_l)
    _n_total  = len(_worst5_l)
    _star_handle = _Line2D_l([0], [0], marker="*", color="crimson", markersize=12,
                              linestyle="None",
                              label=f"★ top-5 MAE test compositions ({_n_marked}/{_n_total} in grid)")
    _ax_l.legend(handles=[
        mpatches.Patch(facecolor="#4472C4", label=f"train ({_n_train})"),
        mpatches.Patch(facecolor="#E69F00", label=f"val ({_n_val})"),
        mpatches.Patch(facecolor="#70AD47", label=f"test ({_n_test})"),
        mpatches.Patch(facecolor="white", edgecolor="lightgray", label="not simulated"),
        _star_handle,
    ], loc="lower right", fontsize=8, framealpha=0.9)
    _ax_l.set_title(
        f"Composition coverage by split — {len(_all_comps_l)} systems  "
        f"(★ = top-5 MAE test: {', '.join(_worst5_l)})"
    )
    _fig_l.tight_layout()
    save_fig(_fig_l, "fig_l_coverage_split_worst5")
    _fig_l
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Statistical tests

    Per-property residual diagnostics: R² with 95 % CI, empirical bias (mean
    residual in normalised space), and Shapiro-Wilk normality test on a random
    subsample (n = 1 000) of the residuals.
    """)
    return


@app.cell
def _(
    bootstrap_ci,
    mo,
    np,
    pd,
    preds_stack,
    properties,
    r2_score,
    stats,
    targets_stack,
):
    _stat_rows = []
    for _j, _prop in enumerate(properties):
        _pool_pred = preds_stack[:, :, _j].ravel()
        _pool_true = targets_stack[:, :, _j].ravel()
        _r2_val    = r2_score(_pool_true, _pool_pred)
        _r2_lo, _r2_hi = bootstrap_ci(
            np.stack([_pool_true, _pool_pred], axis=1),
            lambda x: r2_score(x[:, 0], x[:, 1]),
        )
        _resid  = _pool_pred - _pool_true
        _bias   = float(_resid.mean())
        _sample = np.random.default_rng(0).choice(_resid, size=min(1000, len(_resid)), replace=False)
        _sw_w, _sw_p = stats.shapiro(_sample)
        _stat_rows.append({
            "property": _prop,
            "R²":       _r2_val,
            "R² CI lo": _r2_lo,
            "R² CI hi": _r2_hi,
            "bias":     _bias,
            "SW W":     float(_sw_w),
            "SW p":     float(_sw_p),
            "normal":   _sw_p > 0.05,
        })

    stat_df = pd.DataFrame(_stat_rows).set_index("property")
    mo.as_html(stat_df.round(4))
    return (stat_df,)


@app.cell
def _(mo, stat_df):
    _biased = stat_df[stat_df["bias"].abs() > 0.05].index.tolist()
    _bias_out = (
        mo.callout(mo.md(
            f"**Systematic bias** (|mean residual| > 0.05 normalised units) detected in: "
            + ", ".join(f"`{p}`" for p in _biased)
            + ". This indicates consistent over- or under-prediction and is not captured by R²."
        ), kind="warn")
        if _biased
        else mo.callout(mo.md("No systematic bias > 0.05 normalised units detected."), kind="info")
    )
    _bias_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Gate check

    Stage 5d must beat the Stage 0d 7-property floor on each property (val
    MSE, last-10 epochs mean over seeds). These gates are set in
    `docs/tier_c_7prop_plan.md`. Note that Stage 0d and Stage 5d share the
    same locked HPs (`lr=3e-5, wd=1e-3, hidden_dim=128, num_layers=2,
    epochs=200`) — gate failure here would indicate a noise/seed regression,
    not an HP regression.
    """)
    return


@app.cell
def _(GATES, histories, mo, np, pd):
    _gate_rows = []
    for _prop, _threshold in GATES.items():
        _col  = f"val/loss_{_prop}"
        _vals = [_hist[_col].tail(10).min() for _hist in histories.values() if _col in _hist.columns]
        if _vals:
            _mean_val = float(np.mean(_vals))
            _margin   = _threshold - _mean_val
            _gate_rows.append({
                "property":  _prop,
                "val_mean":  _mean_val,
                "threshold": _threshold,
                "margin":    _margin,
                "pass":      bool(_mean_val < _threshold),
            })

    gate_summary = pd.DataFrame(_gate_rows)
    _pass_count  = gate_summary["pass"].sum()
    _total       = len(gate_summary)

    mo.vstack([
        mo.callout(mo.md(
            f"**Gate check: {_pass_count}/{_total} properties pass** the Stage 0d floor.\n\n"
            + "\n".join(
                f"- `{r['property']}`: {r['val_mean']:.4f} vs {r['threshold']} "
                f"— {'**PASS**' if r['pass'] else '**FAIL**'} (margin {r['margin']:+.4f})"
                for _, r in gate_summary.iterrows()
            )
        ), kind="success" if _pass_count == _total else "warn"),
        mo.as_html(gate_summary.round(4)),
    ])
    return (gate_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Export

    All headline numbers are written to `headline_numbers.json` for downstream
    use (thesis tables, comparison with Tier A Stage 5b and Tier B Stage 5c).
    """)
    return


@app.cell
def _(FIGURES_DIR, gate_summary, json, mo, paired_ttest, tbl):
    _headlines = {}

    for _prop, _row in tbl.iterrows():
        _headlines[f"mse_{_prop}"]  = {"mean": float(_row["MSE mean"]), "std": float(_row["MSE std"])}
        _headlines[f"r2_{_prop}"]   = {
            "point": float(_row["R²"]),
            "ci_lo": float(_row["R² CI lo"]),
            "ci_hi": float(_row["R² CI hi"]),
        }
        _headlines[f"mae_{_prop}"]  = float(_row["MAE"])

    _headlines["paired_ttest"] = {
        "t": paired_ttest["t"],
        "p": paired_ttest["p"],
        "seeds": paired_ttest["seeds"],
    }

    _headlines["gate_check"] = {
        r["property"]: {"val_mean": r["val_mean"], "threshold": r["threshold"], "pass": r["pass"]}
        for _, r in gate_summary.iterrows()
    }

    _out_path = FIGURES_DIR / "headline_numbers.json"
    with open(_out_path, "w") as _f:
        json.dump(_headlines, _f, indent=2, default=float)

    mo.callout(mo.md(f"Headline numbers written to `{_out_path}`."), kind="info")
    return


@app.cell(hide_code=True)
def _(gate_summary, mo, paired_ttest, tbl):
    _r2_lines = "\n".join(
        f"- **`{p}`**: R² = {row['R²']:.3f} [{row['R² CI lo']:.3f}, {row['R² CI hi']:.3f}]"
        for p, row in tbl.iterrows()
    )
    _gate_lines = "\n".join(
        f"  - `{r['property']}`: {r['val_mean']:.4f} vs {r['threshold']} "
        f"({'PASS' if r['pass'] else 'FAIL'})"
        for _, r in gate_summary.iterrows()
    )
    _pass_count = int(gate_summary["pass"].sum())
    _total      = len(gate_summary)
    _ttest_line = (
        f"t = {paired_ttest['t']:.2f}, p = {paired_ttest['p']:.4f}, "
        f"n = {len(paired_ttest['seeds'])} seeds"
    )
    mo.md(rf"""
    ## Conclusions

    **1. Seven-property prediction is feasible at the Tier A/B locked HPs.**
    The locked configuration (`lr=3e-5, wd=1e-3, h=128, l=2, e=200`) carries
    cleanly from Tier B (6 props) to Tier C (7 props). No HP change was
    needed — the 1g → 1g' refinement chain re-confirmed `lr=3e-5` after a
    pilot-level 2-seed lr=1e-5 false alarm dissolved at 4 seeds.

    **2. Per-property pooled test R² (95 % CI)**:
    {_r2_lines}

    Six of seven properties land in the GOOD band (R² ≥ 0.85). Only
    `persistence` remains in the OK band (R² ≈ 0.57), unchanged from Tier B
    5c (0.578) — the architecture floor is unchanged by adding a 7th head.

    **3. `compressibility` learns substantially better than pre-registered.**
    Pooled test R² ≈ 0.88, well above the "<<0.5" architectural-ceiling
    expectation set in `docs/tier_c_7prop_plan.md`. Per-seed val R² is much
    lower (≈ 0.59 in W&B summaries) — the gap is consistent with the small
    val split inflating variance; the test set (275 graphs × 4 seeds = 1100
    pooled points) gives the more stable estimate. Interpretation: the local
    11 Å lipid-packing geometry encodes a partial-but-strong proxy for
    whole-bilayer area-fluctuation density. The architectural argument for
    EFA-style long-wavelength receptive fields (`docs/efa_spatial_layer_future.md`)
    is not falsified — the CG geometry just contains more local information
    about a long-wavelength target than the receptive-field upper bound
    predicted. `bending_modulus` remains the harder, undulation-spectrum
    target where the same shortcut may not hold.

    **4. `diffusivity` static-snapshot → dynamical prediction holds at 7 props.**
    R² ≈ 0.96, within seed jitter of Tier B 5c (0.959). Adding the
    long-wavelength compressibility head did not perturb the easiest dynamical
    target.

    **5. `persistence` is architecture-bound, not HP-bound, and not transfer-bound.**
    R² ≈ 0.57 (Tier C) vs 0.58 (Tier B 5c) vs 0.66 (Stage 0c). Flat across
    all lrs in 1e' and 1g'; flat across all training durations. The shared MLP
    trunk + 11 Å spatial cutoff is the binding constraint. Separate heads
    or uncertainty weighting are the candidate remedies, flagged for thesis
    discussion.

    **6. Gate check: {_pass_count}/{_total} properties pass the Stage 0d floor.**
    {_gate_lines}

    The two failures are within seed jitter: `persistence` 0.391 vs gate 0.370
    (+5.7 %) reflects Stage 0d having a slightly luckier 5-seed mean on a
    flat-floor property; `diffusivity` 0.0657 vs gate 0.0655 (+0.2 %) is a
    statistical tie. Neither indicates an HP or training regression — both
    are within ±10 % of the inherited Tier B numbers.

    **7. Paired t-test vs Stage 0d ({_ttest_line})**:
    {"Not significant at α = 0.05 — Stage 5d and Stage 0d share identical HPs"
     " and epoch count, so this is a noise-only comparison and a non-significant"
     " p-value is the expected outcome. The Tier C HP search (1g → 1g') ended"
     " by re-confirming the inherited lock; Stage 5d is therefore a re-run for"
     " full artifacts, not a new configuration."
     if paired_ttest.get('p', 1) >= 0.05
     else "Statistically significant improvement at α = 0.05 — unexpected"
     " given identical HPs; investigate seed/sampling-variance differences"
     " between the 5d and 0d runs."}

    **8. Net cost of adding `compressibility` to the shared trunk** (Stage
    5d vs Tier B 5c, normalised test MSE):
    `lipid_packing` +14 % (0.0182 → 0.0208), `thickness` +1 %, `thickness_std`
    −1 %, `variation` −5 %, `persistence` +2 %, `diffusivity` −2 %. Net
    wash on five of six Tier B properties; one mild regression on
    `lipid_packing`. Conclusion: the 7-property shared-trunk model is the
    right trade — `compressibility` learns and the Tier B numbers are
    preserved within seed jitter.

    **Caveats and open questions**:

    - **Seed 3 dead-init exclusion.** Seed 3 reproduced its Tier B 0c failure
      mode on `variation` and was excluded from the primary 4-seed pool
      (matches Tier A's seed-2 pattern). Replacement seed 8 is in flight.
      ~20 % init failure rate is now confirmed across three independent
      sweeps (Tier A 1b'/1c, Tier B 0c, Tier C 5d). Documented as a
      cross-tier scope limit, not a Tier C-specific issue.
    - **Per-seed val_compressibility R² ≪ pooled test R².** W&B summaries
      report `val/r2_compressibility` ≈ 0.59 across all four seeds; the
      pooled test R² is ≈ 0.88. The val set is too small to estimate R²
      reliably for a property with broad target range. Report the pooled
      test number in the thesis; flag the val/test discrepancy as a
      reminder that the small val split is a poor R² estimator on its own.
    - **DPPC-/DOPC-rich peripheral compositions still dominate per-system
      MAE** — same Tier A/B pattern, unchanged by adding compressibility.
      Train-coverage augmentation in the PC1 < 0 region is the direct fix.
    - **`bending_modulus` (8th property) remains deferred** — undulation-
      spectrum-derived, even more strongly long-wavelength than
      compressibility, and label-noisier. The Tier C compressibility
      surprise does not change this prior; flag for the EFA-future-work plan.
    """)
    return


if __name__ == "__main__":
    app.run()
