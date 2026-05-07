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
    # Stage 5c — Tier B 6-Property Confirmation Analysis

    Final 5-seed confirmation run (`stage_5c_tier_b_confirm`) at the locked
    Tier A hyperparameters (`hidden_dim=128`, `num_layers=2`, `lr=3e-5`,
    `wd=1e-3`, `epochs=200`) now covering all six Tier B properties:
    `lipid_packing`, `thickness`, `thickness_std`, `variation`, `persistence`,
    `diffusivity`.

    **Prerequisites** (run before opening this notebook):
    ```bash
    python scripts/python/download_wandb_runs.py --group stage_5c_tier_b_confirm
    python scripts/python/download_wandb_runs.py --group stage_0c_tier_b
    python scripts/training/linear_baseline.py --stratified
    ```

    **Output**: `results/figures/stage_5c/` — PDF + PNG per figure, `headline_numbers.json`.

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
    # Tier B gates: Stage 0c 6-prop floor (5-seed val_min_last10 mean).
    # Stage 5c must beat all gates to demonstrate HP tuning was worthwhile.
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
        "compressibility": "Cmpressibility (Å³/kT)"
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

    Load the Stage 5c confirmation runs and the Stage 0c floor baseline.
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
        print(f"\nStage-5c runs ({len(runs_df)} total):")
        print(runs_df[["run_name", "seed", "val_min_last10", "test_mse_total"]].to_string(index=False))
    return artifacts, histories, runs_df, s0_artifacts, s0_df


@app.cell
def _(mo, runs_df):
    mo.stop(
        runs_df.empty,
        mo.callout(
            mo.md("**No Stage-5c runs found.** Run `download_wandb_runs.py --group stage_5c_tier_b_confirm` first."),
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
    primary thesis numbers for the Tier B confirmation.
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
        mo.md("**Headline numbers — normalised space** (R² pooled over 5 seeds, 10 000 bootstrap resamples):"),
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

    `persistence` (R² ≈ {tbl.loc['persistence', 'R²']:.3f}) is the only WEAK property —
    consistent with the architecture-limited finding from Stages 0c, 1e, 1e'.
    `diffusivity` (R² ≈ {tbl.loc['diffusivity', 'R²']:.3f}) confirms that a static
    graph embedding can predict a time-averaged dynamical observable.
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Figures

    Each figure is saved as PDF + PNG to `results/figures/stage_5c/`.
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
    _fig_a.suptitle(f"Training dynamics — {N_SEEDS} seeds (mean ± 1 SD shaded)", y=1.01)
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
    _fig_b.suptitle(f"Predicted vs true — pooled over {N_SEEDS} seeds × {N_TEST} test graphs")
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
    _ax_c.set_title("Per-composition test MAE: normalised units, sorted by total error")
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
    _fig_d.suptitle("Residual distributions (pooled over 5 seeds; dashed = Gaussian fit)")
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
    _fig_e.suptitle("GNN vs linear-composition baseline", y=1.01)
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
            "stage": _g.replace("_tier_b", "").replace("_", " "),
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
        _ax_f.set_title("HP search progression — best config per stage")
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
    _ax_g.set_title("Composition space (PCA): split membership + test MAE")

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
    intervals (10 000 resamples, pooled over 5 seeds). The dotted line at R² = 1
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
    _ax_h.set_xlabel("R² (pooled over 5 seeds)")
    _ax_h.set_title("R² with 95 % bootstrap CI: per property")
    _ax_h.set_xlim(-0.1, 1.1)
    _fig_h.tight_layout()
    save_fig(_fig_h, "fig_h_r2_forest")
    _fig_h
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### (i) Paired comparison: Stage 5c vs Stage 0c

    Paired t-test on total test MSE per seed between the Stage 5c confirmation
    run and the Stage 0c floor. Each line is one seed. A significant one-sided
    p-value (Stage 5c < Stage 0c) would confirm that HP tuning improved the GNN
    beyond the random initialisation floor.
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
        print(f"Paired t-test (Stage5c < Stage0c): t={_t_stat:.3f}, p={_p_val:.4f}, n={len(_common_seeds)}")
    else:
        paired_ttest = {"t": float("nan"), "p": float("nan"), "seeds": [], "s5_mse": [], "s0_mse": []}
        print("[SKIP] No common seeds between Stage 5c and Stage 0c")
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
        _ax_i.set_xticklabels(["Stage 0c\n(floor)", "Stage 5c\n(confirmation)"])
        _ax_i.set_ylabel("Test MSE (normalised, total)")
        _ax_i.set_title(
            f"Paired comparison (n={len(paired_ttest['seeds'])} seeds)\n"
            f"t = {paired_ttest['t']:.2f}, p = {paired_ttest['p']:.4f} (one-sided)"
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
            f"**Paired t-test (Stage 5c vs Stage 0c)**: t = {paired_ttest['t']:.2f}, "
            f"p = {paired_ttest['p']:.4f}, n = {len(paired_ttest['seeds'])} seeds. "
            + (
                "Statistically significant improvement at α = 0.05."
                if _sig else
                "**Not significant** at α = 0.05 — HP tuning did not significantly reduce "
                "total test MSE beyond the Stage 0c floor at this sample size."
            )
        ), kind="success" if _sig else "warn")
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
    _ax_j.set_title(f"Percentage error per property (test set, {N_SEEDS} seeds pooled)")
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

    Stage 5c must beat the Stage 0c floor on **all** properties (val MSE,
    last-10 epochs mean over seeds). These gates are set in
    `docs/tier_b_6prop_plan.md` and `analyze_hp_search.ipynb`.
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
            f"**Gate check: {_pass_count}/{_total} properties pass** the Stage 0c floor.\n\n"
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
    use (thesis tables, comparison with Tier A Stage 5b).
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

    **1. Six-property prediction is feasible at Tier A hyperparameters.**
    No negative transfer was observed: Tier A properties (`lipid_packing`,
    `thickness`, `thickness_std`, `variation`) maintain or improve on their
    Stage 5b R² when two additional properties are added to the readout.

    **2. Per-property R² (pooled, 95 % CI)**:
    {_r2_lines}

    **3. `persistence` is architecture-limited.**
    R² ≈ 0.58 is consistent across all seeds and learning rates — the shared
    MLP trunk does not have sufficient capacity to represent the persistence
    length signal alongside the heterogeneity properties. Separate heads or
    uncertainty weighting are the likely remedy (flagged for thesis discussion).

    **4. `diffusivity` confirms that static embedding → dynamic property.**
    R² ≈ 0.96 is the strongest result in the Tier B suite and provides a clean
    thesis story: a single-frame graph embedding of lipid packing geometry can
    predict a time-averaged lateral diffusivity.

    **5. Gate check: {_pass_count}/{_total} properties pass the Stage 0c floor.**
    {_gate_lines}

    **6. Paired t-test vs Stage 0c ({_ttest_line})**:
    {"Not significant at α = 0.05 — total MSE reduction over the 6-property floor is not"
     " statistically distinguishable at n = " + str(len(paired_ttest['seeds'])) + " seeds."
     " Per-property improvements are substantial (R² ≥ 0.87 on five of six properties)"
     " but the aggregate test MSE is dominated by `persistence`."
     if paired_ttest.get('p', 1) >= 0.05
     else "Statistically significant improvement at α = 0.05."}

    **Caveats and open questions**:
    - Errors concentrate on DPPC- and DOPC-rich compositions at the periphery
      of the training cloud. Coverage augmentation (more extreme-composition
      simulations) is the likely fix — see the generalisation map (figure g).
    - The `variation` property still shows seed-fragility (~20 % bad-init rate),
      consistent with Stage 5b. Planned seed pool {0, 1, 3, 4, 5} excludes the
      known dead-init seed 2.
    - Tier C (+compressibility, +bending_modulus) is likely floor-bound until
      the spatial channel is extended (see `docs/efa_spatial_layer_future.md`).
    """)
    return


if __name__ == "__main__":
    app.run()
