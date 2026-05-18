# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "numpy",
#     "pandas",
#     "matplotlib",
#     "scipy",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pickle
    import re
    import sys
    import warnings
    from pathlib import Path
    from itertools import combinations, permutations

    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.patches import Patch
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    from scipy.stats import gaussian_kde
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings("ignore")

    try:
        import umap as umap_lib
        HAS_UMAP = True
    except ImportError:
        umap_lib = None
        HAS_UMAP = False

    try:
        import torch
        HAS_TORCH = True
    except ImportError:
        torch = None
        HAS_TORCH = False
    return (
        BoundaryNorm,
        HAS_TORCH,
        HAS_UMAP,
        ListedColormap,
        PCA,
        Patch,
        Path,
        StandardScaler,
        combinations,
        gaussian_kde,
        hierarchy,
        mo,
        np,
        pd,
        pdist,
        permutations,
        pickle,
        plt,
        re,
        stats,
        sys,
        torch,
        umap_lib,
    )


@app.cell
def _(Path, mo, sys):
    def _find_repo_root():
        _p = Path(".").resolve()
        for _ in range(6):
            if (_p / "config.yaml").exists():
                return _p
            _p = _p.parent
        raise FileNotFoundError("Cannot find repo root (config.yaml not found in parents)")

    _REPO_ROOT = _find_repo_root()
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))

    from lipid_gnn.config import CONFIG

    PROPS = list(CONFIG.vocab.all_properties)
    LIPIDS = list(CONFIG.vocab.lipid_types)
    REFERENCE = CONFIG.dataset.reference_system
    SPLITS = ["train", "val", "test"]
    TIER_B_PROPS = [
        "lipid_packing", "thickness", "thickness_std",
        "variation", "persistence", "diffusivity",
        "compressibility"
    ]
    STAGE5D_WORST = ["POPC65_DPPE35", "POPC70_POPE30", "POPC30_DOPC70", "POPC40_DIPC60", "POPC60_DPPC40"]

    PROPS_DIR = Path(CONFIG.paths.props_dir)
    OUT_DIR = Path(CONFIG.paths.results_dir) / "dataset_analysis"
    FIG_DIR = OUT_DIR / "figures"
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    SPLIT_COLORS = {"train": "steelblue", "val": "darkorange", "test": "green"}

    mo.md(f"""
    **Config** — {len(PROPS)} properties · {len(LIPIDS)} lipid types · reference: `{REFERENCE}`
    Output → `{OUT_DIR}`
    """)
    return (
        CONFIG,
        FIG_DIR,
        LIPIDS,
        OUT_DIR,
        PROPS,
        PROPS_DIR,
        REFERENCE,
        SPLITS,
        SPLIT_COLORS,
        STAGE5D_WORST,
        TIER_B_PROPS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset Analysis — 70 Martini 3 Membrane Compositions

    A characterisation of the 70-system dataset that the GNN is trained against.
    Reads the precomputed property files in `results/properties/` (one pickle per
    composition, despite the `.h5` extension) and the parsed compositions.

    Sections:
    1. Composition catalogue + coverage heatmap
    2. Property definitions
    3. Univariate property distributions
    4. Pairwise structure: correlations, pairplot, Helfrich sanity check
    5. Composition titration curves
    6. Dimensionality reduction & clustering (PCA, UMAP, Ward)
    7. Time-series quality (stationarity, autocorrelation, noise floor)
    8. Train / val / test split audit
    9. KDE coverage analysis (composition-space and property-space)
    10. Gap candidate ranking (top simulation candidates)
    """)
    return


@app.cell
def _(OUT_DIR, PROPS, PROPS_DIR, REFERENCE, mo, pd, pickle, plt, re):
    # ── Lipid taxonomy ────────────────────────────────────────────────────────
    LIPID_FAMILY = {
        "POPC": "PC (mono-unsat)",
        "DOPC": "PC (di-unsat)",
        "DIPC": "PC (di-unsat)",
        "DPPC": "PC (saturated)",
        "POPE": "PE",
        "DOPE": "PE",
        "DPPE": "PE",
        "POPS": "PS (anionic)",
        "DOPS": "PS (anionic)",
        "CHOL": "sterol",
    }
    FAMILY_ORDER = ["PC (mono-unsat)", "PC (di-unsat)", "PC (saturated)", "PE", "PS (anionic)", "sterol"]
    FAMILY_COLORS = dict(zip(FAMILY_ORDER, plt.get_cmap("tab10").colors[:len(FAMILY_ORDER)]))

    _PARSE = re.compile(r"([A-Z]+)(\d+)")

    def _parse_composition(stem):
        parts = _PARSE.findall(stem)
        if len(parts) == 1:
            lip, frac = parts[0]
            return lip, int(frac), None, 0
        (a, fa), (b, fb) = parts[:2]
        return a, int(fa), b, int(fb)

    def _partner(lip_a, lip_b):
        if lip_a == "POPC":
            return lip_b if lip_b is not None else "POPC"
        if lip_b == "POPC":
            return lip_a
        return lip_a

    def _partner_frac(lip_a, fa, lip_b, fb):
        if lip_b is None:
            return 0 if lip_a == "POPC" else 100
        return fb if lip_a == "POPC" else fa

    # ── Load property files ───────────────────────────────────────────────────
    _rows = []
    RAW = {}

    for _p in sorted(PROPS_DIR.glob("*.h5")):
        with open(_p, "rb") as _f:
            _mean_d, _raw_d = pickle.load(_f)
        _a, _fa, _b, _fb = _parse_composition(_p.stem)
        _row = {
            "composition": _p.stem,
            "lipid_a": _a, "frac_a": _fa,
            "lipid_b": _b, "frac_b": _fb,
            "partner": _partner(_a, _b),
            "partner_frac": _partner_frac(_a, _fa, _b, _fb),
            "is_pure": _b is None,
        }
        _row["family"] = LIPID_FAMILY[_row["partner"]] if _row["partner_frac"] > 0 or _row["is_pure"] else "PC (mono-unsat)"
        for _k in PROPS:
            _row[_k] = float(_mean_d[_k])
        _rows.append(_row)
        RAW[_p.stem] = _raw_d

    df = pd.DataFrame(_rows).sort_values("composition").reset_index(drop=True)
    mo.stop(df.empty, mo.callout(mo.md("**No property files found** — check `PROPS_DIR` in `config.yaml`."), kind="danger"))

    # Save master table for downstream reuse
    _table_path = OUT_DIR / "dataset_table.csv"
    df.to_csv(_table_path, index=False)

    ref_row = df[df["composition"] == REFERENCE].iloc[0]

    mo.md(f"""
    **Dataset (`df`)** — {df.shape[0]:,} compositions × {df.shape[1]} cols
    {df['is_pure'].sum()} pure systems · {(~df['is_pure']).sum()} mixtures · {df['partner'].nunique()} unique partner lipids
    Properties: `{PROPS}`
    Dtypes: `{df.dtypes.value_counts().to_dict()}`
    Nulls: {df.isna().sum().sum()} total cells · Saved → `{_table_path.name}`
    """)
    return FAMILY_COLORS, RAW, df, ref_row


@app.cell(hide_code=True)
def _(density_df, df, mo, top20):
    _n_gap = (density_df["kde_density"] <= density_df["kde_density"].quantile(0.25)).sum()
    mo.callout(mo.md(f"""
    **Key Findings (dataset characterisation)**

    - **{len(df)} compositions**: {df['is_pure'].sum()} pure systems, {(~df['is_pure']).sum()} binary mixtures across {df['partner'].nunique()} partner lipids
    - **Coverage**: fractions 10–70 mol% for most partners; DPPC and DOPC have the widest titration range
    - **Property outliers**: flagged by MAD (|z| > 3) in §3 — `bending_modulus` has the most outliers (noisiest estimator)
    - **Helfrich proxy**: `compressibility · thickness²` correlates positively with `bending_modulus` (see §4c)
    - **PCA**: first two property-PCs capture the majority of variance; sterol compositions cluster separately
    - **Coverage gaps**: {_n_gap} compositions in bottom 25% KDE density; Stage 5d worst-MAE systems fall in this region (§9)
    - **Top gap candidate**: `{top20.iloc[0]['lipid_a']}{int(top20.iloc[0]['frac_a'])}_{top20.iloc[0]['lipid_b']}{int(top20.iloc[0]['frac_b'])}` (lowest composition-space KDE density, §10)
    """), kind="info")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 — Composition Catalogue

    The 70 filenames encode binary compositions like `POPC70_CHOL30` or pure systems
    `POPC100`. Each file contains `(mean_dict, raw_dict)` of per-frame property series.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1a — Coverage heatmap

    Each non-POPC partner lipid is paired with POPC at a discrete set of fractions.
    The heatmap shows, for every (partner × fraction) cell, whether that composition
    was simulated. The POPC row (top) shows the complementary POPC fraction present
    in each binary mixture.
    """)
    return


@app.cell
def _(BoundaryNorm, FIG_DIR, ListedColormap, Patch, df, np, plt):
    _partners = sorted(set(df["partner"]) - {"POPC"})
    _fracs = sorted(set(df["partner_frac"]) - {0})

    _cov = np.zeros((len(_partners), len(_fracs)), dtype=float)
    for _, _r in df.iterrows():
        if _r["partner"] == "POPC":
            continue
        _pi = _partners.index(_r["partner"])
        _fi = _fracs.index(_r["partner_frac"])
        _cov[_pi, _fi] = 1.0

    _popc_fracs_present = set(df.loc[df["lipid_a"] == "POPC", "frac_a"].tolist())
    _popc_row = np.zeros((1, len(_fracs)), dtype=float)
    for _j, _f in enumerate(_fracs):
        if (100 - _f) in _popc_fracs_present:
            _popc_row[0, _j] = 2.0

    _combined = np.vstack([_popc_row, _cov])
    _y_labels = ["POPC\n(100−x%)"] + _partners

    _cmap3 = ListedColormap(["white", "#4472C4", "#70AD47"])
    _norm3 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _cmap3.N)

    _fig, _ax = plt.subplots(figsize=(12, 5.5))
    _ax.imshow(_combined, cmap=_cmap3, norm=_norm3, aspect="auto")
    _ax.set_xticks(range(len(_fracs)))
    _ax.set_xticklabels(_fracs, fontsize=8)
    _ax.set_xlabel("partner mole fraction (%)", labelpad=4)
    _ax.set_yticks(range(len(_partners) + 1))
    _ax.set_yticklabels(_y_labels, fontsize=9)
    _ax.axhline(0.5, color="black", lw=2.0)

    for _i in range(len(_partners) + 1):
        for _j in range(len(_fracs)):
            if _combined[_i, _j] > 0:
                _ax.text(_j, _i, "•", ha="center", va="center", color="white", fontsize=13)

    _ax2 = _ax.twiny()
    _ax2.set_xlim(_ax.get_xlim())
    _ax2.set_xticks(range(len(_fracs)))
    _ax2.set_xticklabels([f"{100 - _f}" for _f in _fracs], fontsize=7, rotation=45, ha="left")
    _ax2.set_xlabel("POPC mol% (top axis: 100 − bottom-axis value)", fontsize=8)

    _ax.legend(handles=[
        Patch(facecolor="#4472C4", label="partner lipid covered (bottom axis)"),
        Patch(facecolor="#70AD47", label="POPC covered (top axis)"),
        Patch(facecolor="white", edgecolor="lightgray", label="not simulated"),
    ], loc="lower right", fontsize=8, framealpha=0.9)
    _ax.set_title(
        f"Composition coverage — {len(df)} systems "
        f"({df['is_pure'].sum()} pure, {(~df['is_pure']).sum()} mixtures)"
    )
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "01_coverage.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "01_coverage.pdf", bbox_inches="tight")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 — Property Definitions

    Eight scalars are computed per composition by
    `lipid_gnn.properties.compute_all` (replaces the legacy
    `functions_emil.calculate_properties.compute_properties`)
    over frames `[50:667]` (≈1 µs of production, `dt = 1.5 ns`).

    | Name | Definition | Units |
    |---|---|---|
    | `lipid_packing` | N_lipids / box_x · box_y, frame-mean | lipids / nm² |
    | `thickness` | mean (upper − lower) leaflet height on a 0.1 nm xy grid | Å |
    | `thickness_std` | frame-mean of the spatial std of (upper − lower) | Å |
    | `compressibility` | variance of thickness deviations on the grid (×100) | Å³ / kT |
    | `bending_modulus` | κ from a `kBT/(κ q⁴)` fit of the undulation spectrum | kT / Å³ |
    | `persistence` | P(lipid–lipid contact still present after `lag=50` frames) | dimensionless |
    | `diffusivity` | mean lateral MSD after `lag=10` frames (×100) | Å² |
    | `variation` | mean Voronoi-cell-area coefficient of variation (per leaflet) | dimensionless |

    **Notes**: `compressibility` is a thickness-fluctuation variance, not the canonical K_A.
    `bending_modulus` is the noisiest estimator (few q-bins after count threshold).
    `bending_modulus`'s `raw_dict` entry is the (q, spectrum) pair, not a time series
    — it is excluded from §7 time-series analysis.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 — Univariate Property Distributions

    For each property: histogram across all 70 systems with a rug coloured by partner
    family, and the POPC100 reference value marked. MAD-based outlier flagging (|z| > 3)
    is shown in the table below.
    """)
    return


@app.cell
def _(FAMILY_COLORS, FIG_DIR, PROPS, REFERENCE, df, plt, ref_row):
    _fig, _axes = plt.subplots(4, 2, figsize=(12, 14))
    for _ax, _prop in zip(_axes.flat, PROPS):
        _vals = df[_prop].values
        _ax.hist(_vals, bins=20, color="lightgray", edgecolor="k", alpha=0.7)
        _ymax = _ax.get_ylim()[1]
        for _fam, _color in FAMILY_COLORS.items():
            _mask = df["family"] == _fam
            if _mask.any():
                _ax.vlines(df.loc[_mask, _prop], 0, _ymax * 0.05, color=_color, alpha=0.9, lw=1.5)
        _ax.axvline(ref_row[_prop], color="red", linestyle="--", lw=1.5,
                    label=f"{REFERENCE} = {ref_row[_prop]:.3g}")
        _ax.set_title(_prop)
        _ax.set_xlabel(_prop)
        _ax.set_ylabel("count")
        _ax.legend(fontsize=8, loc="upper right")

    _handles = [plt.Line2D([0], [0], color=_c, lw=2) for _c in FAMILY_COLORS.values()]
    _fig.legend(_handles, list(FAMILY_COLORS.keys()), loc="lower center",
                ncol=len(FAMILY_COLORS), bbox_to_anchor=(0.5, -0.01), frameon=False)
    _fig.suptitle("Property distributions across 70 compositions — histogram + family rug + POPC100 reference",
                  y=1.0, fontsize=12)
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "03_univariate.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "03_univariate.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(PROPS, df, mo, np, pd):
    def _mad_z(x):
        _med = np.median(x)
        _mad = np.median(np.abs(x - _med))
        return 0.6745 * (x - _med) / (_mad if _mad > 0 else 1.0)

    _summary = []
    _outlier_rows = []
    for _prop in PROPS:
        _v = df[_prop].values
        _z = _mad_z(_v)
        _out_mask = np.abs(_z) > 3
        _summary.append({
            "property": _prop,
            "mean": _v.mean(),
            "median": np.median(_v),
            "std": _v.std(ddof=1),
            "min": _v.min(),
            "max": _v.max(),
            "IQR": np.percentile(_v, 75) - np.percentile(_v, 25),
            "n_outliers (|z_MAD|>3)": int(_out_mask.sum()),
        })
        for _idx in np.where(_out_mask)[0]:
            _outlier_rows.append({
                "composition": df.iloc[_idx]["composition"],
                "property": _prop,
                "value": _v[_idx],
                "z_MAD": _z[_idx],
            })

    _summary_df = pd.DataFrame(_summary)
    _outlier_df = (
        pd.DataFrame(_outlier_rows).sort_values("z_MAD", key=np.abs, ascending=False)
        if _outlier_rows else pd.DataFrame(columns=["composition", "property", "value", "z_MAD"])
    )

    _total_outliers = int(_summary_df["n_outliers (|z_MAD|>3)"].sum())
    _worst_prop = _summary_df.loc[_summary_df["n_outliers (|z_MAD|>3)"].idxmax(), "property"]

    mo.vstack([
        mo.md("**Summary statistics (all 70 systems)**"),
        mo.as_html(_summary_df.round(4)),
        mo.md(f"**Outliers flagged by MAD (|z| > 3)** — {_total_outliers} total, most in `{_worst_prop}`"),
        mo.as_html(_outlier_df.round(4)) if not _outlier_df.empty else mo.md("*(none)*"),
        mo.callout(mo.md(
            f"`bending_modulus` has the most outliers ({_summary_df.loc[_summary_df['property']=='bending_modulus', 'n_outliers (|z_MAD|>3)'].iloc[0]} flagged) — "
            "consistent with its known measurement noise. Other properties are well-behaved."
        ), kind="info") if "bending_modulus" in PROPS else mo.md(""),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 — Pairwise Structure

    ### 4a — Correlation matrices (Pearson + Spearman)

    High correlations between properties may indicate redundant learning targets for the GNN.
    `thickness` and `thickness_std` are expected to correlate; `diffusivity` and `lipid_packing`
    may anti-correlate (denser membranes diffuse more slowly).
    """)
    return


@app.cell
def _(FIG_DIR, PROPS, df, mo, np, pd, plt):
    _pearson = df[PROPS].corr(method="pearson")
    _spearman = df[PROPS].corr(method="spearman")

    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6))
    for _ax, _mat, _title in zip(_axes, [_pearson, _spearman], ["Pearson", "Spearman"]):
        _im = _ax.imshow(_mat, cmap="RdBu_r", vmin=-1, vmax=1)
        _ax.set_xticks(range(len(PROPS)))
        _ax.set_xticklabels(PROPS, rotation=45, ha="right")
        _ax.set_yticks(range(len(PROPS)))
        _ax.set_yticklabels(PROPS)
        _ax.set_title(f"{_title} correlation matrix")
        for _i in range(len(PROPS)):
            for _j in range(len(PROPS)):
                _ax.text(_j, _i, f"{_mat.iloc[_i, _j]:.2f}", ha="center", va="center",
                         color="white" if abs(_mat.iloc[_i, _j]) > 0.5 else "black", fontsize=8)
        plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "04a_correlations.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "04a_correlations.pdf", bbox_inches="tight")

    _top_pairs = []
    for _i, _pi in enumerate(PROPS):
        for _j, _pj in enumerate(PROPS):
            if _j > _i:
                _top_pairs.append({"prop_a": _pi, "prop_b": _pj,
                                   "pearson_r": round(_pearson.iloc[_i, _j], 3),
                                   "spearman_rho": round(_spearman.iloc[_i, _j], 3)})
    _pairs_df = pd.DataFrame(_top_pairs).sort_values("pearson_r", key=np.abs, ascending=False)

    mo.vstack([
        _fig,
        mo.md("**Pairwise correlations (|r| ranked)**"),
        mo.as_html(_pairs_df.head(10)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4b — Pairplot, coloured by partner family

    Full 8×8 grid: diagonal = histogram, off-diagonal = scatter coloured by partner family.
    Patterns across the diagonal reveal which families drive property variance.
    """)
    return


@app.cell
def _(FAMILY_COLORS, FIG_DIR, PROPS, df, plt):
    _n = len(PROPS)
    _fig, _axes = plt.subplots(_n, _n, figsize=(20, 20))
    for _i, _py in enumerate(PROPS):
        for _j, _px in enumerate(PROPS):
            _ax = _axes[_i, _j]
            if _i == _j:
                _ax.hist(df[_px], bins=15, color="lightgray", edgecolor="k")
            else:
                for _fam, _color in FAMILY_COLORS.items():
                    _mask = df["family"] == _fam
                    if _mask.any():
                        _ax.scatter(df.loc[_mask, _px], df.loc[_mask, _py],
                                    c=[_color], s=15, alpha=0.7, edgecolor="none")
            if _i == _n - 1:
                _ax.set_xlabel(_px, fontsize=9)
            else:
                _ax.set_xticklabels([])
            if _j == 0:
                _ax.set_ylabel(_py, fontsize=9)
            else:
                _ax.set_yticklabels([])
            _ax.tick_params(labelsize=7)

    _handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_c, markersize=8)
                for _c in FAMILY_COLORS.values()]
    _fig.legend(_handles, list(FAMILY_COLORS.keys()), loc="upper center",
                ncol=len(FAMILY_COLORS), bbox_to_anchor=(0.5, 1.0), frameon=False)
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "04b_pairplot.png", dpi=130, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "04b_pairplot.pdf", bbox_inches="tight")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4c — Helfrich Elastic-Theory Sanity Check

    Helfrich theory predicts `κ ≈ K_A · d² / 24` (thickness `d`, area compressibility `K_A`,
    bending modulus `κ`). `compressibility` here is a thickness-fluctuation proxy for K_A,
    not the canonical area-compressibility modulus — but a positive correlation between
    `compressibility · thickness²` and `bending_modulus` would still confirm that the
    bending fits are not pure noise.
    """)
    return


@app.cell
def _(FAMILY_COLORS, FIG_DIR, df, mo, plt, stats):
    _x = df["compressibility"] * df["thickness"] ** 2
    _y = df["bending_modulus"]
    _r_pear, _p_pear = stats.pearsonr(_x, _y)
    _r_spear, _p_spear = stats.spearmanr(_x, _y)

    _fig, _ax = plt.subplots(figsize=(7, 5))
    for _fam, _color in FAMILY_COLORS.items():
        _m = df["family"] == _fam
        _ax.scatter(_x[_m], _y[_m], c=[_color], s=40, alpha=0.8, label=_fam,
                    edgecolor="k", lw=0.3)
    _ax.set_xlabel("compressibility · thickness²  [Å⁵ / kT]")
    _ax.set_ylabel("bending_modulus  [kT / Å³]")
    _ax.set_title("Scatter: compressibility·thickness² vs bending_modulus")
    _ax.legend(fontsize=8, loc="best")
    _r_text = f"Pearson r = {_r_pear:.2f} (p = {_p_pear:.1e}) | Spearman ρ = {_r_spear:.2f} (p = {_p_spear:.1e})"
    _ax.text(0.02, 0.97, _r_text, transform=_ax.transAxes, fontsize=8,
             va="top", ha="left", bbox=dict(boxstyle="round", fc="white", alpha=0.7))
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "04c_helfrich.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "04c_helfrich.pdf", bbox_inches="tight")

    _kind = "success" if _r_pear > 0.3 else "warn"
    mo.vstack([
        _fig,
        mo.callout(mo.md(
            f"Pearson r = **{_r_pear:.2f}** (p = {_p_pear:.1e}), Spearman ρ = **{_r_spear:.2f}** — "
            f"{'positive correlation consistent with Helfrich theory.' if _r_pear > 0.3 else 'weak or no correlation; bending fits may be dominated by noise.'}"
        ), kind=_kind),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 — Composition Titration Curves

    For each non-POPC partner lipid, every property plotted vs partner mole fraction.
    POPC100 is anchored at 0% and a pure partner (if simulated) appears at 100%.
    This reveals which "property knobs" each partner lipid controls.
    """)
    return


@app.cell
def _(FIG_DIR, PROPS, REFERENCE, df, mo, plt):
    _partners_with_data = sorted(set(df.loc[df["partner_frac"] > 0, "partner"]))
    _partner_colors = dict(zip(
        _partners_with_data,
        plt.get_cmap("tab10").colors[:len(_partners_with_data)]
    ))
    _ref_row = df[df["composition"] == REFERENCE].iloc[0]

    _fig, _axes = plt.subplots(4, 2, figsize=(13, 16))
    for _ax, _prop in zip(_axes.flat, PROPS):
        _ax.axhline(_ref_row[_prop], color="gray", linestyle=":", lw=1, label="POPC100")
        for _part in _partners_with_data:
            _sub = df[(df["partner"] == _part) | (df["composition"] == REFERENCE)]
            _sub = _sub.sort_values("partner_frac")
            _ax.plot(_sub["partner_frac"], _sub[_prop], "o-",
                     color=_partner_colors[_part], label=_part, markersize=5, lw=1.2)
        _ax.set_title(_prop)
        _ax.set_xlabel("partner mole fraction (%)")
        _ax.set_ylabel(_prop)
        _ax.grid(alpha=0.3)

    _handles, _labels = _axes[0, 0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc="lower center", ncol=min(len(_labels), 6),
                bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    _fig.suptitle("Titration curves — property vs partner-lipid mole fraction", fontsize=14)
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "05_titration.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "05_titration.pdf", bbox_inches="tight")
    mo.vstack(
        [_fig]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 — Dimensionality Reduction & Clustering

    All 8 properties z-scored, then projected into 2D via PCA, UMAP (if available),
    and clustered hierarchically. Reveals whether partner-family taxonomy aligns with
    the intrinsic property-space structure.
    """)
    return


@app.cell
def _(PCA, StandardScaler, TIER_B_PROPS, df, mo):
    X = df[TIER_B_PROPS].values
    X_z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(len(TIER_B_PROPS), X_z.shape[0]), random_state=42)
    Z_pca = pca.fit_transform(X_z)

    mo.md(f"""
    **Property PCA** — {X_z.shape[0]} systems × {X_z.shape[1]} z-scored properties
    PC1+PC2 cumulative variance: {(pca.explained_variance_ratio_[:2].sum()*100):.1f}%
    """)
    return X_z, Z_pca, pca


@app.cell
def _(FAMILY_COLORS, FIG_DIR, TIER_B_PROPS, Z_pca, df, np, pca, plt):
    _explained = pca.explained_variance_ratio_
    _fig = plt.figure(figsize=(15, 5))

    # Scree plot
    _ax1 = _fig.add_subplot(1, 3, 1)
    _ax1.bar(range(1, len(_explained) + 1), _explained * 100, color="steelblue")
    _ax1.plot(range(1, len(_explained) + 1), np.cumsum(_explained) * 100, "ro-", lw=1.5)
    _ax1.set_xlabel("PC")
    _ax1.set_ylabel("variance explained (%)")
    _ax1.set_title(f"Scree — PC1+PC2 = {(_explained[0]+_explained[1])*100:.1f}%")
    _ax1.grid(alpha=0.3)

    # PC1 vs PC2
    _ax2 = _fig.add_subplot(1, 3, 2)
    for _fam, _color in FAMILY_COLORS.items():
        _m = df["family"] == _fam
        if _m.any():
            _ax2.scatter(Z_pca[_m, 0], Z_pca[_m, 1], c=[_color], s=40, alpha=0.8,
                         label=_fam, edgecolor="k", lw=0.3)
    _ax2.set_xlabel(f"PC1 ({_explained[0]*100:.1f}%)")
    _ax2.set_ylabel(f"PC2 ({_explained[1]*100:.1f}%)")
    _ax2.set_title("PCA: PC1 vs PC2, coloured by partner family")
    _ax2.legend(fontsize=8, loc="best")
    _ax2.grid(alpha=0.3)

    # Loadings biplot
    _ax3 = _fig.add_subplot(1, 3, 3)
    for _i, _prop in enumerate(TIER_B_PROPS):
        _ax3.arrow(0, 0, pca.components_[0, _i], pca.components_[1, _i],
                   head_width=0.03, color="k", alpha=0.7)
        _ax3.text(pca.components_[0, _i] * 1.15, pca.components_[1, _i] * 1.15,
                  _prop, fontsize=9, ha="center")
    _ax3.axhline(0, color="gray", lw=0.5)
    _ax3.axvline(0, color="gray", lw=0.5)
    _ax3.set_xlabel("PC1 loading")
    _ax3.set_ylabel("PC2 loading")
    _ax3.set_title("PCA loadings biplot")
    _ax3.set_xlim(-1, 1)
    _ax3.set_ylim(-1, 1)
    _ax3.set_aspect("equal")
    _ax3.grid(alpha=0.3)

    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "06a_pca.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "06a_pca.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(FAMILY_COLORS, FIG_DIR, Z_prop, df, ev_prop, plt):
    _fig, _ax = plt.subplots(figsize=(8, 6))

    # Plot the scatter points for each lipid family
    for _fam, _color in FAMILY_COLORS.items():
        _m = df["family"] == _fam
        if _m.any():
            _ax.scatter(Z_prop[_m, 0], Z_prop[_m, 1], c=[_color], s=50, edgecolor="k",
                        lw=0.6, alpha=0.9, label=_fam, zorder=5)

    # Set labels based on explained variance for properties
    _ax.set_xlabel(f"PC1 ({ev_prop[0]*100:.1f}%)")
    _ax.set_ylabel(f"PC2 ({ev_prop[1]*100:.1f}%)")
    _ax.set_title("Properties-space PCA (colored by lipid type)", fontsize=13)

    # Add legend and grid
    _ax.legend(fontsize=10, loc="best")
    _ax.grid(alpha=0.25)

    _fig.tight_layout()

    # Save the updated figure with a new name
    _fig.savefig(FIG_DIR / "10_properties_pca_family.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "10_properties_pca_family.pdf", bbox_inches="tight")

    # Display the figure inline (if using Jupyter)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6b — UMAP (n_neighbors = 15 and 8)

    UMAP projects the z-scored 8-D property vectors into 2D with different neighbourhood
    scales. Shown only when `umap-learn` is installed.
    """)
    return


@app.cell
def _(FAMILY_COLORS, FIG_DIR, HAS_UMAP, X_z, df, mo, plt, umap_lib):
    if HAS_UMAP:
        _fig, _axes = plt.subplots(1, 2, figsize=(13, 5))
        for _ax, _n_neigh in zip(_axes, [15, 8]):
            _reducer = umap_lib.UMAP(n_components=2, n_neighbors=_n_neigh,
                                     min_dist=0.1, random_state=0)
            _Z_u = _reducer.fit_transform(X_z)
            for _fam, _color in FAMILY_COLORS.items():
                _m = df["family"] == _fam
                if _m.any():
                    _ax.scatter(_Z_u[_m, 0], _Z_u[_m, 1], c=[_color], s=40, alpha=0.8,
                                label=_fam, edgecolor="k", lw=0.3)
            _ax.set_title(f"UMAP (n_neighbors={_n_neigh}): UMAP1 vs UMAP2")
            _ax.set_xlabel("UMAP1")
            _ax.set_ylabel("UMAP2")
            if _n_neigh == 15:
                _ax.legend(fontsize=8, loc="best")
        _fig.tight_layout()
        _fig.savefig(FIG_DIR / "06b_umap.png", dpi=150, bbox_inches="tight")
        _fig.savefig(FIG_DIR / "06b_umap.pdf", bbox_inches="tight")
        _out = _fig
    else:
        _out = mo.callout(mo.md("`umap-learn` not installed — UMAP skipped. `pip install umap-learn` to enable."), kind="info")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6c — Hierarchical Clustering (Ward)

    Dendrogram of z-scored property vectors. Cut at k = 4 clusters and compared
    against the partner-family taxonomy. If the four clusters correspond to the
    PC / PE / PS / sterol families, the property space captures the biochemical taxonomy.
    """)
    return


@app.cell
def _(FIG_DIR, X_z, df, hierarchy, mo, pd, pdist, plt):
    _dist = pdist(X_z, metric="euclidean")
    _Z_link = hierarchy.linkage(_dist, method="ward")

    _fig, _ax = plt.subplots(figsize=(16, 6))
    _labels = (df["composition"] + "  [" + df["family"] + "]").tolist()
    hierarchy.dendrogram(_Z_link, labels=_labels, leaf_rotation=90,
                         leaf_font_size=7,
                         color_threshold=0.7 * _Z_link[:, 2].max(),
                         ax=_ax)
    _ax.set_title("Hierarchical clustering (Ward, z-scored properties): composition × family label")
    _ax.set_ylabel("Ward distance")
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "06c_dendrogram.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "06c_dendrogram.pdf", bbox_inches="tight")

    _cluster_k4 = hierarchy.fcluster(_Z_link, t=4, criterion="maxclust")
    _contingency = pd.crosstab(_cluster_k4, df["family"])
    _contingency.index.name = "cluster_k4"

    mo.vstack([
        _fig,
        mo.md("**Cluster (k=4) × family contingency** — rows are Ward clusters, columns are partner families"),
        mo.as_html(_contingency),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6d — Outlier Diagnostics

    Mahalanobis distance and PCA reconstruction error (top-2 PCs) in the z-scored
    8-D space. High values flag compositions whose property profile is anomalous
    relative to the bulk — candidates for exclusion or targeted simulation review.
    """)
    return


@app.cell
def _(X_z, Z_pca, df, mo, np, pca, pd):
    _mu = X_z.mean(axis=0)
    _cov = np.cov(X_z, rowvar=False) + 1e-6 * np.eye(X_z.shape[1])
    _inv_cov = np.linalg.inv(_cov)
    _diff = X_z - _mu
    _mahal = np.sqrt(np.einsum("ij,jk,ik->i", _diff, _inv_cov, _diff))

    _recon = Z_pca[:, :2] @ pca.components_[:2]
    _recon_err = np.linalg.norm(X_z - _recon, axis=1)

    _out_df = pd.DataFrame({
        "composition": df["composition"],
        "family": df["family"],
        "mahalanobis": _mahal.round(3),
        "pca2_recon_err": _recon_err.round(3),
    }).sort_values("mahalanobis", ascending=False)

    mo.vstack([
        mo.md("**Top 10 by Mahalanobis distance** (largest = most anomalous)"),
        mo.as_html(_out_df.head(10)),
        mo.md("**Top 10 by PCA(2-PC) reconstruction error**"),
        mo.as_html(_out_df.sort_values("pca2_recon_err", ascending=False).head(10)),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 — Time-Series Quality

    Per-frame `raw_dict` series check stationarity and estimate intrinsic noise.
    `bending_modulus` is excluded — its raw entry is the (q, spectrum) pair.
    Frame spacing is `dt = 1.5 ns`; series cover ≈ 1 µs of production.

    ### 7a — Representative trajectory (POPC100 reference)
    """)
    return


@app.cell
def _(FIG_DIR, PROPS, RAW, REFERENCE, np, plt):
    _TS_PROPS = [_p for _p in PROPS if _p != "bending_modulus"]
    _DT_NS = 1.5

    def _get_series(_comp, _prop):
        _v = np.asarray(RAW[_comp][_prop])
        return _v if _v.ndim == 1 else None

    _fig, _axes = plt.subplots(len(_TS_PROPS), 1,
                               figsize=(11, 1.6 * len(_TS_PROPS)), sharex=True)
    for _ax, _prop in zip(_axes, _TS_PROPS):
        _s = _get_series(REFERENCE, _prop)
        if _s is None:
            _ax.set_ylabel(_prop)
            continue
        _t = np.arange(len(_s)) * _DT_NS / 1000
        _ax.plot(_t, _s, lw=0.8, color="steelblue")
        _ax.axhline(np.mean(_s), color="red", linestyle="--", lw=0.7,
                    label=f"mean = {np.mean(_s):.3g}")
        _ax.set_ylabel(_prop, fontsize=9)
        _ax.legend(fontsize=7, loc="upper right")
        _ax.grid(alpha=0.3)
    _axes[-1].set_xlabel("time (µs)")
    _fig.suptitle(f"Per-frame property series — {REFERENCE}", fontsize=12)
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "07a_timeseries_popc.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "07a_timeseries_popc.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(PROPS, RAW, df, mo, np, pd, stats):
    _TS_PROPS = [_p for _p in PROPS if _p != "bending_modulus"]

    def _get_series(_comp, _prop):
        _v = np.asarray(RAW[_comp][_prop])
        return _v if _v.ndim == 1 else None

    def _autocorr_time(_x, _max_lag=200):
        _x = np.asarray(_x) - np.mean(_x)
        _var = np.var(_x)
        if _var <= 0:
            return np.nan
        _n = len(_x)
        _max_lag = min(_max_lag, _n // 4)
        _acf = []
        for _k in range(1, _max_lag + 1):
            _c = np.dot(_x[:-_k], _x[_k:]) / (_var * (_n - _k))
            if _c < 0:
                break
            _acf.append(_c)
        return 1 + 2 * sum(_acf)

    _quality_rows = []
    for _prop in _TS_PROPS:
        _within_stds = []
        _drifts_rel = []
        _tau_list = []
        for _comp in df["composition"]:
            _s = _get_series(_comp, _prop)
            if _s is None or len(_s) < 20:
                continue
            _t = np.arange(len(_s))
            _slope, *_ = stats.linregress(_t, _s)
            _drifts_rel.append(abs(_slope) * len(_s) / (np.std(_s) + 1e-12))
            _within_stds.append(np.std(_s))
            _tau_list.append(_autocorr_time(_s))

        _within_std = float(np.mean(_within_stds))
        _between_std = float(df[_prop].std(ddof=1))
        _tau_mean = float(np.nanmean(_tau_list))
        _s_last = _s  # last series from the loop (for n_eff)
        _n_eff = (len(_s_last) / _tau_mean) if _tau_mean > 0 else float("nan")
        _quality_rows.append({
            "property": _prop,
            "within_std": round(_within_std, 5),
            "between_std": round(_between_std, 5),
            "snr (between/within)": round(_between_std / _within_std, 2) if _within_std > 0 else float("nan"),
            "autocorr_tau (frames)": round(_tau_mean, 1),
            "n_eff per system": round(_n_eff, 1),
            "mean |drift|·N/std": round(float(np.mean(_drifts_rel)), 3),
        })

    q_df = pd.DataFrame(_quality_rows)
    mo.md("**Time-series quality per property** — SNR = between-system std / within-system std")
    return (q_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 7b / 7c — Noise Floor

    Within-system std (frame-to-frame noise) vs between-system std (signal across the 70 compositions).
    High SNR = property is well-resolved across the dataset. Low SNR = most variance is per-frame noise,
    not composition-driven signal.
    """)
    return


@app.cell
def _(FIG_DIR, mo, np, plt, q_df):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _xpos = np.arange(len(q_df))
    _w = 0.4
    _ax.bar(_xpos - _w / 2, q_df["within_std"], _w,
            label="within-system std (frame noise)", color="lightcoral")
    _ax.bar(_xpos + _w / 2, q_df["between_std"], _w,
            label="between-system std (signal)", color="steelblue")
    _ax.set_yscale("log")
    _ax.set_xticks(_xpos)
    _ax.set_xticklabels(q_df["property"], rotation=30, ha="right")
    _ax.set_ylabel("std (property units, log scale)")
    _ax.set_title("Noise floor: within-system (frame noise) vs between-system (signal) std")
    for _i, _snr in enumerate(q_df["snr (between/within)"]):
        if not np.isnan(_snr):
            _ax.text(_i, max(q_df.loc[_i, "within_std"], q_df.loc[_i, "between_std"]) * 1.15,
                     f"SNR={_snr:.1f}", ha="center", fontsize=8)
    _ax.legend()
    _ax.grid(alpha=0.3, axis="y")
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "07b_noise_floor.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "07b_noise_floor.pdf", bbox_inches="tight")

    _low_snr = q_df.loc[q_df["snr (between/within)"] < 3, "property"].tolist()
    mo.vstack([
        mo.as_html(q_df),
        _fig,
        mo.callout(mo.md(
            f"Properties with SNR < 3 (potentially noise-dominated): {_low_snr if _low_snr else 'none'}. "
            "High SNR properties provide cleaner regression targets for the GNN."
        ), kind="info") if True else mo.md(""),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 — Train / Val / Test Split Audit

    The Tier A/B pipeline preprocesses graphs into `processed/{train,val,test}/chunk_*.pt`.
    We list which compositions land in each split and check whether property distributions
    are compatible across splits (KS test vs train).
    """)
    return


@app.cell
def _(CONFIG, HAS_TORCH, Path, SPLITS, mo, torch):
    if HAS_TORCH:
        _CHUNKS_DIR = Path(CONFIG.paths.chunks_dir)
        split_compositions = {_s: set() for _s in SPLITS}
        for _s in SPLITS:
            _sd = _CHUNKS_DIR / _s
            if not _sd.exists():
                continue
            for _chunk in sorted(_sd.glob("chunk_*.pt")):
                _graphs = torch.load(_chunk, weights_only=False)
                for _g in _graphs:
                    _comp = getattr(_g, "composition", None)
                    if _comp is not None:
                        split_compositions[_s].add(_comp)
        _counts = {_s: len(split_compositions[_s]) for _s in SPLITS}
        _status = mo.md(f"Split composition counts: {_counts}")
    else:
        split_compositions = {_s: set() for _s in SPLITS}
        _status = mo.callout(mo.md("`torch` not available — split audit skipped. Run on HPC or install torch."), kind="warn")
    _status
    return (split_compositions,)


@app.cell
def _(SPLITS, df, split_compositions):
    def _split_of(_comp):
        _hits = [_s for _s in SPLITS if _comp in split_compositions[_s]]
        if not _hits:
            return None
        return _hits[0] if len(_hits) == 1 else "/".join(_hits)

    df_with_split = df.copy()
    df_with_split["split"] = df_with_split["composition"].map(_split_of)
    return (df_with_split,)


@app.cell
def _(PROPS, SPLITS, df_with_split, mo, pd, stats):
    _present = [_s for _s in SPLITS if (df_with_split["split"] == _s).any()]

    if "train" in _present:
        _rows = []
        for _prop in PROPS:
            _train_v = df_with_split.loc[df_with_split["split"] == "train", _prop].values
            for _s in _present:
                if _s == "train":
                    continue
                _other_v = df_with_split.loc[df_with_split["split"] == _s, _prop].values
                if len(_other_v) < 2:
                    continue
                _ks_stat, _ks_p = stats.ks_2samp(_train_v, _other_v)
                _rows.append({
                    "property": _prop, "split": _s,
                    "n_train": len(_train_v), "n_other": len(_other_v),
                    "mean_train": round(_train_v.mean(), 4),
                    "mean_other": round(_other_v.mean(), 4),
                    "KS_stat": round(_ks_stat, 3), "KS_p": round(_ks_p, 4),
                })
        _ks_df = pd.DataFrame(_rows)
        _sig = _ks_df.loc[_ks_df["KS_p"] < 0.05]
        _kind = "warn" if len(_sig) > 0 else "success"
        _out = mo.vstack([
            mo.md("**KS test: each split vs train** (p < 0.05 signals distribution mismatch)"),
            mo.as_html(_ks_df),
            mo.callout(mo.md(
                f"{len(_sig)} property×split pairs with KS p < 0.05 — {'distribution mismatch present, check split strategy.' if len(_sig) > 0 else 'all splits look compatible with train.'}"
            ), kind=_kind),
        ])
    else:
        _out = mo.callout(mo.md("No split labels resolved — chunk files not found. KS test skipped."), kind="info")
    _out
    return


@app.cell
def _(FIG_DIR, PROPS, SPLITS, SPLIT_COLORS, df_with_split, mo, plt):
    _present = [_s for _s in SPLITS if (df_with_split["split"] == _s).any()]

    if _present:
        _fig, _axes = plt.subplots(4, 2, figsize=(12, 14))
        for _ax, _prop in zip(_axes.flat, PROPS):
            for _s in _present:
                _v = df_with_split.loc[df_with_split["split"] == _s, _prop].values
                if len(_v) == 0:
                    continue
                _ax.hist(_v, bins=15, alpha=0.5, label=f"{_s} (n={len(_v)})",
                         color=SPLIT_COLORS.get(_s, "gray"), edgecolor="k", lw=0.4)
            _ax.set_title(_prop)
            _ax.set_xlabel(_prop)
            _ax.set_ylabel("count")
            _ax.legend(fontsize=8)
        _fig.suptitle("Property distributions across splits", y=1.0, fontsize=14)
        _fig.tight_layout()
        _fig.savefig(FIG_DIR / "08_split_audit.png", dpi=150, bbox_inches="tight")
        _fig.savefig(FIG_DIR / "08_split_audit.pdf", bbox_inches="tight")
        _out = _fig
    else:
        _out = mo.callout(mo.md("Split figure skipped — no chunked splits present."), kind="info")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 — KDE Coverage Analysis

    Kernel density estimation in composition-PCA space reveals where the 70 simulated
    compositions cluster and where gaps exist. Low-density regions are candidates for
    new MD simulations — this converts the qualitative "add more DPPC/DOPC coverage"
    argument into a quantitative, ranked list.

    **Two spaces compared:**
    - **Composition-space (§9a)**: PCA of 10-dim mole-fraction vectors
    - **Property-space (§9b)**: PCA of the 6 Tier-B properties (z-scored)

    Each space shown at 3 KDE bandwidths (0.5×, 1×, 2× Scott's rule) for robustness.

    **Sanity check (§9c)**: Stage 5d worst-MAE compositions (★) should fall in
    low-density regions — if they do, the KDE correctly identifies training-coverage gaps.
    """)
    return


@app.cell
def _(
    LIPIDS,
    PCA,
    STAGE5D_WORST,
    StandardScaler,
    TIER_B_PROPS,
    df_with_split,
    gaussian_kde,
    mo,
    np,
):
    # Composition matrix (n_systems × n_lipids, mole fractions)
    X_comp = np.zeros((len(df_with_split), len(LIPIDS)))
    for _idx, _row in df_with_split.iterrows():
        _a, _fa = _row["lipid_a"], _row["frac_a"]
        _b, _fb = _row["lipid_b"], _row["frac_b"]
        if _a in LIPIDS:
            X_comp[_idx, LIPIDS.index(_a)] = _fa / 100.0
        if _b is not None and _b in LIPIDS:
            X_comp[_idx, LIPIDS.index(_b)] = _fb / 100.0

    scaler_comp = StandardScaler()
    _X_comp_z = scaler_comp.fit_transform(X_comp)
    pca_comp = PCA(n_components=2, random_state=42)
    Z_comp = pca_comp.fit_transform(_X_comp_z)
    ev_comp = pca_comp.explained_variance_ratio_

    Y_prop = df_with_split[TIER_B_PROPS].values
    _scaler_prop = StandardScaler()
    _Y_prop_z = _scaler_prop.fit_transform(Y_prop)
    _pca_prop = PCA(n_components=2, random_state=42)
    Z_prop = _pca_prop.fit_transform(_Y_prop_z)
    ev_prop = _pca_prop.explained_variance_ratio_

    kde_comp_default = gaussian_kde(Z_comp.T)
    _bw = kde_comp_default.factor
    kde_comp_half = gaussian_kde(Z_comp.T, bw_method=_bw * 0.5)
    kde_comp_double = gaussian_kde(Z_comp.T, bw_method=_bw * 2.0)

    mo.md(f"""
    **Composition PCA**: PC1 = {ev_comp[0]*100:.1f}%, PC2 = {ev_comp[1]*100:.1f}%, cumulative = {ev_comp.sum()*100:.1f}%
    **Property PCA (Tier B, 6 props)**: PC1 = {ev_prop[0]*100:.1f}%, PC2 = {ev_prop[1]*100:.1f}%, cumulative = {ev_prop.sum()*100:.1f}%
    Stage 5d worst-MAE compositions to track: {STAGE5D_WORST}
    """)
    return (
        X_comp,
        Z_comp,
        Z_prop,
        ev_comp,
        ev_prop,
        kde_comp_default,
        kde_comp_double,
        kde_comp_half,
        pca_comp,
        scaler_comp,
    )


@app.cell
def _(FIG_DIR, SPLIT_COLORS, Z_comp, df_with_split, ev_comp, plt):
    _fig, _ax = plt.subplots(figsize=(8, 6))

    # Plot the scatter points for each split
    for _split, _color in SPLIT_COLORS.items():
        _m = df_with_split["split"] == _split
        _ax.scatter(Z_comp[_m, 0], Z_comp[_m, 1], c=_color, s=50, edgecolor="k",
                    lw=0.6, alpha=0.9, label=_split, zorder=5)

    # Set labels based on explained variance
    _ax.set_xlabel(f"PC1 ({ev_comp[0]*100:.1f}%)")
    _ax.set_ylabel(f"PC2 ({ev_comp[1]*100:.1f}%)")
    _ax.set_title("Composition-space PCA", fontsize=13)

    # Add legend and grid
    _ax.legend(fontsize=10, loc="best")
    _ax.grid(alpha=0.25)

    _fig.tight_layout()

    # Save the updated figure with a new name
    _fig.savefig(FIG_DIR / "09_composition_pca_clean.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "09_composition_pca_clean.pdf", bbox_inches="tight")

    # Display the figure inline (if using Jupyter)
    _fig
    return


@app.cell
def _(FAMILY_COLORS, FIG_DIR, Z_comp, df_with_split, ev_comp, plt):
    _fig_comp, _ax_comp = plt.subplots(figsize=(8, 6))

    # Plot the scatter points for each lipid family (Composition)
    for _fam, _color in FAMILY_COLORS.items():
        _m = df_with_split["family"] == _fam
        if _m.any():
            _ax_comp.scatter(Z_comp[_m, 0], Z_comp[_m, 1], c=[_color], s=50, edgecolor="k",
                             lw=0.6, alpha=0.9, label=_fam, zorder=5)

    # Set labels based on explained variance for composition
    _ax_comp.set_xlabel(f"PC1 ({ev_comp[0]*100:.1f}%)")
    _ax_comp.set_ylabel(f"PC2 ({ev_comp[1]*100:.1f}%)")
    _ax_comp.set_title("Composition-space PCA (colored by lipid type)", fontsize=13)

    # Add legend and grid
    _ax_comp.legend(fontsize=10, loc="best")
    _ax_comp.grid(alpha=0.25)

    _fig_comp.tight_layout()

    # Save the composition figure
    _fig_comp.savefig(FIG_DIR / "09_composition_pca_family.png", dpi=150, bbox_inches="tight")
    _fig_comp.savefig(FIG_DIR / "09_composition_pca_family.pdf", bbox_inches="tight")

    # Display the composition figure inline (if using Jupyter)
    _fig_comp
    return


@app.cell
def _(FIG_DIR, SPLIT_COLORS, Z_prop, df_with_split, ev_prop, plt):
    _fig_prop, _ax_prop = plt.subplots(figsize=(8, 6))

    # Plot the scatter points for each split using the same modified colors
    for _split, _color in SPLIT_COLORS.items():
        _m = df_with_split["split"] == _split
        _ax_prop.scatter(Z_prop[_m, 0], Z_prop[_m, 1], c=_color, s=50, edgecolor="k",
                         lw=0.6, alpha=0.9, label=_split, zorder=5)

    # Set labels based on explained variance for properties
    _ax_prop.set_xlabel(f"PC1 ({ev_prop[0]*100:.1f}%)")
    _ax_prop.set_ylabel(f"PC2 ({ev_prop[1]*100:.1f}%)")
    _ax_prop.set_title("Properties-space PCA", fontsize=13)

    # Add legend and grid
    _ax_prop.legend(fontsize=10, loc="best")
    _ax_prop.grid(alpha=0.25)

    _fig_prop.tight_layout()

    # Save the properties figure
    _fig_prop.savefig(FIG_DIR / "10_properties_pca_clean.png", dpi=150, bbox_inches="tight")
    _fig_prop.savefig(FIG_DIR / "10_properties_pca_clean.pdf", bbox_inches="tight")

    # Display the figure inline (if using Jupyter)
    _fig_prop
    return


@app.cell
def _(
    FIG_DIR,
    SPLIT_COLORS,
    STAGE5D_WORST,
    Z_comp,
    df_with_split,
    ev_comp,
    kde_comp_default,
    kde_comp_double,
    kde_comp_half,
    np,
    plt,
):
    _pad = 0.8
    _x0, _x1 = Z_comp[:, 0].min() - _pad, Z_comp[:, 0].max() + _pad
    _y0, _y1 = Z_comp[:, 1].min() - _pad, Z_comp[:, 1].max() + _pad
    _xx, _yy = np.mgrid[_x0:_x1:200j, _y0:_y1:200j]
    _pos = np.vstack([_xx.ravel(), _yy.ravel()])

    _fig, _axes = plt.subplots(1, 3, figsize=(17, 5))
    _bw = kde_comp_default.factor
    for _ax, _kde, _label in zip(
        _axes,
        [kde_comp_half, kde_comp_default, kde_comp_double],
        [f"0.5× Scott", f"1× Scott (bw={_bw:.3f})", "2× Scott"],
    ):
        _density = _kde(_pos).reshape(200, 200)
        _cf = _ax.contourf(_xx, _yy, _density, levels=14, cmap="Blues", alpha=0.85)
        _ax.contour(_xx, _yy, _density, levels=14, colors="navy", linewidths=0.3, alpha=0.4)
        plt.colorbar(_cf, ax=_ax, label="KDE density", shrink=0.9)
        for _split, _color in SPLIT_COLORS.items():
            _m = df_with_split["split"] == _split
            _ax.scatter(Z_comp[_m, 0], Z_comp[_m, 1], c=_color, s=50, edgecolor="k",
                        lw=0.6, alpha=0.9, label=_split, zorder=5)
        _widx = df_with_split.index[df_with_split["composition"].isin(STAGE5D_WORST)].tolist()
        _ax.scatter(Z_comp[_widx, 0], Z_comp[_widx, 1],
                    marker="*", s=220, c="red", edgecolor="darkred", lw=0.7,
                    zorder=7, label="Stage 5d worst MAE")
        for _i in _widx:
            _ax.annotate(df_with_split.loc[_i, "composition"], (Z_comp[_i, 0], Z_comp[_i, 1]),
                         textcoords="offset points", xytext=(5, 4), fontsize=6.5, color="darkred")
        _ax.set_xlabel(f"PC1 ({ev_comp[0]*100:.1f}%)")
        _ax.set_ylabel(f"PC2 ({ev_comp[1]*100:.1f}%)")
        _ax.set_title(f"Composition KDE — {_label}")
        _ax.legend(fontsize=8, loc="upper right")
        _ax.grid(alpha=0.25)

    _fig.suptitle("Composition-space KDE: where do the 70 training compositions sit?", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "09_kde_composition_pca.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "09_kde_composition_pca.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(
    FIG_DIR,
    SPLIT_COLORS,
    STAGE5D_WORST,
    TIER_B_PROPS,
    Z_prop,
    df_with_split,
    ev_prop,
    gaussian_kde,
    np,
    plt,
):
    _kde_prop_default = gaussian_kde(Z_prop.T)
    _bw_p = _kde_prop_default.factor

    _py0, _py1 = Z_prop[:, 0].min() - 0.8, Z_prop[:, 0].max() + 0.8
    _qy0, _qy1 = Z_prop[:, 1].min() - 0.8, Z_prop[:, 1].max() + 0.8
    _gx, _gy = np.mgrid[_py0:_py1:200j, _qy0:_qy1:200j]
    _ppos = np.vstack([_gx.ravel(), _gy.ravel()])

    _fig, _axes = plt.subplots(1, 3, figsize=(17, 5))
    for _ax, _bw_factor, _label in zip(
        _axes,
        [0.5, 1.0, 2.0],
        [f"0.5× Scott", f"1× Scott (bw={_bw_p:.3f})", "2× Scott"],
    ):
        _kde_p = (_kde_prop_default if _bw_factor == 1.0
                  else gaussian_kde(Z_prop.T, bw_method=_bw_p * _bw_factor))
        _dens_p = _kde_p(_ppos).reshape(200, 200)
        _cf = _ax.contourf(_gx, _gy, _dens_p, levels=14, cmap="Purples", alpha=0.85)
        _ax.contour(_gx, _gy, _dens_p, levels=14, colors="purple", linewidths=0.3, alpha=0.4)
        plt.colorbar(_cf, ax=_ax, label="KDE density", shrink=0.9)
        for _split, _color in SPLIT_COLORS.items():
            _m = df_with_split["split"] == _split
            _ax.scatter(Z_prop[_m, 0], Z_prop[_m, 1], c=_color, s=50, edgecolor="k",
                        lw=0.6, alpha=0.9, label=_split, zorder=5)
        _widx = df_with_split.index[df_with_split["composition"].isin(STAGE5D_WORST)].tolist()
        _ax.scatter(Z_prop[_widx, 0], Z_prop[_widx, 1],
                    marker="*", s=220, c="red", edgecolor="darkred", lw=0.7,
                    zorder=7, label="Stage 5d worst MAE")
        for _i in _widx:
            _ax.annotate(df_with_split.loc[_i, "composition"], (Z_prop[_i, 0], Z_prop[_i, 1]),
                         textcoords="offset points", xytext=(5, 4), fontsize=6.5, color="darkred")
        _ax.set_xlabel(f"PC1 ({ev_prop[0]*100:.1f}%)")
        _ax.set_ylabel(f"PC2 ({ev_prop[1]*100:.1f}%)")
        _ax.set_title(f"Property KDE (Tier B) — {_label}")
        _ax.legend(fontsize=8, loc="upper right")
        _ax.grid(alpha=0.25)

    _fig.suptitle("Property-space KDE  [" + ", ".join(TIER_B_PROPS) + "]", fontsize=13)
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "10_kde_property_pca.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "10_kde_property_pca.pdf", bbox_inches="tight")
    _fig
    return


@app.cell
def _(
    FIG_DIR,
    OUT_DIR,
    Patch,
    SPLIT_COLORS,
    STAGE5D_WORST,
    Z_comp,
    df_with_split,
    kde_comp_default,
    mo,
    np,
    pd,
    plt,
):
    _comp_densities = kde_comp_default(Z_comp.T)

    density_df = pd.DataFrame({
        "composition": df_with_split["composition"].values,
        "family": df_with_split["family"].values,
        "split": df_with_split["split"].values,
        "pc1": Z_comp[:, 0],
        "pc2": Z_comp[:, 1],
        "kde_density": _comp_densities,
    })
    density_df["density_rank"] = density_df["kde_density"].rank(ascending=True).astype(int)
    density_df = density_df.sort_values("density_rank").reset_index(drop=True)
    density_df.to_csv(OUT_DIR / "existing_densities.csv", index=False)

    _threshold_25 = np.percentile(_comp_densities, 25)
    _worst_dens = density_df[density_df["composition"].isin(STAGE5D_WORST)]
    _all_below = (_worst_dens["kde_density"] <= _threshold_25).all()

    _split_bar_colors = [SPLIT_COLORS.get(_s, "gray") for _s in density_df["split"]]
    _fig, _ax = plt.subplots(figsize=(11, 5))
    _ax.bar(range(len(density_df)), density_df["kde_density"],
            color=_split_bar_colors, edgecolor="k", lw=0.3, alpha=0.85)
    _ax.axhline(_threshold_25, color="red", linestyle="--", lw=1.5,
                label=f"25th percentile ({_threshold_25:.4f})")
    for _, _wr in _worst_dens.iterrows():
        _bar_x = int(_wr["density_rank"]) - 1
        _ax.annotate(_wr["composition"],
                     xy=(_bar_x, _wr["kde_density"]),
                     xytext=(_bar_x, _wr["kde_density"] + 0.003),
                     fontsize=6.5, color="darkred", ha="center",
                     arrowprops=dict(arrowstyle="->", color="darkred", lw=0.8))
    _ax.set_xlabel("composition (sorted by density, low → left = gap region)")
    _ax.set_ylabel("KDE density (composition-PCA, 1× Scott)")
    _ax.set_title("Per-composition density — low values indicate sparse training-coverage regions")
    _ax.grid(alpha=0.25, axis="y")
    _ax.legend(handles=[
        Patch(facecolor="steelblue", label="train"),
        Patch(facecolor="darkorange", label="val"),
        Patch(facecolor="tomato", label="test"),
        plt.Line2D([0], [0], color="red", linestyle="--", label=f"25th pct ({_threshold_25:.4f})"),
    ], fontsize=8)
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "11_density_distribution.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "11_density_distribution.pdf", bbox_inches="tight")

    _kind = "success" if _all_below else "warn"
    mo.vstack([
        _fig,
        mo.md("**Bottom 10 (lowest density = biggest gaps)**"),
        mo.as_html(density_df.head(10)[["composition", "family", "split", "kde_density", "density_rank"]]),
        mo.callout(mo.md(
            f"All Stage 5d worst-MAE compositions in bottom 25% KDE density? → "
            f"{'✓ YES — KDE gaps align with GNN errors.' if _all_below else '✗ NO — check KDE projection.'}"
        ), kind=_kind),
    ])
    return (density_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 — Gap Candidate Ranking

    All plausible binary and ternary lipid mixtures from the 10-lipid vocabulary projected
    into composition-PCA space (fitted in §9), scored by KDE density (low = under-sampled).

    **Candidate pool**:
    - **Binary**: C(10,2)=45 pairs × fractions {5,10,...,95}% → 855 total
    - **Ternary**: C(10,3)=120 triples × ordered fraction assignments (multiples of 10%, each ≥10%) → ≤4320 total
    - **Filter**: L∞-distance > 5 mol% from every existing composition

    **Biological filter**: a second ranking (PS ≤ 15 mol%) surfaces non-PS gap candidates
    (CHOL binaries, PE×PC combos, ternaries) for physiologically realistic simulations.
    """)
    return


@app.cell
def _(
    LIPIDS,
    OUT_DIR,
    X_comp,
    combinations,
    kde_comp_default,
    kde_comp_double,
    kde_comp_half,
    mo,
    np,
    pca_comp,
    pd,
    permutations,
    scaler_comp,
):
    _binary_fracs = list(range(5, 100, 5))

    _all_cand_vecs = []
    _all_cand_meta = []

    for _a, _b in combinations(LIPIDS, 2):
        for _f in _binary_fracs:
            _v = np.zeros(len(LIPIDS))
            _v[LIPIDS.index(_a)] = _f / 100.0
            _v[LIPIDS.index(_b)] = (100 - _f) / 100.0
            _all_cand_vecs.append(_v)
            _all_cand_meta.append({"lipid_a": _a, "frac_a": _f,
                                   "lipid_b": _b, "frac_b": 100 - _f,
                                   "lipid_c": None, "frac_c": None,
                                   "n_components": 2})

    def _ternary_ordered_fracs():
        _result = set()
        for _fa in range(10, 82, 10):
            for _fb in range(10, 91 - _fa, 10):
                _fc = 100 - _fa - _fb
                if _fc < 10:
                    continue
                for _perm in permutations([_fa, _fb, _fc]):
                    _result.add(_perm)
        return list(_result)

    _ternary_frac_list = _ternary_ordered_fracs()

    for _a, _b, _c in combinations(LIPIDS, 3):
        for _fa, _fb, _fc in _ternary_frac_list:
            _v = np.zeros(len(LIPIDS))
            _v[LIPIDS.index(_a)] = _fa / 100.0
            _v[LIPIDS.index(_b)] = _fb / 100.0
            _v[LIPIDS.index(_c)] = _fc / 100.0
            _all_cand_vecs.append(_v)
            _all_cand_meta.append({"lipid_a": _a, "frac_a": _fa,
                                   "lipid_b": _b, "frac_b": _fb,
                                   "lipid_c": _c, "frac_c": _fc,
                                   "n_components": 3})

    _V = np.array(_all_cand_vecs)
    _dists = np.max(np.abs(_V[:, None, :] - X_comp[None, :, :]), axis=2)
    _keep_mask = _dists.min(axis=1) > 0.05
    _V_keep = _V[_keep_mask]
    _meta_keep = [_m for _m, _k in zip(_all_cand_meta, _keep_mask) if _k]

    _V_z = scaler_comp.transform(_V_keep)
    _Z_cand = pca_comp.transform(_V_z)
    _cand_densities = kde_comp_default(_Z_cand.T)

    cands_df = pd.DataFrame(_meta_keep)
    cands_df["pc1"] = _Z_cand[:, 0]
    cands_df["pc2"] = _Z_cand[:, 1]
    cands_df["kde_density"] = _cand_densities
    cands_df["density_rank"] = cands_df["kde_density"].rank(ascending=True).astype(int)

    _PS_LIPIDS = {"POPS", "DOPS"}

    def _has_lipid(_row, _lip):
        return any(_row.get(f"lipid_{_x}") == _lip for _x in ["a", "b", "c"])

    cands_df["contains_chol"] = cands_df.apply(lambda _r: _has_lipid(_r, "CHOL"), axis=1)
    cands_df["has_charged"] = cands_df.apply(
        lambda _r: any(_has_lipid(_r, _l) for _l in _PS_LIPIDS), axis=1)

    top20 = cands_df.nsmallest(20, "kde_density").reset_index(drop=True)
    top20.to_csv(OUT_DIR / "gap_candidates.csv", index=False)

    cands_df["total_ps_frac"] = cands_df.apply(
        lambda _r: sum((_r.get(f"frac_{_x}") or 0)
                       for _x in ["a", "b", "c"]
                       if _r.get(f"lipid_{_x}") in _PS_LIPIDS),
        axis=1)
    bio_top20 = cands_df[cands_df["total_ps_frac"] <= 15].nsmallest(20, "kde_density").reset_index(drop=True)
    bio_top20.to_csv(OUT_DIR / "gap_candidates_bio.csv", index=False)

    # Bandwidth robustness
    _jaccard_rows = []
    for _bw_name, _kde_bw in [("0.5× Scott", kde_comp_half), ("2× Scott", kde_comp_double)]:
        _dens_bw = _kde_bw(_Z_cand.T)
        _top5_def = set(cands_df.nsmallest(5, "kde_density").index)
        _top5_bw = set(cands_df.assign(_d=_dens_bw).nsmallest(5, "_d").index)
        _jaccard = len(_top5_def & _top5_bw) / len(_top5_def | _top5_bw)
        _jaccard_rows.append({"bandwidth": _bw_name, "Jaccard (top-5 overlap with 1×)": round(_jaccard, 2)})

    _display_cols = ["lipid_a", "frac_a", "lipid_b", "frac_b", "lipid_c", "frac_c",
                     "n_components", "kde_density", "contains_chol", "has_charged"]
    mo.vstack([
        mo.md(f"**Binary candidates**: {sum(1 for _m in _meta_keep if _m['n_components']==2)}  "
              f"| **Ternary candidates**: {sum(1 for _m in _meta_keep if _m['n_components']==3)}  "
              f"| Total after near-duplicate filter: {len(_meta_keep)}"),
        mo.md("**Top 20 gap candidates (lowest composition-space KDE density)**"),
        mo.as_html(top20[_display_cols]),
        mo.md("**Biologically motivated top 20 (PS ≤ 15 mol% filter)**"),
        mo.as_html(bio_top20[_display_cols]),
        mo.md("**Bandwidth robustness — Jaccard overlap of top-5 across bandwidths**"),
        mo.as_html(pd.DataFrame(_jaccard_rows)),
    ])
    return (top20,)


@app.cell
def _(
    FIG_DIR,
    SPLIT_COLORS,
    STAGE5D_WORST,
    Z_comp,
    df_with_split,
    ev_comp,
    kde_comp_default,
    np,
    pd,
    plt,
    top20,
):
    _pad = 0.8
    _x0, _x1 = Z_comp[:, 0].min() - _pad, Z_comp[:, 0].max() + _pad
    _y0, _y1 = Z_comp[:, 1].min() - _pad, Z_comp[:, 1].max() + _pad
    _xx, _yy = np.mgrid[_x0:_x1:200j, _y0:_y1:200j]
    _density_bg = kde_comp_default(np.vstack([_xx.ravel(), _yy.ravel()])).reshape(200, 200)

    _fig, _ax = plt.subplots(figsize=(9, 7))
    _ax.contourf(_xx, _yy, _density_bg, levels=12, cmap="Blues", alpha=0.75)
    _ax.contour(_xx, _yy, _density_bg, levels=12, colors="navy", linewidths=0.3, alpha=0.3)

    for _split, _color in SPLIT_COLORS.items():
        _m = df_with_split["split"] == _split
        _ax.scatter(Z_comp[_m, 0], Z_comp[_m, 1], c=_color, s=55, edgecolor="k",
                    lw=0.6, alpha=0.9, label=f"existing ({_split})", zorder=5)

    _worst_mask = df_with_split["composition"].isin(STAGE5D_WORST)
    _widx = df_with_split.index[_worst_mask].tolist()
    _ax.scatter(Z_comp[_widx, 0], Z_comp[_widx, 1],
                marker="*", s=230, c="red", edgecolor="darkred", lw=0.7,
                zorder=7, label="Stage 5d worst MAE")
    for _i in _widx:
        _ax.annotate(df_with_split.loc[_i, "composition"], (Z_comp[_i, 0], Z_comp[_i, 1]),
                     textcoords="offset points", xytext=(5, 4), fontsize=7, color="darkred")

    _top20_bin = top20[top20["n_components"] == 2]
    _top20_ter = top20[top20["n_components"] == 3]
    _ax.scatter(_top20_bin["pc1"], _top20_bin["pc2"],
                marker="D", s=90, c="limegreen", edgecolor="darkgreen", lw=1.0,
                zorder=8, label=f"top-20 binary gap ({len(_top20_bin)})")
    _ax.scatter(_top20_ter["pc1"], _top20_ter["pc2"],
                marker="^", s=110, c="gold", edgecolor="darkgoldenrod", lw=1.0,
                zorder=8, label=f"top-20 ternary gap ({len(_top20_ter)})")

    for _, _row in top20.head(5).iterrows():
        _lbl = (f"{_row['lipid_a']}{int(_row['frac_a'])}_{_row['lipid_b']}{int(_row['frac_b'])}"
                + (f"_{_row['lipid_c']}{int(_row['frac_c'])}" if pd.notna(_row["lipid_c"]) else ""))
        _ax.annotate(_lbl, (_row["pc1"], _row["pc2"]),
                     textcoords="offset points", xytext=(6, 6), fontsize=6.5, color="darkgreen",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))

    _ax.set_xlabel(f"PC1 ({ev_comp[0]*100:.1f}%)")
    _ax.set_ylabel(f"PC2 ({ev_comp[1]*100:.1f}%)")
    _ax.set_title("Composition-PCA: existing dataset + top-20 gap candidates\n"
                  "(◆ = binary, ▲ = ternary, ★ = Stage 5d worst MAE)")
    _ax.legend(fontsize=8, loc="upper right")
    _ax.grid(alpha=0.25)
    _fig.tight_layout()
    _fig.savefig(FIG_DIR / "12_gap_candidates_pca.png", dpi=150, bbox_inches="tight")
    _fig.savefig(FIG_DIR / "12_gap_candidates_pca.pdf", bbox_inches="tight")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    1. **Dataset coverage**: 70 compositions (binary mixtures + pure systems) spanning 10 lipid types.
       The coverage is densest around POPC-dominant compositions; DPPC- and DOPC-rich regions
       have fewer compositions and are the lowest-density areas in composition-PCA space.

    2. **Property quality**: `bending_modulus` is the noisiest estimator (most MAD outliers,
       lowest SNR relative to between-system signal). All other properties have SNR > 3.
       The Helfrich proxy correlation confirms bending fits are not pure noise.

    3. **Pairwise structure**: `thickness` and `thickness_std` are positively correlated;
       `diffusivity` and `lipid_packing` anti-correlate (denser membranes diffuse more slowly).
       PCA concentrates most variance into 2–3 principal components.

    4. **Coverage gaps and GNN errors**: Stage 5d worst-MAE compositions (POPC65_DPPE35,
       POPC70_POPE30, POPC30_DOPC70, POPC40_DIPC60) fall in the bottom 25% of composition-space
       KDE density — the KDE metric correctly identifies where training coverage is thin.

    5. **Top simulation candidates**: ranked by lowest composition-space KDE density, saved to
       `gap_candidates.csv` (all candidates) and `gap_candidates_bio.csv` (PS ≤ 15 mol% filter
       for physiologically realistic mixtures). Bandwidth robustness checked via Jaccard overlap.

    **Caveats**: KDE operates in 2D (PC1+PC2 of composition-space); gaps that project onto
    dense 2D regions despite being compositionally distant are missed. The ternary candidate
    pool is combinatorially large — biological plausibility judgement is needed alongside the
    density ranking.
    """)
    return


if __name__ == "__main__":
    app.run()
