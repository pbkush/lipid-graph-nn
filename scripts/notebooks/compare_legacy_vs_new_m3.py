# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-learn",
#     "panedr",
# ]
# ///

import marimo

app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import pickle
    import hashlib
    import re
    from pathlib import Path
    from scipy import stats
    from scipy.stats import gaussian_kde
    from sklearn.decomposition import PCA

    try:
        import panedr
        HAS_PANEDR = True
    except ImportError:
        panedr = None
        HAS_PANEDR = False
    return (
        Path,
        HAS_PANEDR,
        PCA,
        gaussian_kde,
        hashlib,
        mo,
        np,
        panedr,
        pd,
        pickle,
        plt,
        re,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Legacy vs new M3 simulations — paired comparison

    The 70 legacy systems (vendored Martini 3 `.itp` files, located at
    `data/membrane_only/<system>/run/`) are being re-run against the upstream
    **M3-Lipid-Parameters** ITPs at `resources/martini3/itp/`. This notebook
    pairs each legacy system with its rerun on `canonical_name` and reports:

    1. Provenance — coverage of the pairing and ITP-file changes.
    2. Per-property comparison — the seven Tier-C properties paired
       per system (legacy vs new), with deltas, distributions, and
       per-composition movers.
    3. Sim-level observables from `prun.edr` — energies, temperature,
       pressure, box geometry, area-per-lipid.
    4. Composition-space view — whether deltas concentrate in any region of
       the 10-lipid simplex.
    5. Conclusions and retraining triggers.
    """
    )


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **Key findings** — populated as the analysis runs.

            - Paired coverage: _filled by §1_.
            - ITP files changed: _filled by §2_.
            - Active properties with material shifts: _filled by §3.1 / §6_.
            - Composition region where Δ concentrates: _filled by §5_.
            - Retraining verdict: _filled by §6_.
            """
        ),
        kind="info",
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("## 0. Paths")


@app.cell
def _(Path, mo):
    repo_root = Path(__file__).resolve().parents[2]

    legacy_props_dir_ui = mo.ui.text(
        value=str(repo_root / "results" / "properties"),
        label="Legacy property pickles dir (`<COMP>.h5`)",
        full_width=True,
    )
    new_props_dir_ui = mo.ui.text(
        value=str(repo_root / "results" / "properties_m3_rerun"),
        label="New property pickles dir",
        full_width=True,
    )
    legacy_runs_dir_ui = mo.ui.text(
        value=str(repo_root / "data" / "membrane_only"),
        label="Legacy run roots (`<COMP>/run/prun.*`)",
        full_width=True,
    )
    new_runs_dir_ui = mo.ui.text(
        value=str(repo_root / "data" / "membrane_only_m3_rerun"),
        label="New run roots",
        full_width=True,
    )
    legacy_itp_dir_ui = mo.ui.text(
        value=str(repo_root / "resources" / "old_ff_mappings"),
        label="Legacy ITP dir (for provenance hashing)",
        full_width=True,
    )
    new_itp_dir_ui = mo.ui.text(
        value=str(repo_root / "resources" / "martini3" / "itp"),
        label="New ITP dir",
        full_width=True,
    )

    mo.vstack(
        [
            legacy_props_dir_ui,
            new_props_dir_ui,
            legacy_runs_dir_ui,
            new_runs_dir_ui,
            legacy_itp_dir_ui,
            new_itp_dir_ui,
        ]
    )
    return (
        legacy_itp_dir_ui,
        legacy_props_dir_ui,
        legacy_runs_dir_ui,
        new_itp_dir_ui,
        new_props_dir_ui,
        new_runs_dir_ui,
    )


@app.cell
def _(Path, legacy_props_dir_ui, new_props_dir_ui):
    LEGACY_PROPS = Path(legacy_props_dir_ui.value)
    NEW_PROPS = Path(new_props_dir_ui.value)
    return LEGACY_PROPS, NEW_PROPS


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 1. Discover paired systems

    Each `<COMP>.h5` is a pickle of `[mean_dict, raw_arrays_dict]` where
    `mean_dict` holds the 8 property scalars (7 active + the dropped
    `bending_modulus`). A system is *paired* iff both `LEGACY_PROPS/<COMP>.h5`
    and `NEW_PROPS/<COMP>.h5` exist.
    """
    )


@app.cell
def _(LEGACY_PROPS, NEW_PROPS, pd):
    def _stems(d):
        return sorted({p.stem for p in d.glob("*.h5")}) if d.exists() else []

    _legacy_systems = _stems(LEGACY_PROPS)
    _new_systems = _stems(NEW_PROPS)

    paired = sorted(set(_legacy_systems) & set(_new_systems))
    legacy_only = sorted(set(_legacy_systems) - set(_new_systems))
    new_only = sorted(set(_new_systems) - set(_legacy_systems))

    coverage = pd.DataFrame(
        {
            "set": ["paired", "legacy_only", "new_only"],
            "n": [len(paired), len(legacy_only), len(new_only)],
            "example": [
                ", ".join(paired[:3]),
                ", ".join(legacy_only[:3]),
                ", ".join(new_only[:3]),
            ],
        }
    )
    return coverage, new_only, paired


@app.cell
def _(coverage, mo, paired):
    mo.vstack(
        [
            mo.as_html(coverage),
            mo.md(f"**Paired systems: {len(paired)}**"),
        ]
    )


@app.cell
def _(mo, new_only, paired):
    if len(paired) == 0:
        _box = mo.callout(
            mo.md(
                "No paired systems found. The new-side property pickles "
                "directory is empty or does not exist. Subsequent cells will "
                "render skeletons only — point `new_props_dir` at the rerun "
                "output once it is produced."
            ),
            kind="warn",
        )
    elif len(new_only) > 0:
        _box = mo.callout(
            mo.md(
                f"{len(new_only)} systems exist only in the new tree. "
                "Treat as in-flight or renamed; not part of the paired comparison."
            ),
            kind="info",
        )
    else:
        _box = mo.callout(
            mo.md(f"{len(paired)} paired systems."),
            kind="info",
        )
    _box


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 2. Provenance: ITP-file changes

    Lipid ITP files drive every downstream observable. Hash each ITP in the
    legacy and new directories and report the diff. Files present in both
    with identical SHA-1 are dropped from the table — only changes, additions,
    and removals are shown.
    """
    )


@app.cell
def _(Path, hashlib, legacy_itp_dir_ui, new_itp_dir_ui, pd):
    def _hash_dir(d):
        d = Path(d)
        out = {}
        if not d.exists():
            return out
        for p in sorted(d.rglob("*.itp")):
            h = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
            out[p.name] = h
        return out

    legacy_itps = _hash_dir(legacy_itp_dir_ui.value)
    new_itps = _hash_dir(new_itp_dir_ui.value)

    _rows = []
    for _name in sorted(set(legacy_itps) | set(new_itps)):
        _l = legacy_itps.get(_name)
        _n = new_itps.get(_name)
        if _l is None:
            _rows.append({"itp": _name, "status": "added_in_new", "legacy_sha1": "", "new_sha1": _n})
        elif _n is None:
            _rows.append({"itp": _name, "status": "removed_in_new", "legacy_sha1": _l, "new_sha1": ""})
        elif _l != _n:
            _rows.append({"itp": _name, "status": "changed", "legacy_sha1": _l, "new_sha1": _n})

    itp_diff = (
        pd.DataFrame(_rows)
        if _rows
        else pd.DataFrame(columns=["itp", "status", "legacy_sha1", "new_sha1"])
    )
    return itp_diff, legacy_itps, new_itps


@app.cell
def _(itp_diff, legacy_itps, mo, new_itps):
    mo.vstack(
        [
            mo.md(
                f"Legacy ITPs hashed: **{len(legacy_itps)}** · "
                f"new ITPs hashed: **{len(new_itps)}** · "
                f"changed/added/removed: **{len(itp_diff)}**"
            ),
            mo.as_html(itp_diff),
        ]
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 3. Per-property comparison

    Load `mean_dict` from each paired pickle. Build a long-form dataframe
    `df_props` with one row per (system, property) holding `legacy`, `new`,
    `delta = new − legacy`, and `pct_delta`.
    """
    )


@app.cell
def _():
    PROP_NAMES = [
        "lipid_packing",
        "thickness",
        "thickness_std",
        "variation",
        "persistence",
        "diffusivity",
        "compressibility",
        "bending_modulus",
    ]
    ACTIVE_PROPS = PROP_NAMES[:-1]  # tier C drops bending_modulus
    return ACTIVE_PROPS, PROP_NAMES


@app.cell
def _(LEGACY_PROPS, NEW_PROPS, PROP_NAMES, paired, pd, pickle):
    def _load_mean(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        m = obj[0] if isinstance(obj, (list, tuple)) else obj
        return {k: float(m[k]) for k in PROP_NAMES if k in m}

    _rows = []
    for _sys in paired:
        try:
            _l = _load_mean(LEGACY_PROPS / f"{_sys}.h5")
            _n = _load_mean(NEW_PROPS / f"{_sys}.h5")
        except Exception as _e:
            print(f"skip {_sys}: {_e}")
            continue
        for _prop in PROP_NAMES:
            if _prop in _l and _prop in _n:
                _rows.append(
                    {
                        "system": _sys,
                        "property": _prop,
                        "legacy": _l[_prop],
                        "new": _n[_prop],
                        "delta": _n[_prop] - _l[_prop],
                    }
                )

    df_props = pd.DataFrame(_rows)
    if not df_props.empty:
        _denom = df_props["legacy"].abs().clip(lower=1e-12)
        df_props["pct_delta"] = 100.0 * df_props["delta"] / _denom
    return (df_props,)


@app.cell
def _(df_props, mo):
    mo.stop(
        df_props.empty,
        mo.callout(
            mo.md(
                "No paired property data — sections 3–6 are skipped. "
                "Point `new_props_dir` at the rerun output and re-run."
            ),
            kind="warn",
        ),
    )

    mo.md(
        f"""
        **Dataset summary (`df_props`)**:
        - **Dimensions**: {df_props.shape[0]:,} rows × {df_props.shape[1]} cols
        - **Systems**: {df_props["system"].nunique()}
        - **Properties**: {df_props["property"].nunique()}
        - **Dtypes**: `{df_props.dtypes.astype(str).to_dict()}`
        - **Nulls**: {df_props.isna().sum().sum()} total cells
        """
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 3.1 Per-property paired summary

    For each property, summarise the paired delta: median, IQR, paired
    Wilcoxon `p`, paired `t` and `p`, and the fraction of systems whose
    `|Δ|` exceeds the legacy-side standard deviation (a rough
    "outside seed jitter" flag).
    """
    )


@app.cell
def _(df_props, np, pd, stats):
    if df_props.empty:
        df_summary = pd.DataFrame()
    else:
        _rows = []
        for _prop, _g in df_props.groupby("property"):
            _d = _g["delta"].to_numpy()
            _legacy = _g["legacy"].to_numpy()
            _sd_legacy = float(np.std(_legacy, ddof=1)) if len(_legacy) > 1 else np.nan
            try:
                _, _w_p = stats.wilcoxon(_d) if np.any(_d != 0) else (np.nan, np.nan)
            except ValueError:
                _w_p = np.nan
            _t_stat, _t_p = stats.ttest_rel(_g["new"], _g["legacy"])
            _rows.append(
                {
                    "property": _prop,
                    "n": len(_d),
                    "median_delta": float(np.median(_d)),
                    "iqr_delta": float(np.subtract(*np.percentile(_d, [75, 25]))),
                    "median_pct": float(np.median(_g["pct_delta"])),
                    "frac_|d|>sd_legacy": float(np.mean(np.abs(_d) > _sd_legacy))
                    if not np.isnan(_sd_legacy)
                    else np.nan,
                    "paired_t": float(_t_stat),
                    "t_p": float(_t_p),
                    "wilcoxon_p": float(_w_p) if not np.isnan(_w_p) else np.nan,
                }
            )
        df_summary = pd.DataFrame(_rows).sort_values("property").reset_index(drop=True)
    return (df_summary,)


@app.cell
def _(df_summary, mo):
    mo.as_html(df_summary.round(4)) if not df_summary.empty else mo.md("_no data_")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 3.2 Paired scatter — legacy vs new

    One panel per property. Points lie on `y = x` if the new force field
    produces the same value as the legacy one. Off-diagonal point clouds
    and y=x-line departures are the visual evidence behind the table above.
    """
    )


@app.cell
def _(PROP_NAMES, df_props, mo, plt):
    if df_props.empty:
        fig_scatter = mo.md("_no data_")
    else:
        _nrows, _ncols = 2, 4
        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(14, 7))
        for _i, _prop in enumerate(PROP_NAMES):
            _ax = _axes.flat[_i]
            _g = df_props[df_props["property"] == _prop]
            if _g.empty:
                _ax.set_visible(False)
                continue
            _ax.scatter(_g["legacy"], _g["new"], s=14, alpha=0.7, color="C0")
            _lo = float(min(_g["legacy"].min(), _g["new"].min()))
            _hi = float(max(_g["legacy"].max(), _g["new"].max()))
            _pad = 0.05 * (_hi - _lo + 1e-9)
            _ax.plot([_lo - _pad, _hi + _pad], [_lo - _pad, _hi + _pad], "k--", lw=0.8, alpha=0.5)
            _ax.set_xlabel(f"{_prop}, legacy")
            _ax.set_ylabel(f"{_prop}, new")
            _ax.set_title(_prop)
        for _j in range(len(PROP_NAMES), _nrows * _ncols):
            _axes.flat[_j].set_visible(False)
        _fig.tight_layout()
        fig_scatter = _fig
    fig_scatter


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 3.3 Bland–Altman — paired mean vs paired delta

    Bland–Altman separates magnitude-dependent bias from systematic shift.
    Horizontal lines at `mean(Δ) ± 1.96·sd(Δ)` mark the limits-of-agreement
    band; a tight band centred on zero means the new FF agrees with the
    legacy one up to seed-jitter.
    """
    )


@app.cell
def _(PROP_NAMES, df_props, mo, plt):
    if df_props.empty:
        fig_ba = mo.md("_no data_")
    else:
        _fig, _axes = plt.subplots(2, 4, figsize=(14, 7))
        for _i, _prop in enumerate(PROP_NAMES):
            _ax = _axes.flat[_i]
            _g = df_props[df_props["property"] == _prop]
            if _g.empty:
                _ax.set_visible(False)
                continue
            _mean_pair = 0.5 * (_g["legacy"] + _g["new"])
            _d = _g["delta"]
            _md = float(_d.mean())
            _sd = float(_d.std(ddof=1)) if len(_d) > 1 else 0.0
            _ax.scatter(_mean_pair, _d, s=14, alpha=0.7, color="C1")
            _ax.axhline(0, color="k", lw=0.6, alpha=0.5)
            _ax.axhline(_md, color="C3", lw=1)
            _ax.axhline(_md + 1.96 * _sd, color="C3", lw=0.8, ls="--")
            _ax.axhline(_md - 1.96 * _sd, color="C3", lw=0.8, ls="--")
            _ax.set_xlabel(f"½(legacy + new), {_prop}")
            _ax.set_ylabel(f"new − legacy, {_prop}")
            _ax.set_title(_prop)
        for _j in range(len(PROP_NAMES), 8):
            _axes.flat[_j].set_visible(False)
        _fig.tight_layout()
        fig_ba = _fig
    fig_ba


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 3.4 Distribution shift per property

    Overlaid KDEs of legacy and new property values on shared axes — a
    complementary view to Bland–Altman that ignores pairing and asks
    whether the marginal distribution has moved.
    """
    )


@app.cell
def _(PROP_NAMES, df_props, gaussian_kde, mo, np, plt):
    if df_props.empty:
        fig_dist = mo.md("_no data_")
    else:
        _fig, _axes = plt.subplots(2, 4, figsize=(14, 7))
        for _i, _prop in enumerate(PROP_NAMES):
            _ax = _axes.flat[_i]
            _g = df_props[df_props["property"] == _prop]
            if _g.empty or len(_g) < 3:
                _ax.set_visible(False)
                continue
            _all = np.concatenate([_g["legacy"], _g["new"]])
            _xs = np.linspace(_all.min(), _all.max(), 200)
            for _col, _color, _label in (("legacy", "C0", "legacy"), ("new", "C3", "new")):
                _vals = _g[_col].to_numpy()
                if np.ptp(_vals) > 0:
                    _kde = gaussian_kde(_vals)
                    _ax.fill_between(_xs, _kde(_xs), alpha=0.35, color=_color, label=_label)
            _ax.set_xlabel(_prop)
            _ax.set_ylabel("density")
            _ax.set_title(_prop)
            _ax.legend(fontsize=8)
        for _j in range(len(PROP_NAMES), 8):
            _axes.flat[_j].set_visible(False)
        _fig.tight_layout()
        fig_dist = _fig
    fig_dist


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 3.5 Top movers

    For each property, the 5 systems with the largest `|Δ|`. The receipt
    for the summary table above: if "thickness shifted by 3 Å on average,"
    these are the systems doing the shifting.
    """
    )


@app.cell
def _(df_props, mo, pd):
    if df_props.empty:
        _out = mo.md("_no data_")
    else:
        _frames = []
        for _prop, _g in df_props.groupby("property"):
            _top = _g.reindex(_g["delta"].abs().sort_values(ascending=False).index).head(5)
            _frames.append(_top[["property", "system", "legacy", "new", "delta", "pct_delta"]])
        df_movers = pd.concat(_frames, ignore_index=True).round(4)
        _out = mo.as_html(df_movers)
    _out


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 3.6 Δ-correlation across properties

    Pearson `corr(Δ_i, Δ_j)` over systems. Tells whether the force-field
    change moves properties coherently (e.g. thickness and thickness_std
    moving together) or independently. Strong off-diagonal structure
    suggests a single physical change driving multiple labels.
    """
    )


@app.cell
def _(PROP_NAMES, df_props, mo, plt):
    if df_props.empty:
        _out = mo.md("_no data_")
    else:
        _wide = df_props.pivot(index="system", columns="property", values="delta")
        _wide = _wide.reindex(columns=[p for p in PROP_NAMES if p in _wide.columns])
        _corr = _wide.corr()
        _fig, _ax = plt.subplots(figsize=(5.5, 4.5))
        _im = _ax.imshow(_corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r")
        _ax.set_xticks(range(len(_corr.columns)))
        _ax.set_yticks(range(len(_corr.index)))
        _ax.set_xticklabels(_corr.columns, rotation=45, ha="right")
        _ax.set_yticklabels(_corr.index)
        _ax.set_title("Pearson corr(Δ_i, Δ_j) across properties")
        _fig.colorbar(_im, ax=_ax, shrink=0.8)
        _fig.tight_layout()
        _out = mo.hstack([_fig, mo.as_html(_corr.round(2))], widths=[1, 1])
    _out


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 4. Sim-level observables (`prun.edr`)

    Parse `prun.edr` from each paired run via `panedr`. For each observable,
    take the mean and standard deviation over the last 50 % of frames and the
    linear-fit slope (drift). One row per (system, source, observable).

    Heavy step — gate behind a switch so the notebook is fast to open.
    """
    )


@app.cell
def _(mo):
    edr_run_ui = mo.ui.switch(value=False, label="Run EDR parsing (heavy)")
    edr_run_ui
    return (edr_run_ui,)


@app.cell
def _(
    Path,
    HAS_PANEDR,
    edr_run_ui,
    legacy_runs_dir_ui,
    mo,
    new_runs_dir_ui,
    np,
    paired,
    panedr,
    pd,
):
    if not edr_run_ui.value or not HAS_PANEDR or len(paired) == 0:
        df_edr = pd.DataFrame()
        if not HAS_PANEDR:
            _status = "_panedr unavailable — `pip install panedr`_"
        elif not edr_run_ui.value:
            _status = "_switch off — flip the switch above to parse `prun.edr`._"
        else:
            _status = "_no paired systems_"
    else:
        _OBSERVABLES = [
            "Potential",
            "LJ (SR)",
            "Coulomb (SR)",
            "Temperature",
            "Pres-XX",
            "Pres-YY",
            "Pres-ZZ",
            "Box-X",
            "Box-Y",
            "Box-Z",
        ]
        _rows = []
        for _source, _root in (
            ("legacy", Path(legacy_runs_dir_ui.value)),
            ("new", Path(new_runs_dir_ui.value)),
        ):
            for _sys in paired:
                _edr = _root / _sys / "run" / "prun.edr"
                if not _edr.exists():
                    continue
                try:
                    _df = panedr.edr_to_df(str(_edr))
                except Exception as _e:
                    print(f"edr fail {_source}/{_sys}: {_e}")
                    continue
                _tail = _df.iloc[len(_df) // 2 :]
                _t = (
                    _tail["Time"].to_numpy()
                    if "Time" in _tail.columns
                    else np.arange(len(_tail))
                )
                for _obs in _OBSERVABLES:
                    if _obs not in _tail.columns:
                        continue
                    _y = _tail[_obs].to_numpy()
                    _slope = float(np.polyfit(_t, _y, 1)[0]) if len(_t) > 1 else np.nan
                    _rows.append(
                        {
                            "system": _sys,
                            "source": _source,
                            "observable": _obs,
                            "mean": float(_y.mean()),
                            "std": float(_y.std(ddof=1)) if len(_y) > 1 else np.nan,
                            "slope_per_time": _slope,
                            "n_frames_tail": len(_y),
                        }
                    )
        df_edr = pd.DataFrame(_rows)
        _status = (
            f"`df_edr`: {df_edr.shape[0]:,} rows"
            if not df_edr.empty
            else "_no EDR files found_"
        )
    mo.md(_status)
    return (df_edr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 4.1 Paired observable means — legacy vs new

    Pivot `df_edr` to wide, one row per `(system, observable)` with
    `legacy` and `new` mean columns. Render scatter panels.
    """
    )


@app.cell
def _(df_edr, mo, plt):
    if df_edr.empty:
        _out = mo.md("_run §4 first_")
    else:
        _wide = df_edr.pivot_table(
            index=["system", "observable"], columns="source", values="mean"
        ).reset_index()
        _obs_list = sorted(_wide["observable"].unique())
        _ncols = 3
        _nrows = (len(_obs_list) + _ncols - 1) // _ncols
        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(4 * _ncols, 3 * _nrows))
        _axes_flat = _axes.flat if hasattr(_axes, "flat") else [_axes]
        for _i, _obs in enumerate(_obs_list):
            _ax = _axes_flat[_i]
            _g = _wide[_wide["observable"] == _obs].dropna(subset=["legacy", "new"])
            if _g.empty:
                _ax.set_visible(False)
                continue
            _ax.scatter(_g["legacy"], _g["new"], s=14, alpha=0.7)
            _lo = float(min(_g["legacy"].min(), _g["new"].min()))
            _hi = float(max(_g["legacy"].max(), _g["new"].max()))
            _pad = 0.05 * (_hi - _lo + 1e-9)
            _ax.plot([_lo - _pad, _hi + _pad], [_lo - _pad, _hi + _pad], "k--", lw=0.8, alpha=0.5)
            _ax.set_xlabel(f"{_obs}, legacy mean")
            _ax.set_ylabel(f"{_obs}, new mean")
            _ax.set_title(_obs)
        for _j in range(len(_obs_list), _nrows * _ncols):
            _axes_flat[_j].set_visible(False)
        _fig.tight_layout()
        _out = _fig
    _out


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ### 4.2 Area per lipid

    APL = `Box-X · Box-Y / (N_lip / 2)` per frame, averaged over the
    last 50 %. `N_lip` is read from the system name (e.g. `POPC30_DOPC70`
    → 30 + 70 = 100 lipids by convention).
    """
    )


@app.cell
def _(df_edr, mo, np, plt, re):
    def _nlip_from_name(name):
        parts = re.findall(r"([A-Z]+)(\d+)", name)
        return sum(int(n) for _, n in parts) if parts else np.nan

    if df_edr.empty:
        _out = mo.md("_run §4 first_")
    else:
        _boxx = df_edr[df_edr["observable"] == "Box-X"][["system", "source", "mean"]]
        _boxy = df_edr[df_edr["observable"] == "Box-Y"][["system", "source", "mean"]]
        _apl = _boxx.rename(columns={"mean": "Lx"}).merge(
            _boxy.rename(columns={"mean": "Ly"}), on=["system", "source"]
        )
        _apl["n_lip"] = _apl["system"].map(_nlip_from_name)
        _apl["apl_nm2"] = _apl["Lx"] * _apl["Ly"] / (_apl["n_lip"] / 2.0)
        _apl_wide = _apl.pivot(index="system", columns="source", values="apl_nm2").dropna()
        if _apl_wide.empty:
            _out = mo.md("_no paired APL data_")
        else:
            _fig, _ax = plt.subplots(figsize=(5, 5))
            _ax.scatter(_apl_wide["legacy"], _apl_wide["new"], s=18, alpha=0.7)
            _lo = float(min(_apl_wide["legacy"].min(), _apl_wide["new"].min()))
            _hi = float(max(_apl_wide["legacy"].max(), _apl_wide["new"].max()))
            _pad = 0.05 * (_hi - _lo + 1e-9)
            _ax.plot([_lo - _pad, _hi + _pad], [_lo - _pad, _hi + _pad], "k--", lw=0.8, alpha=0.5)
            _ax.set_xlabel("APL legacy [nm²]")
            _ax.set_ylabel("APL new [nm²]")
            _ax.set_title("Area per lipid, legacy vs new")
            _out = mo.vstack([mo.as_html(_apl_wide.round(3)), _fig])
    _out


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 5. Composition-space view

    Parse `<system>` into a mole-fraction vector over the lipid types that
    appear in the paired set, embed via PCA, and overlay per-system Δ as
    point colour. Reveals whether shifts concentrate in a corner of the
    composition simplex (e.g. the DPPC- or DOPC-rich corners flagged by
    Stage 5b per-system MAE).
    """
    )


@app.cell
def _(PCA, df_props, np, pd, re):
    def _composition(name):
        parts = re.findall(r"([A-Z][A-Z0-9]+?)(\d+)(?=_|$)", name)
        if not parts:
            m = re.match(r"^([A-Z]+)\d*$", name)
            if m:
                return {m.group(1): 1.0}
            return {}
        total = sum(int(n) for _, n in parts) or 1
        return {lip: int(n) / total for lip, n in parts}

    if df_props.empty:
        df_comp = pd.DataFrame()
        comp_pcs = None
    else:
        _systems = sorted(df_props["system"].unique())
        _comps = [_composition(s) for s in _systems]
        _vocab = sorted({k for c in _comps for k in c})
        _X = np.array([[c.get(k, 0.0) for k in _vocab] for c in _comps])
        if _X.shape[0] >= 2 and _X.shape[1] >= 2:
            _pca = PCA(n_components=2)
            _pcs = _pca.fit_transform(_X)
            df_comp = pd.DataFrame(
                {"system": _systems, "pc1": _pcs[:, 0], "pc2": _pcs[:, 1]}
            )
            comp_pcs = _pca.explained_variance_ratio_
        else:
            df_comp = pd.DataFrame()
            comp_pcs = None
    return comp_pcs, df_comp


@app.cell
def _(PROP_NAMES, comp_pcs, df_comp, df_props, mo, plt):
    if df_comp.empty:
        _out = mo.md("_no composition data_")
    else:
        _fig, _axes = plt.subplots(2, 4, figsize=(14, 7))
        for _i, _prop in enumerate(PROP_NAMES):
            _ax = _axes.flat[_i]
            _g = df_props[df_props["property"] == _prop][["system", "delta"]]
            _merged = df_comp.merge(_g, on="system")
            if _merged.empty:
                _ax.set_visible(False)
                continue
            _mx = float(_merged["delta"].abs().max() or 1.0)
            _sc = _ax.scatter(
                _merged["pc1"],
                _merged["pc2"],
                c=_merged["delta"],
                vmin=-_mx,
                vmax=_mx,
                cmap="RdBu_r",
                s=22,
                edgecolor="k",
                linewidth=0.3,
            )
            _ax.set_title(_prop)
            _ax.set_xlabel(
                f"PC1 ({100 * comp_pcs[0]:.0f}%)" if comp_pcs is not None else "PC1"
            )
            _ax.set_ylabel(
                f"PC2 ({100 * comp_pcs[1]:.0f}%)" if comp_pcs is not None else "PC2"
            )
            _fig.colorbar(_sc, ax=_ax, shrink=0.7)
        for _j in range(len(PROP_NAMES), 8):
            _axes.flat[_j].set_visible(False)
        _fig.tight_layout()
        _out = _fig
    _out


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## 6. Retraining trigger

    Pre-registered rule, evaluated mechanically on `df_summary`:

    - **Trigger fires** if any *active* property (7 Tier-C, excluding
      `bending_modulus`) has `|paired_t| > 3` **or**
      `frac_|d|>sd_legacy > 0.5`.
    - If it fires, Stage 5d weights are evaluated against stale labels and
      test R² / MSE must be recomputed on the new labels before any thesis
      numbers are taken from the rerun.
    """
    )


@app.cell
def _(ACTIVE_PROPS, df_summary, mo):
    if df_summary.empty:
        _box = mo.md("_no data yet_")
    else:
        _active = df_summary[df_summary["property"].isin(ACTIVE_PROPS)]
        _triggers = _active[
            (_active["paired_t"].abs() > 3) | (_active["frac_|d|>sd_legacy"] > 0.5)
        ]
        if _triggers.empty:
            _box = mo.callout(
                mo.md(
                    "**No retraining trigger fired.** All active properties "
                    "move within seed-jitter; Stage 5d numbers are reusable."
                ),
                kind="info",
            )
        else:
            _box = mo.callout(
                mo.md(
                    "**Retraining trigger fired** for: "
                    + ", ".join(f"`{p}`" for p in _triggers["property"])
                    + ". Recompute test metrics on new labels before quoting."
                ),
                kind="warn",
            )
    _box


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Conclusions

    Filled in once paired data exists. Leave the four headings below in
    place and write factual one-liners under each — neutral, no narrative:

    1. **Property shifts** — which of the 7 active properties moved
       materially, and by how much (median Δ, IQR, paired t).
    2. **Where shifts concentrate** — PCA composition region(s) that
       dominate the movers table.
    3. **Sim-level observables** — energy/pressure/box drifts that align
       with or contradict the property shifts.
    4. **Retraining verdict** — whether the trigger in §6 fires, and what
       follow-up is required (recompute test metrics on Stage 5d weights;
       consider re-running Stage 5d if labels move beyond Tier C's
       per-property seed std).
    """
    )


if __name__ == "__main__":
    app.run()
