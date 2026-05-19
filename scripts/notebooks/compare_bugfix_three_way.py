# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "scipy",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.23.6"
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
    import datetime
    from pathlib import Path
    from itertools import combinations
    from scipy import stats
    from scipy.stats import gaussian_kde
    from sklearn.decomposition import PCA

    return (
        PCA,
        Path,
        combinations,
        datetime,
        gaussian_kde,
        hashlib,
        mo,
        np,
        pd,
        pickle,
        plt,
        re,
        stats,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Three-way property + model comparison — bug-fix vs force-field vs seed

    Plan: [docs/compare_bugfix_three_way_plan.md](../../docs/compare_bugfix_three_way_plan.md).

    The current shipped labels (`results/properties/prop_legacy_bugged_random/`)
    were produced by the legacy `functions_emil` pipeline (12 logical bugs,
    unseeded RNG) on legacy GMX-2 trajectories. Moving to the production
    target — `results/properties/prop_m3_bugfixed_s0/` — confounds three
    effects:

    - **Seed effect** (bonus): bug #8 left numpy's global RNG unseeded;
      `legacy_bugged_s0 − legacy_bugged_random` quantifies how much of the
      shipped labels at any single composition is RNG draw vs true label.
    - **Bug-fix effect**: `legacy_bugfixed_s0 − legacy_bugged_s0` (same
      trajectories, same seed, code differs).
    - **Force-field effect**: `m3_bugfixed_s0 − legacy_bugfixed_s0` (same
      code, same seed, trajectories differ).

    Total Δ = seed_Δ + bugfix_Δ + ff_Δ + interaction. §4 partitions the
    variance of the total Δ into the three terms + covariances.

    Sections:
    1. Paths + label-set discovery
    2. ITP and code provenance (the FF arm)
    3. Property-level comparison — summary table, paired scatter,
       Bland–Altman, KDE, top movers per contrast
    4. Variance decomposition (the headline diagnostic)
    5. Composition-space view (PCA with Δ overlay)
    6. Model-level comparison — test MSE / R² across the three seeded
       label sets
    7. Headline callout + retraining verdict
    """)
    return


@app.cell
def _(mo):
    mo.callout(
        mo.md(
            """
            **Key findings** — populated as the analysis runs.

            - Paired coverage across label sets: _filled by §1_.
            - Properties dominated by seed noise: _filled by §3a / §7_.
            - Properties dominated by bug-fix effect: _filled by §4 / §7_.
            - Properties dominated by FF effect: _filled by §4 / §7_.
            - Retraining verdict: _filled by §6 / §7_.
            """
        ),
        kind="info",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. Vocabulary, contrasts, colours

    Names used throughout the notebook: the four label sets (one colour
    per set, kept consistent in every figure), the seven Tier-C
    properties (with `compressibility` renamed to its physically
    correct name `thickness_inhomogeneity` — the loader normalises both
    keys), and the four contrasts that drive every comparison
    downstream.
    """)
    return


@app.cell
def _():
    LSET_TAGS = [
        "legacy_bugged_random",
        "legacy_bugged_s0",
        "legacy_bugfixed_s0",
        "m3_bugfixed_s0",
    ]
    LSET_COLORS = {
        "legacy_bugged_random": "#7f7f7f",
        "legacy_bugged_s0": "#1f77b4",
        "legacy_bugfixed_s0": "#2ca02c",
        "m3_bugfixed_s0": "#d62728",
    }

    PROP_NAMES = [
        "lipid_packing",
        "thickness",
        "thickness_std",
        "thickness_inhomogeneity",
        "persistence",
        "diffusivity",
        "variation",
    ]
    ACTIVE_PROPS = list(PROP_NAMES)

    CONTRASTS = [
        ("seed", "legacy_bugged_s0", "legacy_bugged_random"),
        ("bugfix", "legacy_bugfixed_s0", "legacy_bugged_s0"),
        ("ff", "m3_bugfixed_s0", "legacy_bugfixed_s0"),
        ("total", "m3_bugfixed_s0", "legacy_bugged_random"),
    ]
    CONTRAST_COLORS = {
        "seed": "#9467bd",
        "bugfix": "#2ca02c",
        "ff": "#d62728",
        "total": "#bcbd22",
    }
    return CONTRASTS, LSET_COLORS, LSET_TAGS, PROP_NAMES


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Paths and label-set discovery

    All four property-pickle directories live under `results/properties/`.
    Each `<COMP>.h5` is a `pickle` of `[mean_dict, raw_arrays_dict]`. The
    plan renames `compressibility` → `thickness_inhomogeneity` in the
    bugfixed pipeline (the observable was always thickness inhomogeneity,
    not the area-compressibility modulus); the loader normalises both keys
    to `thickness_inhomogeneity` so cross-pipeline pairing works.
    """)
    return


@app.cell
def _(LSET_TAGS, Path, mo):
    repo_root = Path(__file__).resolve().parents[2]
    props_root = repo_root / "results" / "properties"

    dir_widgets = {
        tag: mo.ui.text(
            value=str(props_root / f"prop_{tag}"),
            label=f"`{tag}` dir",
            full_width=True,
        )
        for tag in LSET_TAGS
    }
    legacy_runs_ui = mo.ui.text(
        value=str(repo_root / "data" / "membrane_only"),
        label="Legacy run roots (for ITP hashing)",
        full_width=True,
    )
    m3_runs_ui = mo.ui.text(
        value=str(repo_root / "data" / "membrane_only_m3_rerun"),
        label="M3-rerun run roots",
        full_width=True,
    )
    wandb_root_ui = mo.ui.text(
        value=str(repo_root / "logs" / "training"),
        label="W&B downloads root (for §6 model loading)",
        full_width=True,
    )

    mo.vstack(
        [
            mo.md("**Property pickle directories**"),
            *[dir_widgets[t] for t in LSET_TAGS],
            mo.md("**Trajectory roots and W&B logs**"),
            legacy_runs_ui,
            m3_runs_ui,
            wandb_root_ui,
        ]
    )
    return dir_widgets, legacy_runs_ui, m3_runs_ui, wandb_root_ui


@app.cell
def _(LSET_TAGS, Path, dir_widgets):
    dirs = {tag: Path(dir_widgets[tag].value) for tag in LSET_TAGS}
    return (dirs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1a. Lipid-rename rules (DIPC ↔ DLPC)

    The M3 resimulation uses `submit_simulations.sh --rename-lipid DIPC=DLPC`
    (M3-Lipid-Parameters renamed di-C18:2 PC). The M3 output dirs therefore
    contain `DLPC*` stems while the legacy sets are `DIPC*`. To pair, every
    loaded stem is normalised under the rules below and the lipid-token
    order is re-canonicalised alphabetically (so `POPC50_DLPC50` →
    `DIPC50_POPC50` matches the legacy `DIPC50_POPC50` after the
    `DLPC=DIPC` rule).

    Rules: comma-separated `OLD=NEW` pairs. Empty → no normalisation
    (system names are paired as-is).
    """)
    return


@app.cell
def _(mo):
    rename_rules_ui = mo.ui.text(
        value="DLPC=DIPC",
        label="Lipid rename rules (comma-separated `OLD=NEW`)",
        full_width=True,
    )
    rename_rules_ui
    return (rename_rules_ui,)


@app.cell
def _(re, rename_rules_ui):
    def parse_rename_rules(text):
        rules = {}
        for chunk in (text or "").split(","):
            chunk = chunk.strip()
            if "=" in chunk:
                old, new = chunk.split("=", 1)
                old, new = old.strip(), new.strip()
                if old and new:
                    rules[old] = new
        return rules

    def canonical_stem(stem, rules):
        if not rules:
            return stem
        _parts = re.findall(r"([A-Z][A-Z0-9]+?)(\d+)(?=_|$)", stem)
        if not _parts:
            _m = re.match(r"^([A-Z]+)(\d*)$", stem)
            if not _m:
                return stem
            _lip, _n = _m.group(1), _m.group(2)
            return f"{rules.get(_lip, _lip)}{_n}"
        _new = [(rules.get(lip, lip), n) for lip, n in _parts]
        _new.sort(key=lambda kv: kv[0])
        return "_".join(f"{lip}{n}" for lip, n in _new)

    rename_rules = parse_rename_rules(rename_rules_ui.value)
    return canonical_stem, rename_rules


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1b. Coverage

    Discover `.h5` files in each of the four label-set directories,
    normalise each stem under the rename rules above, and intersect
    across all four sets. `paired_all` is the post-normalisation
    intersection — used for §3–§5. Each contrast in §3 still works as
    long as its two specific label sets are present, so partial coverage
    is survivable.
    """)
    return


@app.cell
def _(LSET_TAGS, canonical_stem, dirs, pd, rename_rules):
    def _entries(d):
        out = {}
        if not d.exists():
            return out
        for p in sorted(d.glob("*.h5")):
            _canon = canonical_stem(p.stem, rename_rules)
            out[_canon] = p
        return out

    system_paths = {tag: _entries(dirs[tag]) for tag in LSET_TAGS}
    systems_per_set = {tag: sorted(system_paths[tag].keys()) for tag in LSET_TAGS}

    rename_events = []
    for _tag in LSET_TAGS:
        for _canon, _path in system_paths[_tag].items():
            if _path.stem != _canon:
                rename_events.append(
                    {"label_set": _tag, "from": _path.stem, "to": _canon}
                )
    df_renames = pd.DataFrame(rename_events)

    coverage = pd.DataFrame(
        {
            "label_set": LSET_TAGS,
            "directory": [str(dirs[t]) for t in LSET_TAGS],
            "exists": [dirs[t].exists() for t in LSET_TAGS],
            "n_systems": [len(systems_per_set[t]) for t in LSET_TAGS],
            "n_renamed": [
                sum(1 for p in system_paths[t].values() if p.stem != canonical_stem(p.stem, rename_rules))
                for t in LSET_TAGS
            ],
            "example": [", ".join(systems_per_set[t][:3]) for t in LSET_TAGS],
        }
    )

    _sets = [set(systems_per_set[t]) for t in LSET_TAGS]
    paired_all = sorted(set.intersection(*_sets)) if all(_sets) else []
    return coverage, df_renames, paired_all, system_paths, systems_per_set


@app.cell
def _(coverage, df_renames, mo, paired_all):
    _stack = [
        mo.as_html(coverage),
        mo.md(
            f"**Fully paired across all four sets: {len(paired_all)} systems** "
            f"(post-normalisation)"
        ),
    ]
    if not df_renames.empty:
        _stack.append(
            mo.md(f"**Stems normalised by rename rules: {len(df_renames)}**")
        )
        _stack.append(mo.as_html(df_renames.head(20)))
    mo.vstack(_stack)
    return


@app.cell
def _(LSET_TAGS, mo, paired_all, systems_per_set):
    _missing = [t for t in LSET_TAGS if len(systems_per_set[t]) == 0]
    if len(paired_all) == 0:
        if _missing:
            _box = mo.callout(
                mo.md(
                    "**No fully-paired systems** — the following label sets are "
                    "empty or missing: "
                    + ", ".join(f"`{t}`" for t in _missing)
                    + ". §3 onwards will render skeletons only. Each contrast "
                    "renders independently once its two label sets are present."
                ),
                kind="warn",
            )
        else:
            _box = mo.callout(
                mo.md(
                    "All directories exist but the system-name intersection is "
                    "empty — check that the four pipelines wrote compatible "
                    "canonical names."
                ),
                kind="danger",
            )
    else:
        _box = mo.callout(
            mo.md(f"{len(paired_all)} fully-paired systems."), kind="info"
        )
    _box
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Provenance — ITP and code

    The FF contrast (`m3_bugfixed_s0 − legacy_bugfixed_s0`) is driven by
    the ITPs that wrote each trajectory. Hash every `.itp` under the
    legacy and M3-rerun run roots and report the diff. Files identical in
    both trees are omitted; only `added` / `removed` / `changed` rows are
    shown.

    Code provenance is reported through the file mtime of
    `lipid_gnn/properties.py` — confirms the `bugfixed_*` label sets came
    from the post-cleanup pipeline.
    """)
    return


@app.cell
def _(Path, hashlib, legacy_runs_ui, m3_runs_ui, pd):
    def _hash_itps_under(root):
        root = Path(root)
        out = {}
        if not root.exists():
            return out
        for p in sorted(root.rglob("*.itp")):
            try:
                rel = p.relative_to(root).as_posix()
            except ValueError:
                rel = p.name
            out[rel] = hashlib.sha1(p.read_bytes()).hexdigest()[:12]
        return out

    legacy_itps = _hash_itps_under(legacy_runs_ui.value)
    m3_itps = _hash_itps_under(m3_runs_ui.value)

    _rows = []
    for _name in sorted(set(legacy_itps) | set(m3_itps)):
        _l = legacy_itps.get(_name)
        _n = m3_itps.get(_name)
        if _l is None:
            _rows.append(
                {"itp": _name, "status": "added_in_m3", "legacy_sha1": "", "m3_sha1": _n}
            )
        elif _n is None:
            _rows.append(
                {"itp": _name, "status": "removed_in_m3", "legacy_sha1": _l, "m3_sha1": ""}
            )
        elif _l != _n:
            _rows.append(
                {"itp": _name, "status": "changed", "legacy_sha1": _l, "m3_sha1": _n}
            )

    itp_diff = (
        pd.DataFrame(_rows)
        if _rows
        else pd.DataFrame(columns=["itp", "status", "legacy_sha1", "m3_sha1"])
    )
    return itp_diff, legacy_itps, m3_itps


@app.cell
def _(itp_diff, legacy_itps, m3_itps, mo):
    mo.vstack(
        [
            mo.md(
                f"Legacy ITPs hashed: **{len(legacy_itps)}** · "
                f"M3 ITPs hashed: **{len(m3_itps)}** · "
                f"added/removed/changed: **{len(itp_diff)}**"
            ),
            mo.as_html(itp_diff) if len(itp_diff) else mo.md("_no diffs (or no ITPs found)_"),
        ]
    )
    return


@app.cell
def _(Path, datetime, mo):
    _props_py = Path(__file__).resolve().parents[2] / "lipid_gnn" / "properties.py"
    if _props_py.exists():
        _mtime = datetime.datetime.fromtimestamp(
            _props_py.stat().st_mtime
        ).isoformat(timespec="seconds")
        _msg = (
            f"`lipid_gnn/properties.py` mtime: **{_mtime}**. "
            "Both bugfixed label sets must have been computed after the "
            "last edit of this file to be eligible for the bug-fix contrast."
        )
    else:
        _msg = "`lipid_gnn/properties.py` not found — cannot verify code provenance."
    mo.md(_msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Property-level comparison

    Load `mean_dict` from each paired pickle for each of the four label
    sets. Build a wide-form dataframe `df_wide` with one row per
    `(system, property)` and one column per label set, plus derived
    columns for the four contrasts.

    The loader normalises the historical `compressibility` key to
    `thickness_inhomogeneity` so the bug-fix contrast can be computed.
    """)
    return


@app.cell
def _(LSET_TAGS, PROP_NAMES, paired_all, pd, pickle, system_paths):
    def _load_mean(path):
        with open(path, "rb") as f:
            obj = pickle.load(f)
        raw = obj[0] if isinstance(obj, (list, tuple)) else obj
        out = {}
        for k, v in raw.items():
            key = "thickness_inhomogeneity" if k == "compressibility" else k
            out[key] = float(v)
        return out

    _rows = []
    for _sys in paired_all:
        _means = {}
        _ok = True
        for _tag in LSET_TAGS:
            _path = system_paths[_tag].get(_sys)
            if _path is None:
                _ok = False
                break
            try:
                _means[_tag] = _load_mean(_path)
            except Exception as _e:
                print(f"skip {_sys} / {_tag}: {_e}")
                _ok = False
                break
        if not _ok:
            continue
        for _prop in PROP_NAMES:
            _vals = {_tag: _means[_tag].get(_prop) for _tag in LSET_TAGS}
            if any(v is None for v in _vals.values()):
                continue
            _rows.append({"system": _sys, "property": _prop, **_vals})

    df_wide = pd.DataFrame(_rows)
    return (df_wide,)


@app.cell
def _(CONTRASTS, df_wide):
    df_contrasts = df_wide.copy()
    if not df_contrasts.empty:
        for _name, _minuend, _subtrahend in CONTRASTS:
            df_contrasts[f"d_{_name}"] = (
                df_contrasts[_minuend] - df_contrasts[_subtrahend]
            )
    return (df_contrasts,)


@app.cell
def _(df_contrasts, df_wide, mo):
    mo.stop(
        df_wide.empty,
        mo.callout(
            mo.md(
                "**No paired property data across all four label sets.** "
                "Sections 3–5 will render skeletons. Generate the missing "
                "label-set directories and re-run; each contrast still needs "
                "its two specific sets, not all four."
            ),
            kind="warn",
        ),
    )

    mo.md(
        f"""
        **Dataset summary (`df_contrasts`)**:
        - **Dimensions**: {df_contrasts.shape[0]:,} rows × {df_contrasts.shape[1]} cols
        - **Systems**: {df_contrasts["system"].nunique()}
        - **Properties**: {df_contrasts["property"].nunique()}
        - **Dtypes**: `{df_contrasts.dtypes.astype(str).value_counts().to_dict()}`
        - **Nulls**: {df_contrasts.isna().sum().sum()} total cells
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3a. Seed-only sanity panel

    `d_seed = legacy_bugged_s0 − legacy_bugged_random`. Same code, same
    trajectories — only the RNG state differs. This sets the noise floor
    for the bug-fix contrast: a bug-fix effect smaller than the seed
    effect is not distinguishable from RNG variance at single-seed
    resolution.

    `SD_random` here is the within-corpus std of the historically shipped
    labels (`legacy_bugged_random`), which is also the reference dispersion
    used by every "frac \|d\|>SD" column in §3b — so the fractions across
    seed / bugfix / ff are directly comparable.
    """)
    return


@app.cell
def _(PROP_NAMES, df_contrasts, np, pd):
    if df_contrasts.empty:
        df_seed = pd.DataFrame()
    else:
        _rows = []
        for _prop in PROP_NAMES:
            _g = df_contrasts[df_contrasts["property"] == _prop]
            if _g.empty:
                continue
            _d = _g["d_seed"].to_numpy()
            _ref = _g["legacy_bugged_random"].to_numpy()
            _sd_ref = float(np.std(_ref, ddof=1)) if len(_ref) > 1 else np.nan
            _rows.append(
                {
                    "property": _prop,
                    "n": len(_d),
                    "mean_d_seed": float(_d.mean()),
                    "sd_d_seed": float(_d.std(ddof=1)) if len(_d) > 1 else np.nan,
                    "median_|d_seed|": float(np.median(np.abs(_d))),
                    "max_|d_seed|": float(np.max(np.abs(_d))),
                    "SD_random": _sd_ref,
                    "frac_|d_seed|>SD_random": (
                        float(np.mean(np.abs(_d) > _sd_ref))
                        if not np.isnan(_sd_ref) and _sd_ref > 0
                        else np.nan
                    ),
                }
            )
        df_seed = pd.DataFrame(_rows)
    return (df_seed,)


@app.cell
def _(df_seed, mo):
    mo.as_html(df_seed.round(4)) if not df_seed.empty else mo.md(
        "_seed contrast unavailable — need both `legacy_bugged_random` and "
        "`legacy_bugged_s0` populated._"
    )
    return


@app.cell
def _(df_seed, mo):
    if df_seed.empty:
        _box = mo.md("")
    else:
        _flagged = df_seed[df_seed["frac_|d_seed|>SD_random"] > 0.5]
        if _flagged.empty:
            _box = mo.callout(
                mo.md(
                    "Seed noise is below the within-corpus SD on every "
                    "property — bug-fix and FF contrasts can be interpreted "
                    "at single-seed resolution."
                ),
                kind="info",
            )
        else:
            _box = mo.callout(
                mo.md(
                    "Seed noise exceeds within-corpus SD on >50% of systems "
                    "for: "
                    + ", ".join(f"`{p}`" for p in _flagged["property"])
                    + ". Bug-fix conclusions on these properties need "
                    "multi-seed property-side averaging before they're "
                    "actionable."
                ),
                kind="warn",
            )
    _box
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3b. Paired summary table (all four contrasts)

    For every property × contrast, mean ± SD of the Δ vector across the
    70 systems, paired `t` of `minuend` vs `subtrahend`, and the fraction
    of \|d\| beyond the within-corpus SD of `legacy_bugged_random` (the
    common reference). The seed row is diagnostic, not actionable; bugfix
    and ff are the substantive contrasts; total is the "current shipped →
    final product" path.
    """)
    return


@app.cell
def _(CONTRASTS, PROP_NAMES, df_contrasts, np, pd, stats):
    if df_contrasts.empty:
        df_summary = pd.DataFrame()
    else:
        _rows = []
        for _prop in PROP_NAMES:
            _g = df_contrasts[df_contrasts["property"] == _prop]
            if _g.empty:
                continue
            _ref = _g["legacy_bugged_random"].to_numpy()
            _sd_ref = float(np.std(_ref, ddof=1)) if len(_ref) > 1 else np.nan
            for _name, _minuend, _subtrahend in CONTRASTS:
                _d = _g[f"d_{_name}"].to_numpy()
                _m = _g[_minuend].to_numpy()
                _s = _g[_subtrahend].to_numpy()
                try:
                    _t_stat, _t_p = stats.ttest_rel(_m, _s)
                except Exception:
                    _t_stat, _t_p = np.nan, np.nan
                _rows.append(
                    {
                        "property": _prop,
                        "contrast": _name,
                        "n": len(_d),
                        "mean_d": float(_d.mean()),
                        "sd_d": float(_d.std(ddof=1)) if len(_d) > 1 else np.nan,
                        "median_d": float(np.median(_d)),
                        "paired_t": float(_t_stat),
                        "t_p": float(_t_p),
                        "frac_|d|>SD_random": (
                            float(np.mean(np.abs(_d) > _sd_ref))
                            if not np.isnan(_sd_ref) and _sd_ref > 0
                            else np.nan
                        ),
                    }
                )
        df_summary = pd.DataFrame(_rows)
    return (df_summary,)


@app.cell
def _(df_summary, mo):
    if df_summary.empty:
        _out = mo.md("_no data_")
    else:
        _pretty = df_summary.copy()
        _pretty[["mean_d", "sd_d", "median_d", "paired_t", "t_p", "frac_|d|>SD_random"]] = (
            _pretty[["mean_d", "sd_d", "median_d", "paired_t", "t_p", "frac_|d|>SD_random"]].round(4)
        )
        _out = mo.as_html(_pretty)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3c. Per-property quadruple-scatter

    Four panels per property — seed, bug-fix, FF, total. Each panel is a
    minuend-vs-subtrahend scatter with the identity line; off-diagonal
    departures are the visual evidence behind §3b. Axes are shared per
    panel; the four panels per property are not normalised to a common
    range because the contrasts can span different magnitudes.
    """)
    return


@app.cell
def _(CONTRASTS, PROP_NAMES, df_contrasts, mo, plt):
    if df_contrasts.empty:
        _out = mo.md("_no data_")
    else:
        _n_props = len(PROP_NAMES)
        _fig, _axes = plt.subplots(
            _n_props, len(CONTRASTS), figsize=(3.2 * len(CONTRASTS), 2.8 * _n_props)
        )
        for _i, _prop in enumerate(PROP_NAMES):
            _g = df_contrasts[df_contrasts["property"] == _prop]
            for _j, (_name, _minuend, _subtrahend) in enumerate(CONTRASTS):
                _ax = _axes[_i, _j] if _n_props > 1 else _axes[_j]
                if _g.empty:
                    _ax.set_visible(False)
                    continue
                _x = _g[_subtrahend].to_numpy()
                _y = _g[_minuend].to_numpy()
                _ax.scatter(_x, _y, s=12, alpha=0.7, color="C0")
                _lo = float(min(_x.min(), _y.min()))
                _hi = float(max(_x.max(), _y.max()))
                _pad = 0.05 * (_hi - _lo + 1e-9)
                _ax.plot(
                    [_lo - _pad, _hi + _pad],
                    [_lo - _pad, _hi + _pad],
                    "k--",
                    lw=0.7,
                    alpha=0.5,
                )
                _ax.set_xlabel(f"{_subtrahend}\n{_prop}", fontsize=7)
                _ax.set_ylabel(f"{_minuend}\n{_prop}", fontsize=7)
                _ax.set_title(f"{_prop} — {_name}", fontsize=8)
                _ax.tick_params(labelsize=7)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3d. Bland–Altman per contrast

    Per property, one panel per contrast. `mean(minuend, subtrahend)` on
    x; `minuend − subtrahend` on y. Solid red line = mean Δ; dashed red =
    1.96·SD limits-of-agreement band. The seed panel anchors what a
    "no real effect" B–A looks like for this estimator.
    """)
    return


@app.cell
def _(CONTRASTS, PROP_NAMES, df_contrasts, mo, plt):
    if df_contrasts.empty:
        _out = mo.md("_no data_")
    else:
        _n_props = len(PROP_NAMES)
        _fig, _axes = plt.subplots(
            _n_props, len(CONTRASTS), figsize=(3.2 * len(CONTRASTS), 2.6 * _n_props)
        )
        for _i, _prop in enumerate(PROP_NAMES):
            _g = df_contrasts[df_contrasts["property"] == _prop]
            for _j, (_name, _minuend, _subtrahend) in enumerate(CONTRASTS):
                _ax = _axes[_i, _j] if _n_props > 1 else _axes[_j]
                if _g.empty:
                    _ax.set_visible(False)
                    continue
                _mean_pair = 0.5 * (_g[_minuend].to_numpy() + _g[_subtrahend].to_numpy())
                _d = _g[f"d_{_name}"].to_numpy()
                _md = float(_d.mean())
                _sd = float(_d.std(ddof=1)) if len(_d) > 1 else 0.0
                _ax.scatter(_mean_pair, _d, s=12, alpha=0.7, color="C1")
                _ax.axhline(0, color="k", lw=0.5, alpha=0.5)
                _ax.axhline(_md, color="C3", lw=1)
                _ax.axhline(_md + 1.96 * _sd, color="C3", lw=0.7, ls="--")
                _ax.axhline(_md - 1.96 * _sd, color="C3", lw=0.7, ls="--")
                _ax.set_xlabel(f"½(M+S), {_prop}", fontsize=7)
                _ax.set_ylabel(f"d_{_name}", fontsize=7)
                _ax.set_title(f"{_prop} — {_name}", fontsize=8)
                _ax.tick_params(labelsize=7)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3e. KDE overlay across all four label sets

    Per property, KDE of each label set on shared axes. The two
    `legacy_bugged_*` curves should nearly coincide — visible separation
    is RNG-induced label drift across the corpus. Distance between
    `legacy_bugfixed_s0` and `legacy_bugged_s0` is the bug-fix shift in
    marginal distribution shape; between `m3_bugfixed_s0` and
    `legacy_bugfixed_s0` is the FF shift.
    """)
    return


@app.cell
def _(
    LSET_COLORS,
    LSET_TAGS,
    PROP_NAMES,
    df_contrasts,
    gaussian_kde,
    mo,
    np,
    plt,
):
    if df_contrasts.empty:
        _out = mo.md("_no data_")
    else:
        _n_props = len(PROP_NAMES)
        _ncols = 4
        _nrows = (_n_props + _ncols - 1) // _ncols
        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(4 * _ncols, 3 * _nrows))
        _axes_flat = _axes.flat if hasattr(_axes, "flat") else [_axes]
        for _i, _prop in enumerate(PROP_NAMES):
            _ax = _axes_flat[_i]
            _g = df_contrasts[df_contrasts["property"] == _prop]
            if _g.empty or len(_g) < 3:
                _ax.set_visible(False)
                continue
            _all = np.concatenate([_g[t].to_numpy() for t in LSET_TAGS])
            _xs = np.linspace(_all.min(), _all.max(), 200)
            for _tag in LSET_TAGS:
                _vals = _g[_tag].to_numpy()
                if np.ptp(_vals) > 0:
                    _kde = gaussian_kde(_vals)
                    _ax.fill_between(
                        _xs, _kde(_xs), alpha=0.25, color=LSET_COLORS[_tag], label=_tag
                    )
                    _ax.plot(_xs, _kde(_xs), color=LSET_COLORS[_tag], lw=1)
            _ax.set_xlabel(_prop, fontsize=8)
            _ax.set_ylabel("density", fontsize=8)
            _ax.set_title(_prop, fontsize=9)
            _ax.legend(fontsize=6)
            _ax.tick_params(labelsize=7)
        for _j in range(_n_props, _nrows * _ncols):
            _axes_flat[_j].set_visible(False)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 3f. Top-N movers per contrast

    Per property × contrast, the 5 systems with the largest \|Δ\|. The
    receipt for §3b: if "thickness shifted by 3 Å under FF on average",
    these are the systems doing it. Seed movers are candidates for
    "noisy estimator at this composition"; bug-fix movers are likely
    bug-#2/#3/#4 signatures (per-residue persistence, midplane bending);
    FF movers are likely CHOL-containing systems (M2 8-bead → M3 9-bead).
    """)
    return


@app.cell
def _(CONTRASTS, df_contrasts, mo, pd):
    if df_contrasts.empty:
        _out = mo.md("_no data_")
    else:
        _frames = []
        for _prop, _g in df_contrasts.groupby("property"):
            for _name, _minuend, _subtrahend in CONTRASTS:
                _col = f"d_{_name}"
                _top = _g.reindex(
                    _g[_col].abs().sort_values(ascending=False).index
                ).head(5)
                _frames.append(
                    pd.DataFrame(
                        {
                            "property": _prop,
                            "contrast": _name,
                            "system": _top["system"].to_numpy(),
                            "minuend": _top[_minuend].to_numpy(),
                            "subtrahend": _top[_subtrahend].to_numpy(),
                            "delta": _top[_col].to_numpy(),
                        }
                    )
                )
        df_movers = pd.concat(_frames, ignore_index=True).round(4)
        _out = mo.as_html(df_movers)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Variance decomposition (headline diagnostic)

    The total label shift (`d_total = m3_bugfixed_s0 − legacy_bugged_random`)
    decomposes as:

    `Var(d_total) = Var(d_seed) + Var(d_bugfix) + Var(d_ff)
                  + 2·Cov(d_seed, d_bugfix) + 2·Cov(d_seed, d_ff) + 2·Cov(d_bugfix, d_ff)`

    Per property, the six terms are computed across the 70-system pool
    and rendered as a signed stacked bar normalised by `Var(d_total)` —
    so each bar sums to 1.0 by identity. The three variance terms are
    non-negative; the three covariance terms can be either sign.

    This is the **headline diagnostic** of the notebook: it answers "is
    the regression target shift driven by bug-fixes or by force-field?"
    in one figure.
    """)
    return


@app.cell
def _(PROP_NAMES, combinations, df_contrasts, np, pd):
    if df_contrasts.empty:
        df_var = pd.DataFrame()
    else:
        _component_cols = ["d_seed", "d_bugfix", "d_ff"]
        _rows = []
        for _prop in PROP_NAMES:
            _g = df_contrasts[df_contrasts["property"] == _prop]
            if _g.empty:
                continue
            _row = {"property": _prop, "n": len(_g)}
            _row["var_total"] = float(np.var(_g["d_total"].to_numpy(), ddof=1))
            for _c in _component_cols:
                _row[f"var_{_c}"] = float(np.var(_g[_c].to_numpy(), ddof=1))
            for _a, _b in combinations(_component_cols, 2):
                _row[f"cov_{_a}_{_b}"] = float(
                    np.cov(_g[_a].to_numpy(), _g[_b].to_numpy(), ddof=1)[0, 1]
                )
            _rows.append(_row)
        df_var = pd.DataFrame(_rows)
    return (df_var,)


@app.cell
def _(PROP_NAMES, df_var, mo, np, plt):
    if df_var.empty:
        _out = mo.md("_no data_")
    else:
        _terms = [
            ("var_d_seed", "Var(seed)", "#9467bd"),
            ("var_d_bugfix", "Var(bugfix)", "#2ca02c"),
            ("var_d_ff", "Var(ff)", "#d62728"),
            ("cov_d_seed_d_bugfix", "2·Cov(seed,bugfix)", "#c5b0d5"),
            ("cov_d_seed_d_ff", "2·Cov(seed,ff)", "#ff9896"),
            ("cov_d_bugfix_d_ff", "2·Cov(bugfix,ff)", "#ffbb78"),
        ]
        _fig, _ax = plt.subplots(figsize=(10, 5))
        _props = [
            p
            for p in PROP_NAMES
            if p in df_var["property"].values
            and df_var.loc[df_var["property"] == p, "var_total"].iloc[0] > 0
        ]
        _x = np.arange(len(_props))
        _pos_bottom = np.zeros(len(_props))
        _neg_bottom = np.zeros(len(_props))
        for _term, _label, _color in _terms:
            _factor = 2.0 if _term.startswith("cov") else 1.0
            _vals = np.array(
                [
                    _factor
                    * df_var.loc[df_var["property"] == p, _term].iloc[0]
                    / df_var.loc[df_var["property"] == p, "var_total"].iloc[0]
                    for p in _props
                ]
            )
            _pos = np.where(_vals > 0, _vals, 0.0)
            _neg = np.where(_vals < 0, _vals, 0.0)
            _ax.bar(_x, _pos, bottom=_pos_bottom, color=_color, label=_label, edgecolor="k", linewidth=0.3)
            _ax.bar(_x, _neg, bottom=_neg_bottom, color=_color, edgecolor="k", linewidth=0.3, alpha=0.6)
            _pos_bottom = _pos_bottom + _pos
            _neg_bottom = _neg_bottom + _neg
        _ax.axhline(1.0, color="k", lw=0.6, alpha=0.4)
        _ax.axhline(0.0, color="k", lw=0.6)
        _ax.set_xticks(_x)
        _ax.set_xticklabels(_props, rotation=30, ha="right", fontsize=8)
        _ax.set_ylabel("Variance / Var(d_total)")
        _ax.set_title("Decomposition of Var(d_total) per property")
        _ax.legend(fontsize=7, loc="upper right", ncol=2)
        _fig.tight_layout()
        _out = mo.vstack([_fig, mo.as_html(df_var.round(6))])
    _out
    return


@app.cell
def _(df_var, mo, np):
    if df_var.empty:
        _box = mo.md("")
    else:
        _bars = []
        for _, _row in df_var.iterrows():
            _vt = _row["var_total"]
            if _vt <= 0 or np.isnan(_vt):
                continue
            _frac_seed = _row["var_d_seed"] / _vt
            _frac_bug = _row["var_d_bugfix"] / _vt
            _frac_ff = _row["var_d_ff"] / _vt
            _dominant = max(
                [("seed", _frac_seed), ("bugfix", _frac_bug), ("ff", _frac_ff)],
                key=lambda kv: kv[1],
            )
            _bars.append(
                f"- `{_row['property']}` → dominant component: **{_dominant[0]}** "
                f"({_dominant[1]:.0%} of Var(d_total)); "
                f"seed {_frac_seed:.0%}, bugfix {_frac_bug:.0%}, ff {_frac_ff:.0%}."
            )
        _box = (
            mo.callout(mo.md("\n".join(_bars)), kind="info")
            if _bars
            else mo.md("_no variance to decompose_")
        )
    _box
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Composition-space view

    PCA on the 70 × `|lipid types|` mole-fraction matrix → first two
    principal axes. Per property × contrast, scatter the 70 systems in
    PC1/PC2 space coloured by signed Δ; size scales with \|Δ\|. Reveals
    whether each effect concentrates in a region of the composition
    simplex (e.g. seed noise scattered uniformly, FF Δ concentrated in
    the CHOL-rich or DPPC-rich corners).
    """)
    return


@app.cell
def _(PCA, df_contrasts, np, pd, re):
    def _composition(name):
        parts = re.findall(r"([A-Z][A-Z0-9]+?)(\d+)(?=_|$)", name)
        if not parts:
            m = re.match(r"^([A-Z]+)\d*$", name)
            if m:
                return {m.group(1): 1.0}
            return {}
        total = sum(int(n) for _, n in parts) or 1
        return {lip: int(n) / total for lip, n in parts}

    if df_contrasts.empty:
        df_comp = pd.DataFrame()
        comp_pcs = None
    else:
        _systems = sorted(df_contrasts["system"].unique())
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
def _(comp_pcs, df_comp, mo):
    if df_comp.empty or comp_pcs is None:
        _msg = "_no composition embedding (need ≥ 2 systems and ≥ 2 lipid types)_"
    else:
        _msg = (
            f"PCA on the **{df_comp.shape[0]}**-system × "
            f"|vocab|-lipid composition matrix: "
            f"PC1 explains **{100 * comp_pcs[0]:.1f}%**, "
            f"PC2 **{100 * comp_pcs[1]:.1f}%** of composition variance."
        )
    mo.md(_msg)
    return


@app.cell
def _(mo):
    contrast_focus_ui = mo.ui.dropdown(
        options=["seed", "bugfix", "ff", "total"],
        value="bugfix",
        label="Δ contrast to overlay on composition space",
    )
    contrast_focus_ui
    return (contrast_focus_ui,)


@app.cell
def _(PROP_NAMES, comp_pcs, contrast_focus_ui, df_comp, df_contrasts, mo, plt):
    if df_comp.empty or df_contrasts.empty:
        _out = mo.md("_no composition data_")
    else:
        _focus = contrast_focus_ui.value
        _col = f"d_{_focus}"
        _ncols = 4
        _nrows = (len(PROP_NAMES) + _ncols - 1) // _ncols
        _fig, _axes = plt.subplots(_nrows, _ncols, figsize=(4 * _ncols, 3 * _nrows))
        _axes_flat = _axes.flat if hasattr(_axes, "flat") else [_axes]
        for _i, _prop in enumerate(PROP_NAMES):
            _ax = _axes_flat[_i]
            _g = df_contrasts[df_contrasts["property"] == _prop][["system", _col]]
            _merged = df_comp.merge(_g, on="system")
            if _merged.empty:
                _ax.set_visible(False)
                continue
            _mx = float(_merged[_col].abs().max() or 1.0)
            _sc = _ax.scatter(
                _merged["pc1"],
                _merged["pc2"],
                c=_merged[_col],
                vmin=-_mx,
                vmax=_mx,
                cmap="RdBu_r",
                s=20 + 60 * _merged[_col].abs() / (_mx + 1e-9),
                edgecolor="k",
                linewidth=0.3,
            )
            _ax.set_xlabel(
                f"PC1 ({100 * comp_pcs[0]:.0f}%)" if comp_pcs is not None else "PC1",
                fontsize=8,
            )
            _ax.set_ylabel(
                f"PC2 ({100 * comp_pcs[1]:.0f}%)" if comp_pcs is not None else "PC2",
                fontsize=8,
            )
            _ax.set_title(f"{_prop}: d_{_focus}", fontsize=9)
            _ax.tick_params(labelsize=7)
            _fig.colorbar(_sc, ax=_ax, shrink=0.7)
        for _j in range(len(PROP_NAMES), _nrows * _ncols):
            _axes_flat[_j].set_visible(False)
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Model-level comparison

    Three new W&B groups, all trained on the same locked Tier C
    hyperparameters (`lr=3e-5, wd=1e-3, h=128, l=2, e=200`) on 6 seeds
    {0,1,4,5,6,8}, each on one of the three seeded label sets. The
    existing `stage_5d_tier_c_confirm` group is included as an optional
    fourth model — it was trained on `legacy_bugged_random` and so
    represents the property-seed effect on the model side
    (`stage_5d_tier_c_confirm` vs `stage_5d_tier_c_legacy_bugged_s0`),
    distinct from the GNN-init seed.

    Loaders read `test_artifacts.npz` from each W&B group's downloaded
    runs (no live W&B API). Per-property pooled test MSE / R² is
    computed from `test_preds`, `test_targets`, `scaler_mean`,
    `scaler_scale`. Bootstrap CI (95 %) over the 1650-point pool.
    """)
    return


@app.cell
def _():
    GROUP_MAP = {
        "model_legacy_bugged_random": "stage_5d_tier_c_confirm",
        "model_legacy_bugged_s0": "stage_5d_tier_c_legacy_bugged_s0",
        "model_legacy_bugfixed_s0": "stage_5d_tier_c_legacy_bugfixed_s0",
        "model_m3_bugfixed_s0": "stage_5d_tier_c_m3_bugfixed_s0",
    }
    return (GROUP_MAP,)


@app.cell
def _(GROUP_MAP, Path, mo, np, pd, wandb_root_ui):
    def _load_group(group_dir):
        if not group_dir.exists():
            return []
        runs = []
        for _run_dir in sorted(group_dir.iterdir()):
            _npz = _run_dir / "test_artifacts.npz"
            if not _npz.exists():
                continue
            try:
                _a = np.load(_npz, allow_pickle=True)
            except Exception as _e:
                print(f"skip {_run_dir.name}: {_e}")
                continue
            runs.append(
                {
                    "run_name": _run_dir.name,
                    "preds": _a["test_preds"],
                    "targets": _a["test_targets"],
                    "compositions": _a["test_compositions"],
                    "system_idx": _a["test_system_idx"],
                    "scaler_mean": _a["scaler_mean"],
                    "scaler_scale": _a["scaler_scale"],
                    "properties": [str(p) for p in _a["properties"]],
                }
            )
        return runs

    _root = Path(wandb_root_ui.value)
    model_runs = {tag: _load_group(_root / group) for tag, group in GROUP_MAP.items()}

    model_inventory = pd.DataFrame(
        {
            "model_tag": list(GROUP_MAP.keys()),
            "wandb_group": list(GROUP_MAP.values()),
            "n_runs": [len(model_runs[t]) for t in GROUP_MAP.keys()],
            "first_run": [
                model_runs[t][0]["run_name"] if model_runs[t] else ""
                for t in GROUP_MAP.keys()
            ],
        }
    )
    mo.as_html(model_inventory)
    return model_inventory, model_runs


@app.cell
def _(mo, model_inventory):
    mo.stop(
        model_inventory["n_runs"].sum() == 0,
        mo.callout(
            mo.md(
                "**No W&B runs found in any model group.** §6 will render "
                "skeletons only. Once the three new training groups land "
                "(`stage_5d_tier_c_legacy_bugged_s0`, "
                "`stage_5d_tier_c_legacy_bugfixed_s0`, "
                "`stage_5d_tier_c_m3_bugfixed_s0`), download via "
                "`scripts/python/download_wandb_runs.py --group <name>` and "
                "re-run."
            ),
            kind="warn",
        ),
    )

    mo.md(
        "Inventory above lists how many seeds are available per model. "
        "All metrics in §6 are computed across `test_preds`/`test_targets` "
        "pooled across seeds (typically 6 seeds × 275 graphs = 1 650 points)."
    )
    return


@app.cell
def _(model_runs, np, pd):
    def _unscale(arr, mean, scale):
        return arr * scale + mean

    def _metrics_for_runs(runs):
        if not runs:
            return None
        _preds = np.concatenate([r["preds"] for r in runs], axis=0)
        _targs = np.concatenate([r["targets"] for r in runs], axis=0)
        _properties = runs[0]["properties"]
        out = {"properties": _properties, "n": _preds.shape[0]}
        _mse_per_prop = np.mean((_preds - _targs) ** 2, axis=0)
        _ss_res = np.sum((_preds - _targs) ** 2, axis=0)
        _ss_tot = np.sum((_targs - _targs.mean(axis=0, keepdims=True)) ** 2, axis=0)
        _r2 = 1.0 - _ss_res / np.where(_ss_tot > 0, _ss_tot, np.nan)
        out["mse_per_prop"] = _mse_per_prop
        out["r2_per_prop"] = _r2
        out["preds"] = _preds
        out["targets"] = _targs
        return out

    metrics = {_tag: _metrics_for_runs(_runs) for _tag, _runs in model_runs.items()}

    _rows = []
    for _tag, _m in metrics.items():
        if _m is None:
            continue
        for _i, _prop in enumerate(_m["properties"]):
            _rows.append(
                {
                    "model": _tag,
                    "property": _prop,
                    "test_mse": float(_m["mse_per_prop"][_i]),
                    "pooled_r2": float(_m["r2_per_prop"][_i]),
                    "n_points": _m["n"],
                }
            )
    df_metrics = pd.DataFrame(_rows)
    return df_metrics, metrics


@app.cell
def _(df_metrics, mo):
    if df_metrics.empty:
        _out = mo.md("_no model metrics yet_")
    else:
        _wide_mse = df_metrics.pivot(
            index="property", columns="model", values="test_mse"
        ).round(4)
        _wide_r2 = df_metrics.pivot(
            index="property", columns="model", values="pooled_r2"
        ).round(3)
        _out = mo.vstack(
            [
                mo.md("**Test MSE (normalised) per property × model**"),
                mo.as_html(_wide_mse),
                mo.md("**Pooled R² per property × model**"),
                mo.as_html(_wide_r2),
            ]
        )
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6b. Paired t-tests across label sets

    For each pair of models trained on different label sets, the paired
    test points share the same `(seed, system_idx)` triple. Per property,
    pair the per-point squared errors and compute paired `t`. Significant
    `|t|` means the labels did not just shift the targets — they shifted
    the achievable error band.
    """)
    return


@app.cell
def _(df_metrics, metrics, pd, stats):
    PAIRS = [
        ("model_legacy_bugfixed_s0", "model_legacy_bugged_s0", "bugfix"),
        ("model_m3_bugfixed_s0", "model_legacy_bugfixed_s0", "ff"),
        ("model_m3_bugfixed_s0", "model_legacy_bugged_s0", "total"),
        ("model_legacy_bugged_s0", "model_legacy_bugged_random", "seed"),
    ]

    if df_metrics.empty:
        df_paired_t = pd.DataFrame()
    else:
        _rows = []
        for _a, _b, _name in PAIRS:
            _ma = metrics.get(_a)
            _mb = metrics.get(_b)
            if _ma is None or _mb is None:
                continue
            if _ma["preds"].shape != _mb["preds"].shape:
                _rows.append(
                    {
                        "contrast": _name,
                        "model_a": _a,
                        "model_b": _b,
                        "note": (
                            f"shape mismatch {_ma['preds'].shape} vs "
                            f"{_mb['preds'].shape}; can't pair"
                        ),
                    }
                )
                continue
            _sqerr_a = (_ma["preds"] - _ma["targets"]) ** 2
            _sqerr_b = (_mb["preds"] - _mb["targets"]) ** 2
            for _i, _prop in enumerate(_ma["properties"]):
                _t, _p = stats.ttest_rel(_sqerr_a[:, _i], _sqerr_b[:, _i])
                _rows.append(
                    {
                        "contrast": _name,
                        "model_a": _a,
                        "model_b": _b,
                        "property": _prop,
                        "mean_sqerr_a": float(_sqerr_a[:, _i].mean()),
                        "mean_sqerr_b": float(_sqerr_b[:, _i].mean()),
                        "delta_mse": float(_sqerr_a[:, _i].mean() - _sqerr_b[:, _i].mean()),
                        "paired_t": float(_t),
                        "t_p": float(_p),
                        "n_points": _sqerr_a.shape[0],
                    }
                )
        df_paired_t = pd.DataFrame(_rows)
    return (df_paired_t,)


@app.cell
def _(df_paired_t, mo):
    if df_paired_t.empty:
        _out = mo.md("_no paired data_")
    else:
        _pretty = df_paired_t.copy()
        _num_cols = [c for c in _pretty.columns if _pretty[c].dtype.kind in "fc"]
        _pretty[_num_cols] = _pretty[_num_cols].round(5)
        _out = mo.as_html(_pretty)
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6c. Per-system error scatter

    Per property × model, residual `pred − target` against system index,
    colored by composition family (heuristic: which lipid type dominates
    the composition string). Surfaces *where* the new labels help vs hurt.
    If bug fixes are pure noise removal, residuals shrink uniformly; if
    they unmask a structural problem, residuals redistribute toward a
    specific composition region.
    """)
    return


@app.cell
def _(model_runs, np, pd, re):
    def _family(name):
        _parts = re.findall(r"([A-Z][A-Z0-9]+?)(\d+)(?=_|$)", name)
        if not _parts:
            _m = re.match(r"^([A-Z]+)\d*$", name)
            return _m.group(1) if _m else "other"
        _shares = {lip: int(n) for lip, n in _parts}
        if "CHOL" in _shares:
            return "CHOL-mix"
        return max(_shares.items(), key=lambda kv: kv[1])[0]

    _rows = []
    for _tag, _runs in model_runs.items():
        for _run in _runs:
            _resid = _run["preds"] - _run["targets"]
            for _i, _prop in enumerate(_run["properties"]):
                for _k in range(_resid.shape[0]):
                    _comp = str(_run["compositions"][_k])
                    _rows.append(
                        {
                            "model": _tag,
                            "run_name": _run["run_name"],
                            "property": _prop,
                            "system_idx": int(_run["system_idx"][_k]),
                            "composition": _comp,
                            "family": _family(_comp),
                            "residual": float(_resid[_k, _i]),
                            "abs_residual": float(np.abs(_resid[_k, _i])),
                        }
                    )
    df_resid = pd.DataFrame(_rows)
    return (df_resid,)


@app.cell
def _(df_resid, metrics, mo, plt):
    if df_resid.empty:
        _out = mo.md("_no residual data_")
    else:
        _props = sorted(df_resid["property"].unique())
        _models = [t for t in metrics if metrics[t] is not None]
        _families = sorted(df_resid["family"].unique())
        _palette = plt.get_cmap("tab10")
        _fam_color = {f: _palette(i % 10) for i, f in enumerate(_families)}

        _fig, _axes = plt.subplots(
            len(_props), len(_models), figsize=(3.4 * len(_models), 2.4 * len(_props)),
            squeeze=False,
        )
        for _i, _prop in enumerate(_props):
            for _j, _model in enumerate(_models):
                _ax = _axes[_i, _j]
                _g = df_resid[(df_resid["property"] == _prop) & (df_resid["model"] == _model)]
                if _g.empty:
                    _ax.set_visible(False)
                    continue
                for _fam, _gg in _g.groupby("family"):
                    _ax.scatter(
                        _gg["system_idx"],
                        _gg["residual"],
                        s=8,
                        alpha=0.5,
                        color=_fam_color[_fam],
                        label=_fam if (_i == 0 and _j == 0) else None,
                    )
                _ax.axhline(0, color="k", lw=0.5, alpha=0.5)
                _ax.set_title(f"{_prop} — {_model}", fontsize=7)
                _ax.tick_params(labelsize=6)
                _ax.set_xlabel("system_idx", fontsize=7)
                _ax.set_ylabel("residual (norm)", fontsize=7)
        _axes[0, 0].legend(fontsize=6, loc="best")
        _fig.tight_layout()
        _out = _fig
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6d. Cross-model prediction agreement

    For each test point, compute the pairwise mean absolute difference
    between the three seeded models' predictions. If the models *agree*
    on predictions but disagree on residuals, the targets moved, not the
    learned function — clean evidence that the bug-fix/FF changes shifted
    labels without breaking what the GNN can extract from the input
    graph.
    """)
    return


@app.cell
def _(metrics, np, pd):
    _SEEDED = [
        "model_legacy_bugged_s0",
        "model_legacy_bugfixed_s0",
        "model_m3_bugfixed_s0",
    ]
    _avail = [t for t in _SEEDED if metrics.get(t) is not None]
    if len(_avail) < 2:
        df_agree = pd.DataFrame()
    else:
        _shape = metrics[_avail[0]]["preds"].shape
        _shapes_match = all(metrics[t]["preds"].shape == _shape for t in _avail)
        if not _shapes_match:
            df_agree = pd.DataFrame()
        else:
            _props = metrics[_avail[0]]["properties"]
            _rows = []
            for _i, _prop in enumerate(_props):
                _preds_by_model = np.stack(
                    [metrics[_t]["preds"][:, _i] for _t in _avail], axis=0
                )
                _spread = float(_preds_by_model.std(axis=0, ddof=1).mean())
                _mean_resid_spread = float(
                    (_preds_by_model - metrics[_avail[0]]["targets"][:, _i]).std(axis=0, ddof=1).mean()
                )
                _rows.append(
                    {
                        "property": _prop,
                        "mean_pred_spread": _spread,
                        "mean_resid_spread": _mean_resid_spread,
                        "n_models": len(_avail),
                    }
                )
            df_agree = pd.DataFrame(_rows)
    return (df_agree,)


@app.cell
def _(df_agree, mo):
    if df_agree.empty:
        _out = mo.md("_need ≥ 2 seeded models with matched shapes_")
    else:
        _out = mo.as_html(df_agree.round(5))
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Headline callout — verdict

    Aggregates §3a (seed dominance), §4 (variance decomposition), and
    §6a–b (model-side significance) into a four-branch verdict:

    - "seed noise dominates" — bug-fix conclusions need multi-seed
      property-side averaging.
    - "bug-fix changes labels but not model performance" — the GNN
      tolerates the relabel; legacy training results remain interpretable
      up to a label shift.
    - "labels and model both moved — retraining required" — Stage 5d
      numbers cannot be ported forward; the production pipeline needs
      training on the bugfixed labels.
    - "force-field swap is the dominant change" — the regression-target
      shift is driven by trajectory differences, not by code differences.

    Mechanical trigger from the pairwise notebook
    (`|paired_t| > 3` or `frac_|d| > SD_random > 0.5`) carries over per
    contrast.
    """)
    return


@app.cell
def _(df_paired_t, df_seed, df_summary, df_var, mo, np):
    if df_summary.empty:
        _out = mo.md("_verdict pending — no property data_")
    else:
        _msgs = []

        # Seed-dominance flag
        if not df_seed.empty:
            _flagged = df_seed[df_seed["frac_|d_seed|>SD_random"] > 0.5]
            if not _flagged.empty:
                _msgs.append(
                    "- **Seed noise dominates** on: "
                    + ", ".join(f"`{p}`" for p in _flagged["property"])
                )

        # Bug-fix / FF trigger from §3b
        _bug = df_summary[df_summary["contrast"] == "bugfix"]
        _trig_bug = _bug[
            (_bug["paired_t"].abs() > 3) | (_bug["frac_|d|>SD_random"] > 0.5)
        ]
        if not _trig_bug.empty:
            _msgs.append(
                "- **Bug-fix contrast triggers** on: "
                + ", ".join(f"`{p}`" for p in _trig_bug["property"])
            )
        _ff = df_summary[df_summary["contrast"] == "ff"]
        _trig_ff = _ff[(_ff["paired_t"].abs() > 3) | (_ff["frac_|d|>SD_random"] > 0.5)]
        if not _trig_ff.empty:
            _msgs.append(
                "- **FF contrast triggers** on: "
                + ", ".join(f"`{p}`" for p in _trig_ff["property"])
            )

        # Dominant variance component
        if not df_var.empty:
            for _, _row in df_var.iterrows():
                _vt = _row["var_total"]
                if _vt <= 0 or np.isnan(_vt):
                    continue
                _components = {
                    "seed": _row["var_d_seed"] / _vt,
                    "bugfix": _row["var_d_bugfix"] / _vt,
                    "ff": _row["var_d_ff"] / _vt,
                }
                _dominant, _share = max(_components.items(), key=lambda kv: kv[1])
                if _share > 0.5:
                    _msgs.append(
                        f"- `{_row['property']}` Var-decomp: **{_dominant} dominates** ({_share:.0%})"
                    )

        # Model-side regression on any property
        if not df_paired_t.empty and "property" in df_paired_t.columns:
            _ff_model = df_paired_t[
                (df_paired_t["contrast"] == "ff") & (df_paired_t["t_p"] < 0.05)
            ]
            if not _ff_model.empty:
                _msgs.append(
                    "- **Model-side FF effect significant** (p<0.05) on: "
                    + ", ".join(f"`{p}`" for p in _ff_model["property"])
                )

        _body = "\n".join(_msgs) if _msgs else "No triggers fired."
        _out = mo.callout(mo.md(_body), kind="warn" if _msgs else "info")
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    Populated once the data lands. Keep these four headings; write factual
    one-liners under each, neutral, no narrative:

    1. **Seed contribution** — fraction of Var(d_total) explained by
       Var(d_seed) per property. Seed-dominated properties are flagged
       as "needs multi-seed property-side averaging" — single-seed bug-fix
       conclusions are not actionable there.
    2. **Bug-fix contribution** — which properties moved materially under
       bug-fix-only (median Δ, paired t, frac>SD), and which §2 bugs the
       movers list points at.
    3. **Force-field contribution** — which properties moved under the FF
       swap on top of the bug-fix; whether the movers concentrate on
       CHOL-containing or DPPC-/DOPC-rich corners; alignment with the
       Stage 5b per-system MAE concentration.
    4. **Retraining verdict** — for each model contrast (seed, bugfix,
       ff, total), whether mean pooled test R² moved by > 0.05 and whether
       the paired-t in §6b is significant. Recommended action: keep
       Stage 5d numbers / require retraining / require retraining +
       resimulation, per property.

    Open questions to surface, even if absent from the data:
    - Is the property-seed effect (`model_legacy_bugged_random` vs
      `model_legacy_bugged_s0`) larger or smaller than the GNN-init
      seed jitter already reported in `progress.md`? If comparable, the
      property-side seed is irrelevant; if larger, all historical
      single-property-seed numbers need a property-seed-jitter band.
    - Does the bug-fix contrast preferentially affect `persistence` and
      `diffusivity` (bugs #2/#3 and #9 land directly there)? If not, the
      bug-fix is doing something the §2 enumeration didn't predict.
    """)
    return


if __name__ == "__main__":
    app.run()
