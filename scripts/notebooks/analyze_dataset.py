import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Dataset analysis — 70 Martini 3 membrane compositions

    A characterisation of the 70-system dataset that the GNN is trained against.
    Reads the precomputed property files in `results/properties/` (one pickle per
    composition, despite the `.h5` extension) and the parsed compositions.

    Sections:
    1. Composition catalogue + coverage heatmap
    2. Property definitions (annotated from `lipid_gnn.functions_emil.calculate_properties`)
    3. Univariate distributions
    4. Pairwise structure: correlations, pairplot, Helfrich sanity check
    5. Composition titration curves (POPC + X families)
    6. Dimensionality reduction & clustering (PCA, UMAP if available, hierarchical)
    7. Time-series quality (stationarity, autocorrelation, noise floor)
    8. Train/val/test split audit

    **Outputs**: `results/dataset_analysis/figures/*.{png,pdf}` and
    `results/dataset_analysis/dataset_table.csv`.
    """)
    return


@app.cell
def _():
    import pickle
    import re
    import warnings
    from pathlib import Path

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy import stats
    from scipy.cluster import hierarchy
    from scipy.spatial.distance import pdist
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    warnings.filterwarnings('ignore')

    from lipid_gnn.config import CONFIG

    PROPS_DIR = Path(CONFIG.paths.props_dir)
    OUT_DIR   = Path(CONFIG.paths.results_dir) / 'dataset_analysis'
    FIG_DIR   = OUT_DIR / 'figures'
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    PROPS = list(CONFIG.vocab.all_properties)
    LIPIDS = list(CONFIG.vocab.lipid_types)
    REFERENCE = CONFIG.dataset.reference_system  # 'POPC100'

    print(f'props dir: {PROPS_DIR}')
    print(f'output  : {OUT_DIR}')
    print(f'{len(LIPIDS)} lipid types, {len(PROPS)} properties')
    return (
        CONFIG,
        FIG_DIR,
        LIPIDS,
        OUT_DIR,
        PCA,
        PROPS,
        PROPS_DIR,
        Path,
        REFERENCE,
        StandardScaler,
        hierarchy,
        np,
        pd,
        pdist,
        pickle,
        plt,
        re,
        stats,
    )


@app.cell
def _():
    try:
        import umap as umap_lib
        HAS_UMAP = True
    except ImportError:
        umap_lib = None
        HAS_UMAP = False
        print('[INFO] umap-learn not installed — Section 6 falls back to PCA-only.')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1 — Composition catalogue

    The 70 filenames encode binary compositions like `POPC70_CHOL30` or pure systems
    `POPC100`. We parse `(lipid_A, frac_A, lipid_B, frac_B)` and load the
    8 mean properties for each.
    """)
    return


@app.cell
def _(plt, re):
    # Lipid family taxonomy (headgroup + tail saturation)
    LIPID_FAMILY = {
        'POPC': 'PC (mono-unsat)', 'DOPC': 'PC (di-unsat)', 'DIPC': 'PC (di-unsat)',
        'DPPC': 'PC (saturated)', 'POPE': 'PE', 'DOPE': 'PE', 'DPPE': 'PE',
        'POPS': 'PS (anionic)', 'DOPS': 'PS (anionic)', 'CHOL': 'sterol',
    }
    FAMILY_ORDER = ['PC (mono-unsat)', 'PC (di-unsat)', 'PC (saturated)', 'PE', 'PS (anionic)', 'sterol']
    FAMILY_COLORS = dict(zip(FAMILY_ORDER, plt.get_cmap('tab10').colors[:len(FAMILY_ORDER)]))
    _PARSE = re.compile(r'([A-Z]+)(\d+)')

    def parse_composition(stem):
        """Parse 'POPC70_CHOL30' → (POPC, 70, CHOL, 30); 'POPC100' → (POPC, 100, None, 0)."""
        parts = _PARSE.findall(stem)
        if len(parts) == 1:
            lip, frac = parts[0]
            return (lip, int(frac), None, 0)
        (_a, _fa), (_b, _fb) = parts[:2]
        return (_a, int(_fa), _b, int(_fb))

    def partner(lip_a, lip_b):
        """Non-POPC component (or the only component if pure)."""
        if lip_a == 'POPC':
            return lip_b if lip_b is not None else 'POPC'
        if lip_b == 'POPC':
            return lip_a
        return lip_a

    def partner_frac(lip_a, fa, lip_b, fb):
        """Mole fraction of the partner lipid (the non-POPC one)."""
        if lip_b is None:
            return 0 if lip_a == 'POPC' else 100
        return fb if lip_a == 'POPC' else fa  # pure non-POPC (DIPC100, DOPC100, DPPC100)

    return (
        FAMILY_COLORS,
        LIPID_FAMILY,
        parse_composition,
        partner,
        partner_frac,
    )


@app.cell
def _(
    LIPID_FAMILY,
    PROPS,
    PROPS_DIR,
    parse_composition,
    partner,
    partner_frac,
    pd,
    pickle,
):
    def _load_one(path):
        with open(path, 'rb') as _f:
            return pickle.load(_f)

    _rows = []
    RAW = {}
    for _p in sorted(PROPS_DIR.glob('*.h5')):
        _mean_d, _raw_d = _load_one(_p)
        _a, _fa, _b, _fb = parse_composition(_p.stem)
        _row = {
            'composition': _p.stem,
            'lipid_a': _a, 'frac_a': _fa,
            'lipid_b': _b, 'frac_b': _fb,
            'partner': partner(_a, _b),
            'partner_frac': partner_frac(_a, _fa, _b, _fb),
            'is_pure': _b is None,
        }
        _row['family'] = (
            LIPID_FAMILY[_row['partner']]
            if _row['partner_frac'] > 0 or _row['is_pure']
            else 'PC (mono-unsat)'
        )
        for _k in PROPS:
            _row[_k] = float(_mean_d[_k])
        _rows.append(_row)
        RAW[_p.stem] = _raw_d

    df = pd.DataFrame(_rows).sort_values('composition').reset_index(drop=True)
    print(f'loaded {len(df)} compositions')
    return RAW, df


@app.cell
def _(OUT_DIR, df):
    table_path = OUT_DIR / 'dataset_table.csv'
    df.to_csv(table_path, index=False)
    print(f'wrote {table_path}')
    return (table_path,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 1a — Coverage of the composition space

    Each non-POPC partner lipid is paired with POPC at a discrete set of fractions.
    The heatmap below shows, for every (partner × fraction) cell, whether that
    composition was simulated. Pure systems (DIPC100, DOPC100, DPPC100, POPC100)
    appear in the rightmost / dedicated columns.
    """)
    return


@app.cell
def _(FIG_DIR, df, np, plt):
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.patches import Patch

    _partners = sorted(set(df['partner']) - {'POPC'})
    _fracs = sorted(set(df['partner_frac']) - {0})
    _cov = np.zeros((len(_partners), len(_fracs)), dtype=float)

    for _, _r in df.iterrows():
        if _r['partner'] == 'POPC':
            continue
        _pi = _partners.index(_r['partner'])
        _fi = _fracs.index(_r['partner_frac'])
        _cov[_pi, _fi] = 1.0

    _popc_fracs_present = set(df.loc[df['lipid_a'] == 'POPC', 'frac_a'].tolist())
    _popc_row = np.zeros((1, len(_fracs)), dtype=float)
    for _j, _f in enumerate(_fracs):
        _popc_frac = 100 - _f
        if _popc_frac in _popc_fracs_present:
            _popc_row[0, _j] = 2.0
        elif _popc_frac == 0 and (df['is_pure'] & (df['lipid_a'] != 'POPC')).any():
            _popc_row[0, _j] = 2.0

    _combined = np.vstack([_popc_row, _cov])
    _y_labels = ['POPC\n(100−x%)'] + _partners
    _cmap3 = ListedColormap(['white', '#4472C4', '#70AD47'])
    _norm3 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], _cmap3.N)

    _fig, _ax = plt.subplots(figsize=(12, 5.5))
    _ax.imshow(_combined, cmap=_cmap3, norm=_norm3, aspect='auto')
    _ax.set_xticks(range(len(_fracs)))
    _ax.set_xticklabels(_fracs, fontsize=8)
    _ax.set_xlabel('partner mole fraction (%)', labelpad=4)
    _ax.set_yticks(range(len(_partners) + 1))
    _ax.set_yticklabels(_y_labels, fontsize=9)
    _ax.axhline(0.5, color='black', lw=2.0)
    for _i in range(len(_partners) + 1):
        for _j in range(len(_fracs)):
            if _combined[_i, _j] > 0:
                _ax.text(_j, _i, '•', ha='center', va='center', color='white', fontsize=13)
    _ax2 = _ax.twiny()
    _ax2.set_xlim(_ax.get_xlim())
    _ax2.set_xticks(range(len(_fracs)))
    _ax2.set_xticklabels([f'{100 - _f}' for _f in _fracs], fontsize=7, rotation=45, ha='left')
    _ax2.set_xlabel('POPC mol% (read for POPC row only: = 100 − bottom-axis value)', fontsize=8)
    _ax.legend(
        handles=[
            Patch(facecolor='#4472C4', label='partner lipid covered (bottom axis)'),
            Patch(facecolor='#70AD47', label='POPC covered (top axis: 100−x%)'),
            Patch(facecolor='white', edgecolor='lightgray', label='not simulated'),
        ],
        loc='lower right', fontsize=8, framealpha=0.9,
    )
    _ax.set_title(
        f"Composition coverage — {len(df)} systems ({df['is_pure'].sum()} pure, "
        f"{(~df['is_pure']).sum()} mixtures)\n"
        "POPC100 (pure) is present but its partner_frac=0 is excluded from the x-axis"
    )
    plt.tight_layout()
    plt.savefig(FIG_DIR / '01_coverage.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '01_coverage.pdf', bbox_inches='tight')

    print('Pure systems:', sorted(df.loc[df['is_pure'], 'composition'].tolist()))
    print('\nPOPC coverage gaps (x-axis positions where POPC is NOT represented):')
    _gaps = [(100 - _f, _f) for _j, _f in enumerate(_fracs) if _popc_row[0, _j] == 0]
    if _gaps:
        for _popc_f, _partner_f in _gaps:
            print(f'  POPC at {_popc_f:3d}%  (partner at {_partner_f}%) — not simulated')
    else:
        print('  (none — all partner-frac positions have at least one POPC composition)')
    print('  POPC at 100% — covered by POPC100 [not shown on axis, partner_frac=0]')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2 — Property definitions

    Eight scalars are computed per composition by
    `lipid_gnn.functions_emil.calculate_properties.compute_properties` over frames
    `[50:667]` (≈1 µs of production after a 50-frame equilibration discard,
    `dt=1.5 ns`).

    | Name | Definition | Units |
    |---|---|---|
    | `lipid_packing` | N_lipids / box_x · box_y, frame-mean | lipids / nm² |
    | `thickness` | mean (upper − lower) leaflet height on a 0.1 nm xy grid | Å |
    | `thickness_std` | frame-mean of the spatial std of (upper − lower) | Å |
    | `compressibility` | variance of thickness deviations on the grid (×100) | Å³ / kT |
    | `bending_modulus` | κ from a `kBT/(κ q⁴)` fit of the radially-binned undulation spectrum | kT / Å³ |
    | `persistence` | P(lipid–lipid contact still present after `lag=50` frames), `cutoff=0.7 nm` | dimensionless |
    | `diffusivity` | mean lateral squared displacement after `lag=10` frames (×100) | Å² |
    | `variation` | mean Voronoi-cell-area coefficient of variation (per leaflet, then averaged) | dimensionless |

    #### Implementation notes

    - **Thickness** uses `LinearNDInterpolator` on PO4 beads. The leaflet split is
      the largest gap in sorted z (`calculate_properties.py:206-209`). Frames where
      the interpolator returns NaNs anywhere on the grid are silently dropped.
    - **Compressibility** is computed as the variance of `xy_thickness − thickness_series[:, None]`
      on the 0.1 nm grid (`:335-337`) — this is a thickness-fluctuation variance,
      not the canonical area-compressibility modulus K_A (which would use total
      area fluctuations). Treat it as a thickness-curvature noise measure.
    - **Bending modulus** is fit on q > 0.1 nm⁻¹ from the radially-binned 2D-FFT
      spectrum of the midplane height (`:90-110`). Few q-bins survive the bin-count
      threshold, so this property is the noisiest of the eight (flagged in the
      memory bank).
    - **Persistence and diffusivity** sample `probe_size=10` random lipids per frame.
      Both routines have a leaflet-selection identity bug: the if/else branches
      pick the same lipids (`< cutoff` in both, `:238-241` and `:282-285`). Effective
      sampling is leaflet-mixed but still random, so the means remain meaningful;
      flagging here for thesis transparency.
    - **Variation** computes Voronoi CV per leaflet then averages. Cells touching
      the periodic box are clipped to the box (`shapely`).

    The `mean_dict[prop]` scalar is what the GNN regresses against;
    `raw_dict[prop]` is the per-frame series (used in Section 7), except for
    `bending_modulus` whose "raw" entry is the (q_centers, power_spectrum) pair
    from the spectrum fit, not a time series.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3 — Univariate property distributions

    For each property: KDE+rug across the 70 systems, coloured strip by partner
    family, with the POPC100 reference value marked. Outliers (|z|>3 via MAD) are
    listed below.
    """)
    return


@app.cell
def _(FAMILY_COLORS, FIG_DIR, PROPS, REFERENCE, df, mo, plt):
    ref_row = df[df['composition'] == REFERENCE].iloc[0]
    _fig, _axes = plt.subplots(4, 2, figsize=(12, 14))
    for _ax, _prop in zip(_axes.flat, PROPS):
        _vals = df[_prop].values
        _ax.hist(_vals, bins=20, color='lightgray', edgecolor='k', alpha=0.7)
        _ymax = _ax.get_ylim()[1]
        for _fam, _color in FAMILY_COLORS.items():
            _mask = df['family'] == _fam
            if _mask.any():
                _ax.vlines(df.loc[_mask, _prop], 0, _ymax * 0.05, color=_color, alpha=0.9, lw=1.5)
        _ax.axvline(ref_row[_prop], color='red', linestyle='--', lw=1.5, label=f'POPC100 = {ref_row[_prop]:.3g}')
        _ax.set_title(_prop)
        _ax.set_xlabel(_prop)
        _ax.set_ylabel('count')
        _ax.legend(fontsize=8, loc='upper right')
    _handles = [plt.Line2D([0], [0], color=_c, lw=2) for _c in FAMILY_COLORS.values()]
    _fig.legend(_handles, list(FAMILY_COLORS.keys()), loc='lower center', ncol=len(FAMILY_COLORS), bbox_to_anchor=(0.5, -0.01), frameon=False)
    _fig.suptitle('Property distributions across 70 compositions', y=1.0, fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / '03_univariate.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '03_univariate.pdf', bbox_inches='tight')
    mo.output.replace(_fig)
    return (ref_row,)


@app.cell
def _(PROPS, df, np, pd):
    def _mad_z(x):
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        return 0.6745 * (x - med) / (mad if mad > 0 else 1.0)

    _summary = []
    _outlier_rows = []
    for _prop in PROPS:
        _v = df[_prop].values
        _z = _mad_z(_v)
        _out_mask = np.abs(_z) > 3
        _summary.append({
            'property': _prop, 'mean': _v.mean(), 'median': np.median(_v),
            'std': _v.std(ddof=1), 'min': _v.min(), 'max': _v.max(),
            'IQR': np.percentile(_v, 75) - np.percentile(_v, 25),
            'n_outliers (|z_MAD|>3)': int(_out_mask.sum()),
        })
        for _idx in np.where(_out_mask)[0]:
            _outlier_rows.append({'composition': df.iloc[_idx]['composition'], 'property': _prop, 'value': _v[_idx], 'z_MAD': _z[_idx]})

    print('Summary:')
    print(pd.DataFrame(_summary).to_string(index=False))
    print('\nOutliers (|z_MAD| > 3):')
    print(
        pd.DataFrame(_outlier_rows).sort_values('z_MAD', key=np.abs, ascending=False).to_string(index=False)
        if _outlier_rows else '(none)'
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4 — Pairwise structure

    ### 4a — Correlation matrix (Pearson + Spearman)
    """)
    return


@app.cell
def _(FIG_DIR, PROPS, df, plt):
    _pearson = df[PROPS].corr(method='pearson')
    _spearman = df[PROPS].corr(method='spearman')
    _fig, _axes = plt.subplots(1, 2, figsize=(14, 6))
    for _ax, _mat, _title in zip(_axes, [_pearson, _spearman], ['Pearson', 'Spearman']):
        _im = _ax.imshow(_mat, cmap='RdBu_r', vmin=-1, vmax=1)
        _ax.set_xticks(range(len(PROPS)))
        _ax.set_xticklabels(PROPS, rotation=45, ha='right')
        _ax.set_yticks(range(len(PROPS)))
        _ax.set_yticklabels(PROPS)
        _ax.set_title(f'{_title} correlation')
        for _i in range(len(PROPS)):
            for _j in range(len(PROPS)):
                _ax.text(_j, _i, f'{_mat.iloc[_i, _j]:.2f}', ha='center', va='center',
                         color='white' if abs(_mat.iloc[_i, _j]) > 0.5 else 'black', fontsize=8)
        plt.colorbar(_im, ax=_ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(FIG_DIR / '04a_correlations.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '04a_correlations.pdf', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4b — Pairplot, coloured by partner family
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
                _ax.hist(df[_px], bins=15, color='lightgray', edgecolor='k')
            else:
                for _fam, _color in FAMILY_COLORS.items():
                    _mask = df['family'] == _fam
                    if _mask.any():
                        _ax.scatter(df.loc[_mask, _px], df.loc[_mask, _py], c=[_color], s=15, alpha=0.7, edgecolor='none')
            if _i == _n - 1:
                _ax.set_xlabel(_px, fontsize=9)
            else:
                _ax.set_xticklabels([])
            if _j == 0:
                _ax.set_ylabel(_py, fontsize=9)
            else:
                _ax.set_yticklabels([])
            _ax.tick_params(labelsize=7)
    _handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=_c, markersize=8) for _c in FAMILY_COLORS.values()]
    _fig.legend(_handles, list(FAMILY_COLORS.keys()), loc='upper center', ncol=len(FAMILY_COLORS), bbox_to_anchor=(0.5, 1.0), frameon=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / '04b_pairplot.png', dpi=130, bbox_inches='tight')
    plt.savefig(FIG_DIR / '04b_pairplot.pdf', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 4c — Helfrich elastic-theory sanity check

    For a fluid bilayer, Helfrich theory predicts
    `κ ≈ K_A · d² / 24` (with `d` = thickness, `K_A` ≈ area compressibility,
    κ = bending modulus). Emil's `compressibility` is a thickness-fluctuation
    proxy rather than the strict K_A, but a positive correlation between
    `compressibility · thickness²` and `bending_modulus` would still indicate that
    the bending fits are not pure noise.
    """)
    return


@app.cell
def _(FAMILY_COLORS, FIG_DIR, df, plt, stats):
    _x = df['compressibility'] * df['thickness'] ** 2
    _y = df['bending_modulus']
    _r_pear, _p_pear = stats.pearsonr(_x, _y)
    _r_spear, _p_spear = stats.spearmanr(_x, _y)
    _fig, _ax = plt.subplots(figsize=(7, 5))
    for _fam, _color in FAMILY_COLORS.items():
        _m = df['family'] == _fam
        _ax.scatter(_x[_m], _y[_m], c=[_color], s=40, alpha=0.8, label=_fam, edgecolor='k', lw=0.3)
    _ax.set_xlabel('compressibility · thickness²  [Å⁵ / kT]')
    _ax.set_ylabel('bending_modulus  [kT / Å³]')
    _ax.set_title(f'Helfrich proxy:  Pearson r={_r_pear:.2f} (p={_p_pear:.1e})  |  Spearman ρ={_r_spear:.2f} (p={_p_spear:.1e})')
    _ax.legend(fontsize=8, loc='best')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '04c_helfrich.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '04c_helfrich.pdf', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5 — Composition titration curves

    For each non-POPC partner lipid, plot every property vs partner mole fraction
    (0 % = POPC100 anchor; 100 % = pure partner where present). Reveals what
    "property knobs" each partner controls.
    """)
    return


@app.cell
def _(df):
    partners_with_data = sorted(set(df.loc[df['partner_frac'] > 0, 'partner']))
    print('Partners with titration data:', partners_with_data)
    return (partners_with_data,)


@app.cell
def _(FIG_DIR, PROPS, REFERENCE, df, partners_with_data, plt, ref_row):
    _partner_colors = dict(zip(partners_with_data, plt.get_cmap('tab10').colors[:len(partners_with_data)]))
    _fig, _axes = plt.subplots(4, 2, figsize=(13, 16))
    for _ax, _prop in zip(_axes.flat, PROPS):
        _ax.axhline(ref_row[_prop], color='gray', linestyle=':', lw=1, label='POPC100')
        for _part in partners_with_data:
            _m = (df['partner'] == _part) | ((df['lipid_a'] == 'POPC') & (df['frac_a'] == 100))
            _sub = df[_m].sort_values('partner_frac')
            _sub = _sub[(_sub['partner'] == _part) | (_sub['composition'] == REFERENCE)]
            _sub = _sub.sort_values('partner_frac')
            _ax.plot(_sub['partner_frac'], _sub[_prop], 'o-', color=_partner_colors[_part],
                     label=_part, markersize=5, lw=1.2)
        _ax.set_title(_prop)
        _ax.set_xlabel('partner mole fraction (%)')
        _ax.set_ylabel(_prop)
        _ax.grid(alpha=0.3)
    _handles, _labels = _axes[0, 0].get_legend_handles_labels()
    _fig.legend(_handles, _labels, loc='lower center', ncol=min(len(_labels), 6),
                bbox_to_anchor=(0.5, -0.02), frameon=False, fontsize=9)
    _fig.suptitle('Titration curves — property vs partner-lipid fraction', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / '05_titration.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '05_titration.pdf', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6 — Dimensionality reduction & clustering

    All 8 properties z-scored, then projected. Coloured by partner family.
    """)
    return


@app.cell
def _(FAMILY_COLORS, FIG_DIR, PCA, PROPS, StandardScaler, df, mo, np, plt):
    X = df[PROPS].values
    X_z = StandardScaler().fit_transform(X)
    pca = PCA(n_components=min(8, X_z.shape[1]))
    Z_pca = pca.fit_transform(X_z)
    _explained = pca.explained_variance_ratio_

    _fig = plt.figure(figsize=(15, 5))
    _ax = _fig.add_subplot(1, 3, 1)
    _ax.bar(range(1, len(_explained) + 1), _explained * 100, color='steelblue')
    _ax.plot(range(1, len(_explained) + 1), np.cumsum(_explained) * 100, 'ro-', lw=1.5)
    _ax.set_xlabel('PC')
    _ax.set_ylabel('variance explained (%)')
    _ax.set_title(f'Scree — PC1+PC2 = {(_explained[0] + _explained[1]) * 100:.1f}%')
    _ax.grid(alpha=0.3)

    _ax = _fig.add_subplot(1, 3, 2)
    for _fam, _color in FAMILY_COLORS.items():
        _m = df['family'] == _fam
        if _m.any():
            _ax.scatter(Z_pca[_m, 0], Z_pca[_m, 1], c=[_color], s=40, alpha=0.8, label=_fam, edgecolor='k', lw=0.3)
    _ax.set_xlabel(f'PC1 ({_explained[0] * 100:.1f}%)')
    _ax.set_ylabel(f'PC2 ({_explained[1] * 100:.1f}%)')
    _ax.set_title('PCA — coloured by partner family')
    _ax.legend(fontsize=8, loc='best')
    _ax.grid(alpha=0.3)

    _ax = _fig.add_subplot(1, 3, 3)
    for _i, _prop in enumerate(PROPS):
        _ax.arrow(0, 0, pca.components_[0, _i], pca.components_[1, _i], head_width=0.03, color='k', alpha=0.7)
        _ax.text(pca.components_[0, _i] * 1.15, pca.components_[1, _i] * 1.15, _prop, fontsize=9, ha='center')
    _ax.axhline(0, color='gray', lw=0.5)
    _ax.axvline(0, color='gray', lw=0.5)
    _ax.set_xlabel('PC1 loading')
    _ax.set_ylabel('PC2 loading')
    _ax.set_title('Loadings biplot')
    _ax.set_xlim(-1, 1)
    _ax.set_ylim(-1, 1)
    _ax.set_aspect('equal')
    _ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / '06a_pca.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '06a_pca.pdf', bbox_inches='tight')
    mo.output.replace(_fig)
    return X_z, Z_pca, pca


app._unparsable_cell(
    r"""
    if HAS_UMAP:
        _fig, _axes = plt.subplots(1, 2, figsize=(13, 5))
        for _ax, _n_neigh in zip(_axes, [15, 8]):
            _reducer = umap_lib.UMAP(n_components=2, n_neighbors=_n_neigh, min_dist=0.1, random_state=0)
            _Z_u = _reducer.fit_transform(X_z)
            for _fam, _color in FAMILY_COLORS.items():
                _m = df['family'] == _fam
                if _m.any():
                    _ax.scatter(_Z_u[_m, 0], _Z_u[_m, 1], c=[_color], s=40, alpha=0.8, label=_fam, edgecolor='k', lw=0.3)
            _ax.set_title(f'UMAP (n_neighbors={_n_neigh})')
            _ax.set_xlabel('UMAP1')
            _ax.set_ylabel('UMAP2')
            if _n_neigh == 15:
                _ax.legend(fontsize=8, loc='best')
        plt.tight_layout()
        plt.savefig(FIG_DIR / '06b_umap.png', dpi=150, bbox_inches='tight')
        plt.savefig(FIG_DIR / '06b_umap.pdf', bbox_inches='tight')
        return _fig
    else:
        print('UMAP skipped (umap-learn not installed). pip install umap-learn to enable.')
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6c — Hierarchical clustering (Ward)

    Dendrogram of the z-scored property vectors. Cut at k=4 to compare against
    the partner-family taxonomy.
    """)
    return


@app.cell
def _(FIG_DIR, X_z, df, hierarchy, mo, pd, pdist, plt):
    _dist = pdist(X_z, metric='euclidean')
    Z_link = hierarchy.linkage(_dist, method='ward')

    _fig, _ax = plt.subplots(figsize=(16, 6))
    _labels = (df['composition'] + '  [' + df['family'] + ']').tolist()
    hierarchy.dendrogram(Z_link, labels=_labels, leaf_rotation=90, leaf_font_size=7,
                         color_threshold=0.7 * Z_link[:, 2].max(), ax=_ax)
    _ax.set_title('Hierarchical clustering (Ward, z-scored properties)')
    _ax.set_ylabel('Ward distance')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '06c_dendrogram.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '06c_dendrogram.pdf', bbox_inches='tight')
    mo.output.replace(_fig)

    _k = 4
    _clusters = hierarchy.fcluster(Z_link, t=_k, criterion='maxclust')
    print(f'\nCluster (k={_k}) × family contingency:')
    print(pd.crosstab(_clusters, df['family']))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 6d — Outlier diagnostics

    Mahalanobis distance and PCA reconstruction error in the z-scored 8-D space.
    High values flag systems whose property profile is unusual relative to the bulk.
    """)
    return


@app.cell
def _(X_z, Z_pca, df, np, pca, pd):
    _mu = X_z.mean(axis=0)
    _cov = np.cov(X_z, rowvar=False) + 1e-6 * np.eye(X_z.shape[1])
    _inv_cov = np.linalg.inv(_cov)
    _diff = X_z - _mu
    _mahal = np.sqrt(np.einsum('ij,jk,ik->i', _diff, _inv_cov, _diff))
    _recon = Z_pca[:, :2] @ pca.components_[:2]
    _recon_err = np.linalg.norm(X_z - _recon, axis=1)
    _out_df = pd.DataFrame({
        'composition': df['composition'], 'family': df['family'],
        'mahalanobis': _mahal, 'pca2_recon_err': _recon_err,
    }).sort_values('mahalanobis', ascending=False)
    print('Top 10 by Mahalanobis distance:')
    print(_out_df.head(10).to_string(index=False))
    print('\nTop 10 by PCA(top-2) reconstruction error:')
    print(_out_df.sort_values('pca2_recon_err', ascending=False).head(10).to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7 — Time-series quality

    The per-frame `raw_dict` series let us check stationarity and estimate
    intrinsic frame-to-frame noise. `bending_modulus` is excluded — its `raw`
    entry is the (q, |h(q)|²) spectrum, not a time series.

    Frame spacing is `dt = 1.5 ns`; series cover roughly 1 µs of production after
    the equilibration discard.
    """)
    return


@app.cell
def _(FIG_DIR, PROPS, RAW, REFERENCE, mo, np, plt):
    TS_PROPS = [_p for _p in PROPS if _p != 'bending_modulus']
    DT_NS = 1.5  # frame spacing in ns

    def get_series(comp, prop):
        _v = np.asarray(RAW[comp][prop])
        if _v.ndim != 1:
            return None
        return _v

    _fig, _axes = plt.subplots(len(TS_PROPS), 1, figsize=(11, 1.6 * len(TS_PROPS)), sharex=True)
    for _ax, _prop in zip(_axes, TS_PROPS):
        _s = get_series(REFERENCE, _prop)
        _t = np.arange(len(_s)) * DT_NS / 1000  # µs
        _ax.plot(_t, _s, lw=0.8, color='steelblue')
        _ax.axhline(np.mean(_s), color='red', linestyle='--', lw=0.7, label=f'mean = {np.mean(_s):.3g}')
        _ax.set_ylabel(_prop, fontsize=9)
        _ax.legend(fontsize=7, loc='upper right')
        _ax.grid(alpha=0.3)
    _axes[-1].set_xlabel('time (µs)')
    _fig.suptitle(f'Per-frame property series — {REFERENCE}', fontsize=12)
    plt.tight_layout()
    plt.savefig(FIG_DIR / '07a_timeseries_popc.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '07a_timeseries_popc.pdf', bbox_inches='tight')
    mo.output.replace(_fig)
    return TS_PROPS, get_series


@app.cell
def _(TS_PROPS, df, get_series, np, pd, stats):
    def _autocorr_time(x, max_lag=200):
        """Integrated autocorrelation time (sum until first negative ACF)."""
        x = np.asarray(x) - np.mean(x)
        var = np.var(x)
        if var <= 0:
            return np.nan
        n = len(x)
        _max_lag = min(max_lag, n // 4)
        acf = []
        for _k in range(1, _max_lag + 1):
            _c = np.dot(x[:-_k], x[_k:]) / (var * (n - _k))
            if _c < 0:
                break
            acf.append(_c)
        return 1 + 2 * sum(acf)

    _quality_rows = []
    for _prop in TS_PROPS:
        _within_stds = []
        _drifts_rel = []
        _tau_list = []
        _last_n = 0
        for _comp in df['composition']:
            _s = get_series(_comp, _prop)
            if _s is None or len(_s) < 20:
                continue
            _t = np.arange(len(_s))
            _slope, *_ = stats.linregress(_t, _s)
            _drifts_rel.append(abs(_slope) * len(_s) / (np.std(_s) + 1e-12))
            _within_stds.append(np.std(_s))
            _tau_list.append(_autocorr_time(_s))
            _last_n = len(_s)
        _within_std = np.mean(_within_stds)
        _between_std = df[_prop].std(ddof=1)
        _tau_mean = np.nanmean(_tau_list)
        _n_eff = _last_n / _tau_mean if (_tau_mean > 0 and _last_n > 0) else np.nan
        _quality_rows.append({
            'property': _prop,
            'within_std': _within_std,
            'between_std': _between_std,
            'snr (between/within)': _between_std / _within_std if _within_std > 0 else np.nan,
            'autocorr_tau (frames)': _tau_mean,
            'n_eff per system': _n_eff,
            'mean |drift|·N / std': np.mean(_drifts_rel),
        })

    q_df = pd.DataFrame(_quality_rows)
    print('Time-series quality (per property):')
    print(q_df.to_string(index=False))
    return (q_df,)


@app.cell
def _(FIG_DIR, np, plt, q_df):
    _fig, _ax = plt.subplots(figsize=(8, 5))
    _xpos = np.arange(len(q_df))
    _w = 0.4
    _ax.bar(_xpos - _w / 2, q_df['within_std'], _w, label='within-system std (frame noise)', color='lightcoral')
    _ax.bar(_xpos + _w / 2, q_df['between_std'], _w, label='between-system std (signal)', color='steelblue')
    _ax.set_yscale('log')
    _ax.set_xticks(_xpos)
    _ax.set_xticklabels(q_df['property'], rotation=30, ha='right')
    _ax.set_ylabel('std (property units, log scale)')
    _ax.set_title('Noise floor — between-system vs within-system std')
    for _i, _snr in enumerate(q_df['snr (between/within)']):
        _ax.text(_i, max(q_df.loc[_i, 'within_std'], q_df.loc[_i, 'between_std']) * 1.1,
                 f'SNR={_snr:.1f}', ha='center', fontsize=8)
    _ax.legend()
    _ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(FIG_DIR / '07b_noise_floor.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '07b_noise_floor.pdf', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8 — Train/val/test split audit

    The Tier A pipeline preprocesses graphs into `colab_lipid_gnn_subset/processed/{train,val,test}/chunk_*.pt`.
    We list which compositions land in each split, and check whether property
    distributions match across splits (a key concern after the test-narrowness
    fix in early 2026-04).
    """)
    return


@app.cell
def _(CONFIG, Path):
    import torch
    _CHUNKS_DIR = Path(CONFIG.paths.chunks_dir)
    SPLITS = ['train', 'val', 'test']
    split_compositions = {_s: set() for _s in SPLITS}
    for _s in SPLITS:
        _sd = _CHUNKS_DIR / _s
        if not _sd.exists():
            print(f'[skip] {_sd} not present')
            continue
        for _chunk in sorted(_sd.glob('chunk_*.pt')):
            _graphs = torch.load(_chunk, weights_only=False)
            for _g in _graphs:
                _comp = getattr(_g, 'composition', None)
                if _comp is not None:
                    split_compositions[_s].add(_comp)
    for _s in SPLITS:
        print(f'{_s:>5s}: {len(split_compositions[_s]):3d} compositions')
    return SPLITS, split_compositions


@app.cell
def _(SPLITS, df, split_compositions):
    def _split_of(comp):
        hits = [_s for _s in SPLITS if comp in split_compositions[_s]]
        if not hits:
            return None
        return hits[0] if len(hits) == 1 else '/'.join(hits)

    df_with_split = df.assign(split=df['composition'].map(_split_of))
    print(df_with_split['split'].value_counts(dropna=False))
    print()
    print('Compositions per split:')
    for _s in SPLITS:
        _members = sorted((_c for _c, _sp in zip(df_with_split['composition'], df_with_split['split']) if _sp == _s))
        print(f'  {_s} ({len(_members)}):', ', '.join(_members) if _members else '(none)')
    return (df_with_split,)


@app.cell
def _(PROPS, SPLITS, df_with_split, pd, stats):
    _present_splits = [_s for _s in SPLITS if (df_with_split['split'] == _s).any()]
    if 'train' in _present_splits:
        _rows = []
        for _prop in PROPS:
            _train_v = df_with_split.loc[df_with_split['split'] == 'train', _prop].values
            for _s in _present_splits:
                if _s == 'train':
                    continue
                _other_v = df_with_split.loc[df_with_split['split'] == _s, _prop].values
                if len(_other_v) < 2:
                    continue
                _ks_stat, _ks_p = stats.ks_2samp(_train_v, _other_v)
                _rows.append({
                    'property': _prop, 'split': _s,
                    'n_train': len(_train_v), 'n_other': len(_other_v),
                    'mean_train': _train_v.mean(), 'mean_other': _other_v.mean(),
                    'std_train': _train_v.std(), 'std_other': _other_v.std(),
                    'KS_stat': _ks_stat, 'KS_p': _ks_p,
                })
        print(pd.DataFrame(_rows).to_string(index=False))
    else:
        print('No split labels resolved — chunks may not be present.')
    return


app._unparsable_cell(
    r"""
    _present_splits = [_s for _s in SPLITS if (df_with_split['split'] == _s).any()]
    if _present_splits:
        _split_colors = {'train': 'steelblue', 'val': 'orange', 'test': 'tomato'}
        _fig, _axes = plt.subplots(4, 2, figsize=(12, 14))
        for _ax, _prop in zip(_axes.flat, PROPS):
            for _s in _present_splits:
                _v = df_with_split.loc[df_with_split['split'] == _s, _prop].values
                if len(_v) == 0:
                    continue
                _ax.hist(_v, bins=15, alpha=0.5, label=f'{_s} (n={len(_v)})',
                         color=_split_colors.get(_s, 'gray'), edgecolor='k', lw=0.4)
            _ax.set_title(_prop)
            _ax.set_xlabel(_prop)
            _ax.set_ylabel('count')
            _ax.legend(fontsize=8)
        _fig.suptitle('Property distributions across splits', y=1.0, fontsize=14)
        plt.tight_layout()
        plt.savefig(FIG_DIR / '08_split_audit.png', dpi=150, bbox_inches='tight')
        plt.savefig(FIG_DIR / '08_split_audit.pdf', bbox_inches='tight')
        return _fig
    else:
        print('Skipped — no chunked splits present.')
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9 — KDE Coverage Analysis

    Kernel density estimation in composition-PCA space reveals where the 70 simulated
    compositions cluster and where gaps exist. Low-density regions are candidates for
    new MD simulations — this converts the qualitative "add more DPPC/DOPC coverage"
    argument into a ranked, quantitative list.

    **Two spaces are compared:**
    - **Composition-space (§9a)**: PCA of the 10-dim mole-fraction vectors.
    - **Property-space (§9b)**: PCA of the 6 Tier-B properties (z-scored).

    Each space is shown at 3 KDE bandwidths (0.5×, 1×, 2× Scott's rule) to verify
    that the gap structure is bandwidth-robust, not a smoothing artefact.

    **Sanity check (§9c)**: Stage 5b worst-MAE compositions (★) are annotated.
    They should fall in low-density regions of the composition KDE — if they do,
    the metric correctly identifies the training-coverage gap behind the extrapolation
    failures.
    """)
    return


@app.cell
def _(LIPIDS, PCA, StandardScaler, df, np):
    TIER_B_PROPS = ['lipid_packing', 'thickness', 'thickness_std', 'variation', 'persistence', 'diffusivity']
    STAGE5B_WORST = ['POPC30_DOPC70', 'POPC30_DPPC70', 'POPC60_DPPC40', 'POPC40_DIPC60']
    SPLIT_COLORS = {'train': 'steelblue', 'val': 'darkorange', 'test': 'tomato'}

    _X_comp = np.zeros((len(df), len(LIPIDS)))
    for _idx, _row in df.iterrows():
        _a, _fa = _row['lipid_a'], _row['frac_a']
        _b, _fb = _row['lipid_b'], _row['frac_b']
        if _a in LIPIDS:
            _X_comp[_idx, LIPIDS.index(_a)] = _fa / 100.0
        if _b is not None and _b in LIPIDS:
            _X_comp[_idx, LIPIDS.index(_b)] = _fb / 100.0

    scaler_comp = StandardScaler()
    X_comp_z = scaler_comp.fit_transform(_X_comp)
    pca_comp = PCA(n_components=2, random_state=42)
    Z_comp = pca_comp.fit_transform(X_comp_z)
    ev_comp = pca_comp.explained_variance_ratio_
    X_comp = _X_comp

    _Y_prop = df[TIER_B_PROPS].values
    _scaler_prop = StandardScaler()
    _Y_prop_z = _scaler_prop.fit_transform(_Y_prop)
    _pca_prop = PCA(n_components=2, random_state=42)
    Z_prop = _pca_prop.fit_transform(_Y_prop_z)
    ev_prop = _pca_prop.explained_variance_ratio_

    print(f'Composition PCA  PC1={ev_comp[0] * 100:.1f}%  PC2={ev_comp[1] * 100:.1f}%  cumulative={sum(ev_comp) * 100:.1f}%')
    print(f'Property PCA     PC1={ev_prop[0] * 100:.1f}%  PC2={ev_prop[1] * 100:.1f}%  cumulative={sum(ev_prop) * 100:.1f}%')
    print(f'\nStage 5b worst-MAE compositions to track: {STAGE5B_WORST}')
    print('Split membership of worst-MAE:')
    for _c in STAGE5B_WORST:
        _s = df.loc[df['composition'] == _c, 'composition']
        print(f"  {_c}: {'found' if len(_s) else 'not found'}")
    return (
        SPLIT_COLORS,
        STAGE5B_WORST,
        TIER_B_PROPS,
        X_comp,
        Z_comp,
        Z_prop,
        ev_comp,
        ev_prop,
        pca_comp,
        scaler_comp,
    )


@app.cell
def _(
    FIG_DIR,
    SPLIT_COLORS,
    STAGE5B_WORST,
    Z_comp,
    df_with_split,
    ev_comp,
    mo,
    np,
    plt,
):
    from scipy.stats import gaussian_kde

    _kde_default = gaussian_kde(Z_comp.T)
    _bw = float(_kde_default.factor)
    kde_comp_half    = gaussian_kde(Z_comp.T, bw_method=_bw * 0.5)
    kde_comp_default = gaussian_kde(Z_comp.T)
    kde_comp_double  = gaussian_kde(Z_comp.T, bw_method=_bw * 2.0)

    _pad = 0.8
    _x0, _x1 = Z_comp[:, 0].min() - _pad, Z_comp[:, 0].max() + _pad
    _y0, _y1 = Z_comp[:, 1].min() - _pad, Z_comp[:, 1].max() + _pad
    _xx, _yy = np.mgrid[_x0:_x1:200j, _y0:_y1:200j]
    _pos = np.vstack([_xx.ravel(), _yy.ravel()])

    _fig, _axes = plt.subplots(1, 3, figsize=(17, 5))
    for _ax, _kde, _label in zip(
        _axes,
        [kde_comp_half, kde_comp_default, kde_comp_double],
        ['0.5× Scott', f'1× Scott (bw={_bw:.3f})', '2× Scott'],
    ):
        _density = _kde(_pos).reshape(200, 200)
        _cf = _ax.contourf(_xx, _yy, _density, levels=14, cmap='Blues', alpha=0.85)
        _ax.contour(_xx, _yy, _density, levels=14, colors='navy', linewidths=0.3, alpha=0.4)
        plt.colorbar(_cf, ax=_ax, label='KDE density', shrink=0.9)
        for _split, _color in SPLIT_COLORS.items():
            _m = df_with_split['split'] == _split
            _ax.scatter(Z_comp[_m, 0], Z_comp[_m, 1], c=_color, s=50, edgecolor='k', lw=0.6, alpha=0.9, label=_split, zorder=5)
        _widx = df_with_split.index[df_with_split['composition'].isin(STAGE5B_WORST)].tolist()
        _ax.scatter(Z_comp[_widx, 0], Z_comp[_widx, 1], marker='*', s=220, c='red', edgecolor='darkred', lw=0.7, zorder=7, label='Stage 5b worst MAE')
        for _i in _widx:
            _ax.annotate(df_with_split.loc[_i, 'composition'], (Z_comp[_i, 0], Z_comp[_i, 1]),
                         textcoords='offset points', xytext=(5, 4), fontsize=6.5, color='darkred')
        _ax.set_xlabel(f'PC1 ({ev_comp[0] * 100:.1f}%)')
        _ax.set_ylabel(f'PC2 ({ev_comp[1] * 100:.1f}%)')
        _ax.set_title(f'Composition KDE — {_label}')
        _ax.legend(fontsize=8, loc='upper right')
        _ax.grid(alpha=0.25)
    plt.suptitle('Composition-space KDE: where do the 70 training compositions sit?', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / '09_kde_composition_pca.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '09_kde_composition_pca.pdf', bbox_inches='tight')
    mo.output.replace(_fig)
    return gaussian_kde, kde_comp_default, kde_comp_double, kde_comp_half


@app.cell
def _(
    FIG_DIR,
    SPLIT_COLORS,
    STAGE5B_WORST,
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
    for _ax, _bw_factor, _label in zip(_axes, [0.5, 1.0, 2.0], ['0.5× Scott', f'1× Scott (bw={_bw_p:.3f})', '2× Scott']):
        _kde_p = _kde_prop_default if _bw_factor == 1.0 else gaussian_kde(Z_prop.T, bw_method=_bw_p * _bw_factor)
        _dens_p = _kde_p(_ppos).reshape(200, 200)
        _cf = _ax.contourf(_gx, _gy, _dens_p, levels=14, cmap='Purples', alpha=0.85)
        _ax.contour(_gx, _gy, _dens_p, levels=14, colors='purple', linewidths=0.3, alpha=0.4)
        plt.colorbar(_cf, ax=_ax, label='KDE density', shrink=0.9)
        for _split, _color in SPLIT_COLORS.items():
            _m = df_with_split['split'] == _split
            _ax.scatter(Z_prop[_m, 0], Z_prop[_m, 1], c=_color, s=50, edgecolor='k', lw=0.6, alpha=0.9, label=_split, zorder=5)
        _widx = df_with_split.index[df_with_split['composition'].isin(STAGE5B_WORST)].tolist()
        _ax.scatter(Z_prop[_widx, 0], Z_prop[_widx, 1], marker='*', s=220, c='red', edgecolor='darkred', lw=0.7, zorder=7, label='Stage 5b worst MAE')
        for _i in _widx:
            _ax.annotate(df_with_split.loc[_i, 'composition'], (Z_prop[_i, 0], Z_prop[_i, 1]),
                         textcoords='offset points', xytext=(5, 4), fontsize=6.5, color='darkred')
        _ax.set_xlabel(f'PC1 ({ev_prop[0] * 100:.1f}%)')
        _ax.set_ylabel(f'PC2 ({ev_prop[1] * 100:.1f}%)')
        _ax.set_title(f'Property KDE (Tier B) — {_label}')
        _ax.legend(fontsize=8, loc='upper right')
        _ax.grid(alpha=0.25)
    plt.suptitle('Property-space KDE  [' + ', '.join(TIER_B_PROPS) + ']', fontsize=13)
    plt.tight_layout()
    plt.savefig(FIG_DIR / '10_kde_property_pca.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '10_kde_property_pca.pdf', bbox_inches='tight')
    return


@app.cell
def _(
    FIG_DIR,
    OUT_DIR,
    SPLIT_COLORS,
    STAGE5B_WORST,
    Z_comp,
    df_with_split,
    kde_comp_default,
    np,
    pd,
    plt,
):
    _comp_densities = kde_comp_default(Z_comp.T)
    density_df = pd.DataFrame({
        'composition': df_with_split['composition'].values,
        'family': df_with_split['family'].values,
        'split': df_with_split['split'].values,
        'pc1': Z_comp[:, 0],
        'pc2': Z_comp[:, 1],
        'kde_density': _comp_densities,
    })
    density_df['density_rank'] = density_df['kde_density'].rank(ascending=True).astype(int)
    density_df = density_df.sort_values('density_rank').reset_index(drop=True)

    _dens_path = OUT_DIR / 'existing_densities.csv'
    density_df.to_csv(_dens_path, index=False)
    print(f'Saved {_dens_path}')

    _threshold_25 = np.percentile(_comp_densities, 25)
    _worst_dens = density_df[density_df['composition'].isin(STAGE5B_WORST)]
    print(f'\n25th-percentile density: {_threshold_25:.5f}')
    print('Stage 5b worst-MAE in density ranking:')
    print(_worst_dens[['composition', 'kde_density', 'density_rank']].to_string(index=False))
    _all_below = (_worst_dens['kde_density'] <= _threshold_25).all()
    print(f"All worst-MAE in bottom 25% density? → {'✓ YES' if _all_below else '✗ NO (check KDE projection)'}")

    _split_bar_colors = [SPLIT_COLORS.get(_s, 'gray') for _s in density_df['split']]
    _fig, _ax = plt.subplots(figsize=(11, 5))
    _ax.bar(range(len(density_df)), density_df['kde_density'], color=_split_bar_colors, edgecolor='k', lw=0.3, alpha=0.85)
    _ax.axhline(_threshold_25, color='red', linestyle='--', lw=1.5, label=f'25th percentile ({_threshold_25:.4f})')
    for _, _wr in _worst_dens.iterrows():
        _bar_x = int(_wr['density_rank']) - 1
        _ax.annotate(_wr['composition'], xy=(_bar_x, _wr['kde_density']),
                     xytext=(_bar_x, _wr['kde_density'] + 0.003), fontsize=6.5, color='darkred',
                     ha='center', arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))
    _ax.set_xlabel('composition (sorted by density, low → left = gap region)')
    _ax.set_ylabel('KDE density (composition-PCA, 1× Scott)')
    _ax.set_title('Per-composition density — low values indicate sparse train-coverage regions')
    _ax.grid(alpha=0.25, axis='y')
    from matplotlib.patches import Patch as _Patch
    _ax.legend(handles=[
        _Patch(facecolor='steelblue', label='train'),
        _Patch(facecolor='darkorange', label='val'),
        _Patch(facecolor='tomato', label='test'),
        plt.Line2D([0], [0], color='red', linestyle='--', label=f'25th pct ({_threshold_25:.4f})'),
    ], fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG_DIR / '11_density_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '11_density_distribution.pdf', bbox_inches='tight')
    print('\nBottom 10 (lowest density = biggest gaps):')
    print(density_df.head(10)[['composition', 'family', 'split', 'kde_density', 'density_rank']].to_string(index=False))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 10 — Gap Candidate Ranking

    Generate all plausible binary and ternary lipid mixtures from the existing 10-lipid
    vocabulary, project them into composition-PCA space (fitted in §9), score each by
    the KDE density, and rank the top 20 as simulation candidates.

    **Candidate pool**:
    - **Binary**: C(10,2)=45 unordered pairs × fractions {5,10,...,95}% → 855 total,
      filtered: L∞-distance > 5 mol% from every existing composition.
    - **Ternary**: C(10,3)=120 unordered triples × 36 ordered fraction assignments
      (multiples of 10%, each ≥10%, summing to 100%) → ≤4320 total, same filter.

    **Scoring**: KDE density from the composition-PCA KDE (1× Scott, fitted in §9).
    Low density = under-sampled region → prioritise for new MD simulation.

    **Caveats**: The KDE operates in 2-D (PC1+PC2 of composition); gaps that project
    onto dense 2-D regions despite being compositionally distant are missed. Use the
    `existing_densities.csv` ranks alongside biological judgement.
    """)
    return


@app.cell
def _(
    LIPIDS,
    OUT_DIR,
    X_comp,
    kde_comp_default,
    kde_comp_double,
    kde_comp_half,
    np,
    pca_comp,
    pd,
    scaler_comp,
):
    from itertools import combinations, permutations as _permutations

    _binary_fracs = list(range(5, 100, 5))
    _all_cand_vecs = []
    _all_cand_meta = []
    for _a, _b in combinations(LIPIDS, 2):
        for _f in _binary_fracs:
            _v = np.zeros(len(LIPIDS))
            _v[LIPIDS.index(_a)] = _f / 100.0
            _v[LIPIDS.index(_b)] = (100 - _f) / 100.0
            _all_cand_vecs.append(_v)
            _all_cand_meta.append({'lipid_a': _a, 'frac_a': _f, 'lipid_b': _b, 'frac_b': 100 - _f, 'lipid_c': None, 'frac_c': None, 'n_components': 2})
    print(f'Binary candidates generated:  {len(_all_cand_vecs)}')

    def _ternary_ordered_fracs():
        result = set()
        for _fa in range(10, 82, 10):
            for _fb in range(10, 91 - _fa, 10):
                _fc = 100 - _fa - _fb
                if _fc < 10:
                    continue
                for _perm in _permutations([_fa, _fb, _fc]):
                    result.add(_perm)
        return list(result)

    _ternary_frac_list = _ternary_ordered_fracs()
    print(f'Unique ordered ternary fracs:  {len(_ternary_frac_list)}  (expected 36)')

    _n_before = len(_all_cand_vecs)
    for _a, _b, _c in combinations(LIPIDS, 3):
        for _fa, _fb, _fc in _ternary_frac_list:
            _v = np.zeros(len(LIPIDS))
            _v[LIPIDS.index(_a)] = _fa / 100.0
            _v[LIPIDS.index(_b)] = _fb / 100.0
            _v[LIPIDS.index(_c)] = _fc / 100.0
            _all_cand_vecs.append(_v)
            _all_cand_meta.append({'lipid_a': _a, 'frac_a': _fa, 'lipid_b': _b, 'frac_b': _fb, 'lipid_c': _c, 'frac_c': _fc, 'n_components': 3})
    print(f'Ternary candidates generated:  {len(_all_cand_vecs) - _n_before}')

    _V = np.array(_all_cand_vecs)
    _dists = np.max(np.abs(_V[:, None, :] - X_comp[None, :, :]), axis=2)
    _keep_mask = _dists.min(axis=1) > 0.05
    _V_keep = _V[_keep_mask]
    _meta_keep = [_m for _m, _k in zip(_all_cand_meta, _keep_mask) if _k]
    print(f'\nAfter near-duplicate filter (L∞ > 5 mol%): {_V_keep.shape[0]} / {_V.shape[0]} kept')
    print(f"  binary kept:  {sum(1 for _m in _meta_keep if _m['n_components'] == 2)}")
    print(f"  ternary kept: {sum(1 for _m in _meta_keep if _m['n_components'] == 3)}")

    _V_z = scaler_comp.transform(_V_keep)
    _Z_cand = pca_comp.transform(_V_z)
    _cand_densities = kde_comp_default(_Z_cand.T)

    cands_df = pd.DataFrame(_meta_keep)
    cands_df['pc1'] = _Z_cand[:, 0]
    cands_df['pc2'] = _Z_cand[:, 1]
    cands_df['kde_density'] = _cand_densities
    cands_df['density_rank'] = cands_df['kde_density'].rank(ascending=True).astype(int)

    def _has_lipid(row, lip):
        return any(row.get(k) == lip for k in ['lipid_a', 'lipid_b', 'lipid_c'])

    _PS_LIPIDS = {'POPS', 'DOPS'}
    cands_df['contains_chol'] = cands_df.apply(lambda r: _has_lipid(r, 'CHOL'), axis=1)
    cands_df['has_charged'] = cands_df.apply(lambda r: any(_has_lipid(r, _l) for _l in _PS_LIPIDS), axis=1)

    top20 = cands_df.nsmallest(20, 'kde_density').reset_index(drop=True)
    _cands_path = OUT_DIR / 'gap_candidates.csv'
    top20.to_csv(_cands_path, index=False)
    print(f'\nSaved {_cands_path}')

    for _bw_name, _kde_bw in [('0.5× Scott', kde_comp_half), ('2× Scott', kde_comp_double)]:
        _dens_bw = _kde_bw(_Z_cand.T)
        _top5_def = set(cands_df.nsmallest(5, 'kde_density').index)
        _top5_bw = set(cands_df.assign(_d=_dens_bw).nsmallest(5, '_d').index)
        _jaccard = len(_top5_def & _top5_bw) / len(_top5_def | _top5_bw)
        print(f'Bandwidth robustness (Jaccard top-5, 1× vs {_bw_name}): {_jaccard:.2f}')

    _display_cols = ['lipid_a', 'frac_a', 'lipid_b', 'frac_b', 'lipid_c', 'frac_c', 'n_components', 'kde_density', 'contains_chol', 'has_charged']
    print('\nTop 20 gap candidates (lowest composition-space KDE density):')
    print(top20[_display_cols].to_string(index=False))

    _sanity = {
        'CHOL + non-POPC binary': top20[(top20['n_components'] == 2) & top20['contains_chol'] & (top20['lipid_a'] != 'POPC') & (top20['lipid_b'] != 'POPC')],
        'PE×PE or PE×PS combo': top20[top20.apply(lambda r: sum(_has_lipid(r, _l) for _l in ['POPE', 'DOPE', 'DPPE', 'POPS', 'DOPS']) >= 2, axis=1)],
        'ternary with CHOL': top20[(top20['n_components'] == 3) & top20['contains_chol']],
    }
    print()
    for _name, _subset in _sanity.items():
        _status = '✓' if len(_subset) > 0 else '✗ MISSING'
        print(f'  {_status}  {_name}  ({len(_subset)} entries in top-20)')

    cands_df['total_ps_frac'] = cands_df.apply(
        lambda r: sum(r.get(f'frac_{x}') or 0 for x in ['a', 'b', 'c'] if r.get(f'lipid_{x}') in _PS_LIPIDS),
        axis=1,
    )
    bio_top20 = cands_df[cands_df['total_ps_frac'] <= 15].nsmallest(20, 'kde_density').reset_index(drop=True)
    _bio_cands_path = OUT_DIR / 'gap_candidates_bio.csv'
    bio_top20.to_csv(_bio_cands_path, index=False)
    print(f'\nSaved {_bio_cands_path}')
    print('\n── Biologically motivated top-20 (PS ≤ 15 mol% filter applied) ──')
    print(bio_top20[_display_cols].to_string(index=False))

    _sanity_bio = {
        'CHOL + non-POPC binary': bio_top20[(bio_top20['n_components'] == 2) & bio_top20['contains_chol'] & (bio_top20['lipid_a'] != 'POPC') & (bio_top20['lipid_b'] != 'POPC')],
        'PE×PE or PE×PS combo': bio_top20[bio_top20.apply(lambda r: sum(_has_lipid(r, _l) for _l in ['POPE', 'DOPE', 'DPPE', 'POPS', 'DOPS']) >= 2, axis=1)],
        'ternary with CHOL': bio_top20[(bio_top20['n_components'] == 3) & bio_top20['contains_chol']],
    }
    print()
    for _name, _subset in _sanity_bio.items():
        _status = '✓' if len(_subset) > 0 else '✗ MISSING'
        print(f'  {_status}  {_name}  ({len(_subset)} entries in bio top-20)')
    return (top20,)


@app.cell
def _(
    FIG_DIR,
    SPLIT_COLORS,
    STAGE5B_WORST,
    Z_comp,
    df_with_split,
    ev_comp,
    kde_comp_default,
    np,
    plt,
    top20,
):
    _fig, _ax = plt.subplots(figsize=(9, 7))
    _pad = 0.8
    _x0, _x1 = Z_comp[:, 0].min() - _pad, Z_comp[:, 0].max() + _pad
    _y0, _y1 = Z_comp[:, 1].min() - _pad, Z_comp[:, 1].max() + _pad
    _xx, _yy = np.mgrid[_x0:_x1:200j, _y0:_y1:200j]
    _density_bg = kde_comp_default(np.vstack([_xx.ravel(), _yy.ravel()])).reshape(200, 200)
    _ax.contourf(_xx, _yy, _density_bg, levels=12, cmap='Blues', alpha=0.75)
    _ax.contour(_xx, _yy, _density_bg, levels=12, colors='navy', linewidths=0.3, alpha=0.3)
    for _split, _color in SPLIT_COLORS.items():
        _m = df_with_split['split'] == _split
        _ax.scatter(Z_comp[_m, 0], Z_comp[_m, 1], c=_color, s=55, edgecolor='k', lw=0.6, alpha=0.9, label=f'existing ({_split})', zorder=5)
    _widx = df_with_split.index[df_with_split['composition'].isin(STAGE5B_WORST)].tolist()
    _ax.scatter(Z_comp[_widx, 0], Z_comp[_widx, 1], marker='*', s=230, c='red', edgecolor='darkred', lw=0.7, zorder=7, label='Stage 5b worst MAE')
    for _i in _widx:
        _ax.annotate(df_with_split.loc[_i, 'composition'], (Z_comp[_i, 0], Z_comp[_i, 1]),
                     textcoords='offset points', xytext=(5, 4), fontsize=7, color='darkred')
    _top20_bin = top20[top20['n_components'] == 2]
    _top20_ter = top20[top20['n_components'] == 3]
    _ax.scatter(_top20_bin['pc1'], _top20_bin['pc2'], marker='D', s=90, c='limegreen', edgecolor='darkgreen', lw=1.0, zorder=8, label=f'top-20 binary gap ({len(_top20_bin)})')
    _ax.scatter(_top20_ter['pc1'], _top20_ter['pc2'], marker='^', s=110, c='gold', edgecolor='darkgoldenrod', lw=1.0, zorder=8, label=f'top-20 ternary gap ({len(_top20_ter)})')
    for _, _row in top20.head(5).iterrows():
        _label_str = (f"{_row['lipid_a']}{int(_row['frac_a'])}_{_row['lipid_b']}{int(_row['frac_b'])}"
                      + (f"_{_row['lipid_c']}{int(_row['frac_c'])}" if _row['lipid_c'] else ''))
        _ax.annotate(_label_str, (_row['pc1'], _row['pc2']), textcoords='offset points', xytext=(6, 6),
                     fontsize=6.5, color='darkgreen', bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.7))
    _ax.set_xlabel(f'PC1 ({ev_comp[0] * 100:.1f}%)')
    _ax.set_ylabel(f'PC2 ({ev_comp[1] * 100:.1f}%)')
    _ax.set_title('Composition-PCA: existing dataset + top-20 gap candidates\n(◆ = binary, ▲ = ternary, ★ = Stage 5b worst MAE)')
    _ax.legend(fontsize=8, loc='upper right')
    _ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG_DIR / '12_gap_candidates_pca.png', dpi=150, bbox_inches='tight')
    plt.savefig(FIG_DIR / '12_gap_candidates_pca.pdf', bbox_inches='tight')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wrap-up

    Saved figures and tables under `results/dataset_analysis/`:

    - `dataset_table.csv` — one row per composition, parsed composition + 8 properties
    - `existing_densities.csv` — per-composition KDE density + rank in composition-PCA space
    - `gap_candidates.csv` — top-20 simulation candidates ranked by lowest KDE density
    - `figures/01_coverage.{png,pdf}`           — composition coverage heatmap (incl. POPC row)
    - `figures/03_univariate.{png,pdf}`         — property distributions
    - `figures/04a_correlations.{png,pdf}`      — Pearson + Spearman matrices
    - `figures/04b_pairplot.{png,pdf}`          — full pairplot
    - `figures/04c_helfrich.{png,pdf}`          — Helfrich proxy scatter
    - `figures/05_titration.{png,pdf}`          — titration curves
    - `figures/06a_pca.{png,pdf}`               — scree, PC1/PC2, loadings
    - `figures/06b_umap.{png,pdf}`              — UMAP (if available)
    - `figures/06c_dendrogram.{png,pdf}`        — Ward clustering
    - `figures/07a_timeseries_popc.{png,pdf}`   — POPC100 reference trajectories
    - `figures/07b_noise_floor.{png,pdf}`       — within vs between-system std
    - `figures/08_split_audit.{png,pdf}`        — train/val/test distribution check
    - `figures/09_kde_composition_pca.{png,pdf}` — composition-space KDE (3 bandwidths)
    - `figures/10_kde_property_pca.{png,pdf}`   — property-space KDE / Tier B (3 bandwidths)
    - `figures/11_density_distribution.{png,pdf}` — per-composition density bar chart
    - `figures/12_gap_candidates_pca.{png,pdf}` — top-20 gap candidates on PCA map
    """)
    return


@app.cell
def _(FIG_DIR, OUT_DIR, table_path):
    print('Outputs:')
    for _p in sorted(FIG_DIR.glob('*.png')):
        print(f'  {_p.relative_to(OUT_DIR.parent)}')
    print(f'  {table_path.relative_to(OUT_DIR.parent)}')
    return


if __name__ == "__main__":
    app.run()
