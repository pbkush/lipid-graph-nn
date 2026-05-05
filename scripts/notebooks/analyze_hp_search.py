# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "pandas",
#     "numpy",
#     "matplotlib",
#     "seaborn",
#     "pyarrow",
#     "jinja2",
# ]
# ///

import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import warnings
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import marimo as mo

    try:
        import seaborn as sns
        HAS_SEABORN = True
    except ImportError:
        sns = None
        HAS_SEABORN = False
        warnings.warn("seaborn not installed; heatmap will use matplotlib")
    return HAS_SEABORN, Path, json, mo, mticker, np, pd, plt, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # HP Search Analysis

    Loads runs saved by `scripts/python/download_wandb_runs.py`, aggregates over seeds,
    visualises loss curves and training stats, and recommends the best HP combination.

    **Workflow**
    1. Run `python scripts/python/download_wandb_runs.py --group <stage>` first.
    2. Set `GROUP` in the Configuration cell to the downloaded group name.
    3. Run all cells. The recommendation prints at the bottom.

    **Selection rule** (from `docs/gnn_only_hp_search_plan.md`):
    rank by **min val/loss_total over the last 10 epochs, mean over seeds**.
    Test MSE is shown as an overfit guard only — not the selection signal.

    **Sections:**
    1. Configuration
    2. Data Loading
    3. HP Detection & Aggregation
    4. Visualizations (a–g)
    5. Recommendation
    6. Multi-group Comparison
    7. Conclusions
    """)
    return


@app.cell
def _(Path):
    # ── USER CONFIG ──
    GROUP  = "stage_1g_tier_c_lr"
    GROUPS = ["stage_0d_tier_c", "stage_1g_tier_c_lr"]

    GATES = {
        "lipid_packing":   0.019,
        "thickness":       0.067,
        "thickness_std":   0.302,
        "variation":       0.151,
        "persistence":     0.362,
        "diffusivity":     0.059,
        "compressibility": 0.3931,
    }
    OCCAM_TOL = 0.01

    HP_COLS = [
        "comp_mode", "hidden", "num_layers", "lr", "weight_decay",
        "dropout", "batch_size", "rbf_type", "cutoff_type",
    ]

    def _find_repo_root() -> Path:
        p = Path(".").resolve()
        for _ in range(6):
            if (p / "config.yaml").exists():
                return p
            p = p.parent
        raise FileNotFoundError("Cannot find repo root (config.yaml not found in parents)")

    REPO_ROOT = _find_repo_root()
    LOGS_DIR  = REPO_ROOT / "logs" / "training"
    return GATES, GROUP, GROUPS, HP_COLS, LOGS_DIR, OCCAM_TOL, REPO_ROOT


@app.cell
def _(GROUP, REPO_ROOT, plt):
    plt.rcParams.update({"figure.dpi": 120, "font.size": 10})

    FIGURES_DIR = REPO_ROOT / "results" / "training" / GROUP
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    def save_fig(fig, name):
        for _ext in ("pdf", "png"):
            fig.savefig(FIGURES_DIR / f"{name}.{_ext}")
        print(f"  saved → {name}.{{pdf,png}}")

    print(f"FIGURES_DIR: {FIGURES_DIR}")
    return (save_fig,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Data Loading

    `load_group(group_name)` reads all runs from `logs/training/<group>/` and returns
    three objects:
    - `runs_df` – one row per run with config + summary metrics
    - `val_histories` – dict mapping run_id → per-epoch val history DataFrame
    - `sys_histories` – dict mapping run_id → system metrics DataFrame
    """)
    return


@app.cell
def _(GROUP, LOGS_DIR, json, np, pd):

    def _parse_gpu_memory(sys_df: pd.DataFrame) -> float:
        """Peak GPU memory in MB; picks the active GPU (max across all gpu.N.memoryAllocated columns).

        W&B reports all visible GPUs but only the SLURM-allocated one has non-zero values.
        The allocated index varies per job, so we scan all columns and take the maximum.
        """
        if sys_df.empty:
            return float('nan')
        best = float('nan')
        for col in sys_df.columns:
            low = col.lower()
            if 'memoryallocated' in low and 'gpu' in low:
                vals = sys_df[col].dropna()
                if vals.empty or vals.max() == 0:
                    continue
                # Values are percentage (0-100) of total GPU memory
                val = float(vals.max() / 100 * 64 * 1024)  # % of MI210 64 GB -> MB
                if np.isnan(best) or val > best:
                    best = val
        return best


    def _parse_gpu_util(sys_df: pd.DataFrame) -> float:
        """Mean GPU utilisation %; picks the active GPU (highest mean across all gpu.N.gpu columns).

        W&B reports all visible GPUs but only the SLURM-allocated one has non-zero values.
        """
        if sys_df.empty:
            return float('nan')
        best_mean = float('nan')
        for col in sys_df.columns:
            if col.lower().endswith('.gpu') and 'gpu' in col.lower():
                vals = sys_df[col].dropna()
                if vals.empty:
                    continue
                m = float(vals.mean())
                if np.isnan(best_mean) or m > best_mean:
                    best_mean = m
        return best_mean


    def _tail_min(hist: pd.DataFrame, col: str) -> float:
        if hist.empty or col not in hist.columns:
            return float('nan')
        s = hist[col].dropna()
        return float(s.tail(10).min()) if len(s) else float('nan')


    def _tail_mean(hist: pd.DataFrame, col: str) -> float:
        """Mean over last 10 epochs. Used for noisy per-epoch metrics like R²,
        where val_min_last10 (used for MSE) would amplify favourable noise spikes.
        """
        if hist.empty or col not in hist.columns:
            return float('nan')
        s = hist[col].dropna()
        return float(s.tail(10).mean()) if len(s) else float('nan')


    def load_group(group: str) -> tuple[pd.DataFrame, dict, dict]:
        """Load all downloaded runs for *group*.

        Returns
        -------
        runs_df        one row per run with config + derived stats
        histories      {run_id: history_df} for loss plotting
        sys_histories  {run_id: system_df}  for utilisation plotting
        """
        group_dir  = LOGS_DIR / group
        index_path = group_dir / 'runs_index.json'
        if not index_path.exists():
            raise FileNotFoundError(
                f'No index at {index_path}.\n'
                f'Run: python scripts/python/download_wandb_runs.py --group {group}'
            )
        with open(index_path) as f:
            index = json.load(f)

        rows, histories, sys_histories = [], {}, {}

        for entry in index:
            name    = entry['name']
            run_dir = group_dir / name
            cfg     = entry['config']
            props   = cfg.get('properties', [])

            hp      = run_dir / 'history.parquet'
            hist    = pd.read_parquet(hp) if hp.exists() else pd.DataFrame()
            sp      = run_dir / 'system.parquet'
            sys_df  = pd.read_parquet(sp) if sp.exists() else pd.DataFrame()
            sump    = run_dir / 'summary.json'
            summary = json.loads(sump.read_text()) if sump.exists() else {}

            runtime = entry.get('runtime_seconds') or summary.get('_runtime')
            epochs  = cfg.get('epochs', 100)
            vml10   = _tail_min(hist, 'val/loss_total')
            test_mse = summary.get('test/mse_total', float('nan'))

            row = {
                'run_id': entry['id'],
                'run_name': name,
                'group': group,
                'state': entry['state'],
                # scalar config keys flattened
                **{k: v for k, v in cfg.items()
                   if isinstance(v, (int, float, str, bool)) and k != 'properties'},
                'properties_str':   ','.join(props),
                'val_min_last10':   vml10,
                'test_mse_total':   float(test_mse) if test_mse is not None else float('nan'),
                'runtime_seconds':  runtime,
                'sec_per_epoch':    (runtime / epochs) if runtime else float('nan'),
                'peak_gpu_mem_mb':  _parse_gpu_memory(sys_df),
                'mean_gpu_util_pct': _parse_gpu_util(sys_df),
            }
            for prop in props:
                row[f'val_min_{prop}'] = _tail_min(hist, f'val/loss_{prop}')
                row[f'val_r2_{prop}']  = _tail_mean(hist, f'val/r2_{prop}')

            tm, vm = row['test_mse_total'], row['val_min_last10']
            row['gap'] = abs(tm - vm) if not (np.isnan(tm) or np.isnan(vm)) else float('nan')

            rows.append(row)
            histories[entry['id']]     = hist
            sys_histories[entry['id']] = sys_df

        df = pd.DataFrame(rows)
        print(f"Loaded {len(df)} runs from '{group}'")
        return df, histories, sys_histories

    runs_df, histories, sys_histories = load_group(GROUP)

    PROPS = [
        c.replace('val_min_', '')
        for c in runs_df.columns
        if c.startswith('val_min_') and c != 'val_min_last10'
    ]
    print(f'Properties : {PROPS}')
    runs_df.head()
    return PROPS, histories, load_group, runs_df, sys_histories


@app.cell
def _(GROUP, PROPS, mo, runs_df):
    mo.md(f"""
    **Dataset Summary (`runs_df`)**:
    - **Group**: `{GROUP}`
    - **Dimensions**: {runs_df.shape[0]:,} runs × {runs_df.shape[1]} columns
    - **Properties tracked**: {", ".join(f"`{p}`" for p in PROPS)}
    - **Unique seeds**: {runs_df["seed"].nunique() if "seed" in runs_df.columns else "N/A"}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Auto-detect Varying Hyperparameters

    Scan config columns to find which HPs actually vary across runs in this group.
    Fixed HPs are noted; varying ones become the `GROUP_BY` key for seed aggregation.
    """)
    return


@app.cell
def _(runs_df):

    _META = {
        'run_id', 'run_name', 'group', 'state', 'properties_str',
        'val_min_last10', 'test_mse_total', 'gap',
        'runtime_seconds', 'sec_per_epoch', 'peak_gpu_mem_mb', 'mean_gpu_util_pct',
        'properties',
    }
    # Exclude both per-property MSE (val_min_*) and R² (val_r2_*) from HP detection.
    _PROP_VALS = {
        c for c in runs_df.columns
        if (c.startswith('val_min_') and c != 'val_min_last10') or c.startswith('val_r2_')
    }
    _INFRA     = {'epochs', 'batch_size', 'num_workers', 'seed'}

    _dyn_hp_cols = [
        c for c in runs_df.columns
        if c not in _META and c not in _PROP_VALS and c not in _INFRA
    ]

    VARYING_HPS = [c for c in _dyn_hp_cols if runs_df[c].nunique() > 1]
    FIXED_HPS   = [c for c in _dyn_hp_cols if runs_df[c].nunique() <= 1]

    print(f'Varying HPs : {VARYING_HPS}')
    print(f'Fixed HPs   : {[(c, runs_df[c].iloc[0]) for c in FIXED_HPS]}')
    print(f'Seeds       : {sorted(runs_df["seed"].unique())}')
    return (VARYING_HPS,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Aggregate Over Seeds

    Group runs by the varying HPs and compute per-cell statistics:
    - `val_mean` / `val_std`: mean ± std of min val/loss_total over last 10 epochs
    - `test_mean`: mean test MSE
    - `gap`: |test_mean − val_mean| as overfit guard
    - Per-property val MSE and R² averages
    """)
    return


@app.cell
def _(PROPS, VARYING_HPS, np, pd, runs_df):

    GROUP_BY = VARYING_HPS if VARYING_HPS else (['comp_mode'] if 'comp_mode' in runs_df.columns else ['run_name'])


    def agg_cells(df: pd.DataFrame, group_by: list[str]) -> pd.DataFrame:
        """Collapse seed dimension; compute mean/std for each HP cell.

        Per-property metrics:
          val_{prop}  — seed-mean of val_min_last10 of val/loss_{prop}  (selection metric)
          r2_{prop}   — seed-mean of last-10-epoch mean of val/r2_{prop} (reporting metric)
        """
        rows = []
        for keys, grp in df.groupby(group_by, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict = dict(zip(group_by, keys))
            row['n_seeds']      = len(grp)
            row['val_mean']     = grp['val_min_last10'].mean()
            row['val_std']      = grp['val_min_last10'].std()
            row['test_mean']    = grp['test_mse_total'].mean()
            row['test_std']     = grp['test_mse_total'].std()
            row['runtime_mean'] = grp['runtime_seconds'].mean()
            row['runtime_std']  = grp['runtime_seconds'].std()
            row['sec_per_epoch'] = grp['sec_per_epoch'].mean()
            row['gpu_mem_mb']   = grp['peak_gpu_mem_mb'].max()     # max across seeds
            row['gpu_util_pct'] = grp['mean_gpu_util_pct'].mean()
            for prop in PROPS:
                mse_col = f'val_min_{prop}'
                r2_col  = f'val_r2_{prop}'
                row[f'val_{prop}'] = grp[mse_col].mean() if mse_col in grp.columns else float('nan')
                row[f'r2_{prop}']  = grp[r2_col].mean()  if r2_col  in grp.columns else float('nan')
            tm, vm = row['test_mean'], row['val_mean']
            row['gap'] = abs(tm - vm) if not (np.isnan(tm) or np.isnan(vm)) else float('nan')
            rows.append(row)
        return pd.DataFrame(rows).sort_values('val_mean').reset_index(drop=True)


    cells_df = agg_cells(runs_df, GROUP_BY)
    print(f'HP cells: {len(cells_df)}, grouped by {GROUP_BY}')
    cells_df
    return GROUP_BY, agg_cells, cells_df


@app.cell
def _(GROUP_BY, cells_df, mo):
    mo.md(f"""
    **Aggregation Summary (`cells_df`)**:
    - **HP cells**: {len(cells_df)}
    - **Grouped by**: {", ".join(f"`{k}`" for k in GROUP_BY)}
    - **Best val_mean**: {cells_df["val_mean"].min():.4f}
    """)
    return


@app.cell
def _(cells_df, mo):
    mo.as_html(cells_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (a) Val/loss_total curves per HP cell

    Val-loss curves per HP cell, with individual seed traces and mean overlay.
    """)
    return


@app.cell
def _(
    GROUP,
    GROUP_BY,
    cells_df,
    histories,
    mticker,
    np,
    pd,
    plt,
    runs_df,
    save_fig,
):
    _ncols = min(3, len(cells_df))
    _nrows = max(1, int(np.ceil(len(cells_df) / _ncols)))

    _fig_a, _axes = plt.subplots(_nrows, _ncols, figsize=(5 * _ncols, 3.5 * _nrows), squeeze=False)

    _all_val = [h['val/loss_total'].dropna() for h in histories.values() if 'val/loss_total' in h]
    _y_max = float(np.nanpercentile([v.max() for v in _all_val], 95)) if _all_val else 1.0
    _y_min = float(np.nanmin([v.min() for v in _all_val])) if _all_val else 0.0

    for _idx, (_, _cell_row) in enumerate(cells_df.iterrows()):
        _ax = _axes[_idx // _ncols][_idx % _ncols]
        _title = ' | '.join(f'{k}={_cell_row[k]}' for k in GROUP_BY)

        _cell_runs = runs_df.copy()
        for _k in GROUP_BY:
            _cell_runs = _cell_runs[_cell_runs[_k] == _cell_row[_k]]

        _traces = []
        for _, _r in _cell_runs.iterrows():
            _h = histories.get(_r['run_id'], pd.DataFrame())
            if _h.empty or 'val/loss_total' not in _h.columns:
                continue
            _v = _h['val/loss_total'].dropna().values
            _ax.plot(_v, alpha=0.4, linewidth=1, label=f's{_r["seed"]}')
            _traces.append(_v)

        if _traces:
            _ml = min(len(t) for t in _traces)
            _ax.plot(np.mean([t[:_ml] for t in _traces], axis=0),
                    color='black', linewidth=2, label='mean')

        _ax.set_title(_title, fontsize=8)
        _ax.set_xlabel('Epoch')
        _ax.set_ylabel('Val MSE (total)')
        _ax.set_ylim(max(0.0, _y_min * 0.9), _y_max * 1.05)
        _ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        _ax.legend(fontsize=7)

    for _idx in range(len(cells_df), _nrows * _ncols):
        _axes[_idx // _ncols][_idx % _ncols].set_visible(False)

    _fig_a.suptitle(f'Val/loss_total per HP cell — {GROUP}', fontsize=12, fontweight='bold')
    _fig_a.tight_layout()
    save_fig(_fig_a, "fig_a_val_loss_curves")
    _fig_a
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (b) Val/loss per property per HP cell

    Per-property validation loss curves for each HP cell.
    """)
    return


@app.cell
def _(
    GROUP,
    GROUP_BY,
    PROPS,
    cells_df,
    histories,
    mo,
    mticker,
    np,
    pd,
    plt,
    runs_df,
    save_fig,
):
    if PROPS:
        _n_p = len(PROPS)
        _fig_b, _axes_b = plt.subplots(_n_p, len(cells_df),
                                       figsize=(4 * len(cells_df), 3 * _n_p), squeeze=False)

        for _pi, _prop in enumerate(PROPS):
            _col = f'val/loss_{_prop}'
            _prop_vals = [h[_col].dropna() for h in histories.values() if _col in h]
            _pmax = float(np.nanpercentile([v.max() for v in _prop_vals], 95)) if _prop_vals else 1.0
            _pmin = float(np.nanmin([v.min() for v in _prop_vals])) if _prop_vals else 0.0

            for _ci, (_, _cell_row) in enumerate(cells_df.iterrows()):
                _ax = _axes_b[_pi][_ci]
                _cell_runs = runs_df.copy()
                for _k in GROUP_BY:
                    _cell_runs = _cell_runs[_cell_runs[_k] == _cell_row[_k]]

                _traces = []
                for _, _r in _cell_runs.iterrows():
                    _h = histories.get(_r['run_id'], pd.DataFrame())
                    if _h.empty or _col not in _h.columns:
                        continue
                    _v = _h[_col].dropna().values
                    _ax.plot(_v, alpha=0.4, linewidth=1)
                    _traces.append(_v)

                if _traces:
                    _ml = min(len(t) for t in _traces)
                    _ax.plot(np.mean([t[:_ml] for t in _traces], axis=0),
                            color='black', linewidth=2)

                if _pi == 0:
                    _ax.set_title(' | '.join(f'{k}={_cell_row[k]}' for k in GROUP_BY), fontsize=8)
                if _ci == 0:
                    _ax.set_ylabel(_prop, fontsize=9)
                _ax.set_ylim(max(0.0, _pmin * 0.9), _pmax * 1.05)
                _ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

        _fig_b.suptitle(f'Val/loss per property per HP cell — {GROUP}', fontsize=12, fontweight='bold')
        _fig_b.tight_layout()
        save_fig(_fig_b, "fig_b_per_property_val_curves")
        _out_b = _fig_b
    else:
        _out_b = mo.callout(mo.md("No per-property columns found — skipping plot (b)."), kind="warn")
    _out_b
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (c) HP Ranking Table

    Styled ranking table of HP cells sorted by val_mean (lower = better).
    R² is shown alongside MSE as a complementary interpretability signal.
    """)
    return


@app.cell
def _(GROUP, GROUP_BY, PROPS, cells_df, mo):
    _disp_cols = GROUP_BY + ['n_seeds', 'val_mean', 'val_std', 'test_mean', 'gap']
    for _p in PROPS:
        _disp_cols.append(f'val_{_p}')
        _disp_cols.append(f'r2_{_p}')
    _disp_cols = [c for c in _disp_cols if c in cells_df.columns]

    _fmt = {'val_mean': '{:.4f}', 'val_std': '{:.4f}',
            'test_mean': '{:.4f}', 'gap': '{:.4f}'}
    for _p in PROPS:
        _fmt[f'val_{_p}'] = '{:.4f}'
        _fmt[f'r2_{_p}']  = '{:.3f}'

    _styled = (
        cells_df[_disp_cols]
        .style
        .background_gradient(subset=['val_mean'], cmap='RdYlGn_r', axis=0)
        .format(_fmt)
        .set_caption(f'HP Ranking — {GROUP}   (sorted by val_mean ↑ better; R² ↑ better)')
    )
    mo.as_html(_styled)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (d) Val MSE heatmap (val_mean × HP grid)

    Heatmap of val_mean when exactly 2 HPs vary. Skipped otherwise.
    """)
    return


@app.cell
def _(GROUP, HAS_SEABORN, VARYING_HPS, cells_df, mo, plt, save_fig, sns):
    if len(VARYING_HPS) == 2:
        _dim_x, _dim_y = VARYING_HPS[0], VARYING_HPS[1]
        _pivot = cells_df.pivot(index=_dim_y, columns=_dim_x, values='val_mean')

        _fig_w = max(5, len(_pivot.columns) * 1.6)
        _fig_h = max(4, len(_pivot) * 1.3)
        _fig_d, _ax_d = plt.subplots(figsize=(_fig_w, _fig_h))

        if HAS_SEABORN:
            sns.heatmap(_pivot, annot=True, fmt='.4f', cmap='RdYlGn_r',
                        ax=_ax_d, linewidths=0.5, cbar_kws={'label': 'val MSE'})
        else:
            _im = _ax_d.imshow(_pivot.values, cmap='RdYlGn_r', aspect='auto')
            plt.colorbar(_im, ax=_ax_d, label='val MSE')
            _ax_d.set_xticks(range(len(_pivot.columns)))
            _ax_d.set_yticks(range(len(_pivot.index)))
            _ax_d.set_xticklabels(_pivot.columns)
            _ax_d.set_yticklabels(_pivot.index)
            for _i in range(len(_pivot.index)):
                for _j in range(len(_pivot.columns)):
                    _ax_d.text(_j, _i, f"{_pivot.values[_i, _j]:.4f}",
                               ha='center', va='center', fontsize=9)

        _ax_d.set_title(f'val_mean heatmap: {_dim_y} × {_dim_x} — {GROUP}')
        _ax_d.set_xlabel(_dim_x)
        _ax_d.set_ylabel(_dim_y)
        _fig_d.tight_layout()
        save_fig(_fig_d, "fig_d_heatmap")
        _out_d = _fig_d
    else:
        _out_d = mo.callout(
            mo.md(f'Heatmap skipped — need exactly 2 varying HPs, found {len(VARYING_HPS)}: {VARYING_HPS}'),
            kind="info",
        )
    _out_d
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (e) Test MSE vs Val MSE scatter

    Test MSE vs Val MSE scatter to check for overfitting.
    Points near the diagonal indicate good generalisation.
    """)
    return


@app.cell
def _(GROUP, GROUP_BY, cells_df, mo, np, plt, save_fig):
    _valid = cells_df.dropna(subset=['val_mean', 'test_mean'])
    if not _valid.empty:
        _fig_e, _ax_e = plt.subplots(figsize=(6, 5))
        _colors = plt.cm.tab10(np.linspace(0, 1, len(_valid)))

        for _i, (_, _r) in enumerate(_valid.iterrows()):
            _label = ' | '.join(str(_r[k]) for k in GROUP_BY)
            _ax_e.errorbar(
                _r['val_mean'], _r['test_mean'],
                xerr=_r.get('val_std', 0) or 0,
                yerr=_r.get('test_std', 0) or 0,
                fmt='o', color=_colors[_i], markersize=8, capsize=4,
                label=_label,
            )

        _lo = min(_valid['val_mean'].min(), _valid['test_mean'].min()) * 0.95
        _hi = max(_valid['val_mean'].max(), _valid['test_mean'].max()) * 1.05
        _ax_e.plot([_lo, _hi], [_lo, _hi], 'k--', linewidth=1, alpha=0.5, label='test = val')

        _ax_e.set_xlabel('Val MSE (min last 10 epochs, mean over seeds)')
        _ax_e.set_ylabel('Test MSE (mean over seeds)')
        _ax_e.set_title(f'Test MSE vs val MSE — {GROUP}')
        _ax_e.legend(fontsize=7, loc='upper left')
        _fig_e.tight_layout()
        save_fig(_fig_e, "fig_e_test_vs_val")
        _out_e = _fig_e
    else:
        _out_e = mo.callout(mo.md('No runs with both val and test MSE — skipping scatter (e).'), kind="warn")
    _out_e
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (f) Training statistics

    Wall time, peak GPU memory, and compute–quality Pareto scatter per HP cell.
    """)
    return


@app.cell
def _(GROUP, GROUP_BY, cells_df, np, plt, save_fig):
    _x_labels = [' | '.join(str(_r[k]) for k in GROUP_BY) for _, _r in cells_df.iterrows()]
    _x_pos    = np.arange(len(cells_df))

    _fig_f, _axes_f = plt.subplots(1, 3, figsize=(15, 4))

    _ax = _axes_f[0]
    _rt     = (cells_df['runtime_mean'].fillna(0) / 3600).values
    _rt_err = (cells_df['runtime_std'].fillna(0) / 3600).values
    _ax.bar(_x_pos, _rt, yerr=_rt_err, capsize=4, color='steelblue', alpha=0.85)
    _ax.set_xticks(_x_pos)
    _ax.set_xticklabels(_x_labels, rotation=40, ha='right', fontsize=8)
    _ax.set_ylabel('Wall time (h)')
    _ax.set_title('Runtime per HP cell')

    _ax = _axes_f[1]
    _mem = (cells_df['gpu_mem_mb'].fillna(0) / 1024).values  # GB
    _ax.bar(_x_pos, _mem, color='darkorange', alpha=0.85)
    _ax.axhline(64, color='red', linestyle='--', linewidth=1, label='MI210 64 GB')
    _ax.set_xticks(_x_pos)
    _ax.set_xticklabels(_x_labels, rotation=40, ha='right', fontsize=8)
    _ax.set_ylabel('Peak GPU memory (GB)')
    _ax.set_title('Peak GPU memory per HP cell')
    _ax.legend(fontsize=8)

    _ax = _axes_f[2]
    _valid_p = cells_df.dropna(subset=['runtime_mean', 'val_mean'])
    if not _valid_p.empty:
        _colors_p = plt.cm.tab10(np.linspace(0, 1, len(_valid_p)))
        for _i, (_, _r) in enumerate(_valid_p.iterrows()):
            _lbl = ' | '.join(str(_r[k]) for k in GROUP_BY)
            _ax.scatter(_r['runtime_mean'] / 3600, _r['val_mean'],
                       s=60 * _r['n_seeds'], color=_colors_p[_i], alpha=0.85, zorder=3, label=_lbl)
            _ax.annotate(_lbl, (_r['runtime_mean'] / 3600, _r['val_mean']),
                        fontsize=7, textcoords='offset points', xytext=(4, 4))
        _ax.set_xlabel('Total runtime (h)')
        _ax.set_ylabel('Val MSE')
        _ax.set_title('Pareto: runtime vs val MSE')
        _ax.legend(fontsize=6, loc='upper right')
    else:
        _ax.text(0.5, 0.5, 'No runtime data', ha='center', va='center', transform=_ax.transAxes)
        _ax.set_title('Pareto: runtime vs val MSE')

    _fig_f.suptitle(f'Training statistics — {GROUP}', fontsize=12, fontweight='bold')
    _fig_f.tight_layout()
    save_fig(_fig_f, "fig_f_training_stats")
    _fig_f
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## (g) GPU utilisation and CPU memory timeseries

    GPU utilization and CPU memory for the top-3 HP cells (best val_mean).
    """)
    return


@app.cell
def _(GROUP_BY, cells_df, mo, pd, plt, runs_df, save_fig, sys_histories):
    _gpu_util_col = None
    _cpu_mem_col  = None
    _best_util    = -1.0
    for _sys_df in sys_histories.values():
        if _sys_df.empty:
            continue
        for _c in _sys_df.columns:
            _low = _c.lower()
            if _low.endswith('.gpu') and 'gpu' in _low:
                _m = float(_sys_df[_c].dropna().mean()) if not _sys_df[_c].dropna().empty else 0.0
                if _m > _best_util:
                    _best_util    = _m
                    _gpu_util_col = _c
            if _cpu_mem_col is None and 'proc' in _low and 'mem' in _low:
                _cpu_mem_col = _c

    _top3     = cells_df.head(3)
    _n_panels = min(3, len(_top3))

    if _gpu_util_col or _cpu_mem_col:
        print(f'Active GPU util column : {_gpu_util_col}')
        print(f'CPU mem column         : {_cpu_mem_col}')

        _fig_g, _axes_g = plt.subplots(_n_panels, 1, figsize=(11, 3.5 * _n_panels), squeeze=False)

        for _panel_i, (_, _cell_row) in enumerate(_top3.head(_n_panels).iterrows()):
            _ax = _axes_g[_panel_i][0]
            _cell_runs = runs_df.copy()
            for _k in GROUP_BY:
                _cell_runs = _cell_runs[_cell_runs[_k] == _cell_row[_k]]

            _label = ' | '.join(f'{k}={_cell_row[k]}' for k in GROUP_BY)
            _ax2   = None

            for _, _r in _cell_runs.iterrows():
                _sdf = sys_histories.get(_r['run_id'], pd.DataFrame())
                if _sdf.empty or '_runtime' not in _sdf.columns:
                    continue
                _t = _sdf['_runtime'] / 3600
                if _gpu_util_col and _gpu_util_col in _sdf.columns:
                    _ax.plot(_t, _sdf[_gpu_util_col].values, color='royalblue', alpha=0.6, linewidth=1)
                if _cpu_mem_col and _cpu_mem_col in _sdf.columns:
                    if _ax2 is None:
                        _ax2 = _ax.twinx()
                    _ax2.plot(_t, _sdf[_cpu_mem_col].values, color='tomato', alpha=0.6, linewidth=1)

            _ax.set_title(f'Top-{_panel_i + 1}: {_label}', fontsize=9)
            _ax.set_xlabel('Runtime (h)')
            if _gpu_util_col:
                _ax.set_ylabel(f'GPU util % ({_gpu_util_col})', color='royalblue')
            if _ax2 is not None:
                _ax2.set_ylabel(f'CPU mem MB ({_cpu_mem_col})', color='tomato')

        _fig_g.suptitle('GPU utilisation and CPU memory — top-3 HP cells', fontsize=12, fontweight='bold')
        _fig_g.tight_layout()
        save_fig(_fig_g, "fig_g_system_metrics")
        _out_g = _fig_g
    else:
        _sample_cols = []
        for _s in sys_histories.values():
            if not _s.empty:
                _sample_cols = list(_s.columns)[:20]
                break
        _out_g = mo.callout(mo.md(
            f'System metrics not available — system.parquet missing or columns not recognised.\n\n'
            f'Available columns (first run): `{_sample_cols}`'
        ), kind="info")
    _out_g
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Recommendation

    Selection rules (from `docs/gnn_only_hp_search_plan.md` §Verification):
    1. **Primary**: lowest `val_mean` (= mean over seeds of min val/loss_total last 10 epochs).
    2. **Tie-break** within `OCCAM_TOL` of leader: smaller `val_std` → smaller `gap` → smaller model.
    3. **Gate check**: per-property val MSE vs. Stage-5 acceptance thresholds.
    """)
    return


@app.cell
def _(GATES, GROUP, GROUP_BY, OCCAM_TOL, PROPS, cells_df, mo, np):
    if cells_df.empty:
        _out_rec = mo.callout(mo.md("No cells to rank."), kind="warn")
    else:
        _best     = cells_df.iloc[0]
        _best_val = _best['val_mean']

        _hp_lines = "\n".join(f"- **`{k}`**: `{_best[k]}`" for k in GROUP_BY)
        _gate_lines = []
        for _prop, _gate in GATES.items():
            _col = f'val_{_prop}'
            if _col in _best.index and not np.isnan(_best[_col]):
                _val = _best[_col]
                _status = "**PASS**" if _val < _gate else "**FAIL**"
                _pct = (_val - _gate) / _gate * 100
                _diff = f"improvement of {abs(_pct):.1f} %" if _val < _gate else f"decline of {_pct:.1f} %"
                _gate_lines.append(f"  - `{_prop}`: {_status} — val={_val:.4f} vs gate={_gate} ({_diff})")
            else:
                _gate_lines.append(f"  - `{_prop}`: N/A (property not in this group)")

        _r2_lines = []
        for _prop in PROPS:
            _r2_col = f'r2_{_prop}'
            if _r2_col in _best.index and not np.isnan(_best[_r2_col]):
                _r2_val = _best[_r2_col]
                _tag = "GOOD" if _r2_val >= 0.85 else ("OK" if _r2_val >= 0.5 else "WEAK")
                _r2_lines.append(f"  - `{_prop}`: R² = {_r2_val:.3f} [{_tag}]")

        # Tie-break candidates
        _runner_ups = cells_df.iloc[1:]
        _tie_note = ""
        if not _runner_ups.empty and not np.isnan(_best_val):
            _thresh = _best_val * (1 + OCCAM_TOL)
            _close  = _runner_ups[_runner_ups['val_mean'] <= _thresh]
            if not _close.empty:
                _close_strs = [
                    f"`{'|'.join(f'{k}={r[k]}' for k in GROUP_BY)}` val={r['val_mean']:.4f}"
                    for _, r in _close.iterrows()
                ]
                _tie_note = (
                    f"\n\n**Tie-break ({OCCAM_TOL*100:.0f}% tolerance)** — "
                    f"{len(_close)} close competitor(s): " + "; ".join(_close_strs)
                )

        _out_rec = mo.callout(mo.md(f"""
        **Recommended HP combination — {GROUP}**

        {_hp_lines}

        - **val_mean**: {_best_val:.4f} ± {_best['val_std']:.4f}
        - **test_mean**: {_best['test_mean']:.4f}
        - **gap |test − val|**: {_best['gap']:.4f}
        - **n_seeds**: {int(_best['n_seeds'])}
        {_tie_note}

        **Gate check (val MSE vs Stage-5 thresholds)**:
        {chr(10).join(_gate_lines)}

        **Per-property R² (last-10 mean, mean over seeds)**:
        {chr(10).join(_r2_lines) if _r2_lines else "  N/A"}
        """), kind="info")
    _out_rec
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-group Comparison (optional)

    Set `GROUPS = ['stage_0_baseline', 'stage_1_lr', ...]` in the Configuration cell
    to see whether each stage improved on the previous one.
    """)
    return


@app.cell
def _(GROUPS, HP_COLS, agg_cells, load_group, mo, np, pd, plt, save_fig):
    _mg_out = mo.md("*(Multi-group comparison disabled — set `GROUPS` with multiple entries.)*")

    if GROUPS:
        _group_bests = []
        for _g in GROUPS:
            try:
                _gdf, _, _ = load_group(_g)
                _ghps = [c for c in HP_COLS if c in _gdf.columns and _gdf[c].nunique() > 1]
                _gby  = _ghps if _ghps else ['comp_mode']
                _gcells = agg_cells(_gdf, _gby)
                _group_bests.append({
                    'group':     _g,
                    'best_val':  _gcells['val_mean'].min(),
                    'best_test': _gcells['test_mean'].min(),
                })
            except FileNotFoundError as _exc:
                print(f'  Skipping {_g}: {_exc}')

        if _group_bests:
            _gb_df = pd.DataFrame(_group_bests)
            _x = np.arange(len(_gb_df))

            _fig_mg, _ax_mg = plt.subplots(figsize=(max(6, len(_gb_df) * 1.8), 4))
            _ax_mg.bar(_x - 0.2, _gb_df['best_val'],  width=0.38, label='best val MSE',
                       color='steelblue', alpha=0.85)
            _ax_mg.bar(_x + 0.2, _gb_df['best_test'], width=0.38, label='best test MSE',
                       color='darkorange', alpha=0.85)
            _ax_mg.set_xticks(_x)
            _ax_mg.set_xticklabels(_gb_df['group'], rotation=25, ha='right')
            _ax_mg.set_ylabel('MSE')
            _ax_mg.set_title('Best val/test MSE per stage')
            _ax_mg.legend()
            _fig_mg.tight_layout()
            save_fig(_fig_mg, "fig_mg_multi_group")

            _mg_out = mo.vstack([
                _fig_mg,
                mo.as_html(_gb_df.round(4)),
            ])

    _mg_out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusions

    1. **Best HP combination** is reported in the Recommendation section above.
    2. **Gate checks** compare per-property val MSE against Stage-5 acceptance thresholds.
    3. **Per-property R²** provides an interpretable quality signal alongside the MSE selection metric.
    4. **Caveats**: Results depend on the specific group/stage loaded. Re-run with different `GROUP`
       values or set `GROUPS` for cross-stage comparison.
    5. **Next steps**: Use the recommended config for Stage 5 (5-seed confirmation run).
    """)
    return


if __name__ == "__main__":
    app.run()
