"""
Phase 0: Linear Composition Baseline

Trains a Ridge regression model directly on composition fraction vectors derived
from directory names (e.g., POPC80_DOPC20 → [0.8, 0.2, 0, ...]).

This establishes the performance floor and tests whether explicit composition
identity alone is sufficient to predict lipid packing — without any graph structure.

If the linear model achieves Test MSE < 0.5, it proves composition is the dominant
signal and the GNN is failing to leverage even this basic information.

Usage:
    conda run -n lipid_gnn python3 scripts/training/linear_baseline.py
"""

import os
import sys
import re
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score

root_dir = Path(__file__).resolve().parents[2]
DATA_DIR = root_dir / 'data/membrane_only'
RESULTS_DIR = root_dir / 'results/properties'
OUT_DIR = root_dir / 'results/training/lipid_packing/linear_baseline'
OUT_DIR.mkdir(parents=True, exist_ok=True)

# === Lipid type vocabulary (fixed order for reproducibility) ===
LIPID_TYPES = ['POPC', 'DOPC', 'DIPC', 'DPPC', 'POPE', 'DOPE', 'DPPE', 'DOPS', 'POPS', 'CHOL']


def parse_composition(dirname: str) -> np.ndarray:
    """
    Parse a membrane composition directory name into a molar fraction vector.

    Examples:
        POPC100              → [1.0, 0, 0, ...]
        POPC80_DOPC20        → [0.8, 0.2, 0, ...]
        POPC60_CHOL40        → [0.6, 0, 0, ..., 0.4]

    Returns:
        np.ndarray of shape (len(LIPID_TYPES),), float32
    """
    vec = np.zeros(len(LIPID_TYPES), dtype=np.float32)
    # Pattern: e.g. POPC80, DOPC20, CHOL40
    pattern = re.compile(r'([A-Z]+)(\d+)')
    matches = pattern.findall(dirname)

    for lipid, pct_str in matches:
        pct = int(pct_str)
        if lipid in LIPID_TYPES:
            idx = LIPID_TYPES.index(lipid)
            vec[idx] = pct / 100.0
        else:
            print(f"  [WARNING] Unknown lipid type '{lipid}' in '{dirname}'")

    # Normalize (should already sum to ~1.0, but enforce it)
    if vec.sum() > 0:
        vec /= vec.sum()

    return vec


def load_dataset() -> tuple[list[str], np.ndarray, np.ndarray]:
    """
    Load all compositions, their feature vectors, and lipid_packing targets.

    Returns:
        names:   list of composition names, length N
        X:       np.ndarray of shape (N, n_lipid_types)
        y:       np.ndarray of shape (N,)
    """
    compositions = sorted([
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(DATA_DIR / d)
    ])

    names, X_rows, y_vals = [], [], []

    for comp in compositions:
        prop_file = RESULTS_DIR / f"{comp}.h5"
        if not prop_file.exists():
            print(f"  [SKIP] No property file for {comp}")
            continue

        with open(prop_file, 'rb') as f:
            data = pickle.load(f)
        mean_dict = data[0]

        if 'lipid_packing' not in mean_dict:
            print(f"  [SKIP] No lipid_packing in {comp}")
            continue

        names.append(comp)
        X_rows.append(parse_composition(comp))
        y_vals.append(mean_dict['lipid_packing'])

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_vals, dtype=np.float32)
    return names, X, y


def run_baseline():
    print("=" * 60)
    print("  Phase 0: Linear Composition Baseline")
    print("=" * 60)

    # 1. Load
    names, X, y_raw = load_dataset()
    N = len(names)
    print(f"\nLoaded {N} compositions.")

    # 2. Normalize targets (fit on ALL data — for LOO this is correct since each
    #    fold's test point is excluded from prediction; the scaler is the same)
    scaler_y = StandardScaler()
    y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()

    print(f"Target (raw)  : mean={y_raw.mean():.4f}, std={y_raw.std():.4f}")
    print(f"Target (z)    : mean={y.mean():.4f}, std={y.std():.4f}")

    # 3. Leave-One-Out Cross-Validation
    #    With N=70, LOO is the gold standard — every sample is used for testing once.
    loo = LeaveOneOut()
    model = Ridge(alpha=1.0)

    print(f"\nRunning LOO-CV with Ridge(alpha=1.0) on {N} samples...")
    y_pred_loo = cross_val_predict(model, X, y, cv=loo)

    loo_mse  = mean_squared_error(y, y_pred_loo)
    loo_r2   = r2_score(y, y_pred_loo)
    loo_mae  = np.mean(np.abs(y - y_pred_loo))

    print(f"\n{'=' * 40}")
    print(f"  LOO Results (z-scored targets)")
    print(f"{'=' * 40}")
    print(f"  MSE : {loo_mse:.4f}")
    print(f"  MAE : {loo_mae:.4f}")
    print(f"  R²  : {loo_r2:.4f}")

    # Inverse-transform for interpretability
    y_raw_pred = scaler_y.inverse_transform(y_pred_loo.reshape(-1, 1)).ravel()
    raw_mse  = mean_squared_error(y_raw, y_raw_pred)
    raw_mae  = np.mean(np.abs(y_raw - y_raw_pred))
    print(f"\n  MSE (raw Å²): {raw_mse:.6f}")
    print(f"  MAE (raw Å²): {raw_mae:.6f}")

    # Per-sample residuals
    residuals = y - y_pred_loo
    worst_idx = np.argsort(np.abs(residuals))[::-1]
    print(f"\n  Top 5 worst predictions:")
    for i in worst_idx[:5]:
        print(f"    {names[i]:30s} true={y[i]:+.3f}  pred={y_pred_loo[i]:+.3f}  err={residuals[i]:+.3f}")

    # 4. Feature importance (train on full dataset for interpretation)
    model_full = Ridge(alpha=1.0)
    model_full.fit(X, y)
    coefs = model_full.coef_
    print(f"\n  Ridge Coefficients (trained on all data):")
    for lipid, coef in sorted(zip(LIPID_TYPES, coefs), key=lambda x: abs(x[1]), reverse=True):
        bar = '█' * int(abs(coef) * 5)
        sign = '+' if coef >= 0 else '-'
        print(f"    {lipid:6s} : {sign}{abs(coef):.4f} {bar}")

    # 5. Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    scatter = ax.scatter(y, y_pred_loo, c=np.abs(residuals), cmap='RdYlGn_r',
                         alpha=0.8, s=60, edgecolors='k', linewidths=0.4)
    plt.colorbar(scatter, ax=ax, label='|Residual| (z-score)')

    lo = min(y.min(), y_pred_loo.min()) - 0.3
    hi = max(y.max(), y_pred_loo.max()) + 0.3
    ax.plot([lo, hi], [lo, hi], 'r--', linewidth=1.5, label='Perfect prediction')
    ax.set_xlabel('True lipid_packing (z-scored)', fontsize=12)
    ax.set_ylabel('Predicted lipid_packing (z-scored)', fontsize=12)
    ax.set_title(f'Linear Baseline (Ridge, LOO-CV)\nMSE={loo_mse:.3f}  R²={loo_r2:.3f}', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_scatter = OUT_DIR / 'scatter_loo.png'
    fig.savefig(out_scatter, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Scatter plot saved → {out_scatter}")

    # 6. Label distribution vs residuals
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(y, bins=20, color='steelblue', edgecolor='k', alpha=0.8)
    axes[0].set_title('Target Distribution (z-scored)')
    axes[0].set_xlabel('z-score')
    axes[0].set_ylabel('Count')

    axes[1].bar(LIPID_TYPES, coefs, color=['#e63946' if c > 0 else '#457b9d' for c in coefs])
    axes[1].set_title('Ridge Coefficients')
    axes[1].set_xlabel('Lipid Type')
    axes[1].set_ylabel('Coefficient')
    axes[1].axhline(0, color='k', linewidth=0.8)
    plt.xticks(rotation=30, ha='right')
    fig.tight_layout()
    out_dist = OUT_DIR / 'distribution_and_coefs.png'
    fig.savefig(out_dist, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Distribution plot saved → {out_dist}")

    # 7. Summary text file
    summary_path = OUT_DIR / 'baseline_summary.txt'
    with open(summary_path, 'w') as f:
        f.write("Linear Composition Baseline — LOO Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"N compositions : {N}\n")
        f.write(f"MSE (z-scored) : {loo_mse:.4f}\n")
        f.write(f"MAE (z-scored) : {loo_mae:.4f}\n")
        f.write(f"R² (z-scored)  : {loo_r2:.4f}\n")
        f.write(f"MSE (raw Å²)   : {raw_mse:.6f}\n")
        f.write(f"MAE (raw Å²)   : {raw_mae:.6f}\n\n")
        f.write("Ridge Coefficients (sorted by |coef|):\n")
        for lipid, coef in sorted(zip(LIPID_TYPES, coefs), key=lambda x: abs(x[1]), reverse=True):
            f.write(f"  {lipid:6s}: {coef:+.4f}\n")
    print(f"  Summary saved  → {summary_path}")

    print(f"\n{'=' * 60}")
    print(f"  Baseline complete. Results in {OUT_DIR}")
    print(f"{'=' * 60}\n")

    return loo_mse, loo_r2


if __name__ == "__main__":
    run_baseline()
