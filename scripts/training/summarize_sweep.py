"""
summarize_sweep.py

Scans a training results directory and generates a rich Markdown report
comparing all runs based on their FINAL epoch metrics (i.e. convergence state).
"""

import os
import re
import ast
import sys
import datetime
import pandas as pd
from pathlib import Path


# ─────────────────────────────────────────────────────────
# Log Parsing
# ─────────────────────────────────────────────────────────

def parse_training_log(log_path):
    """Parses a training_log.txt to extract config and FINAL epoch metrics."""
    with open(log_path, 'r') as f:
        content = f.read()

    # Extract Config dict (single line after "Config: ")
    config_match = re.search(r"Config: ({.*?})", content)
    if not config_match:
        return None
    try:
        config = ast.literal_eval(config_match.group(1))
    except Exception as e:
        print(f"  [WARN] Could not parse config in {log_path}: {e}")
        return None

    # Extract all logged epoch metrics
    metric_pattern = re.compile(r"Epoch (\d+) \| Train MSE: ([\d.]+) \| Test MSE: ([\d.]+)")
    metrics = metric_pattern.findall(content)
    if not metrics:
        return None

    df_metrics = pd.DataFrame(metrics, columns=['epoch', 'train_mse', 'test_mse']).astype(float)

    # Use the LAST logged epoch — this represents the converged state
    last_row = df_metrics.iloc[-1]

    run_dir = Path(log_path).parent
    return {
        'run_name':      config.get('run_name', run_dir.name),
        'run_dir':       str(run_dir.resolve()),
        'final_test_mse':  round(last_row['test_mse'], 4),
        'final_train_mse': round(last_row['train_mse'], 4),
        'final_epoch':     int(last_row['epoch']),
        'gap':             round(last_row['test_mse'] - last_row['train_mse'], 4),
        'num_layers':    config.get('num_layers'),
        'hidden_dim':    config.get('hidden_dim'),
        'weight_decay':  config.get('weight_decay', 1e-4),
        'batch_size':    config.get('batch_size'),
        'lr':            config.get('learning_rate'),
    }


# ─────────────────────────────────────────────────────────
# Markdown Helpers
# ─────────────────────────────────────────────────────────

def md_table(df, columns, col_labels=None):
    """Renders a subset of dataframe columns as a Markdown table."""
    if col_labels is None:
        col_labels = columns
    header = "| " + " | ".join(col_labels) + " |"
    sep    = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for c in columns:
            val = row[c]
            # Highlight low gap values
            if c == 'gap' and abs(val) < 0.05:
                cells.append(f"**{val}** ✅")
            else:
                cells.append(str(val))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows)


def run_section(rank, row):
    """Generates a detailed Markdown section for a single run."""
    name = row['run_name']
    run_dir = Path(row['run_dir'])
    loss_img  = run_dir / 'loss_curve.png'
    scatter_img = run_dir / 'accuracy_scatter.png'
    log_file  = run_dir / 'training_log.txt'

    lines = [
        f"### #{rank} · `{name}`",
        "",
        f"| Metric | Value |",
        f"| --- | --- |",
        f"| Final Test MSE  | **{row['final_test_mse']}** |",
        f"| Final Train MSE | {row['final_train_mse']} |",
        f"| Gap (Test−Train)| {row['gap']} |",
        f"| Layers          | {row['num_layers']} |",
        f"| Hidden Dim      | {row['hidden_dim']} |",
        f"| Weight Decay    | {row['weight_decay']} |",
        f"| Batch Size      | {row['batch_size']} |",
        f"| Learning Rate   | {row['lr']} |",
        "",
        f"📁 [Open run directory]({run_dir})",
        f"  · [Training Log]({log_file})",
        "",
    ]

    if loss_img.exists():
        lines.append(f"![Loss Curve]({loss_img})")
        lines.append("")
    if scatter_img.exists():
        lines.append(f"![Accuracy Scatter]({scatter_img})")
        lines.append("")

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────
# Main Report Generator
# ─────────────────────────────────────────────────────────

def summarize_sweep(results_dir, top_n=3):
    """Scans a results directory and writes a Markdown summary report."""
    results_path = Path(results_dir).resolve()
    if not results_path.exists():
        print(f"[ERROR] Directory not found: {results_dir}")
        return

    print(f"Scanning: {results_path}")
    all_results = []
    for log_file in sorted(results_path.rglob("training_log.txt")):
        res = parse_training_log(log_file)
        if res:
            all_results.append(res)
        else:
            print(f"  [SKIP] {log_file}")

    if not all_results:
        print("No valid training logs found.")
        return

    df = pd.DataFrame(all_results).sort_values(by='final_test_mse', ascending=True).reset_index(drop=True)

    # ── Build Markdown ──────────────────────────────────
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    report_lines = [
        f"# Hyperparameter Sweep Report",
        f"",
        f"> Generated: {now}  ·  Property: `{results_path.name}`  ·  Runs: {len(df)}",
        f"",
        f"---",
        f"",
        f"## Overview",
        f"",
        f"Metrics are taken from the **last logged epoch** of each run,",
        f"reflecting the converged state of each model.",
        f"",
        f"| Statistic | Value |",
        f"| --- | --- |",
        f"| Total runs | {len(df)} |",
        f"| Best Final Test MSE | **{df['final_test_mse'].min()}** |",
        f"| Median Final Test MSE | {df['final_test_mse'].median():.4f} |",
        f"| Runs with Gap < 0.05 | {(df['gap'].abs() < 0.05).sum()} |",
        f"",
        f"---",
        f"",
        f"## Comparison Table",
        f"",
        f"Sorted by **Final Test MSE** (lower is better).",
        f"A ✅ in the **Gap** column indicates near-zero overfitting (|gap| < 0.05).",
        f"",
    ]

    table_cols   = ['run_name', 'final_test_mse', 'final_train_mse', 'gap', 'num_layers', 'weight_decay', 'batch_size', 'lr']
    table_labels = ['Run Name',  'Test MSE (final)', 'Train MSE (final)', 'Gap',  'Layers', 'Weight Decay', 'Batch Size', 'LR']
    report_lines.append(md_table(df, table_cols, table_labels))
    report_lines.append("")
    report_lines.append("---")
    report_lines.append("")

    # ── Top-N detailed sections ──────────────────────────
    report_lines.append(f"## Top {top_n} Runs — Detailed")
    report_lines.append("")
    report_lines.append(
        f"The following runs achieved the lowest converged Test MSE. "
        f"Each section links to the full training log and embeds the diagnostic plots."
    )
    report_lines.append("")

    for rank, (_, row) in enumerate(df.head(top_n).iterrows(), start=1):
        report_lines.append(run_section(rank, row))
        report_lines.append("---")
        report_lines.append("")

    # ── Footer ──────────────────────────────────────────
    report_lines += [
        "## How to Regenerate",
        "",
        "```bash",
        f"conda run -n lipid_gnn python3 scripts/training/summarize_sweep.py {results_dir}",
        "```",
    ]

    report_md = "\n".join(report_lines)

    # ── Save ────────────────────────────────────────────
    report_path = results_path / "sweep_report.md"
    with open(report_path, 'w') as f:
        f.write(report_md)

    # Also save summary CSV
    csv_path = results_path / "sweep_summary.csv"
    df[table_cols].to_csv(csv_path, index=False)

    print(f"\n✅ Report saved to: {report_path}")
    print(f"✅ CSV saved to:    {csv_path}")


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    results_base = "results/training/lipid_packing"
    if len(sys.argv) > 1:
        results_base = sys.argv[1]
    summarize_sweep(results_base)
