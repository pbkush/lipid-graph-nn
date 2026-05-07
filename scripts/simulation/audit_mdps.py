#!/usr/bin/env python3
"""MDP audit CLI: compare production/equilibration/minimization MDP parameters
across all legacy membrane-only systems and write a human-readable report and
machine-readable freeze record for use by mdp_writer.py (step 4).

Usage:
    python scripts/simulation/audit_mdps.py \
        --systems-root data/membrane_only \
        --output-md docs/mdp_audit_report.md \
        --output-freeze lipid_gnn/martini_pipeline/templates/_audit_freeze.json
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from lipid_gnn.martini_pipeline.analysis import diff_mdps


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit MDP parameters across legacy systems.")
    parser.add_argument("--systems-root", default="data/membrane_only",
                        help="Root directory containing one sub-directory per system.")
    parser.add_argument("--output-md", default="docs/mdp_audit_report.md",
                        help="Path for the human-readable Markdown report.")
    parser.add_argument("--output-freeze",
                        default="lipid_gnn/martini_pipeline/templates/_audit_freeze.json",
                        help="Path for the machine-readable JSON freeze record.")
    parser.add_argument("--gmx-binary", default="gmx",
                        help="GROMACS binary used for `gmx dump -s` (tpr extraction).")
    parser.add_argument("--stages", nargs="+",
                        default=["run", "equilibration", "minimization"],
                        help="Stages to audit.")
    args = parser.parse_args()

    systems_root = os.path.join(_REPO_ROOT, args.systems_root) \
        if not os.path.isabs(args.systems_root) else args.systems_root

    print(f"Auditing MDP parameters in: {systems_root}")
    print(f"Stages: {args.stages}")

    report = diff_mdps(systems_root, stages=tuple(args.stages), gmx_binary=args.gmx_binary)

    for stage, audit in report.stages.items():
        status = "OK" if not audit.deviations else f"{len(audit.deviations)} DEVIATIONS"
        print(f"  {stage:15s}: {audit.n_systems} systems, "
              f"{len(audit.missing_systems)} missing — {status}")

    # Write markdown report
    md_path = args.output_md if os.path.isabs(args.output_md) \
        else os.path.join(_REPO_ROOT, args.output_md)
    os.makedirs(os.path.dirname(md_path), exist_ok=True)
    with open(md_path, "w") as fh:
        fh.write(report.to_markdown())
    print(f"Report written: {md_path}")

    # Write freeze JSON
    freeze_path = args.output_freeze if os.path.isabs(args.output_freeze) \
        else os.path.join(_REPO_ROOT, args.output_freeze)
    os.makedirs(os.path.dirname(freeze_path), exist_ok=True)
    with open(freeze_path, "w") as fh:
        fh.write(report.to_freeze_json())
    print(f"Freeze record written: {freeze_path}")

    if report.total_deviations > 0:
        print(f"\nWARNING: {report.total_deviations} unexpected parameter deviation(s) found.")
        print("Review docs/mdp_audit_report.md before building templates.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
