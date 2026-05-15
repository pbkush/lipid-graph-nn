#!/usr/bin/env python3
"""Scan output roots for completed Martini-pipeline simulations and write CSV.

Designed for the workflow:

    # locally, where the data lives:
    python scripts/python/scan_completed_systems.py --out done.csv

    # upload done.csv to HPC, then:
    bash scripts/bash/submit_simulations.sh \\
        --missing-from-grid popc_interpolation \\
        --completed-csv done.csv \\
        --prod-ns 100 --partition general1

This lets the submitter know which systems are already simulated without
needing the data physically present on the HPC.  Compositions in the CSV
are dropped from the queue before any sbatch is built.

CANONICALISATION: directory names are re-parsed through composition.parse_name
so non-canonically-ordered legacy directories (e.g. ``POPC10_DIPC90`` →
canonical ``DIPC90_POPC10``) are matched correctly by submit_simulations.sh,
which only ever sees grid-generated canonical names.

CSV format (one row per simulated system, header line included):

    canonical_name,source_dir,source_root,status,has_prun_xtc

  canonical_name : canonical composition string (matched by submit_simulations.sh)
  source_dir     : directory name as it appears on disk (may be non-canonical)
  source_root    : root path under which the system was found
  status         : manifest's overall_status, "legacy_no_manifest" if only the
                   run/prun.xtc fallback applies, or "invalid_manifest"
  has_prun_xtc   : true|false (the fallback signal for legacy systems)

A directory that can't be parsed as a composition name (e.g. stray dirs like
`fixtures/` or a typo) is reported on stderr and excluded from the CSV.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from lipid_gnn.martini_pipeline.analysis import summarise_systems
from lipid_gnn.martini_pipeline.composition import parse_name


def _default_output_roots() -> list[Path]:
    """Return [legacy_data_dir, pipeline_output_root] from CONFIG."""
    from lipid_gnn.config import CONFIG
    roots: list[Path] = []
    if hasattr(CONFIG, "paths") and CONFIG.paths is not None:
        if (d := CONFIG.paths.data_dir):
            roots.append(Path(d))
    if CONFIG.martini_pipeline is not None and CONFIG.martini_pipeline.output_root:
        roots.append(Path(CONFIG.martini_pipeline.output_root))
    return roots


def scan_root(root: Path) -> list[dict]:
    """Walk *root* and return one CSV-ready dict per parseable subdir.

    Uses summarise_systems(legacy_fallback=True) so directories without a
    manifest but with a non-empty run/prun.xtc are still reported (with
    status="legacy_no_manifest").
    """
    rows: list[dict] = []
    if not root.is_dir():
        return rows

    for status in summarise_systems(root, legacy_fallback=True):
        source_dir = status.canonical_name  # the raw on-disk directory name
        try:
            canonical = parse_name(source_dir).name
        except ValueError as exc:
            print(
                f"  WARN: skipping unparseable directory {root}/{source_dir!r}: {exc}",
                file=sys.stderr,
            )
            continue

        if status.has_manifest:
            row_status = status.overall_status or "manifest_missing_status"
        elif status.has_prun_xtc:
            row_status = "legacy_no_manifest"
        else:
            # Neither manifest nor a usable prun.xtc — not really "done".
            # summarise_systems returned it because of legacy_fallback=True,
            # but we don't want to mark it as done in the CSV.
            continue

        rows.append({
            "canonical_name": canonical,
            "source_dir":     source_dir,
            "source_root":    str(root),
            "status":         row_status,
            "has_prun_xtc":   "true" if status.has_prun_xtc else "false",
        })

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scan output roots for completed Martini simulations and write CSV."
    )
    parser.add_argument(
        "--output-roots", nargs="+", default=None,
        help="Directories to scan (default: CONFIG.paths.data_dir + martini_pipeline.output_root)",
    )
    parser.add_argument(
        "--out", default=str(_REPO_ROOT / "results" / "completed_systems.csv"),
        help="Output CSV path (default: results/completed_systems.csv)",
    )
    parser.add_argument(
        "--status-filter", nargs="+", default=None,
        help="Include only rows whose status is in this list (default: include all)",
    )
    args = parser.parse_args()

    roots = [Path(r) for r in args.output_roots] if args.output_roots else _default_output_roots()
    if not roots:
        print("ERROR: no output roots configured and none given via --output-roots", file=sys.stderr)
        return 1

    all_rows: list[dict] = []
    canonicals_seen: set[str] = set()  # dedupe across roots, first-seen wins
    for root in roots:
        if not root.is_dir():
            print(f"  INFO: skipping non-existent root {root}", file=sys.stderr)
            continue
        for row in scan_root(root):
            if row["canonical_name"] in canonicals_seen:
                continue
            if args.status_filter and row["status"] not in args.status_filter:
                continue
            canonicals_seen.add(row["canonical_name"])
            all_rows.append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "canonical_name", "source_dir", "source_root", "status", "has_prun_xtc",
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    by_status: dict[str, int] = {}
    for row in all_rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    print(f"Wrote {len(all_rows)} rows to {out_path}")
    for s, n in sorted(by_status.items()):
        print(f"  {s}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
