#!/usr/bin/env python3
"""Print the work queue of missing Martini 3 bilayer compositions.

Identifies which target compositions have not yet been successfully simulated
and prints/writes the result in the requested format.

Usage examples:
    python scripts/simulation/print_work_queue.py --grid dppc_corner
    python scripts/simulation/print_work_queue.py --grid dopc_corner --format json
    python scripts/simulation/print_work_queue.py --grid all --format lines --out /tmp/queue.txt
    python scripts/simulation/print_work_queue.py --grid binary --lipids DPPC DOPC --step 10
    python scripts/simulation/print_work_queue.py --grid ternary --lipids POPC DOPC DPPC --step 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from lipid_gnn.martini_pipeline.analysis import (
    binary_grid,
    dopc_corner_grid,
    dppc_corner_grid,
    missing_compositions,
    ternary_grid,
)


def _build_target_grid(args: argparse.Namespace):
    """Build the target Composition list from CLI args."""
    if args.grid == "dppc_corner":
        return dppc_corner_grid(step=args.step)
    if args.grid == "dopc_corner":
        return dopc_corner_grid(step=args.step)
    if args.grid == "all":
        seen: set[str] = set()
        result = []
        for comp in dppc_corner_grid(step=args.step) + dopc_corner_grid(step=args.step):
            if comp.name not in seen:
                seen.add(comp.name)
                result.append(comp)
        return result
    if args.grid == "binary":
        if not args.lipids or len(args.lipids) != 2:
            print("error: --grid binary requires exactly --lipids A B", file=sys.stderr)
            sys.exit(2)
        return binary_grid(args.lipids[0], args.lipids[1], step=args.step)
    if args.grid == "ternary":
        if not args.lipids or len(args.lipids) != 3:
            print("error: --grid ternary requires exactly --lipids A B C", file=sys.stderr)
            sys.exit(2)
        return ternary_grid(args.lipids, step=args.step)
    print(f"error: unknown --grid value {args.grid!r}", file=sys.stderr)
    sys.exit(2)


def _default_output_roots() -> list[str]:
    """Return [legacy_data_dir, pipeline_output_root] from CONFIG if available."""
    roots: list[str] = []
    try:
        from lipid_gnn.config import CONFIG
        if hasattr(CONFIG, "paths") and CONFIG.paths is not None:
            data_dir = str(CONFIG.paths.data_dir)
            legacy = os.path.join(data_dir, "membrane_only")
            if os.path.isdir(legacy):
                roots.append(legacy)
        if CONFIG.martini_pipeline is not None:
            roots.append(str(CONFIG.martini_pipeline.output_root))
    except Exception:
        pass
    return roots


def _format_table(compositions) -> str:
    rows = []
    for comp in compositions:
        fracs_str = " ".join(f"{k}={v:.2f}" for k, v in comp.fractions.items())
        n = len(comp.fractions)
        kind = "pure" if n == 1 else "bin" if n == 2 else "ternary"
        rows.append((comp.name, kind, fracs_str))

    if not rows:
        return ""

    name_w = max(len(r[0]) for r in rows)
    kind_w = max(len(r[1]) for r in rows)
    frac_w = max(len(r[2]) for r in rows)

    header = f"{'canonical_name':<{name_w}}  {'type':<{kind_w}}  fractions"
    sep = f"{'-' * name_w}  {'-' * kind_w}  {'-' * frac_w}"
    lines = [header, sep]
    for name, kind, fracs in rows:
        lines.append(f"{name:<{name_w}}  {kind:<{kind_w}}  {fracs}")
    return "\n".join(lines)


def _format_json(compositions) -> str:
    items = [
        {"canonical_name": c.name, "fractions": dict(c.fractions)}
        for c in compositions
    ]
    return json.dumps(items, indent=2)


def _format_lines(compositions) -> str:
    return "\n".join(c.name for c in compositions)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print the Martini pipeline work queue (missing compositions).",
    )
    parser.add_argument(
        "--grid",
        choices=["dppc_corner", "dopc_corner", "binary", "ternary", "all"],
        default="all",
        help="Target grid to check (default: all = dppc_corner ∪ dopc_corner).",
    )
    parser.add_argument(
        "--lipids", nargs="+", metavar="LIPID",
        help="Lipid names for --grid binary (2) or --grid ternary (3).",
    )
    parser.add_argument(
        "--step", type=int, default=10,
        help="Grid step in integer percentage points (default: 10).",
    )
    parser.add_argument(
        "--output-roots", nargs="+", metavar="ROOT",
        help="Directories to scan for existing simulations. "
             "Defaults to legacy data/membrane_only/ and pipeline output root from config.",
    )
    parser.add_argument(
        "--require-status", nargs="+", default=["ok"], metavar="STATUS",
        help="Manifest overall_status values counted as present (default: ok).",
    )
    parser.add_argument(
        "--no-legacy-fallback", action="store_true",
        help="Disable legacy fallback: only count manifest-validated systems as present.",
    )
    parser.add_argument(
        "--format", choices=["table", "json", "lines"], default="table",
        dest="fmt",
        help="Output format (default: table). --format is authoritative; suffix is ignored.",
    )
    parser.add_argument(
        "--out", metavar="PATH",
        help="Write output to PATH in addition to stdout.",
    )
    args = parser.parse_args()

    target = _build_target_grid(args)
    output_roots = args.output_roots if args.output_roots else _default_output_roots()
    legacy_fallback = not args.no_legacy_fallback

    missing = missing_compositions(
        target,
        output_roots,
        require_status=tuple(args.require_status),
        legacy_fallback=legacy_fallback,
    )

    n_total = len({c.name for c in target})
    n_missing = len(missing)
    n_present = n_total - n_missing

    if args.fmt == "table":
        body = _format_table(missing)
    elif args.fmt == "json":
        body = _format_json(missing)
    else:
        body = _format_lines(missing)

    if n_missing == 0:
        print("(no missing compositions)", file=sys.stderr)
    else:
        print(body)

    summary = f"\n{n_missing} composition(s) queued, {n_present} already simulated."
    if args.fmt == "table":
        print(summary)

    if args.out:
        with open(args.out, "w") as fh:
            fh.write(body)
            if args.fmt == "table":
                fh.write(summary + "\n")
        print(f"Written to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
