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

    canonical_name,source_dir,source_root,status,has_prun_xtc,sim_ns,sim_ns_source

  canonical_name : canonical composition string (matched by submit_simulations.sh)
  source_dir     : directory name as it appears on disk (may be non-canonical)
  source_root    : root path under which the system was found
  status         : manifest's overall_status, "legacy_no_manifest" if only the
                   run/prun.xtc fallback applies, or "invalid_manifest"
  has_prun_xtc   : true|false (the fallback signal for legacy systems)
  sim_ns         : production length in ns; see sim_ns_source for derivation
  sim_ns_source  : where sim_ns came from, in priority order:
                   - "actual": the authoritative "Statistics over N steps"
                     block at the end of prun.log (mdrun only writes this
                     on clean finish — best signal that the run completed)
                   - "requested_manifest": manifest mdp_params.nsteps_prod
                     (setup value, not produced length)
                   - "requested_log": prun.log MDP echo "nsteps = N"
                     (also setup value)
                   - "" (empty): no signal — sim_ns is unknown

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

import json
import re

from lipid_gnn.martini_pipeline.analysis import summarise_systems
from lipid_gnn.martini_pipeline.composition import parse_name

_DT_PS = 0.02  # Martini 3 production timestep
_LOG_NSTEPS_RE = re.compile(r"^\s*nsteps\s*=\s*(\d+)", re.MULTILINE)
_LOG_DT_RE = re.compile(r"^\s*dt\s*=\s*([\d.eE+-]+)", re.MULTILINE)
# Written by mdrun ONLY on a clean finish.  Multiple lines exist for
# restarted runs (mdrun -append); use the LAST one — it reflects the final
# cumulative step count.
_LOG_STATS_RE = re.compile(
    r"^\s*Statistics over\s+(\d+)\s+steps using\s+\d+\s+frames",
    re.MULTILINE,
)


def _sim_ns_from_log_stats(sim_dir: Path) -> float | None:
    """Authoritative: produced length from prun.log's 'Statistics over N steps'.

    Only present when the run finished cleanly (mdrun writes this in the
    Averages block at the very end).  We pair the step count with the dt
    parsed from the MDP echo at the top of the same log so we don't depend
    on the hardcoded _DT_PS for non-M3 timestep schemes.
    """
    log_path = sim_dir / "run" / "prun.log"
    if not log_path.is_file():
        return None
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    matches = _LOG_STATS_RE.findall(text)
    if not matches:
        return None
    nsteps = int(matches[-1])  # last block wins on -append restarts
    m_dt = _LOG_DT_RE.search(text)
    dt_ps = float(m_dt.group(1)) if m_dt else _DT_PS
    return nsteps * dt_ps / 1000.0


def _sim_ns_from_manifest(sim_dir: Path) -> float | None:
    """Manifest-requested length (mdp_params.nsteps_prod × dt).

    Reflects the setup, not the produced trajectory — a crashed run still
    reports the full requested length.  Used only when the authoritative
    'Statistics over' line is absent.
    """
    manifest_path = sim_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        data = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    mdp = data.get("mdp_params") or {}
    nsteps_prod = mdp.get("nsteps_prod")
    if not isinstance(nsteps_prod, int) or nsteps_prod <= 0:
        return None
    return nsteps_prod * _DT_PS / 1000.0


def _sim_ns_from_log_echo(sim_dir: Path) -> float | None:
    """MDP-echo-requested length: parse 'nsteps = N' from the top of prun.log.

    Same setup-vs-produced caveat as the manifest tier.  Final fallback for
    legacy data with no manifest.
    """
    log_path = sim_dir / "run" / "prun.log"
    if not log_path.is_file():
        return None
    try:
        with log_path.open("r", errors="replace") as fh:
            text = fh.read(64 * 1024)  # MDP echo always at the very top
    except OSError:
        return None
    m_nsteps = _LOG_NSTEPS_RE.search(text)
    if not m_nsteps:
        return None
    nsteps = int(m_nsteps.group(1))
    m_dt = _LOG_DT_RE.search(text)
    dt_ps = float(m_dt.group(1)) if m_dt else _DT_PS
    return nsteps * dt_ps / 1000.0


def _resolve_sim_ns(source_dir_path: Path) -> tuple[float | None, str]:
    """Return (ns, source) where source indicates how the value was derived.

    source ∈ {'actual', 'requested_manifest', 'requested_log', ''}.
    'actual' means the run actually produced that much trajectory (Tier 1).
    The two 'requested_*' tiers mean the run was set up for that length but
    completion isn't confirmed.  Empty string means no signal at all.
    """
    ns = _sim_ns_from_log_stats(source_dir_path)
    if ns is not None:
        return ns, "actual"
    ns = _sim_ns_from_manifest(source_dir_path)
    if ns is not None:
        return ns, "requested_manifest"
    ns = _sim_ns_from_log_echo(source_dir_path)
    if ns is not None:
        return ns, "requested_log"
    return None, ""


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

        sim_ns, sim_ns_source = _resolve_sim_ns(root / source_dir)
        rows.append({
            "canonical_name": canonical,
            "source_dir":     source_dir,
            "source_root":    str(root),
            "status":         row_status,
            "has_prun_xtc":   "true" if status.has_prun_xtc else "false",
            "sim_ns":         f"{sim_ns:.3f}" if sim_ns is not None else "",
            "sim_ns_source":  sim_ns_source,
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
    parser.add_argument(
        "--min-ns", type=float, default=None, metavar="NS",
        help="Drop rows whose production length is below this (in ns).  "
             "Rows with an unknown sim_ns (no manifest and no parseable "
             "prun.log) are also dropped when this filter is active — they "
             "cannot be proven to meet the cutoff.  Use 1000 to gate on "
             "the standard 1 µs production budget.",
    )
    parser.add_argument(
        "--require-actual", action="store_true",
        help="Only accept rows whose sim_ns came from the authoritative "
             "'Statistics over N steps' block (sim_ns_source=actual).  "
             "Without this flag, rows whose length is only the *requested* "
             "setup value (sim_ns_source=requested_manifest or "
             "requested_log) are also accepted.  Use this when you want to "
             "be sure a run actually finished, not just that it was set up "
             "for the right length.",
    )
    parser.add_argument(
        "--merge-with", default=None, metavar="CSV",
        help="Union the scan with rows from an existing CSV.  Useful when "
             "rescanning on HPC where the legacy data lives only on the "
             "local machine — pass the previous done.csv here and the legacy "
             "rows survive even though they aren't on disk at any scanned "
             "root.  Freshly scanned rows take precedence on canonical-name "
             "collisions (status may have changed since the merge CSV was "
             "written).  Status filter, if given, applies to merged rows too.",
    )
    args = parser.parse_args()

    roots = [Path(r) for r in args.output_roots] if args.output_roots else _default_output_roots()
    if not roots:
        print("ERROR: no output roots configured and none given via --output-roots", file=sys.stderr)
        return 1

    def _passes_min_ns(row: dict) -> bool:
        # --require-actual gates first: drop anything that isn't the
        # authoritative finish-line signal.  Empty source is treated as
        # "unknown" and dropped here, which is also what --min-ns would do.
        if args.require_actual and row.get("sim_ns_source") != "actual":
            return False
        if args.min_ns is None:
            return True
        try:
            ns = float(row.get("sim_ns") or "")
        except ValueError:
            return False  # unknown → can't prove it meets the cutoff
        return ns >= args.min_ns

    # First, do the fresh scan into an indexable dict (canonical_name → row).
    # Track scan order separately so net-new rows can be tail-appended in the
    # order they were discovered.
    scanned: dict[str, dict] = {}
    scan_order: list[str] = []
    for root in roots:
        if not root.is_dir():
            print(f"  INFO: skipping non-existent root {root}", file=sys.stderr)
            continue
        for row in scan_root(root):
            name = row["canonical_name"]
            if name in scanned:
                continue
            if args.status_filter and row["status"] not in args.status_filter:
                continue
            if not _passes_min_ns(row):
                continue
            scanned[name] = row
            scan_order.append(name)

    # Compose the output.  Without --merge-with, just take the fresh scan in
    # discovery order.  With --merge-with, preserve the existing CSV's row
    # order as the base (so a diff against the previous file isolates the new
    # additions cleanly): walk the merge CSV, overlay any fresh-scan row
    # in place when the canonical name matches, then tail-append truly net-new
    # fresh-scan rows.  The status_filter still applies to merge-only rows.
    all_rows: list[dict] = []
    merged_in = 0
    overlayed = 0
    if args.merge_with:
        merge_path = Path(args.merge_with)
        if not merge_path.is_file():
            print(f"ERROR: --merge-with {merge_path} does not exist", file=sys.stderr)
            return 1
        with open(merge_path, newline="") as fh:
            reader = csv.DictReader(fh)
            # sim_ns is a newer column; older CSVs may lack it.  We don't
            # require it here so legacy done.csv files still load (rows
            # without sim_ns are treated as "unknown" by --min-ns).
            required = {"canonical_name", "source_dir", "source_root", "status", "has_prun_xtc"}
            missing = required - set(reader.fieldnames or [])
            if missing:
                print(f"ERROR: --merge-with {merge_path} missing columns: "
                      f"{sorted(missing)}", file=sys.stderr)
                return 1
            seen_in_merge: set[str] = set()
            for row in reader:
                name = (row.get("canonical_name") or "").strip()
                if not name or name in seen_in_merge:
                    continue
                seen_in_merge.add(name)
                if name in scanned:
                    # Fresh scan supersedes the merged row (status may have
                    # changed since the CSV was written) but the *position*
                    # stays where the old CSV had it.
                    all_rows.append(scanned[name])
                    overlayed += 1
                else:
                    # Merge-only row: keep it as-is, subject to status_filter
                    # and min-ns gate.  sim_ns may be absent in CSVs written
                    # by older versions of this script — treat that as
                    # unknown for the min-ns check.
                    if args.status_filter and row["status"] not in args.status_filter:
                        continue
                    merge_row = {k: row.get(k, "") for k in [
                        "canonical_name", "source_dir", "source_root",
                        "status", "has_prun_xtc", "sim_ns", "sim_ns_source",
                    ]}
                    if not _passes_min_ns(merge_row):
                        continue
                    all_rows.append(merge_row)
                    merged_in += 1
        # Tail-append fresh-scan rows that weren't in the merge CSV — these
        # are the new additions that will appear at the bottom of a diff.
        for name in scan_order:
            if name not in seen_in_merge:
                all_rows.append(scanned[name])
    else:
        for name in scan_order:
            all_rows.append(scanned[name])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "canonical_name", "source_dir", "source_root",
            "status", "has_prun_xtc", "sim_ns", "sim_ns_source",
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    by_status: dict[str, int] = {}
    for row in all_rows:
        by_status[row["status"]] = by_status.get(row["status"], 0) + 1
    if args.merge_with:
        new_rows = len(all_rows) - merged_in - overlayed
        merge_note = (f" (overlayed {overlayed} + kept {merged_in} from "
                      f"{args.merge_with} + {new_rows} new)")
    else:
        merge_note = ""
    print(f"Wrote {len(all_rows)} rows to {out_path}{merge_note}")
    for s, n in sorted(by_status.items()):
        print(f"  {s}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
