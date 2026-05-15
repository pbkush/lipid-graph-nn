#!/usr/bin/env python3
"""Scan running prun.log files and report projected finish times.

Estimates per-slot throughput from the last two "Writing checkpoint" lines
(or the first/last pair if only two exist), then projects when nsteps will
be reached.  GROMACS only writes a final "Performance:" line after the run
finishes, so checkpoint deltas are the most reliable mid-run signal.

Usage:
    python scripts/simulation/projected_finish.py [ROOT ...]
    python scripts/simulation/projected_finish.py --walltime 48

If no ROOT is given, falls back to martini_pipeline.output_root from config.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

_CKPT_RE = re.compile(
    r"Writing checkpoint, step\s+(\d+)\s+at\s+(.+?)\s*$"
)
_NSTEPS_RE = re.compile(r"^\s*nsteps\s*=\s*(\d+)", re.MULTILINE)
_DT_RE = re.compile(r"^\s*dt\s*=\s*([\d.eE+-]+)", re.MULTILINE)
# GROMACS prints date like "Fri May 15 16:59:25 2026"
_DATE_FMT = "%a %b %d %H:%M:%S %Y"


@dataclass
class SlotReport:
    name: str
    log_path: Path
    nsteps_target: int | None
    last_step: int | None
    steps_per_sec: float | None
    eta: datetime | None
    hours_remaining: float | None
    status: str  # "running", "finished", "stalled", "no-data"


def parse_checkpoints(log_path: Path) -> list[tuple[int, datetime]]:
    ckpts: list[tuple[int, datetime]] = []
    try:
        with log_path.open("r", errors="replace") as fh:
            for line in fh:
                m = _CKPT_RE.search(line)
                if m:
                    try:
                        step = int(m.group(1))
                        when = datetime.strptime(m.group(2).strip(), _DATE_FMT)
                        ckpts.append((step, when))
                    except ValueError:
                        continue
    except OSError:
        return []
    return ckpts


def parse_nsteps_target(log_path: Path) -> int | None:
    """Pull the production nsteps from the MDP header echoed in the .log."""
    try:
        text = log_path.read_text(errors="replace")
    except OSError:
        return None
    m = _NSTEPS_RE.search(text)
    if not m:
        return None
    return int(m.group(1))


def is_finished(log_path: Path) -> bool:
    """Cheap check: a finished run has a 'Performance:' line near the end."""
    try:
        with log_path.open("rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            tail = fh.read().decode(errors="replace")
    except OSError:
        return False
    return "Performance:" in tail or "Finished mdrun" in tail


def analyse_log(name: str, log_path: Path, stall_minutes: float) -> SlotReport:
    if is_finished(log_path):
        return SlotReport(name, log_path, None, None, None, None, None, "finished")

    ckpts = parse_checkpoints(log_path)
    nsteps = parse_nsteps_target(log_path)

    if len(ckpts) < 2:
        return SlotReport(name, log_path, nsteps, None, None, None, None, "no-data")

    first, last = ckpts[0], ckpts[-1]
    dstep = last[0] - first[0]
    dsec = (last[1] - first[1]).total_seconds()
    if dstep <= 0 or dsec <= 0:
        return SlotReport(name, log_path, nsteps, last[0], None, None, None, "no-data")

    rate = dstep / dsec  # steps/sec
    age_min = (datetime.now() - last[1]).total_seconds() / 60.0
    status = "stalled" if age_min > stall_minutes else "running"

    eta = None
    hours_remaining = None
    if nsteps is not None and nsteps > last[0]:
        remaining_sec = (nsteps - last[0]) / rate
        hours_remaining = remaining_sec / 3600.0
        eta = last[1] + timedelta(seconds=remaining_sec)

    return SlotReport(name, log_path, nsteps, last[0], rate, eta, hours_remaining, status)


def find_logs(roots: list[Path]) -> list[tuple[str, Path]]:
    """Locate prun.log files under each root.

    Handles both layouts seen in the wild:
      <root>/<comp>/prun.log
      <root>/<comp>/run/prun.log
    plus the degenerate case where `root` is itself the simulation dir.
    The display name is the first ancestor that isn't a generic "run" subdir.
    """
    out: list[tuple[str, Path]] = []
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        for log in sorted(root.rglob("prun.log")):
            if log in seen:
                continue
            seen.add(log)
            # Walk up past "run" wrappers to find a composition-shaped name.
            parent = log.parent
            name = parent.name
            if name == "run" and parent.parent != parent:
                name = parent.parent.name
            out.append((name, log))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path, help="Output roots to scan")
    ap.add_argument("--walltime", type=float, default=48.0,
                    help="SLURM walltime budget in hours (for warning flag)")
    ap.add_argument("--stall-minutes", type=float, default=20.0,
                    help="Flag run as 'stalled' if last checkpoint older than this")
    ap.add_argument("--only", choices=["running", "stalled", "finished", "no-data", "over"],
                    help="Filter rows by status (or 'over' = projected > walltime)")
    args = ap.parse_args()

    roots = args.roots
    if not roots:
        from lipid_gnn.config import CONFIG
        roots = [Path(CONFIG.martini_pipeline.output_root)]

    logs = find_logs(roots)
    if not logs:
        print(f"No prun.log files found under: {[str(r) for r in roots]}", file=sys.stderr)
        return 1

    reports = [analyse_log(name, p, args.stall_minutes) for name, p in logs]

    header = f"{'composition':<28} {'status':<9} {'step':>11} {'ns/day':>8} {'rem h':>7}  ETA"
    print(header)
    print("-" * len(header))

    over_budget = 0
    for r in reports:
        if args.only == "over":
            if r.hours_remaining is None or r.hours_remaining <= args.walltime:
                continue
        elif args.only and r.status != args.only:
            continue

        step_str = f"{r.last_step:,}" if r.last_step is not None else "-"
        if r.steps_per_sec is not None:
            ns_per_day = r.steps_per_sec * 0.02 / 1000.0 * 86400.0  # dt=0.02 ps
            nsd_str = f"{ns_per_day:6.1f}"
        else:
            nsd_str = "    -"
        rem_str = f"{r.hours_remaining:6.1f}" if r.hours_remaining is not None else "     -"
        eta_str = r.eta.strftime("%Y-%m-%d %H:%M") if r.eta else "-"

        flag = ""
        if r.hours_remaining is not None and r.hours_remaining > args.walltime:
            flag = "  !! OVER"
            over_budget += 1

        print(f"{r.name:<28} {r.status:<9} {step_str:>11} {nsd_str:>8} {rem_str:>7}  {eta_str}{flag}")

    print("-" * len(header))
    total = len(reports)
    running = sum(1 for r in reports if r.status == "running")
    finished = sum(1 for r in reports if r.status == "finished")
    stalled = sum(1 for r in reports if r.status == "stalled")
    nodata = sum(1 for r in reports if r.status == "no-data")
    print(f"  {total} slot(s): {running} running, {finished} finished, "
          f"{stalled} stalled, {nodata} no-data, {over_budget} over {args.walltime:g}h budget")

    return 0


if __name__ == "__main__":
    sys.exit(main())
