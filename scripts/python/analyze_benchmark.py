#!/usr/bin/env python3
"""analyze_benchmark.py — parse HPC benchmark results and recommend hpc_defaults.

Reads per-point directories written by sbatch_benchmark_hpc.sh, parses
gmx mdrun Performance lines, joins rocm-smi GPU telemetry, and ranks points by
aggregate ns/day per node-hour.  With --recommend, prints a config.yaml
hpc_defaults block for the top-scoring point that fits within the memory budget.

Usage
-----
    python scripts/python/analyze_benchmark.py \\
        [--root PATH]              default: results/benchmarks/martini_pipeline/latest
        [--format {csv,json,md,all}]   default: all
        [--mem-headroom-frac F]    fraction of node RAM allowed  (default: 0.70)
        [--node-mem GB]            node RAM in GB               (default: 256)
        [--recommend]              print recommended config.yaml block
        [--allow-partial]          include incomplete points in recommendation
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHMARKS_DIR = REPO_ROOT / "results" / "benchmarks" / "martini_pipeline"


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class PointResult:
    label: str
    sims_per_node: int
    gpus_per_node: int
    cpus_per_sim: int
    mem_per_sim: str
    partition: str
    slot_ns_per_day: list = field(default_factory=list)
    slot_wall_t_s: list = field(default_factory=list)
    aggregate_ns_per_day: float = 0.0
    max_wall_t_s: float = 0.0
    node_hours: float = float("nan")
    score: float = float("nan")
    gpu_util_mean: float = float("nan")
    gpu_power_W_mean: float = float("nan")
    vram_used_MB_max: float = float("nan")
    status: str = "failed"
    n_slots_ok: int = 0
    n_slots_total: int = 0


# ── log parsing ───────────────────────────────────────────────────────────────

def parse_perf(log_path: Path) -> tuple[Optional[float], Optional[float]]:
    """Return (ns_per_day, wall_t_s) from a gmx mdrun log, or (None, None).

    GROMACS log format (relevant lines):
        ...
                   Core t (s)   Wall t (s)        (%)
        Time:       12000.0      1500.0        800.0

        Performance:       57.6       0.417
                     ns/day   hours/ns
    """
    if not log_path.exists():
        return None, None
    text = log_path.read_text(errors="replace")

    # Performance: <ns_per_day>  <hours_per_ns>   (two floats on one line)
    m_perf = re.search(r"^Performance:\s+([\d.]+)\s+[\d.]+", text, re.MULTILINE)
    ns_per_day = float(m_perf.group(1)) if m_perf else None

    # Time: <core_t>  <wall_t>  <pct>
    m_time = re.search(r"^Time:\s+[\d.]+\s+([\d.]+)", text, re.MULTILINE)
    wall_t_s = float(m_time.group(1)) if m_time else None

    return ns_per_day, wall_t_s


# ── rocm-smi parsing ──────────────────────────────────────────────────────────

def parse_rocm_smi(tsv_path: Path) -> tuple[float, float, float]:
    """Return (gpu_util_mean_pct, power_W_mean, vram_used_MB_max).  NaN on failure."""
    nan = float("nan")
    if not tsv_path.exists():
        return nan, nan, nan
    try:
        df = pd.read_csv(tsv_path, sep="\t", comment="#")
        df.columns = [c.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for c in df.columns]
        # Accept various rocm-smi CSV column spellings
        util_col = next((c for c in df.columns if "util" in c or "use" in c), None)
        power_col = next((c for c in df.columns if "power" in c or "pwr" in c), None)
        vram_col = next((c for c in df.columns if "vram" in c), None)
        return (
            float(df[util_col].mean())  if util_col  else nan,
            float(df[power_col].mean()) if power_col else nan,
            float(df[vram_col].max())   if vram_col  else nan,
        )
    except Exception:
        return nan, nan, nan


# ── point loader ──────────────────────────────────────────────────────────────

def load_point(point_dir: Path) -> PointResult:
    meta = json.loads((point_dir / "point_meta.json").read_text())
    p = PointResult(
        label=meta["label"],
        sims_per_node=meta["sims_per_node"],
        gpus_per_node=meta["gpus_per_node"],
        cpus_per_sim=meta["cpus_per_sim"],
        mem_per_sim=meta["mem_per_sim"],
        partition=meta["partition"],
        n_slots_total=meta["sims_per_node"],
    )

    slot_ns: list[float] = []
    slot_wall: list[float] = []
    for slot_dir in sorted(point_dir.glob("slot_*")):
        ns, wall = parse_perf(slot_dir / "bench.log")
        if ns is not None:
            slot_ns.append(ns)
        if wall is not None:
            slot_wall.append(wall)

    p.slot_ns_per_day = slot_ns
    p.slot_wall_t_s = slot_wall
    p.n_slots_ok = len(slot_ns)
    p.status = (
        "ok"         if p.n_slots_ok == p.n_slots_total else
        "incomplete" if p.n_slots_ok > 0                else
        "failed"
    )

    if slot_ns:
        p.aggregate_ns_per_day = sum(slot_ns)
    if slot_wall:
        p.max_wall_t_s = max(slot_wall)
        p.node_hours = p.max_wall_t_s / 3600.0
        if p.node_hours > 0:
            p.score = p.aggregate_ns_per_day / p.node_hours

    p.gpu_util_mean, p.gpu_power_W_mean, p.vram_used_MB_max = \
        parse_rocm_smi(point_dir / "rocm-smi.tsv")

    return p


def load_all_points(root: Path) -> list[PointResult]:
    points_dir = root / "points"
    if not points_dir.exists():
        return []
    return [
        load_point(d)
        for d in sorted(points_dir.iterdir())
        if d.is_dir() and (d / "point_meta.json").exists()
    ]


# ── recommendation ────────────────────────────────────────────────────────────

def _isnan(v: float) -> bool:
    return v != v  # NaN != NaN


def recommend(
    points: list[PointResult],
    node_mem_GB: float,
    headroom: float,
) -> Optional[PointResult]:
    """Return top-scoring point under memory headroom, or None."""
    candidates = [
        p for p in points
        if p.status == "ok"
        and not _isnan(p.score)
        and float(p.mem_per_sim.rstrip("G")) * p.sims_per_node <= headroom * node_mem_GB
    ]
    if not candidates:
        return None
    # rank: score desc; tie-break gpus_per_node asc, then sims_per_node asc
    candidates.sort(key=lambda p: (-p.score, p.gpus_per_node, p.sims_per_node))
    return candidates[0]


# ── output helpers ────────────────────────────────────────────────────────────

def to_dataframe(points: list[PointResult]) -> pd.DataFrame:
    def _r(v: float, n: int = 2) -> Optional[float]:
        return round(v, n) if not _isnan(v) else None

    rows = [
        {
            "label":                      p.label,
            "sims_per_node":              p.sims_per_node,
            "gpus_per_node":              p.gpus_per_node,
            "cpus_per_sim":               p.cpus_per_sim,
            "mem_per_sim":                p.mem_per_sim,
            "aggregate_ns_per_day":       _r(p.aggregate_ns_per_day, 2),
            "max_wall_t_s":               _r(p.max_wall_t_s, 1),
            "node_hours":                 _r(p.node_hours, 4),
            "score_ns_day_per_node_hour": _r(p.score, 1),
            "gpu_util_mean_%":            _r(p.gpu_util_mean, 1),
            "gpu_power_W_mean":           _r(p.gpu_power_W_mean, 1),
            "vram_used_MB_max":           _r(p.vram_used_MB_max, 0),
            "status":                     p.status,
            "slots_ok":                   f"{p.n_slots_ok}/{p.n_slots_total}",
        }
        for p in points
    ]
    return pd.DataFrame(rows)


def _rec_yaml(rec: PointResult) -> str:
    return (
        "  hpc_defaults:\n"
        f"    sims_per_node: {rec.sims_per_node}\n"
        f"    cpus_per_sim: {rec.cpus_per_sim}\n"
        f'    mem_per_sim: "{rec.mem_per_sim}"\n'
        f"    gpus_per_node: {rec.gpus_per_node}\n"
    )


def write_md(df: pd.DataFrame, rec: Optional[PointResult], root: Path) -> None:
    lines = ["# Benchmark summary\n\n", df.to_markdown(index=False), "\n\n"]
    if rec:
        lines += [
            "## Recommended `hpc_defaults`\n\n",
            "Paste into `martini_pipeline:` section of `config.yaml`:\n\n",
            "```yaml\n",
            _rec_yaml(rec),
            "```\n",
        ]
    (root / "summary.md").write_text("".join(lines))


def update_latest_symlink(root: Path) -> None:
    latest = root.parent / "latest"
    if latest.is_symlink():
        latest.unlink()
    try:
        latest.symlink_to(root.name)
    except OSError:
        pass


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Parse HPC benchmark results and recommend hpc_defaults."
    )
    parser.add_argument("--root", type=Path, default=None,
                        help="Benchmark results dir (default: latest dated dir under results/benchmarks/)")
    parser.add_argument("--format", choices=["csv", "json", "md", "all"], default="all")
    parser.add_argument("--mem-headroom-frac", type=float, default=0.70,
                        help="Max fraction of node RAM usable (default: 0.70)")
    parser.add_argument("--node-mem", type=float, default=256.0,
                        help="Node RAM in GB (default: 256; MI210 nodes on Goethe-HLR)")
    parser.add_argument("--recommend", action="store_true",
                        help="Print recommended config.yaml hpc_defaults block")
    parser.add_argument("--allow-partial", action="store_true",
                        help="Include incomplete points in ranking/recommendation")
    args = parser.parse_args()

    # resolve root
    root: Path
    if args.root is not None:
        root = args.root
    else:
        latest = BENCHMARKS_DIR / "latest"
        if latest.is_symlink():
            root = latest.resolve()
        else:
            dirs = sorted(
                (d for d in BENCHMARKS_DIR.iterdir() if d.is_dir() and d.name != "latest"),
                reverse=True,
            ) if BENCHMARKS_DIR.exists() else []
            if not dirs:
                print("ERROR: no benchmark root found; pass --root", file=sys.stderr)
                return 1
            root = dirs[0]

    points = load_all_points(root)
    if not points:
        print(f"ERROR: no benchmark points found under {root}/points/", file=sys.stderr)
        return 1

    incomplete = [p for p in points if p.status != "ok"]
    if incomplete and args.recommend and not args.allow_partial:
        labels = ", ".join(p.label for p in incomplete)
        print(
            f"ERROR: {len(incomplete)} incomplete point(s): {labels}\n"
            "Pass --allow-partial to recommend from partial results.",
            file=sys.stderr,
        )
        return 1

    eligible = points if args.allow_partial else [p for p in points if p.status == "ok"]
    rec = recommend(eligible, args.node_mem, args.mem_headroom_frac) if args.recommend else None

    df = to_dataframe(points)
    fmt = args.format

    if fmt in ("csv", "all"):
        df.to_csv(root / "summary.csv", index=False)
        print(f"Wrote {root / 'summary.csv'}")
    if fmt in ("json", "all"):
        (root / "summary.json").write_text(
            json.dumps([asdict(p) for p in points], indent=2, default=str)
        )
        print(f"Wrote {root / 'summary.json'}")
    if fmt in ("md", "all"):
        write_md(df, rec, root)
        print(f"Wrote {root / 'summary.md'}")

    update_latest_symlink(root)

    print()
    print(df.to_string(index=False))
    print()

    if args.recommend:
        if rec is None:
            print("ERROR: no qualifying point found (all failed or over memory cap)", file=sys.stderr)
            return 1
        print("Recommended hpc_defaults (paste into martini_pipeline: in config.yaml):")
        print(_rec_yaml(rec))

    return 0


if __name__ == "__main__":
    sys.exit(main())
