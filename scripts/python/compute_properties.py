"""CLI for computing bilayer properties from production trajectories.

Replaces the legacy ``scripts/emil/general/calculate_properties.ipynb``
workflow with the new :mod:`lipid_gnn.properties` module.

Default invocation walks ``CONFIG.paths.data_dir`` looking for
``<composition>/run/prun.{xtc,gro}`` pairs and writes
``<out_dir>/<composition>.h5`` pickles whose payload is
``(mean_dict, raw_dict)``, matching the schema consumed by
``lipid_gnn.dataset`` and the comparison notebooks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mdtraj as md

from lipid_gnn.config import CONFIG
from lipid_gnn.io import pkl_save
from lipid_gnn.properties import ALL_PROPERTIES, compute_all


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Compute bilayer properties for one or more compositions. "
            "Writes one <comp>.h5 pickle per composition."
        ),
    )
    p.add_argument(
        "--data-dir", type=Path, default=Path(CONFIG.paths.data_dir),
        help="Root directory containing <composition>/run/prun.{xtc,gro}.",
    )
    p.add_argument(
        "--out-dir", type=Path, default=Path("results/properties"),
        help=(
            "Output directory for <comp>.h5 files. Default 'results/properties/' "
            "keeps the legacy results/properties/ untouched per the cleanup plan."
        ),
    )
    p.add_argument(
        "--composition", nargs="+", default=None,
        help="Composition directory names. Default: every <comp> in --data-dir with prun.{xtc,gro}.",
    )
    p.add_argument(
        "--properties", nargs="+", default=list(ALL_PROPERTIES),
        choices=list(ALL_PROPERTIES) + ["compressibility"],
        help=(f"Which properties to compute. Available: {', '.join(ALL_PROPERTIES)}. "
              "'compressibility' is accepted as an alias for 'thickness_inhomogeneity'."),
    )
    p.add_argument("--frame-start", type=int, default=50,
                   help="First frame to keep (default 50 = drop equilibration).")
    p.add_argument("--frame-stop", type=int, default=667,
                   help="Last frame to keep (default 667 ≈ 1 µs at dt=1.5 ns).")
    p.add_argument("--lag-persistence", type=int, default=50)
    p.add_argument("--lag-diffusivity", type=int, default=10)
    p.add_argument("--probe-size", type=int, default=10,
                   help="Random lipids sampled per frame for persistence/diffusivity.")
    p.add_argument("--seed", type=int, default=0,
                   help="RNG seed. Stochastic properties become deterministic.")
    p.add_argument("--legacy", action="store_true",
                   help="Reproduce historical bugs (see cleanup-plan §2). "
                        "Use only for re-deriving original labels.")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing <comp>.h5 files. Default: skip.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def _discover(data_dir: Path) -> list[str]:
    out = []
    for sub in sorted(data_dir.iterdir()):
        if not sub.is_dir():
            continue
        run = sub / "run"
        if (run / "prun.xtc").exists() and (run / "prun.gro").exists():
            out.append(sub.name)
    return out


def _load_trajectory(comp_dir: Path, frame_start: int, frame_stop: int
                     ) -> tuple[md.Trajectory, tuple[float, float]]:
    xtc = comp_dir / "run" / "prun.xtc"
    gro = comp_dir / "run" / "prun.gro"
    traj = md.load(str(xtc), top=str(gro))[frame_start:frame_stop]
    # box XY parsed from the last line of the .gro file (per properties.md)
    last = gro.read_text().rstrip().splitlines()[-1].split()
    box_xy = (float(last[0]), float(last[1])) if len(last) >= 2 else None
    if box_xy is None:
        box_xy = (
            float(traj.unitcell_lengths[:, 0].mean()),
            float(traj.unitcell_lengths[:, 1].mean()),
        )
    return traj, box_xy


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.composition is None:
        comps = _discover(args.data_dir)
        if not args.quiet:
            print(f"Auto-discovered {len(comps)} compositions under {args.data_dir}")
    else:
        comps = args.composition

    failures = []
    for comp in comps:
        out_h5 = args.out_dir / f"{comp}.h5"
        if out_h5.exists() and not args.overwrite:
            if not args.quiet:
                print(f"[skip] {comp} (exists; pass --overwrite to redo)")
            continue
        comp_dir = args.data_dir / comp
        if not (comp_dir / "run" / "prun.xtc").exists():
            print(f"[miss] {comp}: prun.xtc not found", file=sys.stderr)
            failures.append(comp)
            continue
        if not args.quiet:
            mode = "legacy" if args.legacy else "bugfixed"
            print(f"[run]  {comp} ({mode}, seed={args.seed})")
        t0 = time.time()
        try:
            traj, box_xy = _load_trajectory(comp_dir, args.frame_start, args.frame_stop)
            mean_dict, raw_dict = compute_all(
                traj,
                box_xy=box_xy,
                lag_persistence=args.lag_persistence,
                lag_diffusivity=args.lag_diffusivity,
                probe_size=args.probe_size,
                properties=args.properties,
                seed=args.seed,
                legacy=args.legacy,
            )
            pkl_save(out_h5, (mean_dict, raw_dict))
            if not args.quiet:
                print(f"       → {out_h5} ({time.time() - t0:.1f}s)")
        except Exception as exc:
            print(f"[fail] {comp}: {exc}", file=sys.stderr)
            failures.append(comp)
    if failures:
        print(f"\n{len(failures)} failure(s): {failures}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
