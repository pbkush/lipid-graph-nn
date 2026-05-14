#!/usr/bin/env python3
"""CLI driver: build + run a Martini 3 bilayer for one composition.

Usage examples:
    python scripts/simulation/run_martini_pipeline.py POPC:1.0 --nsteps 5000
    python scripts/simulation/run_martini_pipeline.py POPC:0.7 DOPC:0.3 --prod-ns 50
    python scripts/simulation/run_martini_pipeline.py DIPC:1.0 --prod-ns 0.05 \\
        --out-dir data/martini_pipeline --force-rerun

Composition is given as insane-style ratio strings: LIPID:fraction pairs.
Fractions are normalised so they sum to 1.0 before being passed to the pipeline.
"""
from __future__ import annotations

import argparse
import sys
import os

# Resolve repo root so the script works when invoked directly without install.
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

_DT_PS = 0.02  # production MDP dt (Martini 3); 1 ns = 50 000 steps


def parse_composition(ratio_args: list[str]) -> dict[str, float]:
    """Parse composition tokens into a {lipid: fraction} dict.

    Accepts two forms (mix-and-match across tokens is rejected):

    1. ``LIPID:fraction`` insane-style ratios, e.g. ``POPC:0.7 DOPC:0.3``.
       Fractions are normalised to sum to 1.0.
    2. A single canonical composition name (one token, no colon),
       e.g. ``POPC100`` or ``DOPC70_POPC30``.  Parsed via composition.parse_name.
    """
    if len(ratio_args) == 1 and ":" not in ratio_args[0]:
        # Canonical-name form: a single token with no colon.
        from lipid_gnn.martini_pipeline.composition import parse_name
        try:
            comp = parse_name(ratio_args[0])
        except ValueError as exc:
            raise SystemExit(f"Bad composition name {ratio_args[0]!r}: {exc}")
        return dict(comp.fractions)

    raw: dict[str, float] = {}
    for token in ratio_args:
        if ":" not in token:
            raise SystemExit(
                f"Bad composition token {token!r}. Expected LIPID:fraction "
                "(e.g. POPC:1.0) or a single canonical name (e.g. POPC100)."
            )
        name, val = token.split(":", 1)
        try:
            raw[name.upper()] = float(val)
        except ValueError:
            raise SystemExit(f"Bad fraction in {token!r}: {val!r} is not a number")
    if not raw:
        raise SystemExit("No composition tokens provided")
    total = sum(raw.values())
    if total <= 0:
        raise SystemExit("Composition fractions must sum to a positive number")
    return {k: v / total for k, v in raw.items()}


def main() -> None:
    from lipid_gnn.config import CONFIG
    cfg = CONFIG.martini_pipeline

    parser = argparse.ArgumentParser(
        description="Build and simulate a Martini 3 lipid bilayer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "ratios",
        nargs="+",
        metavar="LIPID:fraction",
        help="Composition as insane-style ratio strings, e.g. POPC:1.0 or POPC:0.7 DOPC:0.3",
    )

    prod_grp = parser.add_mutually_exclusive_group(required=True)
    prod_grp.add_argument(
        "--prod-ns",
        type=float,
        metavar="NS",
        help="Production run length in nanoseconds",
    )
    prod_grp.add_argument(
        "--nsteps",
        type=int,
        metavar="N",
        help="Production run length in steps (alternative to --prod-ns)",
    )

    parser.add_argument(
        "--out-dir",
        default=str(cfg.output_root) if cfg else "data/martini_pipeline",
        help="Root output directory; composition subdir is created inside",
    )
    parser.add_argument(
        "--gmx",
        default=cfg.gmx.executable if cfg else "gmx",
        help="GROMACS executable",
    )
    parser.add_argument(
        "--maxwarn",
        type=int,
        default=cfg.gmx.maxwarn if cfg else 2,
        help="grompp -maxwarn value",
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="RNG seed (integer, or 'random' for a fresh random seed)",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Re-run all stages even if handoff files already exist",
    )
    parser.add_argument(
        "--save-forces",
        action="store_true",
        help="Write forces to production trajectory (larger .trr)",
    )
    parser.add_argument(
        "--nsteps-eq",
        type=int,
        default=cfg.run.nsteps_eq if cfg else 1_000_000,
        help="Equilibration steps",
    )
    parser.add_argument(
        "--nsteps-min",
        type=int,
        default=cfg.run.nsteps_min if cfg else 20_000,
        help="Minimisation steps",
    )
    parser.add_argument(
        "--mdrun-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra arguments passed verbatim to gmx mdrun (place after --)",
    )

    args = parser.parse_args()

    # Resolve composition
    composition = parse_composition(args.ratios)

    # Resolve nsteps_prod
    if args.prod_ns is not None:
        nsteps_prod = round(args.prod_ns * 1000.0 / _DT_PS)  # ns → ps → steps
    else:
        nsteps_prod = args.nsteps

    # Resolve seed
    seed: int | None = None
    if args.seed is not None:
        if args.seed == "random":
            import random
            seed = random.SystemRandom().randint(1, 2**31 - 1)
        else:
            try:
                seed = int(args.seed)
            except ValueError:
                raise SystemExit(f"--seed must be an integer or 'random', got {args.seed!r}")

    # Build canonical composition name for output subdir
    from lipid_gnn.martini_pipeline.composition import Composition
    comp = Composition(composition)
    out_dir = os.path.join(args.out_dir, comp.name)

    from lipid_gnn.martini_pipeline.mdp_writer import MDPParams
    from lipid_gnn.martini_pipeline.pipeline import run as pipeline_run
    from lipid_gnn.martini_pipeline.system_builder import BoxParams

    box = BoxParams(
        xy_nm=cfg.box.xy_nm if cfg else 11.0,
        z_nm=cfg.box.z_nm if cfg else 10.0,
        salt_M=cfg.box.salt_M if cfg else 0.15,
        water_type=cfg.box.water_type if cfg else "W",
        charge_mode=cfg.box.charge_mode if cfg else "auto",
        center=cfg.box.center if cfg else True,
        pbc=cfg.box.pbc if cfg else "rectangular",
    )
    mdp_params = MDPParams(
        nsteps_min=args.nsteps_min,
        nsteps_eq=args.nsteps_eq,
        nsteps_prod=nsteps_prod,
        save_forces=args.save_forces,
    )

    insane_cmd = cfg.insane_cmd if cfg else "insane"
    itp_dir = str(cfg.itp_dir) if cfg else "resources/martini3/itp"

    print(f"Composition: {comp.name}")
    print(f"Output:      {out_dir}")
    print(f"nsteps_prod: {nsteps_prod}")

    result = pipeline_run(
        composition,
        out_dir,
        box=box,
        mdp_params=mdp_params,
        seed=seed,
        gmx_executable=args.gmx,
        mdrun_extra_args=tuple(args.mdrun_args),
        force_rerun=args.force_rerun,
        maxwarn=args.maxwarn,
        insane_cmd=insane_cmd,
        itp_dir=itp_dir,
    )

    print(f"Status:      {result.overall_status}")
    print(f"Manifest:    {result.manifest_path}")


if __name__ == "__main__":
    main()
