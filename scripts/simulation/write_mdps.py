#!/usr/bin/env python3
"""Write Martini 3 MDP files (em.mdp, eq.mdp, run.mdp) from the audit-freeze templates.

Usage:
    python scripts/simulation/write_mdps.py --out-dir /tmp/mdptest
    python scripts/simulation/write_mdps.py --out-dir /tmp/mdptest \\
        --nsteps-min 20000 --nsteps-eq 1000000 --nsteps-prod -1 \\
        --save-forces --gen-seed 42
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from lipid_gnn.martini_pipeline.mdp_writer import MDPParams, write_mdps


def main() -> int:
    parser = argparse.ArgumentParser(description="Write Martini 3 MDP files from audit templates.")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write em.mdp, eq.mdp, run.mdp into.")
    parser.add_argument("--nsteps-min", type=int, default=20_000,
                        help="EM steps (default: 20000).")
    parser.add_argument("--nsteps-eq", type=int, default=1_000_000,
                        help="Equilibration steps (default: 1000000 = 10 ns at dt=0.01).")
    parser.add_argument("--nsteps-prod", type=int, default=-1,
                        help="Production steps; -1 = run until walltime (default: -1).")
    parser.add_argument("--nstenergy-eq", type=int, default=1_000,
                        help="nstenergy for equilibration (default: 1000).")
    parser.add_argument("--save-forces", action="store_true",
                        help="Set nstfout = nstxout-compressed in run.mdp (default: off).")
    parser.add_argument("--gen-seed", type=int, default=None,
                        help="Fixed RNG seed for gen_seed lines. Default: random per call.")
    args = parser.parse_args()

    params = MDPParams(
        nsteps_min=args.nsteps_min,
        nsteps_eq=args.nsteps_eq,
        nsteps_prod=args.nsteps_prod,
        nstenergy_eq=args.nstenergy_eq,
        save_forces=args.save_forces,
        gen_seed=args.gen_seed,
    )

    try:
        written = write_mdps(args.out_dir, params=params)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    for stage, path in written.items():
        print(f"  {stage:15s}: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
