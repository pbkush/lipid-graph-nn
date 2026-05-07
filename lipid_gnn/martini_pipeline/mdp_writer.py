"""MDP file writer for the Martini 3 lipid simulation pipeline.

Reads the audit freeze record (templates/_audit_freeze.json) and renders three
MDP files per simulation: em.mdp (minimisation), eq.mdp (equilibration),
run.mdp (production). Templates live under lipid_gnn/martini_pipeline/templates/.
"""
from __future__ import annotations

import json
import os
import random
import string
from dataclasses import dataclass
from typing import Mapping

_PACKAGE_DIR = os.path.dirname(__file__)
_DEFAULT_FREEZE = os.path.join(_PACKAGE_DIR, "templates", "_audit_freeze.json")
_DEFAULT_TEMPLATES = os.path.join(_PACKAGE_DIR, "templates")

_TEMPLATE_NAMES: dict[str, str] = {
    "minimization": "em.mdp.tmpl",
    "equilibration": "eq.mdp.tmpl",
    "run": "run.mdp.tmpl",
}

_OUTPUT_NAMES: dict[str, str] = {
    "minimization": "em.mdp",
    "equilibration": "eq.mdp",
    "run": "run.mdp",
}

STAGES: tuple[str, ...] = ("minimization", "equilibration", "run")


@dataclass(frozen=True)
class MDPParams:
    nsteps_min: int = 20_000
    nsteps_eq: int = 1_000_000
    nsteps_prod: int = -1
    nstenergy_eq: int = 1_000
    save_forces: bool = False
    gen_seed: int | None = None


def _resolve_seed(gen_seed: int | None) -> int:
    if gen_seed is None:
        return random.SystemRandom().randint(1, 2**31 - 1)
    return gen_seed


def _nstfout(save_forces: bool, run_canonical: Mapping[str, str]) -> int:
    if not save_forces:
        return 0
    return int(run_canonical.get("nstxout-compressed", "75000"))


def _load_freeze(freeze_path: str | os.PathLike) -> dict[str, dict[str, str]]:
    path = str(freeze_path)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"MDP audit freeze record missing: {path}\n"
            "Run scripts/simulation/audit_mdps.py first."
        )
    with open(path) as fh:
        return json.load(fh)


def _build_substitutions(params: MDPParams, seed: int, nstfout: int) -> dict[str, str]:
    return {
        "nsteps_min": str(params.nsteps_min),
        "nsteps_eq": str(params.nsteps_eq),
        "nsteps_prod": str(params.nsteps_prod),
        "nstenergy_eq": str(params.nstenergy_eq),
        "gen_seed": str(seed),
        "nstfout": str(nstfout),
    }


def render_mdp(
    stage: str,
    params: MDPParams,
    canonical: Mapping[str, str],
    template_text: str,
) -> str:
    """Render one MDP template to a string.

    *canonical* should be the freeze record for *stage* (used to derive
    nstfout when save_forces=True for the run stage).  Extra keys in the
    substitution mapping that have no matching placeholder are silently ignored.
    Missing placeholders raise KeyError (strict Template.substitute behaviour).
    """
    seed = _resolve_seed(params.gen_seed)
    nstfout = _nstfout(params.save_forces, canonical)
    subs = _build_substitutions(params, seed, nstfout)
    return string.Template(template_text).substitute(subs)


def write_mdps(
    out_dir: str | os.PathLike,
    *,
    params: MDPParams = MDPParams(),
    freeze_path: str | os.PathLike = _DEFAULT_FREEZE,
    templates_dir: str | os.PathLike = _DEFAULT_TEMPLATES,
) -> dict[str, str]:
    """Write em.mdp, eq.mdp, run.mdp into *out_dir*.

    Returns a dict mapping stage name → written file path.
    A single seed is drawn once per call and shared across all three stages so
    the run dir is internally consistent.  Pass ``params.gen_seed`` explicitly
    for reproducibility.
    """
    freeze = _load_freeze(freeze_path)
    os.makedirs(str(out_dir), exist_ok=True)

    seed = _resolve_seed(params.gen_seed)
    run_canonical = freeze.get("run", {})
    nstfout = _nstfout(params.save_forces, run_canonical)
    subs = _build_substitutions(params, seed, nstfout)

    written: dict[str, str] = {}
    for stage in STAGES:
        tmpl_path = os.path.join(str(templates_dir), _TEMPLATE_NAMES[stage])
        with open(tmpl_path) as fh:
            template_text = fh.read()
        content = string.Template(template_text).substitute(subs)
        out_path = os.path.join(str(out_dir), _OUTPUT_NAMES[stage])
        with open(out_path, "w") as fh:
            fh.write(content)
        written[stage] = out_path

    return written
