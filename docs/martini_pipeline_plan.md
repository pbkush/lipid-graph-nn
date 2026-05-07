# Martini 3 Lipid Simulation Pipeline — Plan & Tracker

Long-term, general-purpose Martini 3 membrane simulation pipeline. Stands as a research deliverable in its own right; newly simulated systems are not necessarily training data. This document is the single source of truth for the plan, progress, and decisions.

Last updated: 2026-05-07.

---

## 1. Goals

1. **Dynamic membrane creation pipeline** — parameterised in lipid types, per-lipid mol fractions, box size, water, ions, temperature, simulation length, force output, etc. Output layout matches the existing `data/membrane_only/<comp>/{equilibration, minimization, run, ...}` tree so downstream graph/dataset code keeps working unchanged when pointed at the new output root.
2. **Capable of simulating any Martini 3 lipid** — registry designed for extension beyond the current 10-lipid pool.
3. **Subgoal — fill the 10-lipid composition coverage** — DPPC- and DOPC-rich corners flagged by the Stage 5b per-system MAE concentration (POPC30_DOPC70 worst).
4. **Later — extend the lipid pool** — add Martini 3 lipids beyond the current 10, expanding `LIPID_TYPES` for future training rounds.

The pipeline targets HPC (Goethe MI210 / ROCm, GROMACS, SLURM) as the production environment; tests must pass locally and on HPC without a real `gmx`.

---

## 2. Where the code lives

- **Package**: `lipid_gnn/martini_pipeline/` — self-contained subpackage. Independence rule: no imports from `lipid_graph.py` / `dataset.py` / `membrane_prop_gnn.py` / training code, and those do not import from the pipeline. Use of MDAnalysis, numpy, etc. is fine where helpful.
- **Tests**: `tests/martini_pipeline/`. Must pass on local and HPC. `gmx` is mocked via a fake binary on `PATH`; the slow end-to-end test is opt-in via env var.
- **CLI drivers**: `scripts/simulation/` (Python scripts following the project's CLI pattern).
- **HPC sbatch wrappers**: `scripts/bash/submit_simulations.sh` and `scripts/bash/sbatch_simulations.sh`, mirroring the `submit_sweep.sh` / `sbatch_sweep.sh` split.
- **Output root**: configurable. Default `data/martini_pipeline/<comp>/` — keeps new and legacy simulations cleanly separated. Layout inside `<comp>/` matches the legacy tree exactly so downstream code is drop-in compatible.
- **Vendored resources**: `resources/martini3/insane.py` (see [Decision 2](#decision-log)).

---

## 3. Module breakdown — `lipid_gnn/martini_pipeline/`

| Module | Responsibility |
|---|---|
| `composition.py` | Validate composition spec `{lipid: mol_fraction}`; canonical naming (e.g. `POPC30_DOPC70`); leaflet count math. |
| `lipid_registry.py` | Registered Martini 3 lipids (data) + `register_lipid()`, `validate_lipid()`, `check_resources()` for adding new entries and verifying that all required `.itp` files / residue names / head-tail definitions / default APL are present. Mutation and validation live here, not in `analysis.py` ([Decision 5](#decision-log)). |
| `system_builder.py` | Wraps `insane.py` to build initial bilayer `.gro` + `topol.top` from a composition spec + box / water / ion params. |
| `mdp_writer.py` | Emit minimization / equilibration / production `.mdp` files from templates with parameter substitution. Templates under `lipid_gnn/martini_pipeline/templates/`. |
| `pipeline.py` | Orchestration: `build → minimize → equilibrate → produce`. Calls `gmx grompp` / `gmx mdrun` directly via `subprocess`. Idempotent (skips stages whose output exists). Writes manifest. |
| `manifest.py` | JSON manifest writer per system: spec, gmx version, mdp hashes, durations, status, seeds, insane args, vendored-insane source URL + version. |
| `analysis.py` | Multi-purpose utility module. Functions: `missing_compositions(target_grid, output_root)`, `summarise_systems(output_root)`, `diff_mdps(systems_root)`. Operates over outputs on disk, not over the registry. |

**Deliberately omitted**: `gromacs_runner.py`. `pipeline.py` calls `gmx` directly; the wrapper layer adds nothing while call sites are linear. Reintroduce only if retry/timeout/error-classification logic accumulates ([Decision 1](#decision-log)).

---

## 4. Config surface

Add a `martini_pipeline:` block to `config.yaml`, parsed in `lipid_gnn/config.py` as a frozen dataclass. CLI flags override per-run; CLI defaults source from `CONFIG`.

```yaml
martini_pipeline:
  output_root: data/martini_pipeline/
  gmx_binary: gmx
  mdp_templates_dir: lipid_gnn/martini_pipeline/templates/
  lipid_itp_dir: resources/martini3/itp/
  insane_path: resources/martini3/insane.py

  # System defaults
  box_xy_nm: ...           # from MDP audit
  water_layer_nm: ...
  salt_M: ...
  temperature_K: ...
  n_lipids_per_leaflet: ...

  # Run defaults
  dt_fs: ...
  nsteps_min: ...
  nsteps_equil: ...
  nsteps_prod: ...
  save_forces: false       # toggles nstfout in production mdp; default off (Decision 3)

  # HPC defaults — populated after benchmark
  hpc_defaults:
    sims_per_node: 8
    cpus_per_sim: ...
    mem_per_sim: ...
    ntomp: ...
```

---

## 5. Test plan — `tests/martini_pipeline/`

| Test | What it covers | Requires GROMACS? |
|---|---|---|
| `test_composition.py` | name canonicalisation, fraction validation, leaflet math | no |
| `test_lipid_registry.py` | current 10 lipids registered, `.itp` paths exist, lookup raises on unknown, `register_lipid()` + `validate_lipid()` round-trip on a synthetic entry | no |
| `test_mdp_writer.py` | templates render with substituted params; round-trip parse to verify key=value | no |
| `test_manifest.py` | manifest schema, hash determinism | no |
| `test_analysis.py` | `missing_compositions` against fake output tree; `diff_mdps` against a fixture set; `summarise_systems` | no |
| `test_system_builder.py` | `insane` mocked via subprocess monkeypatch; verifies command construction; one real-`insane` test guarded by `shutil.which` | optional |
| `test_pipeline.py` | `gmx` mocked (fake binary in `tmp_path`, prepended to PATH); command construction, idempotency, manifest content, error propagation | no |
| `test_e2e_smoke.py` | POPC100, ~100 ps prod, asserts `prun.xtc` exists with ≥ N frames | yes, opt-in via `RUN_MARTINI_E2E=1` |

Mocked `gmx` is what makes local + HPC parity work — neither environment needs a real GROMACS to keep the suite green.

---

## 6. HPC submission layer

Two thin sbatch wrappers, mirroring the `submit_sweep.sh` / `sbatch_sweep.sh` split.

- **`scripts/bash/submit_simulations.sh`** — orchestrator. Takes a list of compositions (or `--missing` to read from `analysis.py`), expands into one "run" per composition, packs onto nodes up to `--sims-per-node` (default 8 on GPU partition), submits one sbatch per batch. CLI flags: `--partition`, `--time`, `--gpus-per-node`, `--cpus-per-sim`, `--mem-per-sim`, `--output-root`, mdp overrides (`--prod-ns`, `--save-forces`, …).
- **`scripts/bash/sbatch_simulations.sh`** — runs inside SLURM. Stages inputs once per node, then backgrounds N `python -m lipid_gnn.martini_pipeline.pipeline` processes, each pinned via `CUDA_VISIBLE_DEVICES=$i` / `HIP_VISIBLE_DEVICES=$i`, each with `OMP_NUM_THREADS` and `gmx mdrun -ntomp` set to fit `cpus_per_task / sims_per_node`. Per-sim logs at `logs/simulations/sim-<jobid>-gpu<i>.{out,err}`.

**GPU vs CPU**: single orchestrator with a GPU/CPU branch ([Decision 4](#decision-log)). `--gpus-per-node 0` short-circuits the GPU-pinning logic and uses `gmx mdrun` CPU mode with thread budget split across N parallel sims. Avoids two near-duplicate scripts.

---

## 7. HPC benchmark

`scripts/simulation/benchmark_hpc.sh` (bash + a small `analyze_benchmark.py`):

- Runs a fixed reference system (POPC100, ~50 k nsteps) under a grid of sbatch params: `{gpus-per-node} × {cpus-per-sim} × {sims-per-node} × {ntomp}`.
- Records `ns/day`, walltime, GPU util (`rocm-smi`), CPU util into a CSV.
- `analyze_benchmark.py` plots `ns/day` per configuration and recommends production defaults.
- Outcome populates `martini_pipeline.hpc_defaults` in `config.yaml`. Run once; reference in the thesis as the sizing justification.

---

## 8. Implementation order & status

### Workflow per step

Every step follows the same git workflow:

1. Before writing any code for the step, create a feature branch off `main`. Naming: `feat/martini-pipeline-step-<NN>-<short-name>` (e.g. `feat/martini-pipeline-step-01-composition`).
2. Implement and commit on the branch only. Tests for the step must pass locally before the branch is opened for merge.
3. When the step is complete, merge back into `main` via PR (or local merge commit per the project's `merge commits only` convention in [`techContext.md`](../.claude/memory-bank/techContext.md) § GitHub / SSH). Delete the feature branch after merge.
4. Update the status table below (`[ ]` → `[x]`) in the same merge, with a one-line note if the step diverged from plan. Record any new design decisions in §9.

`main` always reflects the last completed step. No partial steps land on `main`.

### Status

Status keys: `[ ]` not started · `[~]` in progress · `[x]` done · `[-]` skipped/deferred.

| # | Step | Status | Notes |
|---|---|---|---|
| 1 | `composition.py` + tests | [x] | Token regex uses `[A-Z]+` (not `[A-Z][A-Z0-9]*`) to avoid greedy digit-consumption ambiguity; all Martini lipid names are letter-only. 203 tests pass. |
| 2 | `lipid_registry.py` (data + `register_lipid` + `validate_lipid` + `check_resources`) + tests | [ ] | |
| 3 | **MDP audit** — `analysis.py::diff_mdps()` over the 70 existing systems; freeze templates from dominant settings; document deviations | [ ] | One-shot script doubles as sanity check on legacy data |
| 4 | `mdp_writer.py` + templates derived from audit | [ ] | Add `nsteps_*`, `save_forces` knobs |
| 5 | Vendor `insane.py` into `resources/martini3/insane.py`; record source/version in `thesisStory.md` | [ ] | See Decision 2 |
| 6 | `system_builder.py` + tests | [ ] | |
| 7 | `pipeline.py` + `manifest.py` — local end-to-end on POPC100; reproduce existing POPC100 frame count + mean APL as sanity check | [ ] | |
| 8 | `analysis.py::missing_compositions()` + CLI driver to print DPPC/DOPC corner work queue | [ ] | Subgoal 2 |
| 9 | HPC submission layer (`submit_simulations.sh` + `sbatch_simulations.sh`) | [ ] | Single orchestrator with GPU/CPU branch |
| 10 | HPC benchmark (`benchmark_hpc.sh` + `analyze_benchmark.py`); populate `hpc_defaults` | [ ] | |
| 11 | Subgoal 2 — fill DPPC/DOPC corners on HPC | [ ] | Production run; not a code task |
| 12 | Subgoal 3 — extend lipid pool beyond current 10 | [ ] | Future, after subgoal 2 lands |

Update this table as each step lands. Add a one-line note when a step diverges from plan; record the rationale in the decision log below.

---

## 9. Decision log

Append-only. Each entry: date · decision · rationale.

| # | Date | Decision | Rationale |
|---|---|---|---|
| 1 | 2026-05-07 | No `gromacs_runner.py`; `pipeline.py` calls `gmx` directly via `subprocess` | Wrapper would add no value while call sites are linear. Reintroduce only if retry / timeout / error-classification accumulates. |
| 2 | 2026-05-07 | Vendor `insane.py` into `resources/martini3/insane.py` | License-clean (GPL, thesis use). Ensures reproducibility independent of the user's env. Source URL + version recorded in `thesisStory.md` § Vendored resources. |
| 3 | 2026-05-07 | `save_forces` defaults to `false` | Training uses positions only; forces inflate `.trr` substantially. Toggle exists for future use. |
| 4 | 2026-05-07 | Single orchestrator with GPU/CPU branch, not two sbatch wrappers | Avoids near-duplicate scripts. `--gpus-per-node 0` short-circuits GPU-pinning. |
| 5 | 2026-05-07 | Lipid registry mutation/validation lives in `lipid_registry.py`, not in `analysis.py` | Adding a lipid and verifying its resources are intrinsic registry concerns. `analysis.py` operates over outputs (existing systems, mdp diffs, coverage gaps); the dependency direction would be wrong if validation lived there. Future lipid-pool extension is also naturally a registry edit. |
| 6 | 2026-05-07 | Default output root is `data/martini_pipeline/`, distinct from legacy `data/membrane_only/` | Keeps new vs old simulations cleanly separated; downstream code is drop-in compatible by pointing at the new root. |
| 7 | 2026-05-07 | Canonical composition naming uses descending mol fraction with alphabetical tiebreak (option A) | Deterministic and future-proof. Diverges from legacy `POPC10_DIPC90`-style ordering on binary systems; legacy names still parse (verified by xfail-marked round-trip test in step 1) but `.name` returns the canonical form. New simulations live under `data/martini_pipeline/` so legacy names remain untouched. |
| 8 | 2026-05-07 | Per-step git workflow: feature branch `feat/martini-pipeline-step-<NN>-<short-name>` off `main`, merge back only after the step is complete and tests pass | Aligns with the project's existing short-lived-branch + merge-commits-only convention in `techContext.md`. Keeps `main` reflecting the last completed step at all times; partial work stays off `main`. |

---

## 10. Open questions

- **Final HPC defaults** (`sims_per_node`, `cpus_per_sim`, `mem_per_sim`, `ntomp`) — set after the benchmark in step 10.
- **MDP audit findings** — until step 3 runs, we don't know which fields are constants vs variables across the 70 systems. Parameter surface in `mdp_writer.py` is finalised by the audit, not before.
- **Seed strategy** — whether each system needs a stable seed for reproducibility, and whether seeds become part of the canonical `<comp>` name. Decide before step 7.
- **`bending_modulus` / `compressibility` re-examination** — outside the scope of this pipeline, but new simulations may eventually be used to revisit these targets. Flagged here so the manifest captures enough metadata to support that later.

---

## 11. Cross-references

- Memory bank: [`progress.md`](../.claude/memory-bank/progress.md) § "What's Left to Build" → "Martini 3 lipid simulation pipeline (long-term)"; [`thesisStory.md`](../.claude/memory-bank/thesisStory.md) § 9 "Open questions and next phases" + § Vendored resources.
- Existing submission pattern to mirror: [`scripts/bash/submit_sweep.sh`](../scripts/bash/submit_sweep.sh), [`scripts/bash/sbatch_sweep.sh`](../scripts/bash/sbatch_sweep.sh).
- Coverage gap motivation: [`results/figures/stage_5b/stage_5b_analysis_report.md`](../results/figures/stage_5b/stage_5b_analysis_report.md) (DPPC/DOPC-rich corner errors).
- HPC environment: [`docs/hpc_goethe.md`](hpc_goethe.md).

---

## Appendix A — Step 1 detailed plan: `composition.py`

### A.1 Scope

`lipid_gnn/martini_pipeline/composition.py` defines what a *composition* is and the only operations on it: shape validation, canonical naming, name parsing, leaflet count math. Pure stdlib — no numpy, no MDAnalysis, no registry import.

**Out of scope** (handled in later steps):

- Lipid-name existence in the registry (composed by callers; step 2's `lipid_registry.py`).
- Asymmetric leaflets.
- Anything touching disk, `gmx`, or `insane`.

### A.2 Locked-in design decisions

1. **Canonical name ordering** — descending mol fraction, alphabetical tiebreak (Decision 7). `{POPC: 0.1, DIPC: 0.9}` → `DIPC90_POPC10`.
2. **Percentage representation** — names use integer percentages summing to 100. Fractions like `0.333` are rejected with an informative error; user must pick a compatible spec.
3. **Zero fractions** — rejected (force the user to drop the entry).
4. **Lipid-name case** — strict upper case. `parse_name("popc100")` raises.
5. **`Composition` carries fractions only, not leaflet size** — leaflet count is a build-time concern; `counts_per_leaflet(comp, n)` is a free function.

### A.3 Public API

```python
@dataclass(frozen=True)
class Composition:
    fractions: Mapping[str, float]   # immutable; validated in __post_init__

    @property
    def name(self) -> str: ...                     # canonical, e.g. "DOPC70_POPC30"
    @property
    def lipid_types(self) -> tuple[str, ...]: ...  # ordered as in canonical name

def parse_name(name: str) -> Composition: ...
def counts_per_leaflet(comp: Composition, n_lipids_per_leaflet: int) -> dict[str, int]: ...
def validate_fractions(fractions: Mapping[str, float], tol: float = 1e-6) -> None: ...
```

`__post_init__` calls `validate_fractions`. `tol=1e-6` applies to the sum-to-1 check, not to percentage rounding.

### A.4 Internal helpers

- `_canonical_order(fractions) -> tuple[str, ...]` — descending fraction, alpha tiebreak.
- `_to_integer_percentages(fractions) -> dict[str, int]` — rounds; raises if any entry deviates from its rounded percentage by more than tolerance.
- `_NAME_RE = re.compile(r"^([A-Z][A-Z0-9]*)(\d{1,3})(?:_([A-Z][A-Z0-9]*)(\d{1,3}))*$")` — anchor-based parser.

### A.5 Edge-case matrix

| Input | Expected |
|---|---|
| `{}` | `validate_fractions` raises `ValueError` |
| `{POPC: 1.0}` | valid; name `POPC100` |
| `{POPC: 1.5}` | raises (out of range) |
| `{POPC: -0.1, DOPC: 1.1}` | raises (out of range) |
| `{POPC: 0.6, DOPC: 0.5}` | raises (sum > 1) |
| `{POPC: 0.3, DOPC: 0.6}` | raises (sum < 1) |
| `{POPC: 0.0, DOPC: 1.0}` | raises (zero fraction) |
| `{POPC: 0.5, DOPC: 0.5}` | name `DOPC50_POPC50` (alpha tiebreak) |
| `{POPC: 1/3, DOPC: 1/3, DPPC: 1/3}` | sum-check passes; integer-percent check fails — raises with informative message |
| `parse_name("POPC100")` | `Composition({POPC: 1.0})` |
| `parse_name("DOPC70_POPC30")` | `Composition({DOPC: 0.7, POPC: 0.3})` |
| `parse_name("POPC30_DOPC70")` | parses, but `.name` returns `DOPC70_POPC30` — round-trip is *semantic*, not byte-exact |
| `parse_name("")` | raises |
| `parse_name("POPC30")` | raises (single lipid not at 100) |
| `parse_name("POPC30_DOPC60")` | raises (sums to 90) |
| `parse_name("POPC30_POPC70")` | raises (duplicate lipid) |
| `parse_name("popc30_dopc70")` | raises (lowercase) |
| `parse_name("POPC_30")` | raises (regex mismatch) |
| `counts_per_leaflet(Comp({POPC: 0.3, DOPC: 0.7}), 100)` | `{POPC: 30, DOPC: 70}` |
| `counts_per_leaflet(Comp({POPC: 0.3, DOPC: 0.7}), 33)` | raises (non-integer counts) |

### A.6 Test plan — `tests/martini_pipeline/test_composition.py`

Eight test groups, all stdlib:

1. `test_validate_fractions_*` — happy paths + every `validate_fractions` row of the matrix. Parametrised.
2. `test_canonical_name_*` — pure, binary, ties, three-lipid; verifies the descending-fraction-then-alpha rule explicitly.
3. `test_parse_name_*` — every `parse_name` row, parametrised on (input, expected-or-exception).
4. `test_round_trip_semantic` — over a parametrised list of specs, `parse_name(Composition(spec).name).fractions == spec` (within tol).
5. `test_counts_per_leaflet_*` — happy path + non-integer-count rejection + n=200 cases.
6. `test_immutability` — `Composition` is frozen; mutation raises.
7. `test_legacy_names_parse` — parametrised over `os.listdir("data/membrane_only")`. Every legacy name must `parse_name` cleanly. The byte-exact round-trip is a *separate* assertion expected to fail on legacy binary systems (legacy ordering ≠ canonical ordering); marked `xfail(strict=True)` so a future ordering change flips it to xpass and forces an explicit decision. Skipped if `data/membrane_only/` is absent (HPC nodes without local data).
8. `test_unicode_and_whitespace` — names with leading/trailing whitespace or non-ASCII raise.

### A.7 Layout & dependencies

- New files:
  - `lipid_gnn/martini_pipeline/__init__.py` (empty)
  - `lipid_gnn/martini_pipeline/composition.py`
  - `tests/martini_pipeline/test_composition.py`
- Pytest discovery: existing `tests/` is flat without `__init__.py` and works. Subdirectory should still be discovered. Verify with `pytest tests/martini_pipeline/ -q`; if discovery fails, add `tests/martini_pipeline/__init__.py` (simpler than touching `pyproject.toml`).
- No new entries in `requirements.txt`.
- No `config.yaml` changes in this step; the `martini_pipeline:` block lands with step 4.

### A.8 Acceptance criteria

- `pytest tests/martini_pipeline/test_composition.py -q` passes locally and on HPC.
- `test_legacy_names_parse` xfail count equals the number of legacy multi-lipid systems. Pure-lipid systems (`POPC100`, `DOPC100`, …) round-trip exact.
- Module is < ~150 LOC; no comments unless a non-obvious WHY (per project style).
- Branch `feat/martini-pipeline-step-01-composition` merged into `main`; status table flipped to `[x]`.
