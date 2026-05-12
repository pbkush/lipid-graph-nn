# Martini 3 Lipid Simulation Pipeline — Plan & Tracker

Long-term, general-purpose Martini 3 membrane simulation pipeline. Stands as a research deliverable in its own right; newly simulated systems are not necessarily training data. This document is the single source of truth for the plan, progress, and decisions.

Last updated: 2026-05-12 (step 7 complete).

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
| `test_e2e_smoke.py` | DIPC100, ~50 ps prod, asserts `prun.xtc` + APL in physical range; no legacy comparison | yes, opt-in via `RUN_MARTINI_E2E=1` |

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
| 2 | `lipid_registry.py` (data + `register_lipid` + `validate_lipid` + `check_resources`) + tests | [x] | `_KNOWN_FAMILIES` is a module-level `frozenset` (open for extension); bead cross-check against `node_mapping.json` is a hard assertion. 60 tests pass. |
| 3 | **MDP audit** — `analysis.py::diff_mdps()` over the 70 existing systems; freeze templates from dominant settings; document deviations | [x] | run.mdp: zero deviations (all 70 byte-identical). Equilibration: 7 rlist deviations on CHOL systems (Verlet-buffer auto-tuning, expected). Freeze record committed as `templates/_audit_freeze.json`. |
| 4 | `mdp_writer.py` + templates derived from audit (Appendix D) | [x] | Eq diverges from legacy: `compressibility 3e-4`, `nsteps 1e6`, explicit `gen-vel=yes`. Knobs: `nsteps_*`, `save_forces`, `gen_seed`. 32 tests pass. |
| 5 | Vendor `insane.py` into `resources/martini3/insane.py`; record source/version in `thesisStory.md` (Appendix E) | [x] | **Reworked 2026-05-08 (Decision 26)**: insane.py vendoring removed; `insane` pip package (v1.2.0) used as command instead. `INSANE_PATH` → `INSANE_CMD = "insane"`. 7 tests pass. |
| 6 | `system_builder.py` + tests | [x] | **Reworked 2026-05-08 (Decision 27)**: 32 ITPs from M3-Lipid-Parameters (full lipidome) replace the 9 legacy-copied ITPs. `ffbonded_v2.itp` included (required by v2 lipid files). pbc now always passed explicitly. 27 tests pass. |
| 7 | `pipeline.py` + `manifest.py` — local end-to-end; DIPC100 APL sanity check against physical criteria | [x] | Decisions 22–25, 28–32. Filenames match legacy (`martini_em`, `martini_eq`, `prun`). `-maxwarn 2` default (CLI flag). Seed deterministic from composition hash. `MartiniPipelineConfig` in `config.py`. CLI `run_martini_pipeline.py` with insane-style ratio args + `--nsteps`/`--prod-ns`. 377 tests pass, 7 skipped. |
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
| 9 | 2026-05-07 | `LIPID_REGISTRY` is functional/immutable (`MappingProxyType` + non-mutating `register_lipid`) | Matches `composition.py` pattern; registry is small enough that copy-on-extend is free; eliminates test-pollution bugs. |
| 10 | 2026-05-07 | Bead order hardcoded in registry, cross-checked against `resources/martini_ff_node_mapping.json` by a failing test | Explicit code is reviewable in PRs; hard assertion catches drift before it silently breaks downstream graph construction. |
| 11 | 2026-05-07 | Equilibration/minimization mdps recovered via `gmx dump -s` from `.tpr` (legacy `.mdp` sources not preserved) | `.tpr` is byte-stable and contains the full inputrec; reconstruction is deterministic. |
| 12 | 2026-05-07 | Random-seed-like keys (`gen-seed`, `ld-seed`, `tinit`, `init-step`, `simulation-part`) excluded from deviation reporting | Per-system variation in these keys is intended; including them drowns out real deviations. |
| 13 | 2026-05-07 | Step 4 (`mdp_writer.py`) reads `templates/_audit_freeze.json` at template-build time, fails fast if missing | Enforces audit-then-write order; prevents template defaults drifting from legacy values without explicit re-audit. |
| 14 | 2026-05-07 | Step 4 templates **deliberately diverge from legacy** in equilibration: `compressibility 3e-4 3e-4` (legacy `3e-5`), `nsteps 1_000_000` / 10 ns (legacy 250 000 / 2.5 ns), `nstenergy 1000` (legacy 100), explicit `gen-vel = yes`, `gen-temp = 310`, `gen-seed = -1` | Legacy `3e-5` makes the box relax ~10× slower than τ_p = 5 ps (Berendsen) was tuned for; novel compositions (high-CHOL, DIPC/POPC blends) frequently exit equilibration still drifting at 2.5 ns. The audit captures *what we ran*; the writer captures *what we should run going forward*. Run-stage values are still cloned byte-for-byte from legacy. |
| 15 | 2026-05-07 | Configurable parameters are marked in template files with inline `; [CONFIG: <knob>]` comments | Makes the surface that `MDPParams` (and later `martini_pipeline.*` config) controls discoverable from the template alone; reviewers can audit knob coverage by grepping `[CONFIG:`. |
| 16 | 2026-05-07 | Single-stage equilibration in v1; two-stage (eq1 small-dt + eq2 production-dt) deferred | The 10 ns single-stage already absorbs the legacy reliability gap. Two-stage adds pipeline + manifest complexity; revisit only if step 7+ surfaces equilibration failures on exotic compositions. |
| 18 | 2026-05-07 | Vendored `insane.py` is the 2to3-converted Helgi/Emil-customised copy of Tsjerk Wassenaar's 2014-06-03 build, not the modern Python-3 fork | Parity with lipid templates that built the 70 training systems. **← Superseded by Decision 26 (2026-05-08).** |
| 19 | 2026-05-08 | ITP files sourced from `data/membrane_only/POPC100/toppar/` (9 Martini 3 files; yire1 protein ITPs excluded) | Same files that produced the 70 training systems. **← Superseded by Decision 27 (2026-05-08).** |
| 20 | 2026-05-08 | Build directories are self-contained: each `out_dir` receives its own `toppar/` copy of the ITPs | Avoids relative-path fragility; each build dir can be moved or archived independently. |
| 21 | 2026-05-08 | `LipidEntry.insane_keyword` (not `.name`) used as the `-l` token in `build_command()` | Allows future lipids whose insane identifier differs from their registry name without special-casing in `system_builder.py`. |
| 26 | 2026-05-08 | `insane` pip package (v1.2.0) used via command, not vendored single-file script | Modern Python-3 package; installed in `lipid_gnn` conda env. Vendoring a 2to3-converted 2014 script was a workaround; using the package directly is cleaner and upgradeable. `INSANE_PATH` constant replaced by `INSANE_CMD = "insane"`. Added to `requirements.txt`. Supersedes Decision 18. |
| 27 | 2026-05-08 | 32 ITPs from `github.com/Martini-Force-Field-Initiative/M3-Lipid-Parameters` replace 9 legacy copies | Full lipidome coverage (standard phospholipids, ether phospholipids, plasmalogens, sterols, glycerolipids, ions, solvents, DOTAP) using current upstream parameters. Includes `ffbonded_v2.itp` (required by v2 lipid files). Legacy copies from POPC100 toppar were an old snapshot with 4 unneeded files (nucleobases, sugars, small molecules, matthieu phospholipids). Supersedes Decision 19. |
| 22 | 2026-05-12 | `pipeline.run()` orchestrates one composition; multi-system batching is the submission layer's job (step 9) | Keeps the orchestrator linear and testable; HPC step 9 wraps N parallel invocations. |
| 23 | 2026-05-12 | Stage handoff via per-stage `.gro` copy (`minimized.gro`, `equilibrated.gro`); production reuses `prun.gro` directly | Handoff `.gro` written after `mdrun` exits zero is a strong success marker; no separate `.done` sentinel needed. Production's `prun.gro` is the same file GROMACS writes, so no copy is performed. |
| 24 | 2026-05-12 | Seed derived deterministically from composition name via `sha256(name)[:8]`; CLI overrides; same seed used across all stages | Reproducible without making seeds part of the canonical `<comp>` name. |
| 25 | 2026-05-12 | Manifest rewritten after every stage transition | A killed run still leaves a useful manifest reflecting the last completed stage. |
| 28 | 2026-05-12 | Stage filenames match legacy: `martini_em`, `martini_eq`, `prun` | Pure parity; downstream code references these names today. |
| 29 | 2026-05-12 | `index.ndx` uses default groups (`q\n`) by default; `make_ndx_script` parameter allows callers to pass a custom script | MDPs use `System` group only — custom `Membrane`/`Solute` groups are for analysis, not for grompp. Default is sufficient for all simulations. |
| 30 | 2026-05-12 | `-maxwarn` defaults to 2; configurable via CLI flag `--maxwarn` | Legacy used -maxwarn 5 for em/eq steps. Value 2 is a conservative default; CLI allows override on a per-run basis. |
| 31 | 2026-05-12 | `nsteps_prod = -1` in config; CLI requires `--nsteps N` or `--prod-ns N` (mutually exclusive, one required) | Production run length is the most consequential knob; forcing an explicit value prevents accidental zero-length runs. |
| 32 | 2026-05-12 | CLI `run_martini_pipeline.py` uses insane-style ratio strings (`POPC:1.0 DOPC:0.3`) | Ergonomic for single-system invocations; fractions are normalised so raw counts (e.g. `POPC:7 DOPC:3`) also work. |

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

---

## Appendix B — Step 2 detailed plan: `lipid_registry.py`

### B.1 Scope

`lipid_gnn/martini_pipeline/lipid_registry.py` defines what a *registered Martini 3 lipid* is and the read/extend/verify operations on the registry. Pure stdlib — no numpy, no MDAnalysis, no `insane` import, no `composition` import (independence rule).

#### In scope

- A frozen `LipidEntry` dataclass capturing the metadata the rest of the pipeline needs.
- A `LIPID_REGISTRY` mapping for the current 10 lipids: `DIPC`, `DOPC`, `DPPC`, `POPC`, `DOPE`, `DPPE`, `POPE`, `DOPS`, `POPS`, `CHOL`.
- `register_lipid(registry, entry) -> registry'` — returns a *new* registry dict with the entry added. Pure / non-mutating.
- `validate_lipid(entry)` — shape validation: required fields, types, non-empty strings, unique bead names, `family` in known set.
- `check_resources(entry, *, itp_dir=None, node_mapping_path=None)` — on-disk verification. Each path is optional: `None` skips that check. Returns a `ResourceCheck` dataclass with per-check booleans and an aggregated `ok` flag plus `errors: list[str]`.
- `get_lipid(name)` / `lipid_names()` accessors for the default registry.

#### Out of scope (handled in later steps)

- Anything that loads/imports `insane.py` or runs `gmx`. Step 5 vendors insane and step 6's `system_builder.py` calls it. The registry only declares an `insane_keyword` field; verifying it against vendored insane is a step-6 test.
- Mol-fraction / composition validation — that's `composition.py`.
- `.itp` content rewriting or merging — pipeline-level concern.
- Asymmetric leaflets, lipid mixtures, charge balance — composition + system_builder concerns.

### B.2 Locked-in design decisions

1. **Registry is functional, not mutable.** `register_lipid` returns a new dict. The module-level `LIPID_REGISTRY` is a `MappingProxyType` over a private dict; callers cannot mutate it. Rationale: matches the immutability pattern in `composition.py`; thesis-scale registry is small, copying is free, and we never need cross-test mutation.
2. **Bead order is hardcoded in the registry, then *cross-checked* against `resources/martini_ff_node_mapping.json` in tests.** Source-of-truth lives in code (explicit, reviewable, version-controlled with the pipeline) but a test asserts agreement with the existing node-mapping resource so drift is caught loudly.
3. **`check_resources` paths are injected, not read from `CONFIG`.** `config.yaml`'s `martini_pipeline:` block lands in step 4. Until then, callers (and tests) pass paths explicitly. Once step 4 lands, a thin convenience wrapper in `pipeline.py` will fill defaults from `CONFIG`; the registry stays config-free.
4. **`family` is a closed enum-like string set**: `{"phospholipid", "sterol"}` for the current 10. Adding a new family (e.g. `"glycolipid"`, `"sphingolipid"`) requires extending the set explicitly — `validate_lipid` rejects unknown families. Forces a conscious decision when extending the pool (subgoal 3).
5. **No `insane.py` parsing in this step.** `insane_keyword` is metadata only; verification is deferred to step 6 where a real insane parser exists. Step 2 cannot fail-or-pass on something it doesn't have access to.

### B.3 Public API

```python
@dataclass(frozen=True)
class LipidEntry:
    name: str                        # canonical short name, e.g. "POPC"
    resname: str                     # residue name in topology; usually == name
    itp_file: str                    # basename of the .itp that declares the moleculetype
    moleculetype: str                # name in [moleculetype] block; usually == name
    beads: tuple[str, ...]           # canonical bead order, matches node_mapping.json
    family: str                      # "phospholipid" | "sterol"
    insane_keyword: str              # -l flag value for insane; usually == name

LIPID_REGISTRY: Mapping[str, LipidEntry]   # immutable; 10 default entries

def get_lipid(name: str) -> LipidEntry: ...
def lipid_names() -> tuple[str, ...]: ...
def register_lipid(registry: Mapping[str, LipidEntry], entry: LipidEntry) -> dict[str, LipidEntry]: ...
def validate_lipid(entry: LipidEntry) -> None: ...

@dataclass(frozen=True)
class ResourceCheck:
    lipid: str
    itp_present: bool | None         # None = check skipped
    moleculetype_declared: bool | None
    beads_match_node_mapping: bool | None
    errors: tuple[str, ...]
    @property
    def ok(self) -> bool: ...        # True iff no errors and no failed checks

def check_resources(
    entry: LipidEntry,
    *,
    itp_dir: str | os.PathLike | None = None,
    node_mapping_path: str | os.PathLike | None = None,
) -> ResourceCheck: ...
```

### B.4 The 10 default entries

All ten lipids are present in `resources/martini_ff_node_mapping.json` with full bead lists; the registry mirrors those exactly. Mapping verified during planning:

| name | resname | itp_file | moleculetype | family | beads (count) |
| --- | --- | --- | --- | --- | --- |
| `DIPC` | `DIPC` | `martini_v3.0.0_phospholipids_v1.itp` | `DIPC` | phospholipid | 12 |
| `DOPC` | `DOPC` | `martini_v3.0.0_phospholipids_v1.itp` | `DOPC` | phospholipid | 12 |
| `DPPC` | `DPPC` | `martini_v3.0.0_phospholipids_v1.itp` | `DPPC` | phospholipid | 12 |
| `POPC` | `POPC` | `martini_v3.0.0_phospholipids_v1.itp` | `POPC` | phospholipid | 12 |
| `DOPE` | `DOPE` | `martini_v3.0.0_phospholipids_v1.itp` | `DOPE` | phospholipid | 12 |
| `DPPE` | `DPPE` | `martini_v3.0.0_phospholipids_v1.itp` | `DPPE` | phospholipid | 12 |
| `POPE` | `POPE` | `martini_v3.0.0_phospholipids_v1.itp` | `POPE` | phospholipid | 12 |
| `DOPS` | `DOPS` | `martini_v3.0.0_phospholipids_v1.itp` | `DOPS` | phospholipid | 12 |
| `POPS` | `POPS` | `martini_v3.0.0_phospholipids_v1.itp` | `POPS` | phospholipid | 12 |
| `CHOL` | `CHOL` | `martini_v3.0_sterols_v1.0.itp` | `CHOL` | sterol | 9 |

Bead lists copy the exact arrays in `resources/martini_ff_node_mapping.json` for each `name`. `insane_keyword` equals `name` for all ten.

### B.5 Internal helpers

- `_ITP_MOLECULETYPE_RE = re.compile(r"^\s*\[\s*moleculetype\s*\]\s*$", re.IGNORECASE)` — anchors the section header.
- `_parse_moleculetypes(itp_text: str) -> set[str]` — scans an `.itp` text body, collects every name following a `[ moleculetype ]` header (skipping the comment line). Used by `check_resources`.
- `_load_node_mapping(path) -> dict[str, list[str]]` — `json.load` wrapper.

### B.6 Edge-case matrix

| Input | Expected |
|---|---|
| `validate_lipid(LipidEntry("POPC", "POPC", "...itp", "POPC", ("NC3","PO4",...), "phospholipid", "POPC"))` | passes |
| `validate_lipid(LipidEntry("", ...))` | raises (empty name) |
| `validate_lipid(LipidEntry("popc", ...))` | raises (lowercase — registry uses upper case, mirrors `composition.py`) |
| `validate_lipid(LipidEntry("X", ..., family="lipid"))` | raises (unknown family) |
| `validate_lipid(LipidEntry("X", ..., beads=()))` | raises (no beads) |
| `validate_lipid(LipidEntry("X", ..., beads=("A","A")))` | raises (duplicate bead) |
| `validate_lipid(LipidEntry("X", ..., itp_file=""))` | raises (empty itp_file) |
| `register_lipid(R, entry)` where `entry.name in R` | raises `ValueError("duplicate")` |
| `register_lipid(R, entry)` where `validate_lipid(entry)` would fail | raises (validation runs first) |
| `register_lipid(R, entry)` happy path | returns a new mapping equal to `dict(R)` plus `{entry.name: entry}` |
| `get_lipid("POPC")` | returns the registered entry |
| `get_lipid("NOPE")` | raises `KeyError` with helpful message listing known names |
| `check_resources(entry)` (all paths None) | `ok=True`, all check fields None |
| `check_resources(entry, itp_dir=tmp/with-file-and-moleculetype)` | `itp_present=True`, `moleculetype_declared=True` |
| `check_resources(entry, itp_dir=tmp/empty)` | `itp_present=False`, `errors` mentions missing file |
| `check_resources(entry, itp_dir=tmp/file-without-moleculetype)` | `itp_present=True`, `moleculetype_declared=False`, `errors` informative |
| `check_resources(entry, node_mapping_path=fake_json_with_match)` | `beads_match_node_mapping=True` |
| `check_resources(entry, node_mapping_path=fake_json_with_mismatch)` | `beads_match_node_mapping=False`, `errors` includes diff |
| `check_resources(entry, node_mapping_path=fake_json_missing_lipid)` | `beads_match_node_mapping=False`, `errors` says "lipid not in node mapping" |
| `check_resources(entry, itp_dir=does/not/exist)` | `itp_present=False`, `errors` mentions missing dir |

`ResourceCheck.ok` is `True` iff every non-None check field is `True` and `errors` is empty.

### B.7 Test plan — `tests/martini_pipeline/test_lipid_registry.py`

Seven test groups, all stdlib + pytest:

1. **`test_default_registry_complete`** — asserts `set(lipid_names()) == {"DIPC","DOPC","DPPC","POPC","DOPE","DPPE","POPE","DOPS","POPS","CHOL"}` (exactly these 10).
2. **`test_default_entries_validate`** — parametrised over the 10 names; calls `validate_lipid(get_lipid(name))`. None should raise.
3. **`test_default_beads_match_node_mapping`** — loads `resources/martini_ff_node_mapping.json` and asserts each registry entry's `beads` tuple equals (in order) the keys in the JSON for that name. This is the source-of-truth cross-check from Decision B.2.2. Skipped (with explicit reason) if `resources/martini_ff_node_mapping.json` is absent on the host.
4. **`test_get_lipid_unknown_raises`** — `get_lipid("NOPE")` raises `KeyError`, message includes `"NOPE"` and at least one known name.
5. **`test_register_lipid_*`** — happy round-trip on a synthetic `"FAKE"` entry; original registry unchanged (immutability); duplicate raises; malformed entry raises; returned object is a plain dict with the new entry.
6. **`test_validate_lipid_invalid`** — parametrised over the failure rows of the edge-case matrix.
7. **`test_check_resources_*`** — uses `tmp_path`:
   - `test_check_resources_skipped_when_paths_none` — `ok=True`, all check fields None.
   - `test_check_resources_itp_present_and_moleculetype_declared` — write a minimal `.itp` containing `[ moleculetype ]\n; molname nrexcl\nFAKE 1\n` into `tmp_path / "fake.itp"`, point an entry at it, assert green.
   - `test_check_resources_itp_missing` — `tmp_path` empty.
   - `test_check_resources_moleculetype_missing` — `.itp` exists but doesn't declare the lipid.
   - `test_check_resources_node_mapping_match` — write a JSON file mapping `"FAKE": {"A": "...", "B": "..."}`, entry has `beads=("A","B")`, assert green.
   - `test_check_resources_node_mapping_mismatch` — JSON has different beads/order; assert red with informative error.
   - `test_check_resources_node_mapping_missing_lipid` — JSON doesn't have the lipid at all.
8. **`test_check_resources_against_legacy_data`** — `@skipif(not _HAS_LEGACY_DATA)`, parametrised over the 10 default lipids; runs `check_resources(entry, itp_dir=data/membrane_only/POPC100/toppar/, node_mapping_path=resources/martini_ff_node_mapping.json)` and asserts `ok=True`. This is the integration test that proves the registry is internally consistent with the data we already have.

### B.8 Layout & dependencies

- New files:
  - `lipid_gnn/martini_pipeline/lipid_registry.py`
  - `tests/martini_pipeline/test_lipid_registry.py`
- No new entries in `requirements.txt`.
- No `config.yaml` changes; the `martini_pipeline:` block lands with step 4.
- No imports from or exports to `composition.py` or any other pipeline module.

### B.9 Acceptance criteria

- `pytest tests/martini_pipeline/test_lipid_registry.py -q` passes locally and on HPC (legacy-data and node-mapping tests skip cleanly when those paths are absent on a fresh HPC node).
- All 10 default entries' bead lists agree with `resources/martini_ff_node_mapping.json` byte-for-byte and order-for-order.
- Module is < ~250 LOC; no comments unless a non-obvious WHY (per project style).
- No regressions in the existing test suite (`pytest -q` total count grows by exactly the new test count).
- Branch `feat/martini-pipeline-step-02-lipid-registry` merged into `main` via `--no-ff`; status table flipped to `[x]` and any divergence from this plan recorded as a one-line note plus a Decision-log entry if a design choice changed.

### B.10 New decisions to log on completion

If implementation matches this plan, append two entries to §9:

- **Decision 9** — `LIPID_REGISTRY` is functional/immutable (`MappingProxyType` + non-mutating `register_lipid`). Rationale: matches `composition.py`; registry is small enough that copy-on-extend is free; eliminates an entire class of test-pollution bugs.
- **Decision 10** — Bead order is hardcoded in the registry and cross-checked against `resources/martini_ff_node_mapping.json` by a unit test, rather than loaded from JSON at import time. Rationale: explicit code is reviewable in PRs; the cross-check catches drift loudly; keeps the registry usable on hosts without the resource file.

If the implementation diverges, capture the actual decision instead.

---

## Appendix C — Step 3 detailed plan: MDP audit (`analysis.py::diff_mdps()`)

### C.1 Scope

A one-shot audit that produces (a) the **freeze record**: the canonical, parameter-by-parameter MDP settings used in the legacy 70 systems for production, equilibration, and minimization; and (b) the **deviation report**: any per-system variation. Output drives step 4 (`mdp_writer.py`) — which keys become template knobs vs. hardcoded constants — and doubles as a sanity check on legacy data integrity.

The audit is the first piece of `lipid_gnn/martini_pipeline/analysis.py`; the module is multi-purpose ([§3](#3-module-breakdown--lipid_gnnmartini_pipeline)) but only `diff_mdps()` and its helpers land in step 3. `missing_compositions()` and `summarise_systems()` arrive in step 8.

#### What's in scope for step 3

- `diff_mdps(systems_root, *, stages=("run", "equilibration", "minimization"), gmx_binary="gmx") -> MDPAuditReport` — pure function over a directory tree of legacy systems.
- A `MDPAuditReport` dataclass with: per-stage mode-value-per-key tables, per-stage list-of-deviations, list of missing files per system per stage.
- Internal helpers: `_parse_mdp_file(path) -> dict[str, str]`, `_dump_tpr_inputrec(tpr_path, gmx_binary) -> dict[str, str]`, `_normalise_kv(raw) -> str` (collapses whitespace, lowercases booleans, sorts space-separated lists where order is irrelevant — `tc-grps` etc.).
- A CLI driver `scripts/simulation/audit_mdps.py` that prints the report and writes:
  - `docs/mdp_audit_report.md` — human-readable summary (canonical values + deviations).
  - `lipid_gnn/martini_pipeline/templates/_audit_freeze.json` — machine-readable freeze record consumed by step 4 to derive templates.
- Tests at `tests/martini_pipeline/test_analysis_diff_mdps.py`. `gmx dump` is mocked via a fake binary on `PATH`; one opt-in test uses real `gmx` against legacy data.

#### What's out of scope for step 3

- `mdp_writer.py` itself, template files, parameter substitution — step 4.
- Comparison against canonical Martini 3 reference mdps (Marrink lab). The audit captures *what we ran*, not *what we should have run*.
- Changing the legacy data. The audit is read-only.
- `summarise_systems()`, `missing_compositions()` — step 8.

### C.2 Locked-in design decisions

1. **Read `run.mdp` from disk; recover `eq.mdp` / `em.mdp` from `.tpr` via `gmx dump -s`.** Verified during planning: the legacy tree has only `run.mdp` per system; equilibration/minimization mdps were not preserved as files but the inputrec is fully recoverable from the matching `.tpr`. This makes the audit machine-only — no human bookkeeping needed.
2. **Audit runs locally, not on HPC.** The 70 systems live under `data/membrane_only/`; `gmx dump` is fast (< 100 ms per tpr). No need for SLURM. Tests mock `gmx` so CI stays gmx-free.
3. **Deviations are reported, not fixed.** If two systems disagree on a key, the audit lists the disagreement and picks the **mode** (most frequent value) as the freeze value. The mode is what step 4 templates emit by default; deviations are documented in `docs/mdp_audit_report.md` for manual review. This avoids quietly "correcting" old data.
4. **MDP parser is whitespace-tolerant but type-naive.** Values are stored as normalised strings (e.g. `"310"`, `"3e-5 3e-5"`, `"v-rescale"`). Type promotion (int/float/bool) is a step-4 concern, where templates know each field's expected type.
5. **Comments and blank lines are dropped.** They contain no semantically loaded information for the audit. The freeze record is a flat `{key: value}` dict per stage.
6. **Key normalisation: hyphens vs. underscores.** GROMACS accepts both (`tc-grps` ≡ `tc_grps`). The parser canonicalises to the hyphenated form (matches `gmx dump` output) so on-disk `run.mdp` and dumped `.tpr` records compare cleanly.
7. **One audit per stage**, not per-system, in the report. The output asserts a single canonical value per `(stage, key)` plus a list of `(system, key, value)` triples for any divergence. Per-system identity (verified during planning: all 70 `run.mdp` md5-identical) means the deviation list will likely be empty for `run`; em/eq may differ in `gen_seed` or `tinit` and that is acceptable.
8. **`gen_seed` is excluded from the deviation count.** Random seed is supposed to differ per system; reporting it as a deviation is noise. The freeze record stores `gen_seed = "RANDOM"` (sentinel) and `mdp_writer.py` substitutes a fresh seed per build.

### C.3 Public API

```python
@dataclass(frozen=True)
class MDPDeviation:
    system: str           # composition name, e.g. "POPC30_DOPC70"
    stage: str            # "run" | "equilibration" | "minimization"
    key: str              # canonical hyphenated key, e.g. "ref-t"
    value: str            # the system's value
    canonical: str        # the audit's mode value for this (stage, key)


@dataclass(frozen=True)
class MDPStageAudit:
    stage: str
    n_systems: int                      # systems contributing to this stage
    canonical: Mapping[str, str]        # mode value per key (the freeze record)
    deviations: tuple[MDPDeviation, ...]
    missing_systems: tuple[str, ...]    # systems where the source file/.tpr was absent


@dataclass(frozen=True)
class MDPAuditReport:
    stages: Mapping[str, MDPStageAudit]   # keyed by stage name

    @property
    def total_deviations(self) -> int: ...
    def to_markdown(self) -> str: ...     # human report
    def to_freeze_json(self) -> str: ...  # machine record consumed by step 4


def diff_mdps(
    systems_root: str | os.PathLike,
    *,
    stages: tuple[str, ...] = ("run", "equilibration", "minimization"),
    gmx_binary: str = "gmx",
    skip_keys: frozenset[str] = frozenset({"gen-seed", "ld-seed", "tinit", "init-step"}),
) -> MDPAuditReport: ...
```

### C.4 Stage source map

| Stage | Source per system | Recovery |
| --- | --- | --- |
| `run` | `<system>/run.mdp` | direct file read |
| `equilibration` | `<system>/equilibration/martini_eq.tpr` | `gmx dump -s ...tpr` → parse `inputrec:` block |
| `minimization` | `<system>/minimization/martini_em.tpr` | `gmx dump -s ...tpr` → parse `inputrec:` block |

`_dump_tpr_inputrec` invokes `subprocess.run([gmx_binary, "dump", "-s", tpr_path], capture_output=True, text=True, timeout=30)` and parses the `inputrec:` section by splitting `key = value` lines until the first non-indented line. Errors propagate as `RuntimeError` with the gmx stderr appended.

### C.5 Internal helpers

- `_parse_mdp_file(path) -> dict[str, str]` — strips comments (`;`-prefixed), splits on `=`, normalises key (lower + hyphenate), normalises value (collapse whitespace).
- `_dump_tpr_inputrec(tpr_path, gmx_binary) -> dict[str, str]` — runs `gmx dump`, parses the `inputrec:` block, normalises like the mdp parser.
- `_canonicalise_key(key: str) -> str` — `tc_grps` → `tc-grps`, `epsilon_r` → `epsilon-r`. Lowercased.
- `_mode(values: Sequence[str]) -> str` — most frequent; ties broken by first-seen order.
- `_collect_for_stage(systems_root, stage, gmx_binary) -> dict[system, dict[key, value]]` — orchestrates per-stage collection. Records absent files in a separate `missing_systems` set.
- `_audit_stage(by_system: dict, skip_keys) -> MDPStageAudit` — computes mode per key, lists deviations.

### C.6 Edge-case matrix

| Input | Expected |
|---|---|
| `systems_root` empty | report with all stages having `n_systems=0`, no canonical, no deviations |
| All 70 systems byte-identical at `run.mdp` (verified during planning) | `run` stage canonical = legacy values; `deviations = ()` |
| Single system has a 1-key disagreement at `equilibration` | `MDPStageAudit.deviations` contains one entry; canonical = mode |
| 50/50 split on a key | mode broken by first-seen order; both halves listed as deviations from canonical |
| `gen-seed` differs across all systems | not reported (in `skip_keys`) |
| One system missing `run.mdp` | listed in `missing_systems`; excluded from canonical computation |
| One system missing `equilibration/martini_eq.tpr` | listed in `missing_systems` for that stage only |
| `gmx` returns non-zero | `RuntimeError` with stderr; orchestrator marks the system as missing for that stage rather than aborting the whole audit |
| `gmx` not on PATH | `FileNotFoundError`; orchestrator marks all `equilibration` + `minimization` as missing; `run` audit still completes |
| MDP file with continuation comments mid-value | parser collapses to single line; whitespace-normalised |
| MDP file with a key declared twice | last value wins (matches GROMACS semantics) |
| Key with `;` inline comment (e.g. `tau_p = 12.0 ;parrinello-rahman is more stable...`) | inline comment stripped; value = `"12.0"` |
| Key with `=` in value (rare; e.g. PME tunings) | first `=` splits |

### C.7 Test plan — `tests/martini_pipeline/test_analysis_diff_mdps.py`

Eight test groups, all stdlib + pytest. `gmx` is mocked via a fake shell script written into `tmp_path` and prepended to `PATH`.

1. **`test_parse_mdp_file_*`** — parametrised over fixture mdps in `tests/martini_pipeline/fixtures/mdp/`:
   - simple key=value
   - inline comment after value
   - duplicate key (last wins)
   - whitespace-only / blank / comment-only lines ignored
   - hyphen vs underscore canonicalised to hyphen
2. **`test_dump_tpr_inputrec_with_mock_gmx`** — fixture: a tiny shell script `gmx` that prints a hand-crafted `inputrec:` block; assert parser extracts expected dict.
3. **`test_dump_tpr_inputrec_gmx_failure`** — mock `gmx` exits non-zero; assert `RuntimeError` with stderr in the message.
4. **`test_dump_tpr_inputrec_gmx_missing`** — point `gmx_binary` at a nonexistent path; assert `FileNotFoundError`.
5. **`test_diff_mdps_synthetic_tree`** — build a `tmp_path` tree with 3 fake systems each containing a `run.mdp` (two identical, one with one differing key); assert canonical = majority value, deviations include the dissenter.
6. **`test_diff_mdps_skips_seed_keys`** — three systems differ only on `gen-seed`; assert no deviations reported.
7. **`test_diff_mdps_missing_files`** — one system missing `run.mdp`; assert system listed in `missing_systems`, excluded from canonical computation.
8. **`test_diff_mdps_to_markdown_and_to_freeze_json`** — round-trip checks: markdown contains stage headers and key list; freeze JSON parses to `{stage: {key: value}}` shape and excludes `skip_keys`.
9. **`test_diff_mdps_legacy_integration`** (`@skipif(not _HAS_LEGACY_DATA)`, `@skipif(not shutil.which("gmx"))`) — runs `diff_mdps(data/membrane_only)` for real, asserts:
   - `n_systems == 70` for `run`
   - `deviations == ()` for `run` (verified during planning via md5)
   - canonical contains expected keys: `dt`, `nsteps`, `ref-t`, `ref-p`, `tau-t`, `tau-p`, `cutoff-scheme`, `coulombtype`, `vdw-type`
   - `dt == "0.02"`, `ref-t == "310"`, `cutoff-scheme == "Verlet"` (case-insensitive), `coulombtype == "cutoff"` (case-insensitive)
   - `equilibration` and `minimization` audits complete (n_systems == 70 each, possibly with 0 deviations once `skip_keys` are filtered)

### C.8 CLI driver — `scripts/simulation/audit_mdps.py`

```bash
python scripts/simulation/audit_mdps.py \
    --systems-root data/membrane_only \
    --output-md docs/mdp_audit_report.md \
    --output-freeze lipid_gnn/martini_pipeline/templates/_audit_freeze.json \
    [--gmx-binary gmx]
```

Uses `argparse`, calls `diff_mdps`, writes both outputs, and exits non-zero if any unexpected deviations are found (i.e. deviations involving keys *not* in `skip_keys`). Step 4 reads `_audit_freeze.json` to populate template defaults; step 4 *fails fast* if the file is missing, forcing the audit to be re-run if the underlying legacy mdps ever change.

### C.9 Layout & dependencies

- New files:
  - `lipid_gnn/martini_pipeline/analysis.py`
  - `tests/martini_pipeline/test_analysis_diff_mdps.py`
  - `tests/martini_pipeline/fixtures/mdp/` — small hand-crafted mdps for parser tests
  - `scripts/simulation/audit_mdps.py`
- Generated (committed):
  - `docs/mdp_audit_report.md`
  - `lipid_gnn/martini_pipeline/templates/_audit_freeze.json`
- No new entries in `requirements.txt`.
- No `config.yaml` changes; the `martini_pipeline:` block lands with step 4.
- `analysis.py` does not import from `composition.py` or `lipid_registry.py` — the audit is purely about mdp content, not chemistry.

### C.10 Acceptance criteria

- `pytest tests/martini_pipeline/test_analysis_diff_mdps.py -q` passes locally and on HPC. Legacy-integration test skips cleanly when `gmx` or legacy data are absent.
- `python scripts/simulation/audit_mdps.py --systems-root data/membrane_only` completes locally without error and produces both output files.
- `docs/mdp_audit_report.md`'s `run` stage shows zero unexpected deviations across the 70 systems (consistent with the planning-time md5 finding).
- `_audit_freeze.json` contains all three stages with non-empty canonical dicts.
- Module is < ~350 LOC; no comments unless a non-obvious WHY (per project style).
- Branch `feat/martini-pipeline-step-03-mdp-audit` merged into `main` via `--no-ff`; status table flipped to `[x]` and any divergence recorded.

### C.11 New decisions to log on completion

If implementation matches this plan, append three entries to §9:

- **Decision 11** — Equilibration/minimization mdps are recovered via `gmx dump -s` from `.tpr` files (legacy `.mdp` sources were not preserved). Rationale: `.tpr` is byte-stable and contains the full inputrec; reconstruction is deterministic.
- **Decision 12** — Random-seed-like keys (`gen-seed`, `ld-seed`, `tinit`, `init-step`) are excluded from deviation reporting. Rationale: per-system variation is intended; including them would drown out real findings.
- **Decision 13** — Step 4 (`mdp_writer.py`) reads `_audit_freeze.json` at template-build time and fails fast if missing. Rationale: enforces the audit-then-write order; prevents template defaults from drifting from the legacy values without an explicit re-audit.

---

## Appendix D — Step 4 detailed plan: `mdp_writer.py`

### D.1 Scope

`lipid_gnn/martini_pipeline/mdp_writer.py` produces the three MDP files required per simulation (`em.mdp`, `eq.mdp`, `run.mdp`) from the frozen audit values plus a small, well-defined set of run-time knobs. Pure stdlib; no imports from `composition.py` / `lipid_registry.py` (independence rule).

#### What's in scope for step 4

- A frozen `MDPParams` dataclass — the run-time knobs (per-stage `nsteps`, `save_forces`, optional `gen_seed`).
- `write_mdps(out_dir, *, params, freeze_path, templates_dir) -> dict[stage, path]` — emits all three files.
- Per-stage `render_mdp(stage, params, canonical, template_text) -> str` — pure string substitution.
- Templates: literal text files under `lipid_gnn/martini_pipeline/templates/{em,eq,run}.mdp.tmpl`, with `${name}` placeholders driven via `string.Template(...).substitute(...)` (strict mode — raises on unknown / missing keys).
- A `scripts/simulation/write_mdps.py` argparse driver for manual one-off generation.
- Tests at `tests/martini_pipeline/test_mdp_writer.py`.

#### What's out of scope for step 4

- Calling `gmx grompp` to validate templates (a `@skipif(not which("gmx"))` opt-in test does this; full pipeline grompp lives in step 7).
- The `martini_pipeline:` block in `config.yaml` — deferred to step 7 (Decision 17). Step 4 takes paths/knobs as arguments with package-relative defaults.
- Composition-dependent settings (per-lipid `tc-grps`, position restraints) — Martini production uses `System` group everywhere.
- Any reading of legacy data — step 4 consumes only the freeze record.

### D.2 Locked-in design decisions

1. **Two-source template strategy.**
   - `run.mdp.tmpl`: byte-exact clone of legacy `data/membrane_only/POPC100/run.mdp` (verified identical across all 70 systems in step 3) with `gen_seed`, `nsteps`, and `nstfout` lines replaced by `${...}` placeholders. Preserves Marrink-lab annotations as documentation.
   - `em.mdp.tmpl` and `eq.mdp.tmpl`: hand-authored from a curated mdp-key allowlist (D.4) populated from the freeze record. Required because `gmx dump -s` outputs inputrec fields that aren't valid `grompp` inputs (`mass-repartition-factor`, `ensemble-temperature-setting`, `nbfgscorr`, auto-derived fourier-grid dims, etc.). Authored once, reviewable in PR.
2. **Freeze JSON drives values; the key set per template is hardcoded in `mdp_writer.py`.** Mismatch (template references key, freeze missing it) raises at render time. Catches drift loudly.
3. **Equilibration deliberately diverges from legacy** (Decision 14):
   - `compressibility = 3e-4 3e-4` (legacy `3e-5`) — fixes the τ_p = 5 ps Berendsen mismatch that left novel compositions still drifting at 2.5 ns.
   - `nsteps = 1_000_000` (10 ns at dt = 0.01) — replaces the legacy 250 000 / 2.5 ns. Negligible cost on MI210; removes per-system inspection.
   - `nstenergy = 1000` (legacy 100) — I/O hygiene only; no stability impact.
   - `gen-vel = yes`, `gen-temp = 310`, `gen-seed = -1` made explicit so a fresh equilibration is deterministic regardless of what minimization wrote.
4. **Run stage values are byte-cloned from legacy.** No deliberate divergence — the legacy production setup is what produced existing training data and we keep new systems comparable.
5. **`save_forces` toggles `nstfout` in `run.mdp` only** (Decision 3). Default `0` (off); when `True`, set to `nstxout-compressed`'s value so position and force frequencies match. Em/eq never write forces.
6. **Seed strategy: per-call `random.SystemRandom().randint(1, 2**31 - 1)` when `params.gen_seed is None`; verbatim when set.** Two builds with all-defaults give different `run.mdp`s by design — reproducibility tests pass an explicit seed. The seed actually used is recorded by `manifest.py` in step 7.
7. **Fail-fast on missing freeze record** (Decision 13): `FileNotFoundError("MDP audit freeze record missing — run scripts/simulation/audit_mdps.py first.")`.
8. **`rlist` omitted from `eq.mdp.tmpl`.** Step 3 found 7 CHOL systems with rlist auto-tuned by GROMACS via `verlet-buffer-tolerance`. Reproducing that requires emitting the tolerance and letting `grompp` derive `rlist` per system. Confirmed safe — rlist is a tuned consequence of the buffer tolerance, not a primary input.
9. **Single-stage equilibration in v1** (Decision 16). Two-stage (eq1 small-dt + eq2 production-dt) deferred until step 7+ surfaces an actual failure on exotic compositions. The 10 ns single-stage already closes the legacy reliability gap.
10. **Configurable parameters are marked in template files** with inline `; [CONFIG: <knob>]` comments (Decision 15). Reviewer audits knob coverage by `grep '\[CONFIG:' templates/*.tmpl`. Resulting `.mdp` files preserve these comments — useful at-a-glance documentation in run dirs.
11. **No Jinja / no YAML.** `string.Template` is sufficient; one extra dependency for a half-dozen substitutions is overkill.

### D.3 Public API

```python
@dataclass(frozen=True)
class MDPParams:
    nsteps_min: int = 20_000           # legacy default
    nsteps_eq: int = 1_000_000         # 10 ns at dt = 0.01 — diverges from legacy 250 000 (Decision 14)
    nsteps_prod: int = -1              # -1 = run until walltime; per-run override expected
    save_forces: bool = False          # Decision 3
    gen_seed: int | None = None        # None → SystemRandom; explicit value → reproducible

STAGES: tuple[str, ...] = ("minimization", "equilibration", "run")

def write_mdps(
    out_dir: str | os.PathLike,
    *,
    params: MDPParams = MDPParams(),
    freeze_path: str | os.PathLike = _DEFAULT_FREEZE,
    templates_dir: str | os.PathLike = _DEFAULT_TEMPLATES,
) -> dict[str, str]:  # {stage: written_path}
    ...

def render_mdp(
    stage: str,
    params: MDPParams,
    canonical: Mapping[str, str],
    template_text: str,
) -> str: ...
```

`_DEFAULT_FREEZE` and `_DEFAULT_TEMPLATES` resolve relative to the package, so callers don't need to know the layout.

### D.4 Template key allowlists

**`em.mdp.tmpl`** — derived from `freeze.minimization`. ~15 keys:

`integrator, dt, nsteps, emtol, emstep, nstcomm, nstxout, nstvout, nstfout, nstlog, nstenergy, cutoff-scheme, nstlist, pbc, verlet-buffer-tolerance, coulombtype, coulomb-modifier, rcoulomb, epsilon-r, vdw-type, vdw-modifier, rvdw, tcoupl, pcoupl, constraints`.

Placeholder: `nsteps = ${nsteps_min}`.

**`eq.mdp.tmpl`** — derived from `freeze.equilibration` plus the Decision-14 overrides. ~30 keys:

The em set, plus `ref-t, tau-t, tc-grps, ref-p, tau-p, pcoupl=Berendsen, pcoupltype=semiisotropic, compressibility=3e-4 3e-4, gen-vel=yes, gen-temp=310, gen-seed=${gen_seed_eq}, refcoord-scaling=No, nstxout-compressed`.

Placeholders: `nsteps = ${nsteps_eq}`, `nstenergy = ${nstenergy_eq}` (default 1000), `gen-seed = ${gen_seed_eq}` (`-1` literal by default; an explicit `MDPParams.gen_seed` substitutes a deterministic int).

**`run.mdp.tmpl`** — clone of legacy `run.mdp`. Three lines edited:

- `gen_seed                 = ${gen_seed}`
- `nsteps                   = ${nsteps_prod}`
- `nstfout                  = ${nstfout}`

All Marrink-lab comment blocks preserved verbatim.

The exact key list per template is committed; future readers can diff template ↔ allowlist to see what was deliberately included/excluded.

### D.5 `[CONFIG:]` markers (Decision 15)

Every line driven by `MDPParams` gets an inline marker. Example excerpt from `eq.mdp.tmpl`:

```text
nsteps                   = ${nsteps_eq}            ; [CONFIG: nsteps_eq]
compressibility          = 3e-4 3e-4               ; [CONFIG: future — currently fixed]
gen-seed                 = ${gen_seed_eq}          ; [CONFIG: gen_seed]
nstenergy                = ${nstenergy_eq}         ; [CONFIG: nstenergy_eq]
```

Markers serve two purposes:

1. **Discoverability** — `grep '\[CONFIG:' templates/*.tmpl` enumerates the live knob surface.
2. **Future-extension hints** — values that *could* become knobs but currently aren't (e.g. `compressibility`, `ref-t`) are tagged `[CONFIG: future — currently fixed]`. Step 7 / step 11 can flip these to live knobs by adding fields to `MDPParams` without redesigning the template.

The markers survive into the rendered `.mdp` files (they're plain mdp comments). Run directories therefore self-document which lines are pipeline-controlled vs. force-field constants.

### D.6 Edge-case matrix

| Input | Expected |
| --- | --- |
| `freeze_path` missing | `FileNotFoundError` with re-run-audit message |
| Template references `${foo}` not in params | `KeyError` from `Template.substitute` (strict) |
| Allowlist key missing from freeze record | `KeyError` at render time naming key + stage |
| `params.save_forces=True` | `run.mdp` `nstfout` = `nstxout-compressed`'s value |
| `params.save_forces=False` | `run.mdp` `nstfout = 0` |
| `params.gen_seed=42` | `42` substituted verbatim into run.mdp |
| `params.gen_seed=None`, called twice | two distinct positive ints (system entropy) |
| `out_dir` doesn't exist | created (`os.makedirs(exist_ok=True)`) |
| `nsteps_prod=-1` | written verbatim (legacy default; production driven by walltime) |
| Re-run with same explicit seed + same out_dir | files byte-identical |
| Re-run with `gen_seed=None` | run.mdp differs in seed line only |

### D.7 Test plan — `tests/martini_pipeline/test_mdp_writer.py`

1. **`test_freeze_missing_raises`** — pointing `freeze_path` at a nonexistent file → `FileNotFoundError` with the audit-rerun hint in the message.
2. **`test_render_em_minimal`** — render with the committed freeze + default params; parse with `_parse_mdp_file` from `analysis.py`; assert every allowlist key is present and matches `freeze.minimization` (or the Decision-14 override).
3. **`test_render_eq_overrides_legacy`** — assert `compressibility == "3e-4 3e-4"`, `nsteps == "1000000"`, `nstenergy == "1000"`, `gen-vel == "yes"`, `gen-temp == "310"`, `pcoupl == "Berendsen"`, `pcoupltype == "semiisotropic"`. These are Decision-14 commitments and a regression here means we silently regressed to legacy.
4. **`test_render_run_clones_legacy`** — render with default params + a fixed seed; parse; assert resulting dict equals `freeze.run` for every shared key. Diff list must be exactly `{gen-seed, nsteps, nstfout}` plus whatever `${...}` placeholders we introduced.
5. **`test_save_forces_toggle`** — `False` → `nstfout=0`; `True` → `nstfout=75000` (== `nstxout-compressed`).
6. **`test_seed_explicit_and_random`** — explicit seed round-trips; `None` produces a positive int in [1, 2**31-1] and two consecutive calls give two different seeds.
7. **`test_write_mdps_roundtrip`** — `write_mdps(tmp_path, params)` writes 3 files; each re-parses cleanly; returned dict maps every stage to a path that exists.
8. **`test_template_strict_substitute`** — synthetic template with unsatisfied `${unknown}` → `KeyError` (not silent).
9. **`test_config_markers_present`** — assert every rendered `.mdp` contains at least one `; [CONFIG:` comment, and that every `MDPParams` field name appears as a `[CONFIG: <field>]` marker somewhere across the three templates. Catches divergence between knob set and template markers.
10. **`test_grompp_smoke`** (`@skipif(not shutil.which("gmx"))`, opt-in via `RUN_MDP_GROMPP=1`) — write all three files; run `gmx grompp` for each stage against `data/membrane_only/POPC100/{run.gro, topol.top, toppar/}` if present; assert exit 0 and zero notes related to unknown mdp options. Demonstrates that the curated allowlists are grompp-clean.

### D.8 CLI driver — `scripts/simulation/write_mdps.py`

```bash
python scripts/simulation/write_mdps.py \
    --out-dir <path> \
    [--nsteps-min 20000] \
    [--nsteps-eq 1000000] \
    [--nsteps-prod -1] \
    [--save-forces] \
    [--gen-seed 12345]
```

Calls `write_mdps`, prints the resulting paths. Exits non-zero on any rendering error.

### D.9 Layout & dependencies

New files:

- `lipid_gnn/martini_pipeline/mdp_writer.py` (~150 LOC)
- `lipid_gnn/martini_pipeline/templates/em.mdp.tmpl`
- `lipid_gnn/martini_pipeline/templates/eq.mdp.tmpl`
- `lipid_gnn/martini_pipeline/templates/run.mdp.tmpl`
- `tests/martini_pipeline/test_mdp_writer.py`
- `scripts/simulation/write_mdps.py`

No new entries in `requirements.txt`. No `config.yaml` changes (Decision 17 — deferred to step 7). No imports from `composition.py` / `lipid_registry.py`.

### D.10 Acceptance criteria

- `pytest tests/martini_pipeline/test_mdp_writer.py -q` passes locally and on HPC. The grompp smoke test skips cleanly when `gmx` or legacy data are absent.
- `python scripts/simulation/write_mdps.py --out-dir /tmp/mdptest` produces three files; each parses and contains the expected `[CONFIG: ...]` markers.
- Rendered `eq.mdp` matches the Decision-14 commitments exactly (compressibility, nsteps, nstenergy, gen-vel, gen-temp, pcoupl).
- Rendered `run.mdp` is byte-equal to legacy `data/membrane_only/POPC100/run.mdp` modulo the three placeholder lines.
- Module < ~200 LOC; no comments unless a non-obvious WHY. Templates committed and reviewable.
- No regressions in existing test suite (`pytest -q` total grows by exactly the new test count).
- Branch `feat/martini-pipeline-step-04-mdp-writer` merged into `main` via `--no-ff`; status table flipped to `[x]`.

### D.11 New decisions to log on completion

If implementation matches this plan, decisions 14–16 in §9 (already appended at planning time) stand as written. If implementation diverges, capture the actual decision under a fresh entry rather than rewriting these.

A 17th decision is implicit and worth recording on completion if confirmed in practice:

- **Decision 17** — `martini_pipeline:` config block in `config.yaml` is added in step 7 (when `pipeline.py` first reads it), not step 4. Rationale: smaller, more focused PRs; step 4 has no need for a global config since every knob is a function argument with a default.

---

## Appendix E — Step 5 detailed plan: vendor `insane.py`

### E.1 Scope

Place a versioned, GPL-clean copy of `insane.py` at `resources/martini3/insane.py`. Establishes the single source-of-truth for membrane construction so step 6 (`system_builder.py`) can call it deterministically across local + HPC + future machines regardless of what is on `PATH`.

#### What's in scope for step 5

- `resources/martini3/insane.py` — 2to3-converted, shebang updated, GPL header prepended.
- `resources/martini3/INSANE_PROVENANCE.md` — source path, retrieval date, 2to3 unified diff, license, attribution, parity-test outcome.
- `INSANE_PATH` constant in `lipid_gnn/martini_pipeline/__init__.py`.
- Smoke tests at `tests/martini_pipeline/test_insane_vendor.py`.
- "Vendored resources" section in `.claude/memory-bank/thesisStory.md`.

#### What's out of scope for step 5

- `system_builder.py` (step 6).
- Adding new lipid templates to `insane.py` (step 12 / subgoal 3).
- Cleanup of `lipid_gnn/functions_emil/insane.py` and other legacy copies — deferred, recorded as a future task.
- Modifying upstream logic beyond the 2to3 mechanical patch and shebang fix.

### E.2 Locked-in design decisions

1. **Vendor the 2to3-converted legacy copy (Option A)**, not Tsjerk's modern fork. Rationale: bit-parity with the Helgi/Emil-customised lipid templates that built the 70 training systems; 2to3 patch is mechanical and fully auditable. A future migration to Tsjerk's modern Python 3 fork (Option C) is explicitly noted as a deferred option.
2. **Mechanical 2to3 only** — `print X` → `print(X)`, `xrange` → `range`, `print >>fh, x` → `print(x, file=fh)`. No functional edits. Any edge case where 2to3 would change semantics is flagged in `INSANE_PROVENANCE.md`.
3. **Shebang** updated from `#!/usr/bin/env python` → `#!/usr/bin/env python3`. This is the only non-2to3 edit; recorded explicitly.
4. **GPL v2 header prepended** verbatim from Tsjerk Wassenaar's upstream (`github.com/Tsjerk/Insane`) with a credit line for Helgi I. Ingolfsson's lipid-template additions and Emil's customisations.
5. **Parity divergence is accepted and documented** (E.7.2 answer: option b). If rebuilding POPC100 with the vendored insane produces a different atom count or layout, the outcome is recorded in `INSANE_PROVENANCE.md` and step 7's sanity check is flagged accordingly.
6. **`INSANE_PATH` constant exposed from `lipid_gnn/martini_pipeline/__init__.py`** so step 6 (`system_builder.py`) does not recompute the path.
7. **Legacy copies kept as-is.** `lipid_gnn/functions_emil/insane.py`, `colab_lipid_gnn_subset/`, `build/lib/` are untouched. Cleanup relegated to a separate future task recorded in the memory bank.
8. **`resources/martini3/` is not a Python package.** No `__init__.py`. `system_builder.py` invokes insane via `subprocess.run([sys.executable, INSANE_PATH, ...])`.

### E.3 Procedure

1. Branch `feat/martini-pipeline-step-05-vendor-insane`.
2. `mkdir -p resources/martini3/`.
3. Copy `lipid_gnn/functions_emil/insane.py` → `resources/martini3/insane.py`.
4. Run `2to3 -w resources/martini3/insane.py`; inspect diff; remove the `insane.py.bak` artifact.
5. Update shebang line. `chmod +x resources/martini3/insane.py`.
6. Prepend GPLv2 header from upstream with attribution.
7. Write `resources/martini3/INSANE_PROVENANCE.md` (source, date, diff summary, license, parity outcome).
8. Smoke test: `python3 resources/martini3/insane.py --help` exits 0.
9. Parity check (if `gmx` available locally): rebuild POPC100 in `tmp/`, record atom/lipid counts vs. legacy `data/membrane_only/POPC100/run.gro`. Write outcome to `INSANE_PROVENANCE.md`.
10. Write `tests/martini_pipeline/test_insane_vendor.py` (five tests; one opt-in parity test).
11. Update `lipid_gnn/martini_pipeline/__init__.py` with `INSANE_PATH`.
12. Append "Vendored resources" to `.claude/memory-bank/thesisStory.md`.
13. Flip status table to `[x]`, append Decision 18, merge `--no-ff`, delete branch.

### E.4 Test plan — `tests/martini_pipeline/test_insane_vendor.py`

1. **`test_insane_path_exists`** — `os.path.isfile(INSANE_PATH)` and `os.access(..., os.X_OK)`.
2. **`test_insane_python3_parseable`** — `ast.parse(open(INSANE_PATH).read())` succeeds.
3. **`test_insane_help_exits_zero`** — `subprocess.run([sys.executable, INSANE_PATH, "--help"], timeout=20)` exits 0; output mentions `-l` and `-x`.
4. **`test_insane_version_marker`** — file contains `previous = "20140603.11.TAW"` so upstream drift is caught.
5. **`test_insane_path_importable`** — `from lipid_gnn.martini_pipeline import INSANE_PATH` works and path resolves to an existing file.
6. **`test_insane_parity_popc100`** (opt-in `RUN_INSANE_PARITY=1`, `@skipif(not _HAS_LEGACY_DATA)`) — rebuild POPC100 in `tmp_path`; assert lipid count and total atom count match legacy ± 0.

### E.5 Acceptance criteria

- `pytest tests/martini_pipeline/test_insane_vendor.py -q` passes locally and on HPC (parity test skips cleanly when data absent).
- `python3 resources/martini3/insane.py --help` exits 0.
- `INSANE_PROVENANCE.md` documents source, 2to3 diff, license, parity outcome.
- `from lipid_gnn.martini_pipeline import INSANE_PATH` works.
- No regressions.

### E.6 New decisions to log on completion

- **Decision 18** — Vendored `insane.py` is the 2to3-converted Helgi/Emil-customised build of Tsjerk Wassenaar's 2014-06-03 insane (not the modern Python-3 fork). A future migration to Option C (both versions side-by-side) is explicitly possible when Step 12 extends the lipid pool. Rationale: parity with existing 70-system lipid templates; mechanical 2to3 patch is auditable; no functional changes.

---

## Appendix F — Step 6 detailed plan: `system_builder.py`

### F.1 Scope

Build the initial bilayer `.gro` + finalised `topol.top` for a given composition spec. Wraps `insane.py` (vendored in step 5), stages the 9 Martini 3 ITP files alongside it, and produces a self-contained build directory that `pipeline.py` (step 7) can hand directly to `gmx grompp`. Also generates `index.ndx` via `gmx make_ndx`.

#### What's in scope for step 6

- Sub-step 6a: vendor the 9 Martini 3 ITP files to `resources/martini3/itp/` (copied from `data/membrane_only/POPC100/toppar/`, minus the two yire1 protein ITPs).
- `MARTINI3_ITP_DIR` constant in `lipid_gnn/martini_pipeline/__init__.py`.
- `lipid_gnn/martini_pipeline/system_builder.py`:
  - `BoxParams` frozen dataclass — box geometry + solvent/ion settings.
  - `BuildResult` frozen dataclass — paths + parsed statistics from the completed build.
  - `build_command()` — pure function returning the `insane.py` argv list; unit-testable without touching the filesystem.
  - `build_system()` — runs insane, finalises topology, stages ITPs, generates index.ndx.
- `tests/martini_pipeline/test_system_builder.py` (≥ 10 tests, insane mocked via a fake script).

#### What's out of scope for step 6

- `pipeline.py` orchestration (step 7).
- `gmx grompp` / `gmx mdrun` invocations.
- HPC submission layer (steps 9–10).
- Adding lipid templates to `insane.py` (step 12).

### F.2 Locked-in design decisions

1. **Self-contained build directories** (F.10 answer 1a). Each `out_dir` contains: `run.gro`, `topol.top`, `toppar/` (9 ITP copies), `insane.log`, `index.ndx`. No symlinks, no shared ITP pool. Note: a shared `toppar/` pool (option b) would be more space-efficient and may be worth revisiting when Step 12 adds many new lipids.
2. **ITP source: copy from `data/membrane_only/POPC100/toppar/`** (F.10 answer 2a). These 9 files are known-good for the current 10-lipid pool; provenance recorded in F.2 note below. Note: if new lipids require ITPs not covered by these files, a fresh download from the Marrink group (option b) becomes necessary — flagged for Step 12.
3. **`molecule_counts` parsed from finalised `topol.top`** (F.10 answer 3a). The `[ molecules ]` section is the authoritative count after insane runs; no count is maintained in-memory alongside.
4. **`index.ndx` is generated** (F.10 answer 4). `gmx make_ndx -f run.gro -o index.ndx` run immediately after insane exits 0. If `gmx` is absent, the step is skipped with a warning (index.ndx is optional for `gmx grompp` but useful for analysis and step 7's sanity checks).
5. **`LipidEntry.insane_keyword` used as `-l` token** (F.10 answer 5). Allows lipids whose insane identifier differs from their registry name (e.g. future additions) without special-casing.
6. **Topology finalisation replaces the insane-generated `#include "martini.itp"` line** with the 9 Martini 3 include lines (`#include "toppar/<itp>"` for each of the 9 files). All other lines are preserved verbatim.

### F.3 Public API — `system_builder.py`

```python
@dataclass(frozen=True)
class BoxParams:
    xy_nm: float = 11.0
    z_nm: float = 10.0
    salt_M: float = 0.15
    water_type: str = "W"
    charge_mode: str = "auto"
    center: bool = True
    pbc: str = "rectangular"

@dataclass(frozen=True)
class BuildResult:
    out_dir: str
    gro_path: str          # <out_dir>/run.gro
    top_path: str          # <out_dir>/topol.top
    ndx_path: str | None   # <out_dir>/index.ndx  (None if gmx absent)
    log_path: str          # <out_dir>/insane.log
    molecule_counts: dict[str, int]   # from [ molecules ] section
    n_membrane_beads: int  # sum of lipid-residue atom counts (parsed from gro header)
    n_solvent_atoms: int   # total_atoms - n_membrane_beads
    total_atoms: int       # second line of .gro file
    walltime_s: float      # time insane subprocess took

def build_command(
    composition: dict[str, float],
    box: BoxParams,
    out_gro: str,
    out_top: str,
) -> list[str]:
    """Return the argv list for insane.py. Pure function."""
    ...

def build_system(
    composition: dict[str, float],
    out_dir: str,
    *,
    box: BoxParams = BoxParams(),
    insane_path: str = INSANE_PATH,
    itp_dir: str = MARTINI3_ITP_DIR,
    gmx_executable: str = "gmx",
) -> BuildResult:
    """Build bilayer, finalise topology, stage ITPs, generate index. Raises on non-zero exit."""
    ...
```

`composition` maps registry lipid names → mol fractions (need not sum to 1; internally converted to insane integer ratios matching `build_command`'s expected form).

### F.4 `build_system()` procedure

1. `os.makedirs(out_dir, exist_ok=True)`.
2. Run `build_command()` → `subprocess.run(..., capture_output=True, text=True)`. Write stdout+stderr to `insane.log`. Raise `RuntimeError` on non-zero exit.
3. **Topology finalisation**: read `topol.top`; replace any line matching `#include "martini*.itp"` with the 9 `#include "toppar/<itp>"` lines; write back in place.
4. **ITP staging**: `shutil.copy(itp_dir/<itp>, out_dir/toppar/<itp>)` for each of the 9 files.
5. **index.ndx**: `shutil.which(gmx_executable)` — if found, run `gmx make_ndx -f <gro> -o <ndx> << "q\n"` (stdin `q` exits make_ndx immediately, writing the default index groups). Skip + warn if not found.
6. **Parse stats**: read `.gro` line 2 for `total_atoms`; parse `[ molecules ]` from `topol.top` for `molecule_counts`; compute `n_membrane_beads` and `n_solvent_atoms` from counts × bead-counts per residue (obtained from lipid registry where possible; otherwise fall back to gro-residue scan).
7. Return `BuildResult`.

### F.5 Topology finalisation detail

The line to replace is any line of the form (case-insensitive, ignoring surrounding whitespace):

```text
#include "martini*.itp"
```

Replace with the 9 ordered lines (matching the legacy topol.top include order):

```text
#include "toppar/martini_v3.0.0.itp"
#include "toppar/martini_v3.0.0_ions_v1.itp"
#include "toppar/martini_v3.0.0_nucleobases_v1.itp"
#include "toppar/martini_v3.0.0_phospholipids_v1.itp"
#include "toppar/martini_v3.0.0_phospholipids_v1_matthieu.itp"
#include "toppar/martini_v3.0.0_small_molecules_v1.itp"
#include "toppar/martini_v3.0.0_solvents_v1.itp"
#include "toppar/martini_v3.0.0_sugars_v1.itp"
#include "toppar/martini_v3.0_sterols_v1.0.itp"
```

If the pattern matches zero or more than one line, raise `ValueError` with a diagnostic message (zero matches → insane output format changed; > 1 → unexpected multi-include).

### F.6 `build_command()` logic

Mol fractions are converted to insane integer ratios: multiply all fractions by a common scale so ratios are positive integers (e.g. `{POPC: 0.7, DOPC: 0.3}` → `POPC:70 DOPC:30`; use `round()` then check sum is preserved). `-l` flags use `LipidEntry.insane_keyword` from the registry. If a lipid is not in the registry, fall back to the name itself (forward-compat with `register_lipid`).

Full flag list:

```text
sys.executable  INSANE_PATH
-o  <gro>  -p  <top>
-x  <xy>  -y  <xy>  -z  <z>
-l  <kw1>:<r1>  [-l  <kw2>:<r2>  ...]
-sol  <water_type>
-salt  <salt_M>
-charge  <charge_mode>
[-center]  (if box.center)
[-pbc  <pbc>]  (if box.pbc != "rectangular")
```

### F.7 Edge-case matrix

| Case | Expected behaviour |
| --- | --- |
| insane exits non-zero | `RuntimeError` with last 500 chars of stderr |
| `martini.itp` pattern not found in topol.top | `ValueError("topology finalisation: expected exactly 1 martini include …")` |
| ITP source file missing | `FileNotFoundError` before insane runs (preflight check) |
| `gmx` absent | index.ndx skipped; `ndx_path = None`; warning logged to stdout |
| `out_dir` already exists | `exist_ok=True`; existing `run.gro` overwritten |
| Composition fractions do not sum to 1 | Normalise silently (insane accepts ratios, not absolute fractions) |
| Single-lipid composition | Works; `-l POPC:100` |

### F.8 Test plan — `tests/martini_pipeline/test_system_builder.py`

All tests use a **fake insane script** (`tmp_path/fake_insane.py`) that writes a minimal `.gro` (5 atoms) and a minimal `topol.top` (with the single `#include "martini.itp"` line + `[ system ]` + `[ molecules ]` section) and exits 0.

1. **`test_build_command_single_lipid`** — `build_command({"POPC": 1.0}, BoxParams(), ...)` → argv contains `"-l", "POPC:100"` and the four box flags.
2. **`test_build_command_binary_mixture`** — `{"POPC": 0.7, "DOPC": 0.3}` → two `-l` flags with integer ratios summing to 100.
3. **`test_build_command_center_flag`** — `BoxParams(center=True)` → `"-center"` in argv; `center=False` → not present.
4. **`test_build_command_pbc_nondefault`** — `BoxParams(pbc="hexagonal")` → `"-pbc", "hexagonal"` in argv.
5. **`test_build_system_creates_gro`** — full `build_system()` with fake insane + fake ITP dir; assert `BuildResult.gro_path` exists.
6. **`test_build_system_topology_finalised`** — read written `topol.top`; assert it contains `#include "toppar/martini_v3.0.0.itp"` and does **not** contain `#include "martini.itp"`.
7. **`test_build_system_itps_staged`** — assert `<out>/toppar/martini_v3.0.0.itp` exists.
8. **`test_build_system_log_written`** — assert `insane.log` exists and contains fake insane's stdout.
9. **`test_build_system_molecule_counts`** — fake topol.top has `POPC 196` + `W 5305`; assert `result.molecule_counts == {"POPC": 196, "W": 5305}`.
10. **`test_build_system_insane_failure`** — fake insane exits 1; assert `RuntimeError` raised.
11. **`test_build_system_no_martini_include`** — fake topol.top has no `#include "martini.itp"` line; assert `ValueError` raised.
12. **`test_build_system_no_gmx_skips_ndx`** — pass `gmx_executable="gmx_does_not_exist_XYZ"`; assert `result.ndx_path is None` and no exception.

### F.9 Layout & dependencies

```text
resources/martini3/itp/           ← 9 Martini 3 ITP files (sub-step 6a)
lipid_gnn/martini_pipeline/
    __init__.py                   ← add MARTINI3_ITP_DIR
    system_builder.py             ← new
tests/martini_pipeline/
    test_system_builder.py        ← new
```

No new external dependencies. Imports: `os`, `re`, `shutil`, `subprocess`, `sys`, `time`, `dataclasses`, `lipid_gnn.martini_pipeline` (`INSANE_PATH`, `MARTINI3_ITP_DIR`), `lipid_gnn.martini_pipeline.lipid_registry` (`LIPID_REGISTRY`).

### F.10 Open questions (answered 2026-05-08)

1. **Build directory strategy** → **a) self-contained**. Note: shared toppar pool (option b) may be worth revisiting at Step 12 when many new lipids are added.
2. **ITP source** → **a) copy from `data/membrane_only/POPC100/toppar/`**. Note: if new lipids require ITPs not in this set, switch to a fresh Marrink download (option b) at Step 12.
3. **`molecule_counts` source** → **a) parse finalised `topol.top`**.
4. **index.ndx** → **YES — generate via `gmx make_ndx`**; skip with warning if `gmx` absent.
5. **`-l` token** → **`LipidEntry.insane_keyword`** from registry.

### F.11 New decisions to log on completion

- **Decision 19** — ITP files sourced from `data/membrane_only/POPC100/toppar/` (the 9 Martini 3 files, minus the two yire1 protein ITPs). These are the same files that produced the 70 training systems. A future switch to a fresh Marrink download is flagged for Step 12 if new lipids require ITPs not covered by this set.
- **Decision 20** — Build directories are self-contained: each `out_dir` receives its own `toppar/` copy of the 9 ITPs. Avoids relative-path fragility and allows each build dir to be moved or archived independently. Shared-pool alternative deferred to Step 12.
- **Decision 21** — `LipidEntry.insane_keyword` (not `.name`) used as the `-l` token. Allows future lipids whose insane identifier differs from their registry name without special-casing in `system_builder.py`.

---

## Appendix G — Step 7 detailed plan: `pipeline.py` + `manifest.py`

### G.1 Scope

End-to-end orchestration of a single composition: `system_builder` → write mdps → `gmx grompp + mdrun` for minimization → equilibration → production. Idempotent (skips stages whose handoff output exists). Per-system JSON manifest records spec, mdp hashes, gmx version, seeds, durations, status, and the insane invocation. Gated on a DIPC100 sanity check against absolute physical criteria (APL in published Martini 3 range, no blow-up) — not a legacy byte-comparison (insane version and ITP parameters differ from legacy; POPC100 was manually restarted and is an outlier).

#### What's in scope for step 7

- `lipid_gnn/martini_pipeline/pipeline.py`
- `lipid_gnn/martini_pipeline/manifest.py`
- `martini_pipeline:` block in `config.yaml` + parser in `lipid_gnn/config.py` (Decision 17)
- `scripts/simulation/run_martini_pipeline.py` — single-composition CLI driver
- `scripts/simulation/sanity_check_dipc100.py` — APL physical-criteria comparator
- `tests/martini_pipeline/test_pipeline.py` — mocked `gmx`
- `tests/martini_pipeline/test_manifest.py`
- `tests/martini_pipeline/test_e2e_smoke.py` — opt-in `RUN_MARTINI_E2E=1`

#### What's out of scope for step 7

- HPC submission (step 9), benchmark (step 10), corner-fill production (step 11).
- `analysis.py::missing_compositions` (step 8).
- Multi-system orchestration: the pipeline runs one composition; batching is a layer above.
- New lipids (step 12).

### G.2 Locked-in design decisions

1. **One composition per invocation.** `pipeline.run(composition, out_dir, ...)` builds + simulates a single system. Multi-system batching is the submission layer's job (step 9). Keeps `pipeline.py` linear and testable.
2. **Per-stage subdirectories**, matching legacy layout exactly: `<out>/{minimization,equilibration,run}/`. Initial bilayer + finalised topology + index.ndx live at `<out>/`; per-stage mdp + tpr + trajectory + log live inside the stage dir. Downstream graph/dataset code keying off this layout (`data/membrane_only/<comp>/run/prun.xtc`) continues to work unchanged when pointed at `data/martini_pipeline/`.
3. **Stage filename convention matches legacy** (proposed; G.10 Q1): `martini_em.{tpr,gro,trr,edr,log}` + `minimized.gro` handoff in `minimization/`; `martini_eq.{tpr,gro,xtc,edr,log,cpt}` + `equilibrated.gro` handoff in `equilibration/`; `prun.{tpr,gro,xtc,edr,log,cpt}` in `run/`. Rationale: pure parity; downstream code references these names today.
4. **Idempotency via handoff files.** Stage is skipped iff its handoff file exists: minimization → `minimized.gro`; equilibration → `equilibrated.gro`; production → `prun.gro`. Detect-and-skip is logged in the manifest under `stage_<name>.status = "skipped"`. Rationale: handoff `.gro` is written *after* `gmx mdrun` exits zero, so its presence is a strong success marker.
5. **`pipeline.py` calls `gmx` directly via `subprocess`** (Decision 1). No wrapper layer.
6. **Seed strategy: deterministic by default.** `gen_seed` derived from a stable hash of the composition name (e.g. `int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)`); CLI flag `--seed N` overrides; `--seed random` requests a fresh random seed. Same seed re-used across all stages (insane RNG, em, eq, prod) of one composition. Rationale: makes the canonical `<comp>` name reproducible without making seeds part of the directory name. (Closes one of the step-7 open questions in §10.)
7. **MDP placement: per-stage subdirs.** `<out>/minimization/em.mdp`, `<out>/equilibration/eq.mdp`, `<out>/run/run.mdp`. Each subdir is self-contained (mdp + tpr + outputs). Rationale: cleaner than scattering 3 mdps at the root; legacy had only 1 surviving mdp at root so no precedent is being broken.
8. **No retry logic.** First failure → write manifest with `status: "failed_at_<stage>"`, raise; user reruns (or HPC step 9 wrapper retries). Decision 1's "reintroduce wrapper only if retry accumulates" still holds.
9. **`config.yaml` block added now (Decision 17).** Frozen dataclass `MartiniPipelineConfig` with output_root, mdp/insane/itp paths, system defaults (box, salt, temp), run defaults (`nsteps_*`, `save_forces`), seed strategy. CLI flags on `run_martini_pipeline.py` override per-run.

### G.3 Public API — `pipeline.py`

```python
@dataclass(frozen=True)
class StageResult:
    name: str                # "minimization" | "equilibration" | "production"
    status: str              # "ok" | "skipped" | "failed"
    walltime_s: float
    grompp_cmd: list[str]
    mdrun_cmd: list[str]
    tpr_path: str
    final_gro_path: str
    log_path: str
    error: str | None = None

@dataclass(frozen=True)
class PipelineResult:
    composition: dict[str, float]
    out_dir: str
    build: BuildResult                # from system_builder
    stages: tuple[StageResult, ...]   # in execution order
    manifest_path: str
    overall_status: str               # "ok" | "failed_at_<stage>"

def run(
    composition: dict[str, float],
    out_dir: str,
    *,
    box: BoxParams = BoxParams(),
    mdp_params: MDPParams = MDPParams(),
    seed: int | None = None,          # None → derived from composition name hash
    gmx_executable: str = "gmx",
    mdrun_extra_args: tuple[str, ...] = (),  # e.g. ("-ntomp", "4")
    force_rerun: bool = False,        # disables idempotency check
) -> PipelineResult: ...
```

Stage helpers (private):

```python
def _run_grompp(stage_dir, mdp, gro_in, top, ndx, tpr_out, gmx) -> tuple[list[str], float]: ...
def _run_mdrun(stage_dir, deffnm, gmx, extra_args) -> tuple[list[str], float]: ...
def _stage_minimize(out_dir, mdp_path, ndx_path, top_path, gro_in, gmx, extra) -> StageResult: ...
def _stage_equilibrate(...) -> StageResult: ...
def _stage_produce(...) -> StageResult: ...
def _derive_seed(composition_name: str) -> int: ...
```

### G.4 Public API — `manifest.py`

```python
@dataclass(frozen=True)
class Manifest:
    schema_version: str          # "1.0"
    composition: dict[str, float]
    canonical_name: str          # from composition.canonicalise
    out_dir: str
    created_utc: str             # ISO-8601
    gmx_version: str | None      # parsed from `gmx --version` (None if absent)
    insane_version: str          # parsed from INSANE_PATH header ("20140603.11.TAW")
    insane_cmd: list[str]        # actual argv used
    seed: int
    box: dict                    # BoxParams as dict
    mdp_params: dict             # MDPParams as dict
    mdp_hashes: dict[str, str]   # {"em.mdp": "sha256:...", "eq.mdp": ..., "run.mdp": ...}
    stages: list[dict]           # serialised StageResult per stage
    build_stats: dict            # molecule_counts, total_atoms, n_membrane_beads
    overall_status: str
    host: dict                   # {"hostname": ..., "platform": ..., "python": ...}

def write_manifest(path: str, m: Manifest) -> None: ...
def read_manifest(path: str) -> Manifest: ...
def hash_file(path: str) -> str: ...          # "sha256:<hex>"
def detect_gmx_version(gmx_exe: str) -> str | None: ...
def detect_insane_version(insane_path: str) -> str: ...
```

### G.5 Per-stage procedure

For each stage, in order, with handoff detection:

| Stage | Input gro | MDP | Output deffnm | Handoff |
| --- | --- | --- | --- | --- |
| minimization | `<out>/run.gro` | `minimization/em.mdp` | `martini_em` | `minimization/minimized.gro` |
| equilibration | `minimization/minimized.gro` | `equilibration/eq.mdp` | `martini_eq` | `equilibration/equilibrated.gro` |
| production | `equilibration/equilibrated.gro` | `run/run.mdp` | `prun` | `run/prun.gro` |

Per-stage steps:

1. If `handoff` exists and not `force_rerun` → return `StageResult(status="skipped")`.
2. `gmx grompp -f <mdp> -c <gro_in> -p <top> -n <ndx> -o <stage>/<deffnm>.tpr -maxwarn 1` (capture stdout/stderr → `<stage>/grompp.log`).
3. `gmx mdrun -deffnm <stage>/<deffnm> [+extra]` (stdout/stderr → `<stage>/mdrun.log`; GROMACS also writes its own `<deffnm>.log`).
4. `shutil.copy(<stage>/<deffnm>.gro, <stage>/<handoff>)`.
5. Return `StageResult(status="ok")`.

On any non-zero exit: capture last 500 chars of stderr, write to `error`, return `status="failed"`. `pipeline.run` then writes manifest and raises.

### G.6 Manifest write timing

Written **after every stage transition** (not only at the end), so a crashed/killed run still leaves a manifest reflecting the last completed stage. Use `json.dumps(..., indent=2)` for readability; atomic write via tempfile + rename.

### G.7 `config.yaml` block (Decision 17)

```yaml
martini_pipeline:
  output_root: data/martini_pipeline
  insane_path: resources/martini3/insane.py
  itp_dir: resources/martini3/itp
  mdp_templates_dir: lipid_gnn/martini_pipeline/templates
  mdp_freeze: lipid_gnn/martini_pipeline/templates/_audit_freeze.json

  box:
    xy_nm: 11.0
    z_nm: 10.0
    salt_M: 0.15
    water_type: W
    charge_mode: auto
    center: true
    pbc: rectangular

  run:
    nsteps_min: 20000
    nsteps_eq: 1000000        # 10 ns at dt = 0.01 (Decision 14)
    nsteps_prod: -1           # -1 → caller must supply via CLI
    nstenergy_eq: 1000
    save_forces: false        # Decision 3
    seed_strategy: deterministic   # "deterministic" | "random" | <int>

  gmx:
    executable: gmx
    mdrun_extra_args: []
```

Parser: `MartiniPipelineConfig` frozen dataclass in `lipid_gnn/config.py`. Backwards compatible: if the block is absent, defaults apply.

### G.8 Edge-case matrix

| Case | Expected behaviour |
| --- | --- |
| `gmx` absent | `pipeline.run` raises `FileNotFoundError("gmx not found on PATH")` immediately (preflight); test suite still passes because `gmx` is mocked in unit tests |
| handoff file exists, `force_rerun=False` | Stage status `"skipped"`; subsequent stages still run normally |
| `force_rerun=True` | All stages re-execute; tpr/gro/etc. overwritten |
| grompp `-maxwarn` exceeded | grompp returns non-zero → `status="failed"`, manifest written, exception raised |
| mdrun killed externally (SIGKILL) | Handoff gro absent → on rerun, stage re-executes from scratch (no checkpoint-restart in v1; G.10 Q4 deferred) |
| `out_dir` already contains a partial run | Same as above: missing handoff → re-execute |
| Composition not in registry | `KeyError` raised by `system_builder.build_command` (lipid keyword fallback to name still works but ITP-bead check would fail later) |
| `nsteps_prod = -1` (config sentinel) | CLI driver requires explicit `--prod-ns N`; raises if omitted |

### G.9 Test plan

**`test_pipeline.py`** — `gmx` mocked as a fake binary on PATH (`tmp_path/bin/gmx`) that writes minimal stage outputs and exits 0.

1. `test_run_creates_per_stage_dirs` — invoke `run`; assert `minimization/`, `equilibration/`, `run/` exist.
2. `test_run_writes_tpr_per_stage` — assert `martini_em.tpr`, `martini_eq.tpr`, `prun.tpr` exist.
3. `test_run_writes_handoff_files` — `minimized.gro`, `equilibrated.gro`, `prun.gro` exist.
4. `test_grompp_invocations` — inspect captured argv: each stage calls `gmx grompp -f <mdp> -c <gro_in> -p <top> -n <ndx> -o <tpr>`.
5. `test_mdrun_invocations` — each stage calls `gmx mdrun -deffnm <...>`; extra args propagate.
6. `test_idempotency_skips_completed_stage` — pre-touch `minimization/minimized.gro`; assert minimization stage is `"skipped"` and grompp not called for it.
7. `test_force_rerun_overrides_idempotency` — same setup, `force_rerun=True`; assert minimization re-runs.
8. `test_failure_at_equilibration_writes_manifest` — fake `gmx` exits non-zero on eq stage; assert manifest written with `overall_status == "failed_at_equilibration"` and exception raised.
9. `test_seed_deterministic` — same composition → same seed across two `run()` invocations (without seed override).
10. `test_seed_override` — `seed=12345` → that seed appears in manifest and in mdp `gen_seed`.

**`test_manifest.py`**

1. `test_manifest_round_trip` — write then read, identity preserved.
2. `test_mdp_hash_changes_with_content` — modify one byte → hash differs.
3. `test_detect_insane_version` — parses version from `importlib.metadata.version("insane")`.
4. `test_detect_gmx_version_absent` — non-existent executable → returns `None`, no raise.
5. `test_manifest_schema_fields` — required fields all present in the JSON output.

**`test_e2e_smoke.py`** — opt-in `RUN_MARTINI_E2E=1`. Runs DIPC100 with `nsteps_prod=5000` (50 ps). Asserts:

1. `prun.xtc` exists, has ≥ 2 frames.
2. `manifest.json` parses, `overall_status == "ok"`.
3. Mean area per lipid (computed via MDAnalysis) is within `[0.55, 0.80] nm²` (DIPC Martini 3 literature ~0.68 nm²; wide tolerance for short run).
4. No NaN in `prun.edr` energy terms (system did not blow up).

**`scripts/simulation/sanity_check_dipc100.py`** (run manually, not in pytest):

- Runs DIPC100 end-to-end with `nsteps_prod = 5_000_000` (50 ns).
- Computes mean APL from `prun.xtc` (MDAnalysis, last 10 ns averaged).
- Checks against absolute physical criteria: APL ∈ [0.62, 0.75] nm², bilayer thickness ∈ [3.5, 4.5] nm, no energy blow-up.
- Writes a one-line PASS/FAIL summary plus a JSON metrics file.
- **Not a legacy-comparison**: insane version and ITP parameters differ from legacy 70 systems; POPC100 legacy is an outlier (manually restarted, 2× nsteps).

### G.10 Open questions (need user input before implementation)

1. **Stage filename convention.** Match legacy (`martini_em`, `martini_eq`, `prun`) — proposed in G.2.3 — or rationalise to `em`, `eq`, `prod` and update downstream readers? Recommend match-legacy for least churn; rationalise only if downstream code is already being touched.
2. **`index.ndx` parity.** Legacy `index.ndx` was built with custom groups (`name 18 Membrane`, `name 17 Solute`). Step 6 currently uses default groups only (`q\n`). Options:
   - (a) Extend `system_builder` to take a `make_ndx_script` argument and pass the legacy script for parity.
   - (b) Add the custom groups in `pipeline.py` before grompp (separate `_finalise_ndx()` step).
   - (c) Leave default groups; verify mdps don't reference custom group names. If they don't, the parity gap is cosmetic.
   Need confirmation that nothing in the equilibration/production mdps (or freeze record) references `Membrane`/`Solute` groups. If so, recommend (a).
3. **MDP `-maxwarn` value.** Legacy may have used `-maxwarn 1`; some Martini 3 grompp warnings are benign. Use `-maxwarn 1` by default, override via CLI?
4. **Production checkpoint-restart.** Legacy `prun_prev.cpt` suggests `gmx mdrun -cpi` was used. For step 7 — defer (no checkpoint logic; failed prod restarts from scratch) or implement now? Recommend defer; HPC step 9 is the right place for checkpoint logic.
5. **`nsteps_prod` default.** `MDPParams.nsteps_prod = -1` (sentinel). Should the pipeline require an explicit value (raise on `-1`) or apply a sensible default (e.g. legacy's 50 ns = 5 000 000 steps)? Recommend require-explicit; the production length is the most consequential knob.
6. **Sanity check tolerance.** APL ±5 % is suggested; legacy POPC100 mean APL ≈ 0.65 nm². If the +37-atom solvent divergence (Decision 18) shifts pressure coupling slightly, APL may drift > 5 %. Should the tolerance be wider? Recommend start at ±5 %, widen if step 7's first POPC100 run fails it.
7. **CLI driver argument shape.** `run_martini_pipeline.py POPC:1.0` (insane-style ratio string) vs `--composition '{"POPC": 1.0}'` (JSON) vs `--composition-file composition.yaml`? Recommend insane-style ratio string for ergonomics; JSON/YAML for programmatic callers.

### G.11 Layout & dependencies

```text
lipid_gnn/martini_pipeline/
    pipeline.py                  ← new (~250 LOC)
    manifest.py                  ← new (~120 LOC)
lipid_gnn/
    config.py                    ← extend with MartiniPipelineConfig dataclass
config.yaml                      ← add martini_pipeline: block (Decision 17)
scripts/simulation/
    run_martini_pipeline.py      ← new CLI driver
    sanity_check_popc100.py      ← new manual-run sanity comparator
tests/martini_pipeline/
    test_pipeline.py             ← new (≥10 tests, gmx mocked)
    test_manifest.py             ← new (≥5 tests)
    test_e2e_smoke.py            ← new (opt-in)
```

No new external dependencies. Internal imports: `system_builder` (build), `mdp_writer` (write_mdps), `composition` (canonical_name), `lipid_registry` (lipid validation), `lipid_gnn.config` (CONFIG).

### G.12 Acceptance criteria

- `pytest tests/martini_pipeline/ -q` passes locally and on HPC.
- `run_martini_pipeline.py POPC:1.0 --prod-ns 0.05` (50 ps) completes end-to-end when `gmx` is available, producing `<out>/POPC100/{run.gro, topol.top, index.ndx, toppar/, minimization/, equilibration/, run/, manifest.json}`.
- Manifest schema validated by `test_manifest.py`.
- Idempotency: rerunning `run_martini_pipeline.py POPC:1.0 ...` skips all three stages.
- POPC100 50-ns sanity check (`sanity_check_popc100.py`) reports APL within ±5 % of legacy and frame count == legacy.
- Step status table row 7 flipped to `[x]`.

### G.13 New decisions to log on completion

- **Decision 22** — `pipeline.py` orchestrates one composition; multi-system batching is the submission layer's job (step 9). Rationale: keeps the orchestrator linear and testable; HPC step 9 already plans to wrap N parallel invocations.
- **Decision 23** — Stage handoff via per-stage `.gro` copy (`minimized.gro`, `equilibrated.gro`). Idempotency markers are these handoff files. Rationale: handoff `.gro` is written *after* `mdrun` exits zero, so its presence is a strong success marker; no separate `.done` sentinel needed.
- **Decision 24** — Seed is derived deterministically from composition name by default (`sha256(name)[:8]`); CLI overrides. Same seed used across insane, em, eq, prod for one composition. Rationale: reproducible without making seeds part of the canonical `<comp>` name; closes step-7 open question on seed strategy from §10.
- **Decision 25** — Manifest is rewritten after every stage transition (not only at the end). Rationale: a killed run still leaves a useful manifest.
- *Plus any decisions arising from G.10 answers (filenames, ndx parity, maxwarn, checkpoint, nsteps_prod default, tolerance, CLI shape).*
