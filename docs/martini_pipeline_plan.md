# Martini 3 Lipid Simulation Pipeline — Plan & Tracker

Long-term, general-purpose Martini 3 membrane simulation pipeline. Stands as a research deliverable in its own right; newly simulated systems are not necessarily training data. This document is the single source of truth for the plan, progress, and decisions.

Last updated: 2026-05-14 (step 10c complete — production routing to general1).

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
| 8 | `analysis.py::missing_compositions()` + CLI driver to print DPPC/DOPC corner work queue | [x] | Decisions 33–36. Grid generators: `binary_grid`, `ternary_grid` (full symmetric simplex), `dppc_corner_grid`, `dopc_corner_grid`. CHOL capped at 40%. CLI `print_work_queue.py` with `--grid`, `--format`, `--out`. 451 tests pass (32 new). |
| 9 | HPC submission layer (`submit_simulations.sh` + `sbatch_simulations.sh`) | [x] | Decisions 37–42. Multi-token `--compositions`, `--queue-file`, `--missing-from-grid`; GPU/CPU branch; `gpu_test` guards; `--dry-run`. 476 tests pass (25 new). |
| 10 | HPC benchmark (`benchmark_hpc.sh` + `analyze_benchmark.py`); populate `hpc_defaults` | [ ] | GPU sweep merged; awaiting first real run to land values |
| 10b | CPU benchmark on general1 (`benchmark_hpc_general1.sh` + worker; populate `hpc_defaults_cpu`) | [x] | Decisions 50–53. 7-point sweep × 40 cores each; reuses v2025.4 tpr; mpi_ranks dimension probes domain decomp. 24 tests pass (8 new). |
| 10c | Route `submit_simulations.sh` to general1 (`hpc_defaults_cpu`, `sbatch_simulations_general1.sh`) | [x] | Decisions 58–62. Partition-aware dispatch + `--mpi-ranks-per-sim` flag + CPU worker with OMP pinning. 32 dry-run tests pass (7 new). |
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
| 33 | 2026-05-12 | Legacy `data/membrane_only/` systems counted as present via filesystem fallback (`run/prun.xtc` exists), not via manifest | Legacy systems pre-date the manifest format; re-simulating them just to mint manifests is wasteful. Opt-out via `legacy_fallback=False`. |
| 34 | 2026-05-12 | `dppc_corner_grid` and `dopc_corner_grid` use a ≥ 50 % cutoff for the named lipid | Natural split point; matches the binary midpoint. Revisit if the resulting queue is too large for HPC step 11. |
| 35 | 2026-05-12 | CHOL capped at 40 % in all generated corner grids | Martini 3 sterol-content guidance + legacy precedent (`POPC60_CHOL40` is the max-CHOL system in the training set). |
| 36 | 2026-05-12 | Work-queue ordering/prioritisation lives in step 9, not step 8; step-8 CLI emits alphabetically-sorted-by-partner groups | Scoring logic belongs in the submission layer; the planner is read-only. |

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

---

## Appendix H — Step 8 detailed plan: `analysis.py::missing_compositions()` + work-queue CLI

### H.1 Scope

Identify which compositions in a target grid have not yet been simulated and emit a work queue the HPC submission layer (step 9) can consume. Specifically: close the DPPC- and DOPC-rich corner gaps flagged by the Stage 5b per-system MAE analysis ([`results/figures/stage_5b/fig_c_per_system_mae.png`](../results/figures/stage_5b/fig_c_per_system_mae.png) — POPC30_DOPC70 is the worst legacy system; the 70 legacy systems are POPC-anchored and provide near-zero coverage when POPC fraction falls below ~30 %).

#### What's in scope for step 8

- `lipid_gnn/martini_pipeline/analysis.py::missing_compositions(target_grid, output_root, *, require_status="ok")` — pure function returning the subset of `target_grid` not yet present (or not yet successfully simulated) under `output_root`.
- `lipid_gnn/martini_pipeline/analysis.py::summarise_systems(output_root)` — companion read-only function that produces a per-system status table (canonical name, build status, walltime, manifest path, missing-handoff list). Lands together because it shares the directory-walking logic.
- Grid generators in `analysis.py`: `dppc_corner_grid()`, `dopc_corner_grid()`, `binary_grid(lipid_a, lipid_b, step=10)`, `ternary_grid(lipids, step=10)`. Pure functions returning lists of `Composition`.
- `scripts/simulation/print_work_queue.py` — CLI driver. Prints / writes a work queue with filters; emits JSON or a one-composition-per-line text format consumable by `scripts/bash/submit_simulations.sh` in step 9.
- `tests/martini_pipeline/test_missing_compositions.py` — coverage of grid generators, missing-set detection, status filters, fake-tree fixtures.

#### What's out of scope for step 8

- Actually submitting simulations to SLURM — step 9.
- Benchmarking and HPC defaults — step 10.
- Running the queued simulations to fill the corners — step 11 (production run, not a code task).
- Extending the lipid pool beyond the current 10 — step 12.
- New analytics (e.g. APL distributions over the new systems): out-of-scope until simulations exist; revisit after step 11.

### H.2 Locked-in design decisions

1. **Detection of "already simulated" is manifest-driven, not directory-existence.** A composition counts as built only if `<output_root>/<canonical_name>/manifest.json` exists AND its `overall_status` field matches the filter (default `"ok"`). Half-built or failed runs are reported as missing, not as present. Rationale: directory existence is a weak signal; the manifest is the single source of truth for "did this simulation actually finish".
2. **Both legacy `data/membrane_only/` and new `data/martini_pipeline/` are searched.** `missing_compositions` accepts a list of output roots, deduplicates by canonical name across them. Rationale: subgoal 2 is to *fill gaps*, so any legacy system that already covers a target point counts. Legacy systems lack our manifest, so a fallback "directory exists + has run/prun.xtc" check is used for them — documented as Decision 33 below.
3. **Canonical name is the only join key.** `composition.Composition(...).name` resolves alias issues (e.g. legacy `POPC10_DIPC90` vs canonical `DIPC90_POPC10`). Rationale: keeps the join clean and forces every comparison through the canonicaliser already tested in step 1.
4. **Grid generators return `Composition` objects, not raw dicts.** Validates fractions on construction and ensures the join key is always available. Rationale: catches bad grids at generation time, not at submission time.
5. **Step granularity is configurable but defaults to 10 %.** All legacy multi-lipid systems use 10 % increments except the DOPS-rich edge (2/4/6/8 %). Default 10 grid is dense enough for the corner-fill but the function takes a `step` parameter for future refinement. Rationale: matches existing data density; finer grids can be requested explicitly without rewriting.
6. **CLI emits two outputs side by side**: a human-readable table on stdout AND an optional machine-readable file (`--out path.json`). Rationale: same workflow supports interactive auditing and pipeline plumbing without two scripts.
7. **No mutation of disk state.** `missing_compositions` and the CLI are strictly read-only. The CLI never creates output directories or stub manifests; that responsibility belongs to step 9. Rationale: same independence rule as analysis.py's existing functions (`diff_mdps`, `summarise_systems`).

### H.3 Public API — `analysis.py` additions

```python
@dataclass(frozen=True)
class SystemStatus:
    canonical_name: str
    output_root: str
    out_dir: str                       # absolute
    has_manifest: bool
    overall_status: str | None         # None if no manifest
    has_prun_xtc: bool                 # legacy-fallback signal
    walltime_s: float | None           # sum of stage walltimes from manifest

def summarise_systems(
    output_root: str | os.PathLike,
    *,
    legacy_fallback: bool = False,
) -> list[SystemStatus]: ...

def missing_compositions(
    target_grid: Sequence[Composition],
    output_roots: Sequence[str | os.PathLike],
    *,
    require_status: str | tuple[str, ...] = "ok",
    legacy_fallback: bool = True,
) -> list[Composition]:
    """Return the subset of *target_grid* not yet successfully simulated.

    A composition is considered present iff any *output_root* contains a
    subdir matching its canonical name AND either:
      - a manifest.json with overall_status ∈ require_status, or
      - legacy_fallback=True and a non-empty run/prun.xtc exists.
    """

def binary_grid(
    lipid_a: str,
    lipid_b: str,
    *,
    step: int = 10,
    include_pure: bool = True,
) -> list[Composition]: ...

def ternary_grid(
    lipids: Sequence[str],
    *,
    step: int = 10,
    include_edges: bool = True,
) -> list[Composition]: ...

def dppc_corner_grid(step: int = 10) -> list[Composition]: ...
def dopc_corner_grid(step: int = 10) -> list[Composition]: ...
```

`dppc_corner_grid` and `dopc_corner_grid` are concrete subsets chosen to match the gap pattern from Stage 5b:

- **DPPC corner**: all binaries `DPPC_X` with `DPPC ≥ 50 %` for `X ∈ {DOPC, DIPC, DOPE, DPPE, POPE, CHOL, POPC}` plus pure `DPPC100` if not already present.
- **DOPC corner**: all binaries `DOPC_X` with `DOPC ≥ 50 %` for `X ∈ {DPPC, DIPC, DOPE, DPPE, POPE, CHOL, POPC}` plus pure `DOPC100`.

Pure singletons are included once globally (idempotent: `missing_compositions` will dedupe by canonical name).

### H.4 Public API — `scripts/simulation/print_work_queue.py`

```text
usage: print_work_queue.py [-h]
                           [--grid {dppc_corner,dopc_corner,binary,ternary,all}]
                           [--lipids LIPIDS [LIPIDS ...]]
                           [--step STEP]
                           [--output-roots ROOT [ROOT ...]]
                           [--require-status STATUS [STATUS ...]]
                           [--no-legacy-fallback]
                           [--out PATH]
                           [--format {table,json,lines}]
```

Behaviour:

- `--grid dppc_corner` / `dopc_corner` → uses the named helper.
- `--grid binary --lipids DPPC DOPC --step 10` → binary grid for that pair.
- `--grid ternary --lipids POPC DOPC DPPC --step 20` → ternary simplex.
- `--grid all` → union of `dppc_corner_grid()` and `dopc_corner_grid()`.
- Default `--output-roots`: `[CONFIG.paths.data_dir, CONFIG.martini_pipeline.output_root]` (covers legacy and new trees).
- `--format table` (default) prints aligned text to stdout; `--format json` emits a JSON array of `{canonical_name, fractions}` objects; `--format lines` emits one canonical name per line.
- Exit code: 0 if any work queued, 0 also if queue is empty (prints `(no missing)` to stderr); reserve non-zero for IO/parse failure only — empty is a valid result.

Example:

```bash
$ python scripts/simulation/print_work_queue.py --grid dppc_corner
canonical_name          step  fractions
----------------------  ----  -------------------------
DPPC100                 pure  DPPC=1.00
DPPC90_DIPC10           bin   DPPC=0.90 DIPC=0.10
DPPC90_DOPC10           bin   DPPC=0.90 DOPC=0.10
…
DPPC50_POPE50           bin   DPPC=0.50 POPE=0.50

42 compositions queued, 0 already simulated.
```

### H.5 Internal data flow

```text
target_grid (List[Composition])  ──┐
                                    │
output_roots (Sequence[Path])  ─────┼──►  missing_compositions()
                                    │            │
                                    │            ▼
                            for each root:  walk root/<name>/
                                            │     │
                                            │     ├─ manifest.json present?  → SystemStatus
                                            │     └─ legacy fallback: run/prun.xtc present?
                                            ▼
                                  Union of present canonical_names
                                            │
                                            ▼
                              target_grid − present  →  list[Composition]
```

Implementation notes:

- `summarise_systems` walks `output_root` exactly once per call; `missing_compositions` calls it per root and unions the canonical-name set. O(N_systems) on the filesystem; no per-comp re-walk.
- Manifest parsing is best-effort: a corrupt `manifest.json` is logged as `overall_status="invalid_manifest"` and the system is treated as missing (so the work queue conservatively re-runs it).
- Legacy fallback gate: opt-out via `--no-legacy-fallback` for users who only want to count manifest-validated systems.

### H.6 Edge-case matrix

| Case | Expected behaviour |
| --- | --- |
| Empty `target_grid` | `missing_compositions` returns `[]`; CLI prints `(no missing)`; exit 0 |
| `output_roots` empty / non-existent | Treat as zero present systems; entire grid is missing; CLI warns once on stderr |
| Manifest exists but `overall_status="failed_at_equilibration"` | System counted as missing (default filter is `"ok"` only) |
| Manifest exists with `overall_status="ok"` but no `run/prun.xtc` on disk | Still counted as present (manifest is authoritative; missing xtc is a separate problem surfaced by `summarise_systems`) |
| Legacy system with `run/prun.xtc` and no manifest | Counted as present iff `legacy_fallback=True` |
| Duplicate target compositions (e.g. `DOPC50_DPPC50` and `DPPC50_DOPC50`) | Deduplicated by canonical name before the missing-set computation |
| Non-integer percentages in the grid (e.g. `step=7`) | Each generated comp is validated by `Composition.__post_init__`; bad ones raise at grid generation, not at the missing-set step |
| Target lipid not in `LIPID_REGISTRY` | Grid generators warn and proceed (registry membership isn't required to *name* a composition); simulation-time failure is the registry's concern, not the planner's |

### H.7 Test plan — `tests/martini_pipeline/test_missing_compositions.py`

1. `test_binary_grid_count` — `binary_grid("POPC", "DOPC", step=10)` returns 9 mixtures (10 %–90 %) plus 2 pures = 11 entries.
2. `test_binary_grid_step_5` — `step=5` returns 19 mixtures + 2 pures = 21.
3. `test_ternary_grid_simplex_count` — `ternary_grid(["POPC", "DOPC", "DPPC"], step=20)` returns the correct number of simplex points (15 for 20 %).
4. `test_dppc_corner_grid_includes_pure_dppc` — `DPPC100` is in the result.
5. `test_dopc_corner_grid_includes_pure_dopc` — `DOPC100` is in the result.
6. `test_corner_grids_use_canonical_names` — every entry's `.name` round-trips via `parse_name`.
7. `test_missing_compositions_empty_grid` — empty input → empty output.
8. `test_missing_compositions_empty_root` — non-existent root → entire grid returned.
9. `test_missing_compositions_with_manifest_ok` — fake tree with one valid manifest → that comp removed from the missing set.
10. `test_missing_compositions_with_manifest_failed` — fake tree with `overall_status="failed_at_equilibration"` → comp stays in the missing set.
11. `test_missing_compositions_legacy_fallback_on` — fake legacy tree with `run/prun.xtc` but no manifest → comp counted as present.
12. `test_missing_compositions_legacy_fallback_off` — same tree with `legacy_fallback=False` → comp counted as missing.
13. `test_missing_compositions_union_across_roots` — same comp present in root A, target queries roots [A, B] → not missing.
14. `test_missing_compositions_canonical_aliasing` — present-name `POPC10_DOPC90` and target `DOPC90_POPC10` resolve as the same; comp removed.
15. `test_summarise_systems_returns_status_per_dir` — fake tree with 3 systems → 3 `SystemStatus` entries with correct fields.
16. `test_invalid_manifest_counted_missing` — corrupt JSON → treated as missing by `missing_compositions`.

Fakes: each test builds a tiny `tmp_path/<comp>/{manifest.json,run/prun.xtc}` skeleton inline (no real GROMACS). The manifest skeleton uses `Manifest(schema_version="1.0", ..., overall_status="ok")` serialised via `manifest.write_manifest`.

### H.8 Layout & dependencies

```text
lipid_gnn/martini_pipeline/
    analysis.py                          ← extend (add 5 functions + SystemStatus)
scripts/simulation/
    print_work_queue.py                  ← new CLI
tests/martini_pipeline/
    test_missing_compositions.py         ← new
docs/
    martini_pipeline_plan.md             ← status table row 8 → [x] + Decision 33
```

No new external dependencies. Internal imports: `composition.Composition`, `manifest.read_manifest`, `lipid_gnn.config.CONFIG`.

### H.9 Acceptance criteria

- `pytest tests/martini_pipeline/test_missing_compositions.py -q` passes locally and on HPC.
- `python scripts/simulation/print_work_queue.py --grid dppc_corner` prints a non-empty table on the current repo state (since no DPPC-corner systems exist in `data/membrane_only/` beyond pure `DPPC100`).
- `python scripts/simulation/print_work_queue.py --grid dopc_corner` prints a non-empty table (legacy data lacks high-DOPC mixtures except `POPC30_DOPC70`-style POPC-anchored ones).
- `print_work_queue.py --grid dppc_corner --format json --out /tmp/queue.json` writes a valid JSON array consumable by the step-9 submission orchestrator.
- Re-running the CLI after one composition lands in `data/martini_pipeline/<comp>/` with a `manifest.json` `overall_status="ok"` reduces the queue by exactly one.
- Step status table row 8 flipped to `[x]`.

### H.10 Open questions (need user input before implementation)

1. **Corner definition boundary.** "DPPC-rich" = `DPPC ≥ 50 %`? Or stricter (`≥ 60 %` to focus on the *corner*, not the *edge*)? Recommend `≥ 50 %`, matching the natural split point; revisit if the resulting queue is too large for HPC step 11.
2. **CHOL handling in corner grids.** CHOL is special: it's a sterol, not a phospholipid, and legacy CHOL systems max out at 40 % (`POPC60_CHOL40`). Should `dppc_corner_grid` include `DPPC60_CHOL40`? Recommend yes, but cap CHOL at 40 % everywhere in the grid (consistent with the Martini 3 sterol-content guidance and legacy practice).
3. **Ternary corner expansion.** The Stage 5b worst points are all binaries; should the corner grids also queue ternaries like `DPPC60_DOPC30_POPC10`? Recommend defer to a follow-up after the binary corners fill, since ternary count explodes (15 systems per step=20 simplex × C(10,3) lipid triples).
4. **Step-8 vs step-9 dividing line for `--max-queue N`.** Should the CLI itself be able to cap the queue length (priority-sorted) or is that a submission-orchestrator concern? Recommend keep step-8 unbounded and let step-9 prioritise; otherwise step-8 grows scoring logic that belongs in the submission layer.
5. **Should `summarise_systems` walk both the legacy and new roots in one call?** Or is "summarise a single root" the right unit and the CLI does the union? Recommend single-root primitive + CLI-level union; keeps the function composable.
6. **Output file convention.** `--out /tmp/queue.json` or `--out /tmp/queue.txt` — should the format be inferred from the suffix, or driven only by `--format`? Recommend `--format` is authoritative; suffix is advisory. Avoids surprise behaviour.

### H.11 New decisions to log on completion

- **Decision 33** — Legacy `data/membrane_only/` systems counted as present via filesystem fallback (`run/prun.xtc` exists), not via manifest. Rationale: legacy systems pre-date the pipeline's manifest format; re-simulating them just to mint manifests is wasteful. Opt-out via `legacy_fallback=False`.
- **Decision 34** — `dppc_corner_grid` and `dopc_corner_grid` use a `≥ 50 %` cutoff for the named lipid (per H.10 Q1 answer; adjust if user picks otherwise).
- **Decision 35** — CHOL capped at 40 % in all generated grids (Martini 3 sterol guidance + legacy precedent).
- **Decision 36** — Work-queue ordering/prioritisation lives in step 9, not step 8. The step-8 CLI emits an unsorted (or alphabetically sorted) queue.
- *Plus any decisions arising from H.10 Q2-Q6.*

---

## Appendix I — Step 9 detailed plan: HPC submission layer

### I.1 Scope

Submit the work queue from step 8 to Goethe-HLR (AMD MI210 / ROCm / SLURM), running N Martini simulations in parallel per node. Mirrors the established `submit_sweep.sh` / `sbatch_sweep.sh` split (which packs N training runs onto one SLURM job) but for the simulation pipeline rather than training.

#### What's in scope for step 9

- `scripts/bash/submit_simulations.sh` — orchestrator (login-node side). Resolves the composition list (explicit `--compositions`, `--missing-from-grid`, or `--queue-file`), packs onto SLURM batches, submits one `sbatch` per batch with per-slot `RUN_<i>_COMP=…` env vars baked in.
- `scripts/bash/sbatch_simulations.sh` — worker (compute-node side). Activates the conda env, loads the ROCm module, fans out N parallel `python scripts/simulation/run_martini_pipeline.py` processes, each pinned via `HIP_VISIBLE_DEVICES=$i` (GPU mode) or `OMP_NUM_THREADS` carve-out (CPU mode), waits for all, surfaces the worst exit code.
- `tests/martini_pipeline/test_submit_simulations.py` — bash-level tests via subprocess + `--dry-run` (no real `sbatch`).
- `docs/martini_pipeline_plan.md` — status table row 9 → `[x]`, decisions 37–N appended.

#### What's out of scope for step 9

- The HPC benchmark (`benchmark_hpc.sh` + `analyze_benchmark.py`) — that's step 10. Step 9 ships with **conservative pre-benchmark defaults** (documented in I.2 Decision a) and a `martini_pipeline.hpc_defaults` config stub that step 10 fills in.
- Production corner-fill execution — step 11 (not a code task).
- Checkpoint-restart (`gmx mdrun -cpi`). The orchestrator can re-invoke a partial system idempotently via `pipeline.run()`'s stage-skip logic ([Decision 23](#decision-log)); explicit `-cpi` restart-from-checkpoint is deferred.
- Top-level "submission summary" manifest aggregating per-sim status. Each system's manifest is already authoritative; aggregation can come from `summarise_systems()` re-run on the new root.
- Retries on transient sim failure. First failure → manifest with `overall_status="failed_at_<stage>"`; user resubmits the queue (which by default skips already-`ok` systems via `--missing-from-grid`).

### I.2 Locked-in design decisions

1. **Two-script split mirrors `submit_sweep.sh` / `sbatch_sweep.sh` exactly.** Login-node orchestrator parses args, queries `print_work_queue.py` / config, builds the per-slot env-var bundle, and issues one `sbatch` per N-sim batch. Compute-node worker reads the env vars and fans out. Same `RUN_<i>_<KEY>=…` indirect-lookup pattern.
2. **One composition per Python process.** Each backgrounded slot runs `python scripts/simulation/run_martini_pipeline.py <composition> --output-root <root> --nsteps <N>`; multi-system batching lives in the bash worker, not in `pipeline.py` ([Decision 22](#decision-log)).
3. **Output root lives on `/work`, not on `$HOME`.** Default for HPC runs is `/work/$GROUP/$USER/lipid-data/martini_pipeline/`. Local-run default stays at the repo-relative `data/martini_pipeline/` from `MartiniPipelineConfig.output_root`. The orchestrator flips between them based on `--output-root` (with HPC default sourced from a new config knob `martini_pipeline.hpc_output_subpath`, defaulting to `martini_pipeline/` under `hpc.work_subpath`).
4. **Composition list sources are exclusive.** Exactly one of `--compositions A B C`, `--missing-from-grid <grid_name>`, or `--queue-file <path>` must be supplied. Rationale: avoid silent precedence bugs.
5. **GPU/CPU as a single script with a branch ([Decision 4](#decision-log)).** `--gpus-per-node 0` short-circuits the GPU-pinning block; `gmx mdrun -nb cpu -ntomp <K>` is used with thread budget split across the N parallel sims.
6. **Pre-benchmark defaults are intentionally conservative.** `--sims-per-node 4` (GPU partition) and `--time 24:00:00`. Rationale: corner systems are ≤ 12k atoms, but production length will land at ≥ 100 ns per system; under-packing burns walltime but cannot fail, over-packing risks OOM/contention. Step 10 raises this with data.
7. **All HPC environment plumbing flows through `config.yaml`.** No hardcoded module names, partitions, accounts, or paths in the bash scripts; everything reads via `python scripts/python/print_config_var.py <dotted.key>` (matching `submit_sweep.sh`). New config keys live under `hpc.*` (existing block) and `martini_pipeline.hpc_defaults.*` (new sub-block, step-10-filled).
8. **`gpu_test` guard rails mirror `submit_sweep.sh` exactly.** Time cap at 08:00:00; max 2 sbatch jobs; warn-and-cap or error-and-abort with the same messages.
9. **Per-slot logs land at `logs/simulations/sim-<jobid>-gpu<i>.{out,err}`.** Matches the `logs/sweeps/sweep-<jobid>-gpu<i>.{out,err}` precedent. The orchestrator's own stdout summary is captured by SLURM at `logs/simulations/submit-<jobid>.out` (same convention).
10. **Worker reads inputs from `/work`, writes outputs to `/work`, no `/local` staging.** Same root-cause analysis as `sbatch_sweep.sh`'s big comment: `/local` is tmpfs, pages get evicted under shared-node memory pressure, file reads turn into `FileNotFoundError`. ITPs and MDP templates are read from the repo on `$HOME` (small, hot, cache-friendly); per-sim outputs go directly to `/work/.../martini_pipeline/<comp>/`.
11. **Failure isolation: one bad sim cannot kill its node-mates.** Each slot is backgrounded (`&`), all PIDs collected, `wait` runs to completion. Exit code reported is the worst per-slot exit code (same as `sbatch_sweep.sh` lines 113–121). Each system writes its own manifest with `overall_status="failed_at_*"` so the next `--missing-from-grid` rerun picks it up automatically.
12. **Idempotency is inherited, not re-implemented.** `pipeline.run()` already skips stages whose handoff `.gro` exists ([Decision 23](#decision-log)). Resubmitting an interrupted batch with `--missing-from-grid` returns only the systems whose manifest doesn't have `overall_status="ok"` — and `pipeline.run()` resumes them from the last completed stage. No restart logic needed in the bash layer.

### I.3 Public API — `scripts/bash/submit_simulations.sh`

```text
usage: submit_simulations.sh
    (--compositions COMP [COMP ...]
     | --missing-from-grid {dppc_corner,dopc_corner,binary,ternary,all}
     | --queue-file PATH)
    [--lipids LIPID [LIPID ...]]      # required for --missing-from-grid {binary,ternary}
    [--step N]                        # default 10
    [--prod-ns FLOAT | --nsteps N]    # exactly one required
    [--save-forces]
    [--maxwarn N]                     # default 2
    [--nsteps-eq N] [--nsteps-min N]
    [--output-root PATH]              # default: WORK/.../martini_pipeline/
    [--partition NAME]                # default from CONFIG.hpc.partition_train
    [--time HH:MM:SS]                 # default 24:00:00
    [--gpus-per-node N]               # default 8; set 0 for CPU partition
    [--sims-per-node N]               # default 4 pre-benchmark (Decision 37)
    [--cpus-per-sim N]                # default 8
    [--mem-per-sim SIZE]              # default 16G
    [--ntomp N]                       # default = cpus-per-sim
    [--max-queue N]                   # cap total sims submitted (alphabetical priority)
    [--dry-run]                       # print sbatch commands, do not submit
```

Behaviour:

- **Composition resolution**:
  - `--compositions` → use the literal list (validated via `python -c "from lipid_gnn.martini_pipeline.composition import parse_name; ..."` round-trip).
  - `--missing-from-grid` → shell out to `python scripts/simulation/print_work_queue.py --grid <X> --step <step> [--lipids ...] --format lines` and pipe the canonical names back. Inherits the union-of-output-roots logic from step 8 automatically.
  - `--queue-file` → read one comp name per line (blank lines and `#`-comments skipped).
- **Packing**: `N_total = len(comp_list)`; `N_batches = ceil(N_total / sims_per_node)`. One `sbatch` per batch. Each batch packs the next `sims-per-node` composition names into `RUN_0_COMP, RUN_1_COMP, …` env vars; the worker reads them via indirect expansion.
- **Resource calculation** (mirrors `submit_sweep.sh` line 134–137 for `--mem` workaround):
  - `TOTAL_CPUS = cpus_per_sim × N_sims_in_batch`
  - `TOTAL_MEM  = mem_per_sim × N_sims_in_batch` (numeric scaling on the suffix)
  - `--gres=gpu:N_sims_in_batch` when `gpus_per_node > 0`; otherwise `--gres=none`.
- **Frozen knobs at submission time**: the orchestrator resolves `--prod-ns`/`--nsteps`, `--maxwarn`, `--save-forces`, `--output-root`, plus all HPC defaults (conda env, ROCm module, account, work subpath) and bakes them into the export list. Queue wait cannot cause drift.
- **`--dry-run`**: prints the exact `sbatch …` invocation for every batch and the per-slot composition assignment, but does not call `sbatch`. Required for tests and for sanity-checking large queues before burning quota.
- **Summary stdout** (after submission, mirroring `submit_sweep.sh` line 161–179):

  ```text
  Submitting simulations  (2026-05-13 09:42)
    partition       : gpu
    time            : 24:00:00
    sims-per-node   : 4
    output-root     : /work/cellmembrane/pberger/lipid-data/martini_pipeline
    Total comps     : 35
    Batches         : 9 (up to 4 sims/node)

    Job 8127492  batch 1/9  N_SIMS=4  cpus=32  mem=64G
      [GPU 0]  DPPC100
      [GPU 1]  DPPC90_DIPC10
      …
  ```

### I.4 Public API — `scripts/bash/sbatch_simulations.sh`

Static SBATCH directives at the top of the file (same pattern as `sbatch_sweep.sh`):

```bash
#SBATCH --job-name=lipid-sim
#SBATCH --mail-user=pberger@fias.uni-frankfurt.de
#SBATCH --account=cellmembrane
#SBATCH --output=logs/simulations/submit-%j.out
#SBATCH --error=logs/simulations/submit-%j.err
```

Dynamic resources (`--partition`, `--time`, `--gres`, `--cpus-per-task`, `--mem`) are passed by the orchestrator on the `sbatch` command line, not via `#SBATCH` directives.

Inside the script:

1. `set -euo pipefail; mkdir -p logs/simulations`.
2. `source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate "$(python …conda_env)"; module load "$(python …module_rocm)"`.
3. Set `cd "$HOME/lipid-graph-nn"`.
4. Resolve `WORK="/work/$GROUP/$USER/$WORK_SUBPATH"` (matches `sbatch_sweep.sh` line 26).
5. Resolve `OUTPUT_ROOT="${OUTPUT_ROOT:-$WORK/martini_pipeline}"` (set in orchestrator's export list).
6. Resolve `N_SIMS="${N_SIMS_PER_NODE:-1}"`.
7. For `i in 0..N_SIMS-1`:
   - Indirect-lookup `COMP="${!COMP_VAR}"` where `COMP_VAR="RUN_${i}_COMP"`.
   - Logs at `logs/simulations/sim-${SLURM_JOB_ID}-gpu${i}.{out,err}`.
   - Subshell:
     - `export HIP_VISIBLE_DEVICES="$i"; export CUDA_VISIBLE_DEVICES="$i"` (GPU mode; skipped if `GPUS_PER_NODE=0`).
     - `export OMP_NUM_THREADS="$NTOMP"`.
     - Run `python scripts/simulation/run_martini_pipeline.py "$COMP" --output-root "$OUTPUT_ROOT" --nsteps "$NSTEPS" --maxwarn "$MAXWARN" [--save-forces] --mdrun-args "-ntomp $NTOMP [-nb cpu]" >"$LOGOUT" 2>"$LOGERR"`.
   - `&` into background; collect PID.
8. After the loop, `wait` on every PID; report worst exit code (same pattern as `sbatch_sweep.sh` line 110–124).

### I.5 Internal data flow

```text
print_work_queue.py --grid X --format lines
            │
            ▼
submit_simulations.sh
  ├─ validates comps via parse_name round-trip
  ├─ batches into ceil(N_total / sims_per_node)
  ├─ per batch:
  │    EXPORT_VARS = ALL,OUTPUT_ROOT=<root>,N_SIMS_PER_NODE=<k>,NSTEPS=<n>,
  │                  MAXWARN=<m>,SAVE_FORCES=<bool>,NTOMP=<t>,GPUS_PER_NODE=<g>,
  │                  RUN_0_COMP=<c0>, RUN_1_COMP=<c1>, …
  │    sbatch --partition=… --time=… --gres=gpu:k --cpus-per-task=… --mem=… \
  │           --export=$EXPORT_VARS scripts/bash/sbatch_simulations.sh
  ▼
sbatch_simulations.sh  (compute node)
  ├─ activates env, loads rocm
  ├─ for i in 0..k-1:
  │    HIP_VISIBLE_DEVICES=$i  python scripts/simulation/run_martini_pipeline.py "$RUN_${i}_COMP" …  &
  └─ wait; exit worst_rc
            │
            ▼
For each composition: out_dir/<comp>/{minimization,equilibration,run,manifest.json}
            │
            ▼
Next `submit_simulations.sh --missing-from-grid …` sees the completed manifests
and produces a shorter queue. Loop until queue is empty.
```

### I.6 Edge-case matrix

| Case | Expected behaviour |
| --- | --- |
| Empty queue (all comps already `ok`) | Orchestrator prints `(no missing)`, exits 0 without submitting |
| Invalid composition name in `--compositions` | Fail fast on the login node before any `sbatch` |
| `--queue-file` with blank lines / `#` comments | Silently skipped |
| `--missing-from-grid binary` without `--lipids` | Fail fast with usage; mirrors print_work_queue.py's check |
| `gpu_test` partition with > 2 batches | Error and exit (same as `submit_sweep.sh` line 142–148) |
| `gpu_test` partition with `--time > 08:00:00` | Warn, cap to 08:00:00 |
| Mid-run SLURM kill / preemption | Each slot's pipeline writes its current manifest stage; re-submission with `--missing-from-grid` picks up where it left off |
| One sim of four fails mid-batch | Other three continue; node exits with the failed slot's rc; failed slot's manifest has `overall_status="failed_at_…"` |
| `--gpus-per-node 0` (CPU partition) | Skip `HIP_VISIBLE_DEVICES` export; `--mdrun-args "-nb cpu -ntomp $NTOMP"`; `--gres=none` |
| `--dry-run` | Print `sbatch …` per batch + per-slot comp assignment; do not invoke `sbatch` |
| `OUTPUT_ROOT` already contains `<comp>/` with completed manifest | `pipeline.run()` skips all stages (idempotent); single GPU minute wasted on the no-op |
| W&B-style air-gapped node | Pipeline does not use W&B; no special handling |
| Duplicate `RUN_<i>_COMP` (orchestrator bug) | Worker's `pipeline.run()` is idempotent; second run is a no-op. Still worth a unit test in I.7. |

### I.7 Test plan — `tests/martini_pipeline/test_submit_simulations.py`

All tests invoke `submit_simulations.sh --dry-run` via `subprocess.run([…], capture_output=True)`. No real `sbatch`. Where the script needs to invoke `print_work_queue.py`, the tests can rely on the real (read-only) implementation.

1. `test_explicit_compositions_packed_correctly` — `--compositions A B C D E --sims-per-node 2 --dry-run --prod-ns 100` ⇒ 3 batches (2, 2, 1).
2. `test_sims_per_node_default_4` — no `--sims-per-node` ⇒ default 4 reflected in batch sizes.
3. `test_missing_grid_dppc_corner_dry_run` — `--missing-from-grid dppc_corner --output-root <tmp_empty>` ⇒ 35 comps (the full grid, since the tmp root is empty).
4. `test_missing_grid_dppc_corner_partial_present` — populate tmp_root with one `DPPC100/manifest.json overall_status=ok` ⇒ queue has 34 entries.
5. `test_queue_file_strips_comments_and_blanks` — `--queue-file <tmp_file>` containing `# header\nDPPC100\n\nDIPC100` ⇒ 2 comps queued.
6. `test_exclusive_source_modes` — passing both `--compositions` and `--queue-file` ⇒ exit non-zero with usage message.
7. `test_invalid_comp_name_fails_fast` — `--compositions NOTALIPID99` ⇒ non-zero exit, error mentions parse failure.
8. `test_missing_prod_length_required` — neither `--prod-ns` nor `--nsteps` ⇒ usage error.
9. `test_mutually_exclusive_prod_length` — both `--prod-ns 50` and `--nsteps 1000000` ⇒ usage error.
10. `test_gpu_test_max_two_batches_enforced` — 12 comps × `--sims-per-node 4 --partition gpu_test` ⇒ 3 batches ⇒ error and exit 1.
11. `test_gpu_test_time_cap_warning` — `--partition gpu_test --time 12:00:00` ⇒ warning + capped to `08:00:00` in dry-run output.
12. `test_dry_run_does_not_invoke_sbatch` — `--dry-run` prints `sbatch …` lines but never runs the binary (PATH stubbed sbatch records invocation; should remain empty).
13. `test_resource_scaling` — `--sims-per-node 4 --cpus-per-sim 8 --mem-per-sim 16G` ⇒ dry-run shows `--cpus-per-task=32` and `--mem=64G`.
14. `test_cpu_branch` — `--gpus-per-node 0` ⇒ dry-run output contains `--gres=none` and `-nb cpu` in the mdrun args.
15. `test_output_root_override` — `--output-root /custom/path` ⇒ dry-run shows `OUTPUT_ROOT=/custom/path` in the export list.
16. `test_max_queue_caps_total` — 35-entry queue, `--max-queue 5` ⇒ 5 comps submitted; remainder logged to stderr.
17. `test_sbatch_script_parses_run_i_comp` — directly invoke `sbatch_simulations.sh` under a stubbed-`python` / stubbed-`module` / stubbed-`conda` environment with `RUN_0_COMP=DPPC100 RUN_1_COMP=DOPC100 N_SIMS_PER_NODE=2 SLURM_JOB_ID=test`; check stub-python receives the right `argv[1]` per slot. (Hardest test; may be skipped if the bash stubbing is too brittle — covered by I.9 acceptance criterion E2E instead.)

Tests 1–16 are deterministic and run locally without SLURM. Test 17 is allowed to be `skipUnless(shutil.which("bash"))` or similar.

### I.8 Layout & dependencies

```text
scripts/bash/
    submit_simulations.sh                ← new (orchestrator)
    sbatch_simulations.sh                ← new (worker)
tests/martini_pipeline/
    test_submit_simulations.py           ← new
logs/simulations/                        ← created on first submission (.gitkeep)
docs/
    martini_pipeline_plan.md             ← status row 9 → [x] + Decisions 37–N
config.yaml                              ← add martini_pipeline.hpc_defaults stub
lipid_gnn/config.py                      ← parse hpc_defaults sub-block
```

New config block (step-10 fills in real values):

```yaml
martini_pipeline:
  hpc_defaults:                # pre-benchmark conservative values; step 10 overwrites
    sims_per_node: 4
    cpus_per_sim: 8
    mem_per_sim: 16G
    ntomp: 8
    gpus_per_node: 8
```

No new Python dependencies. Bash dependencies: `python scripts/python/print_config_var.py` (already used by `submit_sweep.sh`), `sbatch`/`squeue` (HPC only; tests stub them out).

### I.9 Acceptance criteria

- `pytest tests/martini_pipeline/test_submit_simulations.py -q` passes locally (no `sbatch` needed).
- `bash scripts/bash/submit_simulations.sh --missing-from-grid dppc_corner --prod-ns 100 --dry-run` prints a coherent batch plan with 9 batches × ≤ 4 sims/batch for the current 35-entry queue.
- `bash scripts/bash/submit_simulations.sh --compositions DIPC100 --prod-ns 10 --dry-run` prints a single 1-sim batch with the correct `sbatch …` line.
- On HPC: a real submission of `--compositions DIPC100 --prod-ns 0.5` (smoke) lands one job, exits cleanly, leaves a `manifest.json overall_status=ok` under `$OUTPUT_ROOT/DIPC100/`. *(Manual; not in CI.)*
- After the smoke job lands, re-running `submit_simulations.sh --missing-from-grid all --prod-ns 100 --dry-run` correctly omits `DIPC100` from the queue.
- Step status table row 9 flipped to `[x]`.

### I.10 Open questions (need user input before implementation)

1. **Default `--sims-per-node` before step 10's benchmark.** Recommend `4`. Rationale: 8-GPU node, 8 MI210s, but each `gmx mdrun` slot is bounded by ITP-init + per-step memory writes; halving the GPU count leaves room for the CPU side. Step 10 may raise to 8 if benchmark says so. Override?
2. **Output root on HPC: `/work/.../martini_pipeline/` vs `/work/.../lipid-data/martini_pipeline/`?** The existing legacy layout puts simulations under `/work/.../lipid-data/data/membrane_only/`; matching means new sims land at `/work/.../lipid-data/martini_pipeline/`. Recommend the latter (single root for all simulation data; mirrors the `lipid-data/` umbrella). Override?
3. **Composition prioritisation for `--missing-from-grid` and `--max-queue`.** Right now the planner emits alphabetically grouped output. Options for step 9: (a) keep alphabetical, (b) interleave corners (DPPC, DOPC) round-robin, (c) put pure singletons first (they're the cheapest), (d) put the systems flagged by Stage 5b MAE first. Recommend (a) — alphabetical is simplest and step 11 can resort if needed.
4. **Per-sim mdrun thread count when `--ntomp` is unset.** Default to `--cpus-per-sim` (1:1). Override?
5. **CPU-only fallback path.** When `--gpus-per-node 0`, do we want a separate default `--partition` (e.g. `cpu`)? Or require the user to pass `--partition cpu` explicitly? Recommend explicit — keeps the GPU/CPU branch in the script trivial.
6. **Retry-once mode for transient failures.** Some HPC failures are flaky (node memory pressure, ROCm init race). Recommend defer to a later step; first-failure → manifest with `failed_at_*` is fine. Worth implementing now if the user expects frequent flake.
7. **Email notifications on job failure.** `sbatch --mail-type=FAIL` is cheap and useful for long queues. Add? Recommend yes (consistent with the existing `submit_sweep.sh` pattern which already wires `--mail-user`).
8. **`logs/simulations/` retention.** Per-sim logs grow with the queue. Recommend no auto-cleanup in step 9; surface a `.gitkeep` so the directory exists but the contents are gitignored.
9. **Smoke test acceptance.** Should the I.9 manual smoke test be a separate `bash scripts/simulation/sanity_check_hpc.sh` driver (mirroring `sanity_check_dipc100.py`)? Recommend skip — `sanity_check_dipc100.py` already works locally; HPC parity is verified by the dry-run tests + a one-time manual job.

### I.11 New decisions to log on completion

- **Decision 37** — Two-script split for the HPC submission layer, mirroring `submit_sweep.sh` / `sbatch_sweep.sh` line-for-line in structure. Rationale: convention parity, easy review.
- **Decision 38** — Pre-benchmark defaults: `sims-per-node 4`, `cpus-per-sim 8`, `mem-per-sim 16G`, `time 24:00:00`. Conservative; step 10 promotes to data-driven values.
- **Decision 39** — Read inputs from `/work`, write outputs to `/work`; no `/local` staging. Rationale: same tmpfs-eviction story as `sbatch_sweep.sh` lines 41–57.
- **Decision 40** — Each backgrounded slot is a separate `python scripts/simulation/run_martini_pipeline.py` invocation. Failures of one slot do not abort node-mates; worst exit code is surfaced (mirror of `sbatch_sweep.sh` lines 113–121).
- **Decision 41** — Idempotency is inherited from `pipeline.run()` (Decision 23); no separate restart / checkpoint logic at the bash layer. Resubmission with `--missing-from-grid` is the canonical retry mechanism.
- **Decision 42** — `--dry-run` is a first-class testing affordance; all unit tests use it.
- **Decision 49** — Composition token `DIPC` maps to v2 moleculetype `DLPC`. The M3-Lipid-Parameters v2 set (per [K.B. Pedersen et al., ACS Central Science 2025](https://doi.org/10.1021/acscentsci.5c00755)) renamed di-C18:2 PC from legacy `DIPC` to `DLPC`. We keep `DIPC` as the user-facing composition token (matches legacy 70-system directory naming and `composition.py` parsing) but pass `DLPC` to insane and into `topol.top`. Discovered when sanity_check_dipc100 failed at grompp with `No such moleculetype DIPC`; legacy systems remain unaffected because they ship their own v1 ITPs in `toppar/`.
- *Plus any decisions arising from I.10 Q1–Q9.*

---

## Appendix J — Step 10 detailed plan: HPC benchmark & `hpc_defaults` calibration

### J.1 Scope

Run a structured throughput sweep on Goethe-HLR (AMD MI210 / ROCm) using the step-9 submission layer, parse `gmx mdrun` performance, and produce data-backed values for `martini_pipeline.hpc_defaults` in `config.yaml`. The sweep must be small enough to finish in a single wallclock day (< 16 GPU-hours total) and large enough to land defaults that survive corner-fill production (step 11) without re-benchmarking.

#### What's in scope for step 10

- `scripts/simulation/benchmark_hpc.sh` — orchestrator: submits one SLURM job per `(sims_per_node, cpus_per_sim, gpus_per_node)` config point, all running the same reference system for a short, fixed length.
- `scripts/simulation/sbatch_benchmark_hpc.sh` — worker: identical structure to `sbatch_simulations.sh`, but runs a stripped-down "production-only" pipeline (no insane / minimization / equilibration on every point — those are amortised once and reused).
- `scripts/python/analyze_benchmark.py` — Python CLI: ingests the per-point logs, parses `Performance: <ns/day>` from `prun.log`, samples `rocm-smi` snapshots, writes `results/benchmarks/martini_pipeline/<date>/summary.csv` + `summary.json` + `summary.md`, and prints a recommended `hpc_defaults` block.
- `tests/martini_pipeline/test_analyze_benchmark.py` — unit tests for the log parser, ns/day aggregation, and recommendation logic (no SLURM).
- `docs/martini_pipeline_plan.md` — status table row 10 → `[x]`, decisions 43–N appended, recommended defaults pasted into the decision log.
- `config.yaml` — `martini_pipeline.hpc_defaults` overwritten with benchmark-derived values; pre-benchmark values moved to a comment for traceability.

#### What's out of scope for step 10

- Re-benchmarking after every code change. The benchmark is a one-shot calibration; only re-run if GROMACS, ROCm, or the lipid pool changes materially.
- Benchmarking the CPU partition (`general1`). MI210 nodes are the production target; CPU defaults remain at the conservative step-9 fallback. Add a CPU-only benchmark row if step 11 ever overflows to CPU.
- Benchmarking `nstlist` / Verlet-buffer tuning. The current MDP templates ([Step 3 audit](#step-3-mdp-audit)) are frozen; tuning them belongs in a separate MDP-revision step.
- Tuning per-system (e.g. CHOL-heavy systems may scale differently). The benchmark recommends one set of defaults; per-system overrides land via CLI flags if step 11 reveals outliers.
- Auto-rewriting `config.yaml` programmatically. `analyze_benchmark.py` prints the recommended block; user inspects and pastes (same flow as the [Step 3 audit](#step-3-mdp-audit)).

### J.2 Locked-in design decisions

1. **One reference system, one production length.** POPC100 at 100k nsteps (1 ns at dt=0.01 ps). Rationale: POPC is mid-pool in atom count, single-component, no charged groups, no CHOL ring strain — closest to "typical" of the 70-system corpus. 100k nsteps gives `gmx mdrun` enough work to amortise startup (load-balancing settles after ~5k steps) while keeping each point ≤ 5 min on the fastest config.
2. **Re-use a single pre-built `prun.tpr`.** Step 10 does *not* re-run insane / minimization / equilibration for every config point. The benchmark orchestrator (a) runs the pipeline once locally to produce `prun.tpr`, (b) stages that single `.tpr` on `/work/.../benchmark/POPC100/`, (c) every benchmark point launches `gmx mdrun -s prun.tpr -deffnm prun-bench-<i>` directly. Rationale: insane + min + eq are config-independent and dominate wallclock; benchmarking them is wasted quota.
3. **Sweep is a curated subset, not a Cartesian product.** Cartesian over `sims_per_node ∈ {1,2,4,8} × cpus_per_sim ∈ {4,8,16} × gpus_per_node ∈ {1,2,4,8}` = 48 points. We test 9 points covering the realistic operating region (see J.3 sweep table). Rationale: most Cartesian cells are infeasible (e.g. `sims_per_node=8` with `cpus_per_sim=16` ⇒ 128 CPUs, > node capacity).
4. **Each benchmark point runs the *full* `sims_per_node` simultaneously.** Pack-induced contention (PCIe, memory bandwidth, ROCm context switching) is the variable we're actually measuring; benchmarking a lone slot misses it. ns/day is summed across slots; the headline metric is **aggregate ns/day per node-hour**.
5. **rocm-smi sampling is optional but on by default.** A 5 s polling loop runs in the background and dumps GPU utilisation / power / VRAM to `rocm-smi.tsv`. Adds zero overhead; helpful for the thesis figure but not required by the recommendation logic.
6. **Reproducibility: each point uses the same `-seed`.** mdrun's `gen-vel` seed is fixed via the existing pipeline's deterministic-seed strategy ([Decision 22](#decision-log)). Two runs of the same config point should produce identical trajectories byte-for-byte (modulo `nstlog` ordering); ns/day variance comes from system noise only.
7. **Output convention**: `results/benchmarks/martini_pipeline/<ISO date>/{summary.csv, summary.json, summary.md, points/<i>/prun.log, points/<i>/rocm-smi.tsv}`. Versioned by date so re-benchmarks don't overwrite history. Symlink `results/benchmarks/martini_pipeline/latest -> <date>` for tooling.
8. **Recommendation logic is mechanical, not heuristic.** `analyze_benchmark.py` ranks points by `aggregate_ns_per_day / node_hours` (higher = better) and picks the top-scoring point whose `mem_per_sim × sims_per_node ≤ 0.7 × node_mem` (memory headroom). Tie-break: prefer smaller `gpus_per_node` (leaves room for other jobs). Output is auditable: the CSV shows the score for every point.
9. **Single-job vs many-job submission.** Each benchmark point is a separate `sbatch`. Rationale: SLURM accounting (`sacct -j <id> --format=JobID,Elapsed,MaxRSS`) is per-job; bundling points into one job loses per-config memory data. Cost: ≤ 9 jobs queued, easy for `gpu_test` quota.
10. **`gpu_test` is the default partition.** The headline production partition is `gpu` (24-hour time limit), but the benchmark itself fits comfortably in `gpu_test` (8-hour cap, faster scheduling). Each point ≤ 10 min; full sweep ≤ 90 wallclock min if serialised, ≤ 15 min if all 9 jobs run in parallel.

### J.3 Sweep table

`gpu_test` caps a single job at **≤ 4 GPUs in parallel** (in addition to the 8 h walltime and 2-job-per-user limits). Points must stay within that envelope; this still covers the realistic operating region for ≤ 12k-atom corner systems, which never need 8 GPUs.

The 7 curated points:

| # | sims_per_node | gpus_per_node | cpus_per_sim | total CPU | total GPU | mem | note |
|---|---|---|---|---|---|---|---|
| 1 | 1 | 1 | 16 | 16 | 1 | 16 G | dedicated GPU, max CPU per slot |
| 2 | 2 | 2 | 8  | 16 | 2 | 32 G | 1 sim per GPU baseline |
| 3 | 4 | 4 | 4  | 16 | 4 | 64 G | 1 sim per GPU, CPU-starved |
| 4 | 4 | 4 | 8  | 32 | 4 | 64 G | 1 sim per GPU, balanced — **expected winner** |
| 5 | 4 | 4 | 16 | 64 | 4 | 64 G | 1 sim per GPU, CPU-rich |
| 6 | 2 | 1 | 8  | 16 | 1 | 32 G | 2 sims share 1 GPU — PCIe contention probe |
| 7 | 8 | 4 | 4  | 32 | 4 | 128 G | 2 sims per GPU, max packing within gpu_test cap |

Points 6–7 probe whether one MI210 can timeshare multiple `gmx mdrun` slots — relevant if POPC100 is GPU-underutilised. Drop them if step 9 reveals `--gres=gpu:N` is enforced strictly per slot (cannot allocate fewer GPUs than sims). Note that `gpu_test` allows GPU sharing across *different* jobs from different users on the same node, but within a single job `HIP_VISIBLE_DEVICES` masking is what creates the over-subscription tested here.

CPU baseline (formerly point 8 on `general1`) dropped: `general1`'s GROMACS is the 2022 build which has already shown divergent behaviour (ion naming, MDP-key acceptance) from the v2025.4 ROCm module used for production. Mixing the two in one calibration sweep would compare apples to oranges; if a CPU number is needed later for the thesis figure, run it as a separate one-off with an explicit points file.

Total quota: 7 GPU jobs; ≤ 10 min each, ≤ 70 GPU-min serialised, ≤ 30 wallclock min if 2 schedule in parallel under the `gpu_test` per-user cap.

### J.4 Public API — `scripts/simulation/benchmark_hpc.sh`

```text
usage: benchmark_hpc.sh
    [--reference-comp NAME]      # default: POPC100
    [--nsteps N]                 # default: 100000
    [--points PATH]              # default: scripts/simulation/benchmark_points.tsv
    [--output-root PATH]         # default: results/benchmarks/martini_pipeline/<ISO date>
    [--partition NAME]           # default: gpu_test
    [--time HH:MM:SS]            # default: 00:30:00 per point
    [--build-tpr-only]           # rebuild prun.tpr from scratch, exit
    [--dry-run]                  # print sbatch commands, do not submit
```

Behaviour:

1. **Prereq check**: confirm `--reference-comp` system has a fully-built `prun.tpr` under `<output-root>/<comp>/`. If missing, run `python scripts/simulation/run_martini_pipeline.py --stop-after equilibration` locally (or on a single `sbatch` if local quota tight), then exit (user re-runs the benchmark proper).
2. **Read sweep points** from a TSV: `sims_per_node\tgpus_per_node\tcpus_per_sim\tmem_per_sim\tlabel`.
3. **Per point**, submit one `sbatch scripts/simulation/sbatch_benchmark_hpc.sh` with `--export=ALL,POINT_INDEX=<i>,REFERENCE_TPR=<path>,SIMS_PER_NODE=…,…`. Resource line mirrors step 9: `--cpus-per-task=$((CPUS * SIMS))`, `--mem=$((MEM * SIMS))G`, `--gres=gpu:GPUS_PER_NODE`.
4. **Print summary** to stdout: one line per submitted job, with the recommended `analyze_benchmark.py` invocation at the bottom.

### J.5 Public API — `scripts/simulation/sbatch_benchmark_hpc.sh`

Per-point worker. Identical preamble to `sbatch_simulations.sh` (conda + module load) plus:

1. `cd "$OUTPUT_ROOT/points/$POINT_INDEX"` (created by orchestrator).
2. Stage `REFERENCE_TPR` via symlink (no copy — `/work` is durable).
3. **Background `rocm-smi` sampler** (5 s interval, into `rocm-smi.tsv`).
4. For `i in 0..SIMS_PER_NODE-1`: subshell that exports `HIP_VISIBLE_DEVICES=$((i % GPUS_PER_NODE))`, launches `gmx mdrun -s prun.tpr -deffnm prun-bench-$i -ntomp $CPUS_PER_SIM -nsteps $NSTEPS -resethway` in background. The `-resethway` flag tells GROMACS to reset performance counters at the halfway mark, excluding startup (load-balancing) overhead from the ns/day measurement.
5. `wait` all PIDs; kill the `rocm-smi` sampler; exit with worst rc.

`prun.log` (per slot) is left in place; `analyze_benchmark.py` parses each.

### J.6 Public API — `scripts/python/analyze_benchmark.py`

```text
usage: analyze_benchmark.py
    [--root PATH]                # default: results/benchmarks/martini_pipeline/latest
    [--format {csv,json,md,all}] # default: all
    [--mem-headroom-frac FLOAT]  # default: 0.70 (J.2 Decision 8)
    [--node-mem GB]              # default: 256 (MI210 node spec)
    [--recommend]                # print recommended config.yaml block to stdout
```

Steps:

1. Walk `<root>/points/*/prun-bench-*.log`; for each slot, regex `^Performance:\s+(\S+)\s+(ns/day)`.
2. Aggregate per point: `aggregate_ns_per_day = sum(per-slot)`; `node_hours = (elapsed_s / 3600) * 1` (one node always).
3. Compute `score = aggregate_ns_per_day / node_hours`.
4. Read each point's `rocm-smi.tsv` and add columns `gpu_util_mean`, `gpu_power_W_mean`, `vram_used_MB_max`.
5. Write `summary.csv` (one row per point), `summary.json` (same data, machine-readable), `summary.md` (markdown table, plus the recommendation block).
6. With `--recommend`: filter points where `mem_per_sim × sims_per_node ≤ mem_headroom_frac × node_mem`, sort by score desc, tie-break by `gpus_per_node` asc, print the top point's `hpc_defaults` YAML block to stdout.

### J.7 Internal data flow

```text
benchmark_hpc.sh
    │   reads scripts/simulation/benchmark_points.tsv
    │
    ├─ ensures REFERENCE_TPR exists under results/benchmarks/.../<comp>/
    │
    └─ per point i: sbatch sbatch_benchmark_hpc.sh
            │   --export=POINT_INDEX=i,SIMS_PER_NODE=k,…,REFERENCE_TPR=…
            ▼
        sbatch_benchmark_hpc.sh  (compute node)
            ├─ rocm-smi → rocm-smi.tsv (background)
            ├─ for i in 0..k-1:
            │      gmx mdrun -s prun.tpr -deffnm prun-bench-$i -ntomp K -resethway &
            └─ wait
                    │
                    ▼
              points/<i>/prun-bench-{0..k-1}.{log,edr,xtc}
                    │
                    ▼
   analyze_benchmark.py
            ├─ parse all prun-bench-*.log → ns/day per slot
            ├─ aggregate + rocm-smi join → summary.csv
            └─ --recommend → YAML block to stdout
                    │
                    ▼
            user pastes into config.yaml; status row 10 flipped to [x]
```

### J.8 Edge-case matrix

| Case | Expected behaviour |
| --- | --- |
| `REFERENCE_TPR` missing | `benchmark_hpc.sh` errors with a one-line "run `--build-tpr-only` first" message |
| One benchmark point times out | Its `prun.log` lacks the `Performance:` line; `analyze_benchmark.py` flags it as `status=incomplete`, omits from ranking |
| `gpu_test` quota exceeded (> 2 jobs simultaneous) | Orchestrator detects via `--partition gpu_test` and submits sequentially with `--dependency=afterany:<prev_jobid>` |
| `--gres=gpu:N` with `gpus_per_node < sims_per_node` (points 8–9) | `HIP_VISIBLE_DEVICES=$((i % gpus_per_node))` round-robins; flag in summary as `oversubscribed_gpu=True` |
| `rocm-smi` not on PATH | Sampler logs a one-line warning, continues without it; `analyze_benchmark.py` populates GPU columns with `NaN` |
| Cosmic ray: one slot crashes mid-run | Same as timeout case; report partial points, score the rest |
| Re-run on the same date | Outputs go under `<date>__<run-index>/`; `latest` symlink updated atomically |
| `analyze_benchmark.py` finds two points within 5 % score | Tie-break by `gpus_per_node` asc, then by `sims_per_node` asc; print both in `summary.md` for the user |
| `--recommend` on an incomplete sweep | Refuse with non-zero exit unless `--allow-partial` is passed |
| User wants to add a 10th point post-hoc | Edit `benchmark_points.tsv`, run `benchmark_hpc.sh --only-new`; new point sbatched, re-run `analyze_benchmark.py` |

### J.9 Test plan — `tests/martini_pipeline/test_analyze_benchmark.py`

Unit tests around `analyze_benchmark.py` only (the bash drivers are integration-only and exercised manually on HPC; mirrors the step-9 split where `submit_simulations.sh` has dry-run tests but `sbatch_simulations.sh` is HPC-only).

1. `test_parse_perf_line` — single `Performance: 542.3 ns/day` line extracted correctly from a fixture log.
2. `test_parse_perf_missing_returns_none` — log without a `Performance:` line returns `None`, point marked incomplete.
3. `test_aggregate_two_slots` — two slot logs (300 + 320 ns/day) → aggregate 620 ns/day.
4. `test_score_calculation` — 620 ns/day in 0.5 node-hours → score 1240; matches expected within 1e-6.
5. `test_rocm_smi_join` — synthetic `rocm-smi.tsv` joined; `gpu_util_mean` computed correctly.
6. `test_recommendation_picks_top_score_under_mem_cap` — three synthetic points; one disqualified by mem headroom; recommendation is the top-scoring of the remaining two.
7. `test_recommendation_tie_break_by_gpu_count` — two points within 5 %; one uses 4 GPUs, one uses 8; 4-GPU point wins.
8. `test_recommend_refuses_partial_sweep` — one of three points incomplete; `--recommend` exits non-zero without `--allow-partial`.
9. `test_summary_md_renders` — sanity check that the markdown table renders with the right column headers.
10. `test_yaml_block_round_trips` — recommended YAML block parses back into a `MartiniPipelineHpcDefaultsConfig` cleanly.

Plus one bash-level dry-run test for `benchmark_hpc.sh`:

11. `test_benchmark_dry_run_emits_one_sbatch_per_point` — `--dry-run` against the canonical 9-point TSV → 9 `[DRY RUN] sbatch …` lines.

### J.10 Layout & dependencies

```text
scripts/simulation/
    benchmark_hpc.sh                       ← new (orchestrator)
    sbatch_benchmark_hpc.sh                ← new (worker)
    benchmark_points.tsv                   ← new (default sweep table)
scripts/python/
    analyze_benchmark.py                   ← new
tests/martini_pipeline/
    test_analyze_benchmark.py              ← new
    fixtures/benchmark/                    ← new (sample prun.log, rocm-smi.tsv)
results/benchmarks/martini_pipeline/
    <ISO date>/{points/,summary.{csv,json,md}}   ← runtime output
    latest -> <date>                       ← symlink
docs/
    martini_pipeline_plan.md               ← status row 10 → [x] + Decisions 43–N
config.yaml                                ← hpc_defaults overwritten with data-derived values
```

Python deps: `pandas` (CSV → markdown), already in `lipid_gnn` env. No new bash deps; `rocm-smi` is part of the loaded ROCm module.

### J.11 Acceptance criteria

- `pytest tests/martini_pipeline/test_analyze_benchmark.py -q` passes locally.
- `bash scripts/simulation/benchmark_hpc.sh --dry-run` prints 9 coherent `sbatch …` invocations.
- On HPC: real sweep completes in < 90 wallclock min; `summary.csv` has 9 rows; `summary.md` includes the recommendation block. *(Manual; not in CI.)*
- `python scripts/python/analyze_benchmark.py --recommend` prints a YAML block syntactically compatible with `config.yaml`'s `martini_pipeline.hpc_defaults`.
- Pasting the block into `config.yaml` and running the existing pipeline tests (`pytest tests/martini_pipeline/ -q`) still passes (sanity check on schema).
- `docs/martini_pipeline_plan.md` row 10 flipped to `[x]`; thesis story updated with the recommendation rationale.
- `thesisStory.md` gains a one-paragraph note linking to `results/benchmarks/.../<date>/summary.md` as the sizing justification (per §7 of this plan).

### J.12 Open questions (need user input before implementation)

1. **Reference system.** Recommend POPC100 (rationale in J.2 Decision 1). Alternative: a 3-system average (POPC100, DPPC100, CHOL40_DPPC60) to capture lipid-class spread. Cost: 3× sweep, 3× quota. Accept POPC100-only, or expand?
2. **Production length per point.** Recommend 100k nsteps (1 ns at dt=0.01 ps). For very fast configs this is ~30 s of GPU time, possibly too short to amortise the `-resethway` reset window. Bump to 200k? Or detect dynamically (run until `nsteps >= 100k AND elapsed >= 60 s`)?
3. **Sweep size — 9 points or finer?** 9 points is the recommended curated set (J.3). Finer (20+ points sweeping `cpus_per_sim` at half-step granularity) would land a better fit but cost ~3× quota and 3× analysis-figure complexity. Recommend 9 for step 10 and a single follow-up "fine sweep" only if step 11 reveals a regime where the recommendation is dominated by `cpus_per_sim` sensitivity.
4. **Re-use eq trajectories from existing systems?** The 70 legacy systems already have `prun.tpr` under `data/membrane_only/POPC100/run/`. Symlinking that file as the benchmark reference saves the ~10 min eq run. Caveat: legacy `prun.tpr` was built with a slightly different MDP (per [Step 3 audit](#step-3-mdp-audit)); strictly speaking, the benchmark should use the frozen step-4 templates. Recommend: re-run eq once via `--build-tpr-only` for cleanliness.
5. **`rocm-smi` integration depth.** Recommend background sampler + simple aggregation (mean util, mean power, peak VRAM). Anything fancier (per-step util, temperature curves) is thesis-figure-only, not recommendation-driving. Accept simple aggregation, or want the richer profile?
6. **Auto-edit `config.yaml`?** Recommend no — print the YAML block, user pastes (audit-friendly, matches the existing audit-freeze pattern). Override with a `--apply` flag?
7. **Where to store benchmark artifacts.** Recommend `results/benchmarks/martini_pipeline/<date>/` (gitignored, results-dir). Alternative: commit `summary.{csv,md}` so future contributors can read the rationale without HPC access. Recommend committing `summary.md` + `summary.csv` only (small; thesis-cited); ignoring `prun.log` and `rocm-smi.tsv` (large).
8. **CPU partition (`general1`) coverage.** Recommend defer to a separate sub-step. Worth adding a single CPU point (8 sims/node, no GPU) to ground the GPU vs CPU speedup figure in the thesis? Cost: 1 extra `sbatch`, no extra code (existing `sbatch_benchmark_hpc.sh` handles `gpus_per_node=0`).
9. **Memory headroom fraction.** Recommend 0.70 (J.2 Decision 8). Alternatives: 0.50 (very conservative, leaves room for memory spikes on cholesterol-rich systems) or 0.85 (squeezes throughput). Pick one before implementation; affects the recommendation but not the data collection.
10. **Variance estimate — single run or three?** Recommend single per point. Goethe-HLR's MI210 nodes have shown ≤ 2 % run-to-run variance in prior projects; benchmark precision is dominated by the curated-point coverage, not by sampling. Triple-replicate balloons the sweep to 27 jobs.

### J.13 New decisions to log on completion

- **Decision 43** — Step 10 benchmarks POPC100 only (J.12 Q1 answer pending) at 100k nsteps, using `-resethway` to exclude load-balancing startup from the ns/day measurement.
- **Decision 44** — Sweep is a 9-point curated subset of the `sims_per_node × cpus_per_sim × gpus_per_node` space (J.3), not a Cartesian product; rationale is throughput per node-hour, not exhaustive scaling laws.
- **Decision 45** — Benchmark TPR is built once via `--build-tpr-only` and re-used across all sweep points; insane + minimization + equilibration are config-independent and pre-amortised.
- **Decision 46** — Recommendation logic is mechanical: top score under a memory-headroom filter (default 70 %), tie-break by smaller `gpus_per_node` for queue friendliness. Auditable via the per-point CSV.
- **Decision 47** — `analyze_benchmark.py --recommend` prints the YAML block; user pastes manually into `config.yaml`. Matches the audit-freeze flow of [Step 3](#step-3-mdp-audit).
- **Decision 48** — `summary.md` and `summary.csv` are committed under `results/benchmarks/martini_pipeline/<date>/` for thesis traceability; raw per-point logs are gitignored.
- *Plus any decisions arising from J.12 Q1–Q10.*

---

## Appendix K — Step 10b detailed plan: CPU benchmark on `general1`

### K.1 Scope

Run a CPU-only throughput sweep on the `general1` partition using the spack-installed GROMACS 2022 (`gmx_mpi`), so we have a calibrated CPU baseline for:

- **Thesis figure** — GPU/CPU speedup ratio for the corner systems (≈ 12 k beads each).
- **Overflow planning** — if GPU quota or hardware availability becomes a bottleneck during step-11 production, we know whether routing a fraction of the queue to `general1` is feasible (in wall-time / CPU-hours per system).
- **A second `hpc_defaults_cpu` block** in `config.yaml`, separate from the GPU defaults, that `submit_simulations.sh --partition general1` could read in a future enhancement.

This is explicitly *separate* from Appendix J:

- Different GROMACS module → different `module load` sequence
- `gmx_mpi` binary (not `gmx`) → different invocation pattern
- No GPUs → one fewer sweep dimension, but multi-rank MPI domain decomposition becomes interesting
- A different `sbatch_benchmark_hpc_cpu.sh` worker (the GPU worker hard-codes `gmx`, `HIP_VISIBLE_DEVICES`, ROCm module)

#### What's in scope

- `scripts/simulation/benchmark_hpc_cpu.sh` — orchestrator analogous to `benchmark_hpc.sh` but always submits to `general1`, never sets `--gres`, loads the spack stack.
- `scripts/simulation/sbatch_benchmark_hpc_cpu.sh` — worker. Loads `mpi/openmpi/5.0.0` then `gromacs/2022.4-gcc-11.3.1-zx2wwcx`. Runs the sweep point as `mpirun -np 1 gmx_mpi mdrun ...` (or `mpirun -np N` for the multi-rank decomp points; see K.3).
- `scripts/simulation/benchmark_points_cpu.tsv` — sweep table with CPU-relevant columns (no `gpus_per_node`, plus `mpi_ranks_per_sim` for the decomp dimension).
- `scripts/python/analyze_benchmark.py` extension — accept a `--cpu` flag (or auto-detect from `point_meta.json["device"]`) and emit a second YAML block: `hpc_defaults_cpu:`.
- `tests/martini_pipeline/test_benchmark_hpc_cpu.py` — bash-level dry-run tests (2–3).
- `config.yaml` — new `martini_pipeline.hpc_defaults_cpu` stub.
- `lipid_gnn/config.py` — parse the new sub-block (small structural addition).
- `docs/martini_pipeline_plan.md` — status table row 10b → `[x]`, Decisions 49+ appended.

#### What's out of scope

- Production-running on `general1`. The benchmark proves feasibility; actually wiring `submit_simulations.sh` to support `general1` is a future step (probably 10c) that loads the spack modules conditionally.
- Comparing GROMACS 2022 vs v2025.4 on the same hardware. We've already documented two divergences (`NA+`/`CL-` ion naming, missing MDP-key acceptance) and don't want this benchmark to drag those in.
- TPR-building on `general1`. We re-use the same `prun.tpr` produced by Phase 1 of the GPU benchmark (built with v2025.4) — see K.2 Decision 3 for the version-mismatch caveat.

### K.2 Locked-in design decisions

1. **Reuse the GPU benchmark's `prun.tpr`.** Phase 1 of `benchmark_hpc.sh` produces a clean POPC100 `prun.tpr` under `/work/.../martini_pipeline/benchmark/POPC100/run/prun.tpr`. The CPU benchmark consumes that same file via `REFERENCE_TPRS`; no separate Phase 1 needed. Saves quota, and we benchmark the *same* physics that v2025.4 will run in production.
2. **GROMACS 2022 reads v2025.4 TPRs.** TPRs are forward-compatible — a 2022 `gmx_mpi mdrun` can run a v2025-built TPR (it'll print a "downgraded" warning at startup but the integration is identical). Validated experimentally before submitting the full sweep (K.9 acceptance criterion 1).
3. **No `mdrun` flags that depend on v2025-only features.** We do *not* pass `-resethway` to the CPU worker — `-resethway` exists in GROMACS 2022 (since 2018), but PME tuning differs between versions and we don't want startup-cost differences contaminating the comparison. (Reverse argument also holds: keeping it removes noise. Pending confirmation in K.12 Q4.)
4. **MPI-rank-per-slot via `mpirun -np <N>`.** `gmx_mpi mdrun` requires `mpirun`. For single-rank slots: `mpirun -np 1 gmx_mpi mdrun -ntomp <K> ...`. For multi-rank: `mpirun -np <N> gmx_mpi mdrun -ntomp <K> ...` and let `gmx_mpi` do the domain decomposition. This adds one sweep dimension absent from the GPU benchmark.
5. **One reference comp.** POPC100 only, matching the GPU benchmark, so the CPU/GPU comparison is on identical physics. Multi-comp extension is a one-line points-file change for later.
6. **Sweep length: same 100 000 steps (2 ns) as GPU benchmark.** CPU is ≈ 30× slower, so 2 ns of POPC100 on 16 cores ≈ 6–10 min per point. Quota: 8 points × ≤ 15 min ≤ 2 CPU-hours total — trivial on `general1`.
7. **Outputs under `results/benchmarks/martini_pipeline/<date>/cpu/`.** Sibling to the GPU sweep under the same date, so `analyze_benchmark.py` can correlate both into one `summary.md`. `point_meta.json` records `"device": "cpu"`.
8. **Recommendation logic separated by device.** `--recommend` picks the best GPU point under the GPU mem-headroom filter; `--recommend --cpu` picks the best CPU point under a similar CPU mem-headroom filter (default 70 %). Two independent YAML blocks — `hpc_defaults:` and `hpc_defaults_cpu:`.
9. **No `HIP_VISIBLE_DEVICES`, no `--gres`.** CPU node; everything goes through `OMP_NUM_THREADS` and `mpirun`'s rank placement. We optionally set `OMP_PLACES=cores OMP_PROC_BIND=close` (proven to help Martini throughput on shared CPU nodes); confirmation in K.12 Q5.
10. **Walltime: 30 min per sbatch.** Generous for a 2-ns POPC100 run on any reasonable CPU layout.

### K.3 Sweep table — `benchmark_points_cpu.tsv`

Need K.12 Q1 (node core count) to finalise. Strawman assuming ≥ 64 cores per `general1` node:

| # | label | sims/node | mpi_ranks/sim | cpus/sim (= ntomp) | total CPUs | rationale |
|---|---|---|---|---|---|---|
| 1 | 1sim_1rank_16omp | 1 | 1 | 16 | 16 | single-sim baseline; thread-only |
| 2 | 1sim_1rank_32omp | 1 | 1 | 32 | 32 | single-sim, max threads — diminishing returns probe |
| 3 | 1sim_2rank_8omp  | 1 | 2 | 8  | 16 | single sim, 2-way MPI decomp + threads |
| 4 | 1sim_4rank_4omp  | 1 | 4 | 4  | 16 | single sim, 4-way MPI decomp — typical small-system sweet spot |
| 5 | 2sim_1rank_16omp | 2 | 1 | 16 | 32 | two parallel sims, no MPI decomp each |
| 6 | 4sim_1rank_8omp  | 4 | 1 | 8  | 32 | four parallel sims — max packing |
| 7 | 4sim_1rank_16omp | 4 | 1 | 16 | 64 | four parallel sims, more threads each |
| 8 | 8sim_1rank_8omp  | 8 | 1 | 8  | 64 | eight parallel sims, packing-dominant regime |

The "aggregate ns/day per node-hour" metric (Decision 46) ranks all 8 cleanly even though the parallelism axis differs.

### K.4 Public API — `scripts/simulation/benchmark_hpc_cpu.sh`

```text
usage: benchmark_hpc_cpu.sh
    [--reference-tpr PATH]          # required; usually .../benchmark/POPC100/run/prun.tpr
    [--nsteps N]                    # default 100000
    [--points PATH]                 # default benchmark_points_cpu.tsv
    [--output-root PATH]            # default results/benchmarks/martini_pipeline/<ISO date>/cpu
    [--partition NAME]              # default general1 (CLI override allowed for testing)
    [--time HH:MM:SS]               # default 00:30:00 per point
    [--dry-run]
```

Behaviour:

1. Verify `--reference-tpr` exists. If not, fail with "run benchmark_hpc.sh first to build it".
2. For each row of the points TSV: build the export env (`REFERENCE_TPR`, `SIMS_PER_NODE`, `MPI_RANKS_PER_SIM`, `CPUS_PER_SIM`, `NSTEPS`), then submit one `sbatch sbatch_benchmark_hpc_cpu.sh`.
3. Mirror the GPU script's chaining (`afterany:<prev>`) so SLURM serialises submissions politely on `general1`'s QOS.
4. `--dry-run` prints the would-be `sbatch` line per point, no real submission.

### K.5 Public API — `scripts/simulation/sbatch_benchmark_hpc_cpu.sh`

Static SBATCH directives:

```bash
#SBATCH --job-name=lipid-bench-cpu
#SBATCH --mail-user=pberger@fias.uni-frankfurt.de
#SBATCH --mail-type=FAIL
#SBATCH --account=cellmembrane
#SBATCH --output=logs/benchmarks/cpu-%j.out
#SBATCH --error=logs/benchmarks/cpu-%j.err
```

Dynamic resources (`--partition`, `--time`, `--cpus-per-task`, `--mem`) come from the orchestrator's command line.

Inside the worker:

1. `set -euo pipefail`
2. `module purge`
3. `module load mpi/openmpi/5.0.0`
4. `module load gromacs/2022.4-gcc-11.3.1-zx2wwcx`
5. (no conda activation needed — analyse step is on the login node afterwards)
6. `cd "$BENCH_POINT_DIR"`
7. Symlink `REFERENCE_TPR` to local `prun.tpr` (same as GPU worker)
8. For each slot `i in 0..SIMS_PER_NODE-1`:
   - `mkdir slot_$i; cd slot_$i; ln -s ../prun.tpr .`
   - Background: `OMP_NUM_THREADS=$CPUS_PER_SIM OMP_PLACES=cores OMP_PROC_BIND=close mpirun -np $MPI_RANKS_PER_SIM gmx_mpi mdrun -s prun.tpr -deffnm prun-bench -ntomp $CPUS_PER_SIM -nsteps $NSTEPS >prun.log 2>&1 &`
9. `wait` all PIDs, propagate worst exit code.

### K.6 `analyze_benchmark.py` extension

- Walk `<root>/cpu/points/*/prun-bench-*.log` (note the extra `cpu/` segment), parse `Performance:` lines exactly as for GPU.
- Per-point columns added to `summary.csv`: `device`, `mpi_ranks_per_sim`.
- New CLI flag: `--cpu` toggles which device's points are considered for `--recommend`. Defaults to GPU (preserves existing behaviour).
- `summary.md` gains a "CPU baseline (general1)" section with its own ranking table.
- YAML block for `--recommend --cpu`:

  ```yaml
  hpc_defaults_cpu:
    sims_per_node: <N>
    mpi_ranks_per_sim: <R>
    cpus_per_sim: <K>
    mem_per_sim: "<XG>"
  ```

### K.7 Internal data flow

```text
GPU benchmark_hpc.sh  Phase 1
        │  builds /work/.../benchmark/POPC100/run/prun.tpr  (v2025.4)
        ▼
benchmark_hpc_cpu.sh  --reference-tpr <that path>
        ├─ per point: sbatch sbatch_benchmark_hpc_cpu.sh
        │      env: REFERENCE_TPR, SIMS_PER_NODE, MPI_RANKS_PER_SIM,
        │           CPUS_PER_SIM, NSTEPS, BENCH_POINT_DIR
        ▼
sbatch_benchmark_hpc_cpu.sh  (general1 node)
        ├─ module load mpi/openmpi/5.0.0
        ├─ module load gromacs/2022.4-gcc-11.3.1-zx2wwcx
        ├─ for i in 0..N-1:
        │      mpirun -np R gmx_mpi mdrun -ntomp K -s prun.tpr ... &
        └─ wait
                │
                ▼
        points/<label>/slot_<i>/prun-bench.log
                │
                ▼
analyze_benchmark.py --root <date> --recommend --cpu
        → prints hpc_defaults_cpu YAML block
```

### K.8 Edge-case matrix

| Case | Expected behaviour |
| --- | --- |
| `--reference-tpr` missing | Fail-fast with hint to run `benchmark_hpc.sh` first |
| GROMACS 2022 refuses the v2025 TPR | Fail-fast at the first sbatch; manual fallback documented in K.12 Q2 |
| `mpirun` invokes more ranks than available cores | mdrun errors at startup; per-slot log records, run counted as `status=incomplete` in analyze |
| `OMP_NUM_THREADS × MPI_RANKS > cores_per_node` | Same as above; we keep the table tight to avoid this |
| `general1` `MaxSubmitJobs` exceeded | `afterany` chaining minimises pending count; if still hit, user resubmits remaining via `--points` with a trimmed TSV |
| `--dry-run` | Prints would-be sbatch line per point, no real submission |
| Multiple `general1` jobs from the same user collide on cores | SLURM allocates per `--cpus-per-task`; cgroup-fenced — no inter-job CPU contention. The within-job slot-vs-slot contention IS what we're measuring. |

### K.9 Acceptance criteria

1. **Compatibility smoke**: a single `mpirun -np 1 gmx_mpi mdrun -s <v2025.4_built_tpr>` on `general1` runs ≥ 100 steps without errors. Run manually before the full sweep.
2. `pytest tests/martini_pipeline/test_benchmark_hpc_cpu.py -q` passes (dry-run tests only; no real sbatch).
3. `bash scripts/simulation/benchmark_hpc_cpu.sh --reference-tpr <tpr> --dry-run` prints 8 coherent `[DRY RUN] sbatch ...` lines.
4. Real sweep completes in < 2 hours wallclock.
5. `python scripts/python/analyze_benchmark.py --recommend --cpu` prints a syntactically valid `hpc_defaults_cpu:` YAML block.
6. `summary.md` includes both GPU and CPU ranking tables, with a derived "GPU/CPU speedup ratio" row (top-GPU ns/day ÷ top-CPU ns/day per node).
7. Plan doc row 10b → `[x]`.

### K.10 Layout & dependencies

```text
scripts/simulation/
    benchmark_hpc_cpu.sh                       ← new
    sbatch_benchmark_hpc_cpu.sh                ← new
    benchmark_points_cpu.tsv                   ← new
scripts/python/
    analyze_benchmark.py                       ← extended with --cpu flag
tests/martini_pipeline/
    test_benchmark_hpc_cpu.py                  ← new
logs/benchmarks/                               ← created at runtime
results/benchmarks/martini_pipeline/
    <ISO date>/cpu/{points/, summary_cpu.csv}  ← runtime output (sibling to GPU)
config.yaml                                    ← + hpc_defaults_cpu stub
lipid_gnn/config.py                            ← parse hpc_defaults_cpu sub-block
docs/martini_pipeline_plan.md                  ← row 10b + Decisions 50–N
```

No new Python deps. New bash deps: `mpirun` (from the openmpi module).

### K.11 Status-table row

To be added before row 11 once we agree on the design:

```markdown
| 10b | CPU benchmark on general1 (`benchmark_hpc_cpu.sh` + sbatch worker; populate `hpc_defaults_cpu`) | [ ] | |
```

### K.12 Open questions (need user input before implementation)

1. **Cores per `general1` node?** I've assumed ≥ 64 in the strawman sweep table. If the node is 32-core, drop points 7–8 and add a 32-core single-sim entry. If 128-core, double everything for points 5–8.

2. **Version-mismatch behaviour.** Does GROMACS 2022 actually accept a v2025.4-built `prun.tpr`? Two answers possible:
   - (a) **Yes, with a downgrade warning.** Keep K.2 Decision 1 (TPR reuse) and proceed.
   - (b) **No, refuses outright.** Fall back to building a separate `prun.tpr` with the 2022 module on `general1` first (extra Phase 1, ~30 min CPU). Recommend testing (a) manually with a 100-step run before the full sweep (K.9 criterion 1).

3. **MPI-decomp dimension worth keeping?** Martini systems with ≤ 12 k beads are at the lower edge of where MPI domain decomposition pays off — communication overhead can dominate. Points 3–4 of K.3 probe this. Recommend keeping them: confirming "1-rank-N-threads beats N-ranks-1-thread for our systems" is a useful thesis bullet. Override?

4. **`-resethway` on CPU benchmark?** GPU benchmark uses it to exclude DLB warm-up. CPU has different PME-tuning behaviour in 2022; including it might bias the comparison. Recommend **off** for CPU (steady-state from step 1 is closer to what production overflow would see). Override?

5. **Thread pinning.** Default `OMP_PLACES=cores OMP_PROC_BIND=close` is generally helpful on shared CPU nodes (avoids OS thread migration). Adds ~5–10 % throughput on prior projects. Recommend **on**. Override?

6. **`hpc_defaults_cpu` schema.** Strawman in K.6 is `{sims_per_node, mpi_ranks_per_sim, cpus_per_sim, mem_per_sim}`. Should it also include `partition: general1` and `module_gromacs_cpu: gromacs/2022.4-gcc-11.3.1-zx2wwcx` so future `submit_simulations.sh --partition general1` can route automatically? Recommend yes — keeps the bash side dumb.

7. **Naming.** `benchmark_hpc_cpu.sh` keeps things grouped under `benchmark_*` (good for tab completion), but I could also call it `benchmark_general1.sh`. Recommend the device-typed name (`_cpu`); a future Intel/AMD CPU benchmark on a different partition would just be another `_cpu` script with a different module load.

8. **Step number.** Calling this **10b** keeps the main step-10 deliverable focused (GPU defaults for the production target). Alternative: bump to step **10.5** or just **10.1**. Recommend **10b** for clarity in the status table.

9. **mpirun launch model.** `mpirun -np 1 gmx_mpi mdrun -ntomp K` vs `gmx_mpi mdrun -ntomp K` (with `mpirun` implicit via SLURM's `srun`). Recommend explicit `mpirun -np <R>` so the number of MPI ranks is unambiguous and matches what we want to measure. `srun` is an option if `mpirun` misbehaves on `general1`; need to test.

10. **Memory headroom.** General1 nodes are typically 256–512 GB; Martini POPC100 is < 1 GB per slot. Mem is never the bottleneck on CPU — we can drop the headroom filter from K.6's recommendation logic for the CPU branch. Recommend dropping it (always pick top-score CPU point).

### K.13 New decisions to log on completion

- **Decision 50** — CPU benchmark on `general1` is a separate orchestrator + worker (`benchmark_hpc_general1.sh` + `sbatch_benchmark_hpc_general1.sh`), not a flag on the GPU benchmark. Per K.12 Q7 the script is named by partition (`_general1`) rather than by device (`_cpu`), keeping room for a future Intel-CPU partition with its own script and module stack. Rationale: different GROMACS module (spack `gromacs/2022.4-gcc-11.3.1-zx2wwcx`), different mdrun binary (`mpirun -np <R> gmx_mpi mdrun`), different parallelism dimension. Mixing them in one script would dwarf the actual benchmarking logic in conditionals.
- **Decision 51** — CPU benchmark builds its **own** `prun.tpr` on `general1` with the 2022 toolchain by default (Phase 1 setup job, lands under `$WORK/.../general1_benchmark/POPC100/`). `--reference-tpr PATH` skips Phase 1 entirely and uses any existing TPR (e.g. the GPU benchmark's v2025.4 output). Rationale: the GPU cluster was full when we needed CPU numbers, so the CPU benchmark must be runnable without GPU quota first. TPR forward-compatibility (v2025 → 2022) is a nice fallback but not the default path.
- **Decision 58** — Phase 1 setup on `general1` uses a 4-line wrapper (`_gmx_mpi_wrapper.sh`) that does `exec mpirun -np 1 gmx_mpi "$@"`, passed to `run_martini_pipeline.py` as `--gmx`. Lets the existing pipeline (which has a single `gmx_executable` parameter shared between grompp and mdrun) work unmodified on a partition where only `gmx_mpi` is available. grompp picks up a small MPI-init overhead per call (acceptable; 3 grompp invocations per pipeline run); mdrun runs single-rank under mpirun and parallelises via `-ntomp 40` OpenMP threads.
- **Decision 52** — Sweep adds an `mpi_ranks_per_sim` dimension absent from the GPU sweep (K.12 Q3 kept), because CPU domain decomposition is meaningful below the GPU-throughput regime. Points 3–4 (`1sim_2rank_20omp`, `1sim_4rank_10omp`) probe whether 2/4-way MPI decomp beats pure threading on ≤ 12 k-bead systems.
- **Decision 53** — `hpc_defaults_cpu` is a separate config block from `hpc_defaults` (which stays GPU-only). It includes `partition: "general1"`, `module_gromacs_cpu`, and `module_mpi` (per K.12 Q6) so a future `submit_simulations.sh --partition general1` extension can route automatically without re-introspecting the analyzer output.
- **Decision 54** — CPU recommendation logic drops the memory-headroom filter (per K.12 Q10): `general1` nodes have 256+ GB RAM and Martini POPC100 fits in < 1 GB per slot; mem is never the bottleneck. Tie-break order changes from GPU's `(gpus_per_node, sims_per_node)` to `(mpi_ranks_per_sim, sims_per_node)` — prefer fewer MPI ranks for less comm overhead.
- **Decision 55** — `-resethway` is **off** for CPU benchmarks (per K.12 Q4 recommendation). GROMACS 2022 PME-tuning behaviour differs from v2025.4; including the half-run reset would bias the GPU/CPU comparison via version-specific warm-up. CPU benchmarks measure steady-state throughput from step 0.
- **Decision 56** — Thread pinning **on** for CPU worker: `OMP_PLACES=cores OMP_PROC_BIND=close` (per K.12 Q5). Validated to add ~5–10 % throughput on shared CPU nodes in prior projects by avoiding OS thread migration.
- **Decision 57** — Explicit `mpirun -np <R>` invocation for *every* slot, including 1-rank ones (per K.12 Q9). Keeps the number of MPI ranks unambiguous and matches the form needed for the multi-rank decomp points; no implicit `srun` magic to debug if it misbehaves on `general1`.

---

## Appendix L — Step 10c detailed plan: route production work to `general1`

### L.1 Scope

Wire the step-9 submission layer (`submit_simulations.sh`) to support `--partition general1`, so corner-fill production (step 11) and any future bulk runs can dispatch to the CPU partition transparently. Without this step, `hpc_defaults_cpu` (calibrated by step 10b) is a dead-letter — `submit_simulations.sh` only reads `hpc_defaults` and only knows how to load the ROCm gromacs module.

The benchmark proved the CPU pipeline works end-to-end on `general1` (insane → grompp → mpirun-gmx_mpi mdrun, all with the spack 2022 toolchain). Step 10c lifts that proof into the production submission layer, reusing the wrapper + module-load patterns we already debugged.

#### What's in scope

- `scripts/bash/submit_simulations.sh` — partition-aware dispatch. When `--partition general1` (or any future CPU partition listed in a small registry), read `hpc_defaults_cpu` instead of `hpc_defaults`, omit `--gres`, propagate `MPI_RANKS_PER_SIM`, and route to the CPU worker.
- `scripts/bash/sbatch_simulations_general1.sh` — new compute-node worker. Mirrors `sbatch_simulations.sh` (same `RUN_<i>_*` env-var protocol, same fan-out + `wait` loop) but loads the spack openmpi + gromacs/2022 modules, uses the `_gmx_mpi_wrapper.sh` shim for the pipeline, and adds OpenMP thread pinning.
- `lipid_gnn/config.py` — parse the new `martini_pipeline.hpc_defaults_cpu` block into a `MartiniPipelineHpcDefaultsCpuConfig` dataclass.
- `tests/martini_pipeline/test_submit_simulations.py` — extend existing bash-level dry-run tests with a `--partition general1` matrix (no real sbatch).
- `docs/martini_pipeline_plan.md` — status table row 10c → `[x]`, Decisions 58–N appended.

#### What's out of scope

- A CPU re-benchmark after this lands. The step-10b numbers are authoritative until the lipid pool or GROMACS version changes.
- Cross-partition load balancing (split a queue across `gpu` + `general1` simultaneously). The user runs `submit_simulations.sh` twice with different `--partition` if they want both.
- New compositions, new mdp templates, new analysis. Pure plumbing.
- A `general1` analog of `gpu_test` quick-iteration partition. `general1` has no time-limit shortcut.

### L.2 Locked-in design decisions

1. **Two workers, one orchestrator.** `submit_simulations.sh` keeps a single CLI surface and dispatches internally based on `--partition`. The two compute-node workers (`sbatch_simulations.sh` GPU and `sbatch_simulations_general1.sh` CPU) stay separate — mirrors the benchmark architecture (10 vs 10b) and avoids piling partition-conditional branching into one large bash file. Matches Decision 50.
2. **Partition-to-config map is explicit, not heuristic.** `submit_simulations.sh` has a small lookup `case` block translating `$PARTITION` into `$DEFAULTS_KEY` and `$WORKER`. No fallback. Unknown partitions fail fast; the user adds a row to the table when they want a new one. Auditable.
3. **`MPI_RANKS_PER_SIM` becomes a first-class CLI flag** (`--mpi-ranks-per-sim N`) with default from `hpc_defaults_cpu.mpi_ranks_per_sim`. Used only by the CPU worker; the GPU worker ignores it.
4. **Resource scaling differs per device.** GPU: `cpus_per_task = sims × cpus_per_sim`, `--gres=gpu:sims`. CPU: `cpus_per_task = sims × mpi_ranks × cpus_per_sim`, no `--gres`. Mem scaling is the same for both: `sims × mem_per_sim`.
5. **CPU worker reuses the existing `_gmx_mpi_wrapper.sh` shim** as the pipeline's `--gmx` argument. The wrapper already picked up the `--map-by :OVERSUBSCRIBE` fix during step 10b; production CPU runs inherit that fix automatically.
6. **Production CPU runs default `mpi_ranks_per_sim = 1`** unless the benchmark recommended otherwise. Production sims are throughput-bound, not latency-bound; multi-rank decomp is only worth it if benchmark data says so. The flag is there for completeness and lets us re-route to the multi-rank winner if step 10b promotes it.
7. **`gpu_test`-style guard rails apply to no other partition.** The orchestrator only enforces `gpu_test`'s 8h-cap-and-max-2 limits when `--partition gpu_test` is in play, unchanged from step 9.
8. **Module names live in config.yaml, not in scripts.** The step-10b `hpc_defaults_cpu` block already carries `module_gromacs_cpu` and `module_mpi_cpu`. Worker reads via `print_config_var.py`. Matches step 9 Decision 7.
9. **Idempotency is unchanged.** `pipeline.run()`'s stage-skip logic is identical regardless of `gmx_executable`. A partial CPU run resumes from the last completed stage; `--missing-from-grid` queries the same manifests.
10. **No partition flag in manifest.** The composition's manifest already records `mdrun_cmd` per stage; that's the source of truth for "this system was run on CPU vs GPU". `summarise_systems()` doesn't need partition awareness for step 11.

### L.3 Public API changes — `submit_simulations.sh`

Additions to the usage block:

```text
[--mpi-ranks-per-sim N]    # CPU only; default from hpc_defaults_cpu.mpi_ranks_per_sim
```

No other CLI changes. `--partition general1` and friends just work after step 10c.

Internal changes:

- Single `case` block translates `$PARTITION` into `$DEFAULTS_KEY` and `$WORKER`.
- `_cfg()` invocations use `martini_pipeline.${DEFAULTS_KEY}.*` instead of the hard-coded `hpc_defaults`.
- `--gres=gpu:N` line conditioned on `$DEFAULTS_KEY == hpc_defaults`.
- `EXPORT_VARS` adds `MPI_RANKS_PER_SIM=…` when on CPU; the GPU worker ignores it (it never reads that env var).

### L.4 Public API — `scripts/bash/sbatch_simulations_general1.sh` (new)

Static `#SBATCH` directives identical to the GPU worker except `--output=logs/simulations/sim-cpu-%j.out` and corresponding `--error`. Inside:

1. `set -euo pipefail`
2. Source conda; activate the env.
3. `module purge; module load "$(_cfg ...module_mpi_cpu)"; module load "$(_cfg ...module_gromacs_cpu)"`
4. Validate `OUTPUT_ROOT`, `N_SIMS_PER_NODE`, `MPI_RANKS_PER_SIM`, `CPUS_PER_SIM` env vars
5. `NTOMP_VALUE = CPUS_PER_SIM` (NOT `$SLURM_CPUS_PER_TASK / N_SIMS` — the CPU resource math accounts for ranks explicitly, see L.2 Decision 4)
6. For `i in 0..N_SIMS-1`:
   - Indirect lookup `COMP="${!RUN_${i}_COMP}"`
   - Per-slot `OUT_DIR`, log paths (same convention as GPU)
   - Subshell exports `OMP_NUM_THREADS=$NTOMP_VALUE`, `OMP_PLACES=cores`, `OMP_PROC_BIND=close`. No `HIP_VISIBLE_DEVICES`.
   - `MDRUN_EXTRA="-ntomp $NTOMP_VALUE -nb cpu"`; pipeline invoked with `--gmx "$PWD/scripts/simulation/_gmx_mpi_wrapper.sh"`.
   - `&` into background; collect PID.
7. After loop: `wait` on every PID; report worst exit code.

Notable differences from the GPU worker:

| | GPU worker | CPU worker |
|---|---|---|
| Module set | `gromacs/v2025.4/rocm-…` | `mpi/openmpi/5.0.0` + `gromacs/2022.4-…` |
| `--gmx` to pipeline | `gmx` (default) | `_gmx_mpi_wrapper.sh` |
| GPU pinning | `HIP_VISIBLE_DEVICES=$i` | (none) |
| Thread pinning | (none; threads share GPU bandwidth) | `OMP_PLACES=cores OMP_PROC_BIND=close` |
| `-nb` arg | implicit (GPU) | explicit `-nb cpu` |
| MPI ranks per slot | n/a | wrapper does `mpirun --map-by :OVERSUBSCRIBE -np 1`; multi-rank inside one slot is benchmark-only |

### L.5 Config schema

Step 10b already added `martini_pipeline.hpc_defaults_cpu` to `config.yaml`. Step 10c only adds a Python-side dataclass in `lipid_gnn/config.py`:

```python
@dataclass(frozen=True)
class MartiniPipelineHpcDefaultsCpuConfig:
    sims_per_node: int
    mpi_ranks_per_sim: int
    cpus_per_sim: int
    mem_per_sim: str
    partition: str = "general1"
    module_gromacs_cpu: str = "gromacs/2022.4-gcc-11.3.1-zx2wwcx"
    module_mpi_cpu: str = "mpi/openmpi/5.0.0"

@dataclass(frozen=True)
class MartiniPipelineConfig:
    …
    hpc_defaults: Optional[MartiniPipelineHpcDefaultsConfig] = None
    hpc_defaults_cpu: Optional[MartiniPipelineHpcDefaultsCpuConfig] = None
```

Plus a `_build_martini_pipeline_hpc_defaults_cpu` builder mirroring the existing GPU version.

### L.6 Internal data flow

```text
submit_simulations.sh --missing-from-grid dppc_corner --prod-ns 100 --partition general1
        │
        ├─ resolve composition list (unchanged from step 9)
        ├─ partition lookup: general1 → DEFAULTS_KEY=hpc_defaults_cpu, WORKER=sbatch_simulations_general1.sh
        ├─ read hpc_defaults_cpu.{sims_per_node, mpi_ranks_per_sim, cpus_per_sim, mem_per_sim} via _cfg()
        ├─ pack: N_batches = ceil(N_total / sims_per_node)
        ├─ per batch, build EXPORT_VARS with MPI_RANKS_PER_SIM added; --gres omitted
        └─ sbatch --partition=general1 --cpus-per-task=$(( sims × ranks × cpus_per_sim )) \
                  --mem=... --export="$EXPORT_VARS" sbatch_simulations_general1.sh
                ▼
        sbatch_simulations_general1.sh (compute node)
        ├─ load openmpi + gromacs/2022
        ├─ for each slot: OMP env + run_martini_pipeline.py --gmx _gmx_mpi_wrapper.sh ... &
        └─ wait; surface worst rc
                ▼
        Per-composition outputs at /work/.../martini_pipeline/<comp>/{minimization,equilibration,run,manifest.json}
        — identical layout to GPU runs; summarise_systems doesn't care which partition produced them.
```

### L.7 Edge-case matrix

| Case | Expected behaviour |
| --- | --- |
| `--partition general1` without `hpc_defaults_cpu` in config.yaml | Fail-fast: "hpc_defaults_cpu missing; run benchmark_hpc_general1.sh first or paste a stub" |
| `--compositions A B C --partition general1` | All A, B, C dispatched to general1 worker. No partition-per-comp routing |
| Resubmission after CPU failure | pipeline.run() stage-skip kicks in; resumes via grompp(stage_in.gro) |
| `--mpi-ranks-per-sim 4` on `gpu_test` | Warning + ignored; GPU worker doesn't read MPI_RANKS_PER_SIM |
| `--mpi-ranks-per-sim 4` on `general1` | `--cpus-per-task = sims × 4 × cpus_per_sim`; mpirun -np 4 per slot via wrapper |
| `--dry-run` | Prints sbatch line with CPU worker path and MPI_RANKS_PER_SIM in EXPORT_VARS |
| Unknown partition (`--partition fancy`) | Fail-fast: "unknown partition; add a row to the case block" |
| Job times out mid-corner-fill | Same as benchmark: partial manifest with `failed_at_<stage>`; next `--missing-from-grid` picks it up. CPU runs longer, so wall-time misjudgments are more likely — see L.12 Q4 |
| Accidental double-submit | pipeline.run() is idempotent; second job no-ops or resumes. Worst case is duplicated CPU-hours |

### L.8 Test plan — `tests/martini_pipeline/test_submit_simulations.py` extensions

All extend the existing dry-run subprocess pattern. No real sbatch.

1. `test_general1_dispatches_to_cpu_worker` — `--partition general1` ⇒ sbatch line contains `sbatch_simulations_general1.sh`.
2. `test_general1_no_gres` — `--partition general1` ⇒ no `--gres=` in the sbatch line.
3. `test_general1_mpi_ranks_in_export` — `--partition general1 --mpi-ranks-per-sim 4` ⇒ EXPORT_VARS contains `MPI_RANKS_PER_SIM=4`.
4. `test_general1_cpus_includes_ranks` — `--partition general1 --sims-per-node 2 --mpi-ranks-per-sim 4 --cpus-per-sim 5` ⇒ `--cpus-per-task=40`.
5. `test_gpu_test_still_uses_gpu_worker` — regression guard.
6. `test_unknown_partition_fails_fast` — `--partition fancy` ⇒ non-zero exit, "unknown partition" in stderr.
7. `test_general1_uses_hpc_defaults_cpu_for_defaults` — no explicit `--sims-per-node` ⇒ dry-run reflects `hpc_defaults_cpu.sims_per_node`.

### L.9 Layout & dependencies

```text
scripts/bash/
    submit_simulations.sh                  ← MODIFIED: partition-aware dispatch
    sbatch_simulations.sh                  ← unchanged
    sbatch_simulations_general1.sh         ← NEW
config.yaml                                ← hpc_defaults_cpu already present from step 10b
lipid_gnn/config.py                        ← +MartiniPipelineHpcDefaultsCpuConfig + builder
tests/martini_pipeline/
    test_submit_simulations.py             ← +7 dry-run cases
docs/martini_pipeline_plan.md              ← row 10c → [x] + Decisions 58–N
```

No new Python deps. No new bash deps.

### L.10 Acceptance criteria

1. `pytest tests/martini_pipeline/test_submit_simulations.py -q` passes locally.
2. `bash scripts/bash/submit_simulations.sh --compositions DIPC100 --prod-ns 1 --partition general1 --dry-run` prints a coherent sbatch line: routes to `sbatch_simulations_general1.sh`, no `--gres=`, `MPI_RANKS_PER_SIM=1`.
3. `--partition gpu_test --dry-run` unchanged from step 9.
4. On HPC: a 1-comp smoke (`--compositions DIPC100 --prod-ns 1 --partition general1`) lands one job, exits clean, manifest `overall_status=ok`.
5. `--missing-from-grid dppc_corner --partition general1 --dry-run` works against a partly-populated output root.
6. Plan doc row 10c → `[x]`.

### L.11 Status-table row to add

```markdown
| 10c | Route submit_simulations.sh to general1 (hpc_defaults_cpu, sbatch_simulations_general1.sh) | [ ] | |
```

### L.12 Open questions (need user input before implementation)

1. **Dispatch model**: single `submit_simulations.sh` with internal `case` (L.2 Decision 2 — my recommendation), or two CLI entry points? Recommend single dispatcher — fewer scripts for users to remember, partition is already a required arg.
2. **Missing `hpc_defaults_cpu`**: when `--partition general1` but the YAML block is absent, should we (a) fail-fast, (b) fall back to hard-coded conservative values, or (c) fall back to `hpc_defaults` and warn? Recommend (a) — failing loud means the user notices that the benchmark hasn't calibrated the partition yet.
3. **MPI-rank dimension in production**: expose `--mpi-ranks-per-sim` CLI flag with default from config (recommended) or always force 1?
4. **CPU wall-time defaults**: step 9's `--time 24:00:00` works on GPU; on CPU at ~500 ns/day a 100 ns POPC100 ≈ 5 h, a 200 ns CHOL-heavy system ≈ 10 h — still under 24h. Recommend keep `--time 24:00:00` default for all partitions; user can pass `--time 48:00:00` for long systems.
5. **`gpu_test`-style guard rails for `general1`**: any partition-specific limits we should enforce in the orchestrator? My draft enforces none. Override if your group's QOS is tighter?
6. **Module-name location**: `module_gromacs_cpu` / `module_mpi_cpu` currently live under `hpc_defaults_cpu` (per step 10b). Recommend keep them there — the analyzer already emits them in that block.
7. **Worker naming**: `sbatch_simulations_general1.sh` (partition-typed) vs `sbatch_simulations_cpu.sh` (device-typed). Recommend partition-typed, mirrors Decision 50.
8. **CHOL handling on CPU**: legacy CHOL-rich systems converge slowly (sterol flip-flop); worth a soft warning note in L.13? Recommend yes — documentation only, no code change.
9. **`--smoke` flag**: should the orchestrator gain a `--smoke` shortcut that submits a single short job? Recommend no — feature creep; the manual command is fine.
10. **Backwards compat**: any user scripts that grep for `sbatch_simulations.sh` directly? Recommend a quick `grep -rn` audit before merging.

### L.13 New decisions to log on completion

- **Decision 58** — `submit_simulations.sh` dispatches via an explicit partition lookup table; unknown partitions fail fast.
- **Decision 59** — Two compute-node workers (`sbatch_simulations.sh` GPU + `sbatch_simulations_general1.sh` CPU), one orchestrator. Mirrors benchmark architecture (step 10 + 10b).
- **Decision 60** — `MPI_RANKS_PER_SIM` is a first-class CLI flag (`--mpi-ranks-per-sim N`) and config key. Default from `hpc_defaults_cpu`. Ignored on GPU partitions.
- **Decision 61** — Module names for both GROMACS builds live in `hpc_defaults_cpu` (CPU) and `hpc.module_gromacs` (GPU). Centralises HPC plumbing in `config.yaml` per step 9 Decision 7.
- **Decision 62** — If `hpc_defaults_cpu` is missing and `--partition general1` is requested, fail-fast rather than fall back to GPU defaults.
- *Plus any decisions arising from L.12 Q1–Q10.*
