# `lipid_gnn/functions_emil/` cleanup plan

**Created 2026-05-18.** Living document — update each section as the corresponding subtask is executed.

`lipid_gnn/functions_emil/` is a vendored copy of an earlier (TPS / committor / free-energy-NN) project. It currently holds **11,887 LOC across 12 modules**. Most of it is dead weight for this project. The exercise: classify everything, audit what survives for bugs, rewrite it in proper project shape, then delete the rest.

Out-of-project notebooks (`scripts/emil/general/`, `scripts/emil/free_energy_nn_paper/`) are explicitly **not** considered users — code reachable only from them is "not used."

---

## 1. Module-by-module categorisation

Top-level survey. Each module is **kept** (reachable from in-project code, or its outputs are consumed by this project) or **dropped** (no in-project consumer once the kept symbols are migrated).

| Module | LOC | Verdict | Reason |
| --- | --- | --- | --- |
| [functions.py](../lipid_gnn/functions_emil/functions.py) | 319 | **kept (2 symbols only)** | `pkl_load` / `pkl_save` used by `lipid_gnn/dataset.py`, `scripts/training/prepare_colab_subset.py`, `scripts/training/smoke_test_sweep.py`, `functions_emil/properties_nn.py`. The rest (`trun`, `mkdir`, `copy`, `cancel`, `DRMSD`, `save_log`, `compute_com_dist`) has zero in-project callers. |
| [calculate_properties.py](../lipid_gnn/functions_emil/calculate_properties.py) | 376 | **kept (semantically)** | `compute_properties` is the source of every training label in `results/properties/<comp>.h5`. Only the out-of-project notebook calls it today, but the *project* depends on its output. Subroutines `voronoi_area_cv`, `compute_bending_modulus`, `undulation_model` are internal helpers. |
| `aimmd_analysis.py` | 1 983 | **drop** | AIMMD / TPS analysis — committor checks, validation sets from `committor_check/*.pkl`. Depends on `utils`, `pathensemble`. No in-project caller. |
| `bilayer_builder.py` | 497 | **drop** | Old `build_bilayer()` flow. Superseded by [lipid_gnn/martini_pipeline/](../lipid_gnn/martini_pipeline/) (modern `insane` + `system_builder`). |
| `bilayer_functions.py` | 360 | **drop** | Dependency of `bilayer_builder.py`. |
| `bilayer_params.py` | 46 | **drop** | Hard-coded params for `bilayer_builder.py` (POPC + IRE1 protein, GMX-21). |
| `committor_nn.py` | 1 039 | **drop** | Committor NN training for TPS. |
| `free_energy_nn.py` | 632 | **drop** | FE-NN paper code (`feed_forward`). |
| `insane.py` | 1 655 | **drop** | Legacy Python-2 vendor copy. Already tracked separately by `project_insane_legacy_cleanup` memory; project uses the modern `insane` package via `martini_pipeline/system_builder.py`. Fold into this cleanup or keep separate — see §4. |
| `pathensemble.py` | 2 337 | **drop** | TPS path-ensemble bookkeeping. |
| `properties_nn.py` | 521 | **drop** | Composition → properties FFNN (precursor to the GNN). Replaced by the GNN. The project's composition-only baseline lives in `lipid_gnn/linear_baseline.py` (Ridge), not this. |
| `utils.py` | 2 122 | **drop** | AIMMD utility blob (imported as `aimmd` by `aimmd_analysis.py` / `committor_nn.py`). |

**Kept surface** = `functions.pkl_load`, `functions.pkl_save`, `calculate_properties.compute_properties` (+ its helpers). Everything else can be deleted once §3 lands — except the "possibly useful in the future" subset in §1b.

### 1b. Possibly useful in the future (third bin)

Things with no in-project caller today that plausibly serve a known item on the project roadmap (M3 lipidome extension, embedding evaluation, protein+membrane transfer, long-simulation robustness). The default disposition is still **delete**, but each entry below should be **read, the load-bearing idea ported into a project-grade module, and then the source deleted** — not vendored as-is. If the corresponding roadmap item is decided against, drop without porting.

| Symbol | Module | Plausible future use |
| --- | --- | --- |
| `compute_com_dist(traj, monA_index, monB_index)` | `functions.py` | Periodic-aware centre-of-mass distance between two atom groups with full minimum-image unwrap. Drop-in primitive for protein–lipid or protein–protein distances once the project extends to protein+membrane systems. Current `martini_pipeline/` and analysis code have no equivalent. |
| `DRMSD(trajectory, connections)` | `functions.py` | Distance-RMSD against a reference connection set. Standard reaction-coordinate / order-parameter primitive — relevant if the protein+membrane phase needs a CV for e.g. protein conformational state or membrane defect. |
| `Network` (composition → properties FFNN), `train`, `train_sweep`, `evaluate`, `ZMatrix` (`Dataset`) | `properties_nn.py` | Composition-only FFNN baseline. Stronger competitor than the current `linear_baseline.py` (Ridge) — would tighten the "GNN vs composition" comparison reported in Stage 5b/c/d. Architecture is simple; if used, port the model class + a minimal training loop into `lipid_gnn/ffnn_baseline.py` rather than vendoring the whole 521-LOC module. |
| `transfer_membrane_comp`, `extract_membrane_composition` | `properties_nn.py` | Composition-string ↔ fractions-dict parsing. **Check first** whether `martini_pipeline.canonical_name` / `lipid_registry` already covers this — if yes, drop; if no, the parsing rules are worth porting. |
| `build_bilayer`, `make_new_initial`, `set_up_aimmd` | `bilayer_builder.py` | Builds a bilayer **with embedded protein** (the IRE1 case), salt, equilibration, restraint setup. The current `martini_pipeline/` is pure-bilayer only. For the long-term protein+membrane goal, this is the closest in-house reference for protein placement, even though the GROMACS calls are GMX-21 era and have to be modernised. Treat as **read-only reference**, not as code to import — and only when the protein+membrane work actually begins. |
| `compute_autocorrelation(list_of_series, list_of_times, Tau, dTau, ...)` | `utils.py` | Time-series autocorrelation across an arbitrary set of trajectories. Directly useful for property-convergence diagnostics on the 1 µs production runs (does the 50-frame discard suffice for `persistence`? for `diffusivity`?). Could feed into `compare_legacy_vs_new_m3.py` or a successor convergence notebook. |
| `recover_trr(trr_file, top_file, chunk_size=100)` | `utils.py` | Chunked TRR-recovery for partially-written trajectories. Operational utility for long HPC runs that hit walltime mid-write — the kind of thing `projected_finish.py` flags. |
| `weighted_quantile(data, weights, q)` | `utils.py` | Weighted quantile primitive. Useful if/when the lipidome analysis (§ Section 6 of the M3 plan) wants quantile bands on the descriptor distributions weighted by composition mole-fraction. |
| `feed_forward`, `RNNArchLSTM`, `feed_forward_dataset`, `RNN_dataset` | `free_energy_nn.py` | Generic FFNN / LSTM scaffolding. Mostly redundant with `properties_nn.Network`; keep `properties_nn` and drop these unless the LSTM specifically becomes useful for time-resolved property prediction. |

Everything else in the dropped modules (`aimmd_analysis.py`, `bilayer_functions.py` outside the items above, `bilayer_params.py`, `committor_nn.py`, the rest of `free_energy_nn.py`, `insane.py`, `pathensemble.py`, the rest of `utils.py`) has no identifiable future hook to the project roadmap and should be deleted outright.

### `compute_properties` — actual callers

| Caller | In project? | Disposition |
| --- | --- | --- |
| `scripts/emil/general/calculate_properties.ipynb` | no | irrelevant per task framing |
| `scripts/emil/free_energy_nn_paper/01_calculate_properties.ipynb` | no | irrelevant |
| (post-cleanup) new `scripts/python/compute_properties.py` CLI | yes | will replace the notebook |

### `pkl_load` / `pkl_save` — actual in-project callers

| Caller | What it loads | Migration |
| --- | --- | --- |
| `lipid_gnn/dataset.py:22` | `<comp>.h5` mean-dict from `results/properties/` | inline `pickle.load(open(path, "rb"))` or move into a 30-line `lipid_gnn/io.py`. |
| `scripts/training/prepare_colab_subset.py:13` | same | same |
| `scripts/training/smoke_test_sweep.py:24` | same | same |
| `lipid_gnn/functions_emil/properties_nn.py:18` | n/a — module is dropped | resolved by drop |

The `.h5` files are pickle blobs (no actual HDF5 in them) — `pickle.load` is sufficient. The two-line wrapper does not earn its keep across three call sites.

---

## 2. Audit of `compute_properties` — logical bugs and design issues

Numbered roughly in order of severity. Line refs are to [calculate_properties.py](../lipid_gnn/functions_emil/calculate_properties.py).

1. **Leaflet selection branch is identical for "upper" and "lower"** ([L237-241](../lipid_gnn/functions_emil/calculate_properties.py#L237-L241), and repeated at [L281-285](../lipid_gnn/functions_emil/calculate_properties.py#L281-L285)). Both branches compute `np.where(... < cutoff)` — the "upper leaflet" arm is wrong; it should be `> cutoff`. **Persistence and diffusivity therefore always sample the lower leaflet only.** Bias in symmetric bilayers is small; in asymmetric / cholesterol systems it is not.

2. **Persistence residue-index lookup is mis-indexed.** [L256-259](../lipid_gnn/functions_emil/calculate_properties.py#L256-L259): `contacts_indices = np.where(d[0] < contact_cutoff)[0]` returns positions inside the *flattened* `other_indices` bead-list, not global atom indices. The next line, `[beads[j].residue.index for j in contacts_indices]`, treats those positions as global atom indices into `beads`. Should be `beads[other_indices[j]].residue.index`. Almost the entire `persistence` signal is therefore sampling the wrong residues — and the project trains on it (`persistence` is in Tier B).

3. **Persistence "still in contact at +lag" check is type-incompatible.** [L262-268](../lipid_gnn/functions_emil/calculate_properties.py#L262-L268): `other_indices` is reassigned to `set(residues[...])` (bead indices for one residue), then intersected with `np.where(d[1] < contact_cutoff)[0]` (positions inside the *original* flattened `other_indices` — which has just been overwritten). Two different index spaces. Logic is broken even ignoring bug #2.

4. **`compute_bending_modulus` is fed the half-thickness field, not the midplane.** [L340](../lipid_gnn/functions_emil/calculate_properties.py#L340): `compute_bending_modulus(xy_membrane, ...)` where `xy_membrane = (upper - lower) / 2` ([L224](../lipid_gnn/functions_emil/calculate_properties.py#L224)). Bending modulus from the Helfrich undulation spectrum needs `h(x,y) = (upper + lower) / 2` (midplane height), not the half-thickness. This is the underlying reason `bending_modulus` was found "too noisy/unreliable" and dropped — the spectrum it was fitting is the *peristaltic* mode, not the bending mode. Property is already dropped from the active set, but if we ever re-enable it the fix is two characters.

5. **`compressibility` is not a compressibility modulus.** [L335-337](../lipid_gnn/functions_emil/calculate_properties.py#L335-L337) computes `std(thickness_xy − mean_thickness_per_frame, axis=1)² * 100` and labels the units "Å³/kT". This is the per-frame *spatial variance* of thickness — a measure of bilayer height inhomogeneity, not the area-compressibility modulus `K_A = k_B T ⟨A⟩ / ⟨(δA)²⟩` (or the analogue for thickness fluctuations). Tier C trained on it under the wrong physical interpretation. The numerical signal is real and the GNN learned it (test R² = 0.88); the *name* is wrong. Decision needed: rename to `thickness_inhomogeneity` everywhere, or compute the actual modulus.

6. **Leaflet midplane cutoff uses PO4 only.** [L187](../lipid_gnn/functions_emil/calculate_properties.py#L187): `po4_indices = trajectory.topology.select('name PO4')`. Cholesterol (ROH head) and any non-phospholipid contribute to `heads_indices` ([L188](../lipid_gnn/functions_emil/calculate_properties.py#L188)) but not to the leaflet midplane. The midplane is set by `np.diff(sorted PO4 z)`'s argmax — which is fine for symmetric phospholipid bilayers but is biased on CHOL-containing systems (where CHOL partitions asymmetrically). Affects every property that uses `cutoff` (persistence, diffusivity, leaflet-resolved thickness — though thickness is symmetric so OK; persistence/diffusivity already use only lower leaflet per bug #1).

7. **Raw-series lengths are inconsistent across properties.** Frames whose interpolation hit a NaN are skipped from `xy_thickness` / `xy_membrane` ([L222-224](../lipid_gnn/functions_emil/calculate_properties.py#L222-L224)), so `thickness_series` / `thickness_std_series` / `compressibility_series` use the filtered count while `persistence_series` (skips last `lag_persistence`), `diffusivity_series` (skips last `lag_diffusivity`), `packing_series` (full N), `variation_series` (full N) use different counts. The `raw_dict` time-series can't be co-plotted without re-aligning. Means are fine.

8. **Non-reproducible RNG.** [L238, L244, L262, L282, L288](../lipid_gnn/functions_emil/calculate_properties.py#L238): `np.random.random()` / `np.random.choice` with no seed kwarg. Re-running label computation gives different `persistence` / `diffusivity` numbers. The 70-system labels in `results/properties/` are therefore non-recomputable bit-for-bit.

9. **Diffusivity is relative-pair-displacement, not single-lipid MSD.** [L292-309](../lipid_gnn/functions_emil/calculate_properties.py#L292-L309) centres on lipid `j`, then measures the displacement of lipid `i` in that frame. If `i` and `j` are independent the variance is 2·MSD; in practice neighbour correlations reduce it. The docstring claims "lipid diffusivity (Å²)" — the actual quantity is `⟨|Δr_i − Δr_j|²⟩ × 100` averaged over random pairs. Not wrong, just mis-labelled.

10. **Voronoi tessellation ignores PBC.** [L317-320](../lipid_gnn/functions_emil/calculate_properties.py#L317-L320) builds the tessellation on raw 2-D points and clips to the box; boundary cells are still artefacts. A periodic Voronoi (replicate-and-clip from 9 box copies) would give an unbiased CV.

11. **Hard-coded grid origin and step.** [L182-183](../lipid_gnn/functions_emil/calculate_properties.py#L182-L183): `arange(1.5, box_xy[0]-1.49, .1)`. The 1.5-nm inset is a magic margin to avoid extrapolation in `LinearNDInterpolator`; the 0.1-nm step is a magic resolution. Both depend on `box_xy` being roughly 11×11 nm (the legacy training corpus). For different box sizes the implicit margin scales weirdly.

12. **`numpy` deprecation surface.** [L107](../lipid_gnn/functions_emil/calculate_properties.py#L107): `popt, _ = curve_fit(undulation_model, ...)` with no `p0`. Works but emits warnings on newer scipy. Minor.

### What stays correct

- `lipid_packing = n_lipids / box_area` — fine, deterministic, no RNG.
- `thickness` (mean of upper-lower interp) — physically OK; PO4-only leaflet split is acceptable for the legacy 10-lipid pool (all phospholipids except CHOL).
- `thickness_std` (spatial std of thickness, frame-averaged) — physically OK if interpreted as inhomogeneity (which the project does).

### Decision points to surface before rewriting

- **Recompute the 70-system labels?** If yes, the dataset's `<comp>.h5` files must be regenerated and Tier A–C numbers will shift slightly (mostly on `persistence`, possibly `diffusivity`). If no, the rewrite has to be **bug-compatible** for those two properties — exposing the buggy behaviour behind a `legacy=True` flag — so existing labels remain reproducible.
- **`compressibility` label.** Rename, recompute as the real modulus, or leave alone.
- **Drop `bending_modulus` for good or fix the midplane bug?** It's already excluded from training; fixing it lets future users re-enable it.

---

## 3. Rewrite plan

### Target layout

```
lipid_gnn/
  properties.py            # NEW — public API; replaces functions_emil/calculate_properties.py
  _properties_internals.py # NEW (optional) — Voronoi/interpolation helpers if properties.py grows
  io.py                    # NEW — thin pickle wrapper, replaces functions_emil.functions
scripts/python/
  compute_properties.py    # NEW CLI — replaces scripts/emil/general/calculate_properties.ipynb
tests/
  test_properties.py       # NEW — POPC100 regression test against properties.md reference values
```

Then delete `lipid_gnn/functions_emil/` entirely.

### `lipid_gnn/properties.py` design

Split the monolithic `compute_properties` into independent per-property functions plus an orchestrator:

```python
def compute_lipid_packing(traj) -> tuple[float, np.ndarray]: ...
def compute_thickness(traj, grid=...) -> tuple[float, float, np.ndarray, np.ndarray]:
    # returns (mean, std, thickness_series, thickness_std_series)
def compute_variation(traj, periodic=True) -> tuple[float, np.ndarray]: ...
def compute_persistence(traj, lag=50, probe_size=10, *, rng) -> tuple[float, np.ndarray]: ...
def compute_diffusivity(traj, lag=10, probe_size=10, *, rng) -> tuple[float, np.ndarray]: ...
def compute_thickness_inhomogeneity(traj, grid=...) -> tuple[float, np.ndarray]: ...
    # was "compressibility" — rename pending decision
def compute_bending_modulus(traj, grid=..., kBT=1.0) -> tuple[float, dict]: ...
    # midplane = (upper + lower) / 2  — bug 4 fixed

def compute_all(traj, *, seed: int | None = None, properties: list[str] | None = None,
                legacy: bool = False) -> tuple[dict, dict]:
    """Compute the requested subset; returns (mean_dict, raw_dict).

    legacy=True reproduces the functions_emil bugs bit-for-bit (for label
    re-derivation). legacy=False is the rewrite.
    """
```

Each property function takes the trajectory and its own params, returns `(scalar, series)`. The orchestrator dispatches and assembles the dicts in the existing schema so downstream code (the `<comp>.h5` consumer in `dataset.py`) stays untouched.

### Bug fixes to land in `legacy=False`

| Bug | Fix |
| --- | --- |
| #1 leaflet branch identical | `> cutoff` in the "upper" branch |
| #2 wrong residue-index lookup | use `other_indices[j]` |
| #3 broken contact-still-there check | track contact residues by global residue id; recompute contacts at frame `+lag` directly |
| #4 bending modulus on half-thickness | feed `(upper + lower) / 2` |
| #5 compressibility mislabelled | rename to `thickness_inhomogeneity`; optionally add a separate `area_compressibility` that computes `K_A = kBT ⟨A⟩ / ⟨(δA)²⟩` |
| #6 PO4-only leaflet cutoff | use the full head-bead set (`PO4 ROH …`) plus optional per-residue leaflet assignment for asymmetric systems |
| #7 inconsistent raw-series lengths | document or pad to full N; raw series only used for plots, not training |
| #8 non-reproducible RNG | every stochastic step takes `rng: np.random.Generator`; default seeded from a CLI flag |
| #9 diffusivity is pair-relative | switch to single-lipid lab-frame MSD with explicit PBC unwrap; keep `legacy=True` path for the existing `persistence`/`diffusivity` labels |
| #10 Voronoi ignores PBC | replicate points across the 8 neighbouring boxes, build Voronoi on the 9× set, take the central cells |
| #11 magic grid params | make grid resolution / margin keyword args; default scales with box_xy |
| #12 curve_fit warning | pass `p0=[1.0]` |

### `legacy=True` path

Mirrors `functions_emil/calculate_properties.compute_properties` exactly, including bugs #1–#3 and #6, so the existing `results/properties/<comp>.h5` files can be regenerated bit-for-bit (modulo the RNG — `legacy=True` would seed `np.random.seed(0)` to lock the labels going forward; the current files are non-deterministic, so exact reproduction is not achievable anyway).

Decision deferred: keep `legacy=True` only as a regression test, or expose it as the production path until the 70-system labels are recomputed?

### `lipid_gnn/io.py`

```python
import pickle
from pathlib import Path

def pkl_load(path: Path | str): ...
def pkl_save(path: Path | str, obj) -> None: ...
```

30 lines, no glob magic, no `nglview` import (the legacy `functions.py` imports `cv2`, `nglview`, `matplotlib.animation`, ... at module top, which slows every dataset load). Migrate the 3 in-project call sites to `from lipid_gnn.io import pkl_load`.

### CLI driver — `scripts/python/compute_properties.py`

Replaces the `scripts/emil/general/calculate_properties.ipynb` workflow. Reads from `CONFIG.paths.data_dir` (`data/membrane_only/<comp>/run/prun.{xtc,gro}`), writes to `CONFIG.paths.properties_dir/<comp>.h5`. Flags: `--composition COMP [COMP …]`, `--properties …`, `--seed N`, `--frame-start 50 --frame-stop 667`, `--lag-persistence 50 --lag-diffusivity 10`, `--legacy/--no-legacy`. Per the project's CLI pattern (see `systemPatterns.md`).

### Candidate new properties (orthogonal-signal hunt) — deferred, not v1

**Not in scope for this cleanup** (resolved 2026-05-18). v1 ships exactly the existing 8 properties under bug-fixed implementations. The list below is a parking lot for the follow-on task that decides which to add after the cleanup is finished — kept here so the design notes don't bit-rot in a separate doc.

The current 7-property set is dominated by two physical axes: bilayer geometry (`lipid_packing`, `thickness`) and short-time dynamics (`diffusivity`, `persistence`), with `thickness_std` / `variation` / `compressibility` all measuring spatial inhomogeneity. The embedding may benefit from labels that probe a different signal. Candidates, prioritised by orthogonality × computational cheapness:

| Candidate | Signal | Why it could be orthogonal | Cost |
| --- | --- | --- | --- |
| **Tail order parameter `S_CC`** (classical P2 of consecutive tail beads vs bilayer normal) | Tail ordering — gel-vs-fluid axis | Independent of thickness/diffusivity at fixed composition; sensitive to chain saturation in a way `lipid_packing` is not. Classical M3 descriptor. | Cheap — per-frame angle computation on tail-bead triples. |
| **Hexatic order `ψ₆`** from Voronoi nearest neighbours | Orientational ordering of head-bead lattice | Captures phase behaviour (liquid / hexatic / crystalline) beyond what cell-area CV (`variation`) sees. | Cheap — already have Voronoi infrastructure. |
| **Headgroup tilt** (PO4-GL1 or P-N vector relative to z) | Lipid-axis orientation distribution | Geometry axis decoupled from thickness; standard descriptor in lipid biophysics. | Cheap — single vector per lipid per frame. |
| **Surface tension γ** from `gmx energy` (`#Surf*SurfTen`) | NPT-equilibration diagnostic | Should be ≈ 0 for a properly relaxed tensionless bilayer; non-zero values flag setup issues. Both a quality signal and a label. | Trivial — read from EDR; one number per system. |
| **q-resolved undulation amplitudes `⟨\|h(q)\|²⟩`** at the lowest few q-bins | Long-wavelength membrane response | Once bug #4 is fixed, individual q-bin amplitudes are themselves labels (richer than a single κ scalar fit). Natural target for the EFA reopening — see the protein+membrane / EFA design notes. | Cheap if the height field is already being built for `thickness`. |
| **Tail interdigitation depth** (overlap between opposing-leaflet tail-bead z-distributions) | Inter-leaflet coupling | Independent of thickness (which is head-to-head); relevant for thin bilayers. | Cheap. |
| **Lipid-lipid orientational correlation** g_AB(r) (binary mixtures only) | De-mixing / clustering tendency | Composition-coupled signal — but composition-conditional, so adds info beyond global composition vector. | Medium. |
| **Lateral pressure profile** (per-z stress tensor) | Stress distribution | Strong orthogonality, but expensive and tooling-heavy. | High (requires `gmx grompp` rerun with pressure tensor or post-hoc analysis). |

**For the follow-on task** (after cleanup is finished): `S_CC`, `ψ₆`, headgroup tilt, surface tension γ, and the q-resolved spectrum are the strongest starting candidates — cheap, well-defined, probe distinct physical axes. Decision rule: compute pairwise Pearson on the 70-system labels — anything `|r| > 0.9` against an existing property is redundant and a candidate to drop. Defer interdigitation, g_AB, and lateral pressure to an even later round.

Each new property is added behind the same `compute_all(..., properties=[...])` interface, with its own `compute_<name>(traj, ...)` function and unit test. Adding a property to the active training set is a separate decision (Tier-D-style gate + 5-seed confirmation).

### Tests — `tests/test_properties.py`

**Regression tests** (on real trajectories):

- POPC100 `legacy=False`, `seed=0`: each scalar matches [memory-bank/properties.md](../.claude/memory-bank/properties.md) within ~5 % (slack for the bug fixes — `persistence` and `diffusivity` are expected to shift; `lipid_packing`, `thickness`, `thickness_std`, `variation` should be tight).
- POPC100 `legacy=True`, `seed=0`: matches a stored golden dict within ~1 %. Golden created from one run of the new `legacy=True` path (the current `.h5` files are RNG-uncontrolled and not bit-comparable).

**Mock tests** (on synthetic constructed trajectories with analytic answers). Each is a single `mdtraj.Trajectory` built from scratch — no MD needed — designed to exercise one calculation and its known pitfall.

| Test | Construction | Assertion |
| --- | --- | --- |
| `lipid_packing` exact | Regular grid of N PO4 beads in known Lx, Ly | `lipid_packing == N / (Lx * Ly)` to machine precision |
| `thickness` flat | Two parallel planes at z = Z_lo and z = Z_hi for two leaflets | `thickness == (Z_hi - Z_lo) * 10` Å exactly; `thickness_std == 0` |
| `thickness` corrugated | Upper plane = Z_hi + A·cos(2π x / Lx) (lower flat) | Mean thickness = Z_hi − Z_lo; spatial std = A / √2 (analytic) |
| `lipid_packing` in NPT | Replicate the flat-bilayer frame with box lengths drawn from a known distribution | Recovered packing series matches the analytic per-frame value |
| `variation` lattice | Triangular / square lattice of head beads | Voronoi CV → 0 (within periodic-boundary clipping artefacts; **fails on bug #10 — exercises the PBC-Voronoi fix**) |
| `variation` Poisson | Uniform random points (fixed seed) | CV converges to the analytic Poisson-Voronoi value ≈ 0.36 within Monte-Carlo error |
| `persistence` identity | Two-frame trajectory with frame 1 = frame 0 | `persistence == 1.0` exactly |
| `persistence` decoupled | Frame 1 has every lipid translated by ≫ contact_cutoff | `persistence == 0.0` exactly |
| `persistence` known fraction | Construct N pairs, fix exactly k of them to still be in contact at frame 1 | `persistence == k / N` ± Monte-Carlo std on the probe (**fails on bug #2 and #3**) |
| `diffusivity` ballistic | Each lipid displaced by an exact known vector Δr in frame 1 | Single-lipid lab-frame MSD == \|Δr\|² (**fails on bug #9 — exercises the pair-relative fix**) |
| `diffusivity` PBC unwrap | Lipid moves Lx + ε across the periodic boundary | Recovered displacement = ε, not Lx + ε |
| Leaflet split symmetric | Equal phospholipid populations at ±Z | Cutoff lands at z = 0; upper and lower assignments equal in size |
| Leaflet split asymmetric | Phospholipid populations 60/40 split + cholesterol concentrated on the upper leaflet | After bug-#6 fix using full head-bead set, leaflet assignment recovers the true partition; PO4-only path mis-assigns CHOL — assertion captures the regression |
| `bending_modulus` Helfrich | Synthetic midplane z(x,y,t) = A·cos(q·x) with known q and A; flat thickness field | After bug-#4 fix (midplane, not half-thickness), recovered κ matches the analytic κ from `⟨\|h(q)\|²⟩ = kT / (κ q⁴)` within ~10 %. Pre-fix path recovers garbage on the same input. |
| `bending_modulus` peristaltic-null | Upper and lower planes counter-undulate (thickness oscillates, midplane is flat) | Post-fix bending_modulus is un-measurable / ∞; pre-fix returns a finite spurious value (**captures the bug regression**) |
| `thickness_inhomogeneity` | Reuse the corrugated-thickness construction above | `compressibility_series` (pre-rename) / `thickness_inhomogeneity` (post-rename) = (A / √2)² × 100, analytic |
| RNG reproducibility | Same trajectory, same seed, two runs | `mean_dict` identical bit-for-bit (**fails on bug #8**) |

Each mock test is independently constructed and runs in < 1 s — no fixture trajectory required. Failure of a specific test maps 1-to-1 to a specific bug in §2, so the test suite doubles as a regression gate when re-enabling features (e.g. `bending_modulus`).

### Migration / deletion order

1. Add `lipid_gnn/io.py`. Switch the 3 in-project `pkl_load` import sites to it.
2. Add `lipid_gnn/properties.py` with the per-property functions, `compute_all`, and the `legacy=` switch. Add tests.
3. Add `scripts/python/compute_properties.py` CLI.
4. **Regenerate labels (not optional, per 2026-05-18 decision)**. After the bug fixes land, recompute `results/properties/<comp>.h5` for the 70-system corpus under `legacy=False`, into a fresh directory (e.g. `results/properties_v2/`). The original `results/properties/` is preserved untouched as the historical-bug baseline. This feeds the comparison notebook in §6 (separate task in memory bank). Coordinates with the already-planned legacy→M3-ITP resimulation: the new property pipeline is run on **both** label sets — the bugged-trajectory corpus and the re-simulated M3 corpus — so the three-way comparison in §6 has all data points.
5. Delete `lipid_gnn/functions_emil/` and its `__pycache__` / `.ipynb_checkpoints`. Drop the directory from `pyproject.toml` / `MANIFEST.in` if listed. Then sweep the four other legacy `insane.py` copies enumerated in §4: `colab_lipid_gnn_subset/lipid_gnn/functions_emil/insane.py`, `colab_lipid_gnn_subset/lipid_gnn/functions_emil/.ipynb_checkpoints/insane-checkpoint.py`, and `build/lib/lipid_gnn/functions_emil/insane.py`. Easiest: delete the entire `colab_lipid_gnn_subset/lipid_gnn/functions_emil/` tree (whole subset is legacy reference per `feedback_training_hpc_only`) and `build/` (regenerated on `pip install`). Close the `project_insane_legacy_cleanup` memo as subsumed.

---

## 4. Open questions / decision log

Resolved 2026-05-18:

- [x] **Recompute the 70-system labels.** After the bug fixes land. The pending Tier C retraining (now also needed to save model weights — see [activeContext.md](../.claude/memory-bank/activeContext.md)) becomes the natural moment to evaluate the new labels. The pre-existing comparison framework with the M3-ITP resimulation extends naturally to a three-way comparison (legacy bugged labels / bugfixed labels on legacy trajectories / bugfixed labels on M3-ITP trajectories) — separate notebook task tracked in memory bank.
- [x] **Rename `compressibility` → `thickness_inhomogeneity`** everywhere (label name, plotting axes, properties.md, notebooks). The downstream code can swap by string substitution; the model retrains under the new name on the regenerated labels. A real area-compressibility modulus `K_A` is not added in this round — defer to a later property-expansion round (§ candidate-new-properties).
- [x] **Fix `bending_modulus`** (use midplane `(upper + lower) / 2`, not the half-thickness). Once the regenerated labels exist, evaluate whether the fixed property gives a trainable signal — i.e. is it still the noise-dominated channel it was when dropped, or does the fix produce a label the GNN can learn? If trainable, it goes back into the active set as a candidate 8th property; if still noise-dominated, the drop stands.

Resolved 2026-05-18 (continued):

- [x] **Fold the `insane.py` legacy deletion into this cleanup.** Five copies of the Python-2-era `insane.py` exist in the repo (per `project_insane_legacy_cleanup`):
    1. `lipid_gnn/functions_emil/insane.py`
    2. `lipid_gnn/functions_emil/.ipynb_checkpoints/insane-checkpoint.py`
    3. `colab_lipid_gnn_subset/lipid_gnn/functions_emil/insane.py`
    4. `colab_lipid_gnn_subset/lipid_gnn/functions_emil/.ipynb_checkpoints/insane-checkpoint.py`
    5. `build/lib/lipid_gnn/functions_emil/insane.py`

    (1) and (2) go away as part of the `lipid_gnn/functions_emil/` directory deletion in step 5. (3) and (4) live under `colab_lipid_gnn_subset/`, which is itself legacy reference (`feedback_training_hpc_only` — Colab is no longer the active training path); delete them in the same step. (5) is a build artefact from a `pip install` and can be removed by also blowing away `build/`. The canonical Python-3 `insane` is the pip-installed package; `resources/martini3/insane.py` (vendored CPython-3 version) is what `martini_pipeline/system_builder.py` would fall back to if needed. The separate `project_insane_legacy_cleanup` memo is closed as subsumed once step 5 lands.

Resolved 2026-05-18 (continued):

- [x] **Keep `legacy=True` for now.** Ship it alongside `legacy=False` as a runtime switch, not just a test fixture. Re-evaluate (and possibly drop) after the regenerated labels are validated and the three-way comparison notebook lands. Lets us reproduce historical labels on demand during the comparison phase without resurrecting deleted code.
- [x] **No new properties in v1.** §3 candidate-new-properties (`S_CC`, `ψ₆`, headgroup tilt, surface tension, q-resolved spectrum) are not implemented as part of this cleanup. The cleanup ships exactly the 8 existing properties (7 active + `bending_modulus`) under bug-fixed implementations, with `bending_modulus` re-evaluated for trainability post-fix. Which of the §3 candidates to add is decided **after** the cleanup is finished — separate task, not blocking this one.

All §4 questions resolved.

---

## 5. Status

- 2026-05-18 — Plan written. Execution not started; do not begin without explicit go-ahead.
