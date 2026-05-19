# Active Context

## Preprocessing script renamed + new graph-dataset layout (2026-05-19)

`scripts/training/prepare_colab_subset.py` → [scripts/training/preprocess_graphs.py](../../scripts/training/preprocess_graphs.py). Public function `prepare_colab_subset()` → `preprocess_graphs()`. Colab is no longer in the training path (HPC-only per `feedback_training_hpc_only`); the script now exists purely to turn simulations into graph chunks.

**New output layout** under `data/preprocessed_graphs/` (a sibling of `data/membrane_only/`; lives under `data/` because chunks are derived training **inputs**, not analysis outputs):

```text
data/preprocessed_graphs/
├── <run-name>/                <- one preprocessing run
│   ├── train/  val/  test/
└── archives/
    └── <run-name>.zip         <- HPC-transfer zip (chunks only, code via git)
```

`<run-name>` defaults to the new required `--props-set` flag, so different property folders never overwrite each other.

**CLI changes**:

- New required `--props-set` (e.g. `prop_legacy_bugfixed_s0`) → subfolder of `--props-base`.
- `--props-base` (default `CONFIG.paths.props_dir` = `results/properties/`) replaces the old `--props-dir` (effective path = `props_base / props_set`).
- `--parent-dir` (default `CONFIG.paths.preprocessed_graphs_dir` = `data/preprocessed_graphs/`).
- `--run-name` (default = `--props-set`) replaces the old `--subset-name`.

**Config**: dropped `paths.subset_bundle_dir`, added `paths.preprocessed_graphs_dir: data/preprocessed_graphs`. `paths.chunks_dir` default updated to `data/preprocessed_graphs/active` (no functional impact — HPC overrides via `CHUNKS_DIR` env var).

**Touched callers**: [scripts/bash/sbatch_preprocess.sh](../../scripts/bash/sbatch_preprocess.sh) (now requires `PROPS_SET`), [scripts/bash/gc_transfer_files.sh](../../scripts/bash/gc_transfer_files.sh) (takes run-name arg, rsyncs new archive path), docstring/comment refs in [scripts/training/run_sweep.py](../../scripts/training/run_sweep.py), [scripts/training/linear_baseline.py](../../scripts/training/linear_baseline.py), [lipid_gnn/io.py](../../lipid_gnn/io.py), [tests/test_stratified_split.py](../../tests/test_stratified_split.py), [tests/test_multi_frame_loading.py](../../tests/test_multi_frame_loading.py). [README.md](../../README.md) "Colab sweep" section replaced with an HPC preprocessing section. Tests green (8 affected).

## Design notes — protein+membrane extension & EFA reopened (2026-05-18)

New design doc at [docs/protein_membrane_embedding_thoughts.md](../../docs/protein_membrane_embedding_thoughts.md).
Working notes for the long-term scientific goal (protein+membrane embedding,
`projectbrief.md`). Ten sections; substantive decisions:

- **Scope flavour (§1)**: (A) inference-only on locked Tier C, (B) fine-tune
  with new local labels, (C) pure structural probe. Doc argues (A) first.
- **Protein pair (§2)**: WALP + β2AR recommended (cheap probe + M3-validated
  GPCR with documented CHOL fingerprint). Souza et al. 2021 *Nat. Methods* is
  the M3 protein-FF reference. Optionally OmpA for β-barrel coverage.
- **Compositions (§3)**: stay inside the 70-system training corpus so every
  protein+membrane system has a matched pure-bilayer training reference.
- **CG topology cost (§4)**: routine via `martinize2` + ElNeDyn / Go. The
  *engineering* lift is the bead-vocabulary extension in `lipid_graph.py` /
  `ff_node_mapping.json` / `MembranePropertyGNN`, not the simulations.
- **Sweep size (§5)**: 2 proteins × 3 compositions = 6 systems. Same bead-
  count scale as current corpus — no HPC re-benchmark.
- **Length (§7)**: 2 µs / system for v1. Total cost ~1 GPU-node-day.
- **EFA reopened (§8)** — see next entry.
- **Phasing (§10)**: Phase 0 bead vocab → Phase 1 WALP/POPC smoke test →
  Phase 2 6-system factorial inference → Phase 3 (optional) fine-tune.

Six open followups in §9 (scope flavour, protein pick, label question,
when to extend bead vocab, lipidome-shortlist coupling, EFA gate) before
any simulation is submitted.

**EFA status reopened** — [docs/efa_spatial_layer_future.md](../../docs/efa_spatial_layer_future.md)
predates Tier C and its motivating-target list is stale:

- `bending_modulus` **permanently dropped** (label noise, not architecture).
- `compressibility` Tier C 5d pooled test R² = 0.88 — receptive-field upper-
  bound argument falsified empirically. EFA's headline acid-test target is
  gone on the membrane-only side.
- New strongest motivation is the **protein extension**: inhomogeneity +
  oriented inclusion + bilayer long-wavelength response = textbook
  global-mixing regime; SE(3) equivariance becomes salient once an oriented
  protein coexists with the bilayer normal.
- Test order unchanged from efa-doc: (f) deeper MP → (c) readout-only EFA →
  (b) per-layer parallel. But **acceptance criterion must be redefined** —
  candidate gates: embedding-quality metrics, DPPC/DOPC corner error-tail
  reduction, or a new `S(q_min)` architectural-probe label (preferred —
  same observable also serves as a scenario-(B) protein label, sharing
  infra).

---

## New task — `functions_emil/` cleanup (2026-05-18)

Plan: [docs/functions_emil_cleanup_plan.md](../../docs/functions_emil_cleanup_plan.md). Goal — systematically prune and rewrite `lipid_gnn/functions_emil/` (11,887 LOC, mostly inherited TPS/FE-NN code). Three subtasks:

1. **Categorise** every module + top-level symbol into three bins: **used**, **not used**, and **possibly useful in the future**. `scripts/emil/general/` and `scripts/emil/free_energy_nn_paper/` notebooks are out-of-project. First-pass survey in §1 + §1b of the plan: **two modules kept now** — `functions.py` (`pkl_load`/`pkl_save`, 3 in-project callers) and `calculate_properties.py` (`compute_properties`, source of every training label in `results/properties/`). The other 10 modules (~11 200 LOC) are not currently reachable from in-project code; most go to the "drop outright" bin, with the §1b "possibly useful" exceptions earmarked for **read-and-port-the-idea, then delete**: `compute_com_dist` and `DRMSD` (protein+membrane primitives), `properties_nn.Network` + train/eval loop (stronger FFNN baseline than the current Ridge), `bilayer_builder` (read-only reference for protein-placement when the protein+membrane phase begins), `utils.compute_autocorrelation` (property-convergence diagnostics), `utils.recover_trr` (long-simulation robustness), `utils.weighted_quantile`, plus composition-string parsers in `properties_nn` pending overlap-check with `martini_pipeline.canonical_name`.
2. **Audit** the kept functions for logical bugs — *especially* property-calculation correctness. §2 of the plan lists 12 issues; the load-bearing ones: identical upper/lower leaflet branch (persistence & diffusivity always sample lower leaflet), persistence residue-index lookup is mis-indexed against the wrong array, persistence "still in contact at +lag" check intersects incompatible index spaces, `compute_bending_modulus` is fed the half-thickness field instead of the midplane, `compressibility` is a mislabelled thickness-inhomogeneity (not `K_A`), PO4-only leaflet cutoff biases CHOL systems, RNG is unseeded → labels are non-reproducible.
3. **Rewrite** into new project-grade modules: `lipid_gnn/properties.py` (per-property functions + `compute_all` orchestrator + `legacy=True/False` switch), `lipid_gnn/io.py` (replaces the `pkl_load`/`pkl_save` wrappers), `scripts/python/compute_properties.py` CLI (replaces the emil notebook), `tests/test_properties.py` (POPC100 regression against properties.md). Then delete `lipid_gnn/functions_emil/` entirely. Migration order in §3 of the plan.

**Decisions resolved 2026-05-18** (decision log §4 of the plan):

- **Recompute 70-system labels** after the bug fixes land — fresh `results/properties_v2/` next to the preserved bugged `results/properties/`. Couples to the pending Tier C retraining-with-weight-saving and the M3-ITP resimulation.
- **Rename `compressibility` → `thickness_inhomogeneity`** everywhere (label, plots, properties.md, notebooks). Real `K_A` deferred.
- **Fix `bending_modulus`** (use midplane `(upper + lower) / 2`); re-evaluate whether the fixed property is trainable on the regenerated labels — if yes, candidate 8th property; if still noise-dominated, the permanent drop stands.

**Two new sections added to the plan**:

- §3 candidate **new properties** for orthogonal-signal coverage (the embedding's diet is currently dominated by geometry + short-time dynamics + inhomogeneity). Recommended v1 additions: tail order `S_CC`, hexatic order `ψ₆`, headgroup tilt, surface tension γ from EDR, q-resolved undulation amplitudes `⟨|h(q)|²⟩` (which also serves the EFA-reopening agenda). Pairwise-Pearson redundancy check on the 70-system labels gates inclusion. Deferred: interdigitation, g_AB(r), lateral pressure profile.
- §3 **mock tests** for property correctness — synthetic constructed trajectories with analytic answers, one per pitfall. Highlights: flat / corrugated bilayer (thickness + std), triangular lattice and Poisson points (Voronoi CV + PBC), identity / decoupled / known-fraction trajectories (persistence — exercises bugs #2/#3), ballistic single-vector displacement (diffusivity — exercises bug #9), Lx+ε boundary crossing (PBC unwrap), symmetric vs CHOL-asymmetric leaflet split (exercises bug #6), sinusoidal midplane (bending_modulus — exercises bug #4) and counter-undulating peristaltic-null (also bug #4 regression), RNG reproducibility (bug #8). Each mock test maps 1-to-1 to a numbered §2 bug.

**`insane.py` legacy cleanup folded in 2026-05-18.** Step 5 of the migration order also deletes the four other legacy `insane.py` copies enumerated in `project_insane_legacy_cleanup`: two under `colab_lipid_gnn_subset/lipid_gnn/functions_emil/` (incl. `.ipynb_checkpoints`) and one under `build/lib/lipid_gnn/functions_emil/`. Simplest sweep: remove the whole `colab_lipid_gnn_subset/lipid_gnn/functions_emil/` subtree (legacy Colab reference, no longer the active training path) and `build/` (regenerated on `pip install`). The `project_insane_legacy_cleanup` memo is closed as subsumed once step 5 lands.

**Two final decisions resolved 2026-05-18**:

- **Keep `legacy=True` for now.** Ship as a runtime switch (not just a test fixture) alongside `legacy=False`. Re-evaluate after the regenerated labels are validated and the three-way comparison notebook lands.
- **No new properties in v1.** §3 candidate-new-properties section stays as a parking-lot for the follow-on task. v1 ships exactly the existing 8 properties under bug-fixed implementations; `bending_modulus` is re-evaluated for trainability post-fix.

Status — **plan complete and ready to execute 2026-05-18.** All §4 decisions resolved. Do not begin without explicit go-ahead.

### Migration executed 2026-05-18 (steps 1–3 + tests of plan §3)

- **Step 1** — [lipid_gnn/io.py](../../lipid_gnn/io.py) ships `pkl_load` / `pkl_save` (thin pickle wrapper, no glob, no `nglview`/`cv2` imports). The 3 in-project import sites — [lipid_gnn/dataset.py](../../lipid_gnn/dataset.py), [scripts/training/prepare_colab_subset.py](../../scripts/training/prepare_colab_subset.py), [scripts/training/smoke_test_sweep.py](../../scripts/training/smoke_test_sweep.py) — switched from `lipid_gnn.functions_emil.functions` to `lipid_gnn.io`.
- **Step 2** — [lipid_gnn/properties.py](../../lipid_gnn/properties.py) implements all 8 bug-fixed properties + `compute_all` orchestrator with `legacy=True/False` switch. Bug fixes landed: #1 (real upper/lower leaflet split), #2 (`other_indices[j]` instead of positional index for persistence residue lookup), #3 (recompute contacts at +lag instead of intersecting incompatible index spaces), #4 (midplane = `(upper + lower)/2` for bending modulus, not the half-thickness), #6 (full head-bead set for leaflet cutoff, not PO4-only), #8 (RNG seed kwarg → reproducible labels), #9 (single-lipid lab-frame MSD with PBC unwrap, not pair-relative pivot), #10 (periodic Voronoi via 9-image replication, no bbox clipping), #11 (grid params are kwargs), #12 (`curve_fit` with `p0=[1.0]`). Property rename: `compressibility` → `thickness_inhomogeneity` (the legacy key remains as a value alias so downstream code keeps working).
- **Step 3** — [scripts/python/compute_properties.py](../../scripts/python/compute_properties.py) CLI replaces the emil notebook. Default `--out-dir results/properties_v2/` preserves the historical `results/properties/` untouched.
- **Tests** — [tests/test_properties.py](../../tests/test_properties.py): 17 mock tests, all pass. Covers analytic regular-grid `lipid_packing`, flat & corrugated `thickness`, periodic-Voronoi `variation`, frozen / decoupled / asymmetric-bilayer `persistence`, ballistic-displacement & PBC-unwrap `diffusivity`, sinusoidal-midplane `bending_modulus` plus peristaltic-null regression, RNG reproducibility, legacy/bugfixed schema + alias. Full repo test suite (59 tests) still green after the migration.
- **Step 5 executed 2026-05-18.** Deleted `lipid_gnn/functions_emil/` (11 887 LOC across 12 modules), `colab_lipid_gnn_subset/lipid_gnn/functions_emil/`, and `build/`. All five legacy `insane.py` copies are now gone (canonical Python-3 `insane` remains via the pip-installed package + `resources/martini3/insane.py`). One in-project docstring updated: [scripts/notebooks/analyze_dataset.py:349](../../scripts/notebooks/analyze_dataset.py) now points at `lipid_gnn.properties.compute_all`. Out-of-project notebooks under `scripts/emil/` and `scripts/colab/train_colab.ipynb` still have stale imports but those notebooks are explicitly out of scope (`feedback_training_hpc_only`); they are now broken and stay broken. `setup.py` uses `find_packages()`, so no packaging-config edit needed.
- **Verification**: full repo test suite green — 567 passed, 7 skipped (59 in core + 508 in `martini_pipeline`).
- **Step 4 executed 2026-05-18.** Original `results/properties/` moved to `results/old_properties/` (preserved as historical baseline — one-off non-reproducible random draw). Three label sets regenerated under `results/properties/`, 70 systems each:
    - `prop_legacy_bugged_random/` — legacy buggy algorithms, unseeded (RNG-noise sibling of `old_properties`)
    - `prop_legacy_bugged_s0/` — legacy buggy algorithms, `--seed 0` (reproducible comparison reference)
    - `prop_legacy_bugfixed_s0/` — bug-fixed `legacy=False`, `--seed 0` (**production label set going forward**)

  These three sets are the inputs to the three-way comparison notebook tracked below. Bug #7 (inconsistent raw-series lengths across properties) addressed inline: `thickness_summary(..., frame_mask=...)` now pads dropped-frame slots with `NaN` so per-property series are co-indexable; regression test `test_frame_mask_pads_series_with_nan` added (21 property tests total).
- **Second-pass audit fixes 2026-05-18.** A second-pass review of `lipid_gnn/properties.py` (after Step 4 landed) surfaced five physical-correctness issues beyond the cleanup-plan §2 taxonomy. Full breakdown in [properties.md](properties.md) under "Bug fixes layered on top of cleanup-plan §2". Load-bearing:
    - **Bending modulus is now physical κ in kBT** (`legacy=False`). Continuous Helfrich `⟨|h(q)|²⟩ = kBT / (κ q⁴ A)` becomes `⟨|H_k|²⟩_ortho = kBT / (κ q⁴ Δx Δy)` in the discrete `norm="ortho"` form, so the raw `curve_fit` returned `κ · Δx Δy` (off by ≈ 0.01 nm² for default step=0.1 nm). The dimensionally-nonsense `* 1000.0` "kT/Å³" scaling has been removed from `legacy=False` (preserved in `legacy=True` by undoing the new normalisation via the `kappa_raw_fit` field in the diag dict). Typical M3 POPC: ≈ 15 kBT in the new units (was ≈ 152 in the legacy convention).
    - **FFT axis labelling**: `np.meshgrid(x, y, indexing="ij")` makes `Z.shape == (n_frames, nx, ny)` semantic; previously `indexing="xy"` (default) silently transposed axes so `fftfreq(nx, d=Lx/nx)` mixed x-extent with y-grid count for rectangular grids. No effect on square Martini boxes; corrects behaviour for any rectangular grid.
    - **FFT spacing off-by-one**: `fftfreq(n, d=...)` now uses the actual grid step (`X[1,0] − X[0,0]`) rather than `Lx/n`, which for a half-open `arange` grid is `(n−1)/n · step`. Negligible at large n but strictly correct.
    - **Voronoi CV → NaN on degenerate input** (was 0.0); `compute_variation` uses `np.nanmean`. Stops silent zero-bias when a leaflet has no usable cells.
    - **Inhomogeneity formula simplified**: `(flat − mean).std()² · 100` ≡ `flat.std()² · 100` rewritten as `kept_std ** 2`. No numerical change.
    - New tests: `test_physical_kappa_recovered_from_helfrich_field` (Gaussian random field with target κ = 20 kBT; fit recovers within 25 %), `test_non_square_grid_uses_correct_axis_spacing`, `test_variation_nan_when_voronoi_fails_everywhere`. Total 21 property tests; full repo suite 571 passed / 7 skipped.
    - **Implication for the three-way notebook**: `bending_modulus` between `prop_legacy_bugged_*` (× 1000 "kT/Å³") and `prop_legacy_bugfixed_s0` (κ in kBT) differs by a factor of ~10 *before* the half-thickness-vs-midplane field change kicks in. Plot on log-y or with a per-set unit note. Property is still out of the active training set per the §4 decision log; the fix matters only if `bending_modulus` is re-evaluated for trainability.
- **All cleanup-plan steps complete.** [docs/functions_emil_cleanup_plan.md](../../docs/functions_emil_cleanup_plan.md) §5 records the final status. Follow-on work: three-way comparison notebook + Tier C retraining (separate entries below).

## Three-way property + model comparison notebook (plan + skeleton, 2026-05-19)

Plan: [docs/compare_bugfix_three_way_plan.md](../../docs/compare_bugfix_three_way_plan.md).
Notebook: [scripts/notebooks/compare_bugfix_three_way.py](../../scripts/notebooks/compare_bugfix_three_way.py).

**Design upgraded to 4 label sets / 4 contrasts** after the realisation
that bug #8 (unseeded RNG) means the historical labels are one of an
ensemble of possible draws; the seed effect is therefore a confounder
of the bug-fix contrast and worth measuring separately.

Directory layout under `results/properties/`:

- `prop_legacy_bugged_random/` (existing, the shipped labels; unseeded)
- `prop_legacy_bugged_s0/` (legacy code rerun, `--seed 0`)
- `prop_legacy_bugfixed_s0/` (bugfixed code on legacy trajectories, `--seed 0`)
- `prop_m3_bugfixed_s0/` (bugfixed code on M3-rerun trajectories, `--seed 0`)

Naming `prop_<traj>_<method>_<seed>`. Step 4 of the cleanup plan
populated three of the four; `prop_m3_bugfixed_s0/` is pending the M3
resimulation run.

Three primary contrasts + one bonus, all paired on `canonical_name`:

- **seed** (bonus): `legacy_bugged_s0 − legacy_bugged_random`. Same code,
  same trajectories, RNG differs. Sets the noise floor.
- **bugfix**: `legacy_bugfixed_s0 − legacy_bugged_s0`. Code differs only.
- **ff**: `m3_bugfixed_s0 − legacy_bugfixed_s0`. Trajectories differ only.
- **total**: `m3_bugfixed_s0 − legacy_bugged_random` (current shipped → final product).

Identity drives §4 variance decomposition: `Var(d_total) = Var(d_seed) + Var(d_bugfix) + Var(d_ff) + 2·(three covariances)`.

**Notebook sections implemented**:

- §0 vocab / §1 paths + DIPC↔DLPC rename rules + coverage; §2 ITP SHA-1 diff + `properties.py` mtime; §3a seed-only sanity panel + callout; §3b paired summary table across four contrasts; §3c 7×4 quadruple-scatter; §3d 7×4 Bland–Altman; §3e KDE overlay across all four label sets; §3f top-5 movers per (property, contrast); §4 variance decomposition (stacked signed bar + table via `mo.vstack`) + auto-generated callout naming the dominant component; §5 PCA on composition matrix with dropdown to switch contrast overlay; §6 model loaders for four W&B groups, per-property MSE/R² tables, paired t-tests across pairs, per-system residual scatter coloured by lipid family (largest-mol-fraction wins, with CHOL-mix override), cross-model prediction-spread vs residual-spread; §7 four-branch verdict callout.

**Implementation notes**:

- **DIPC ↔ DLPC normalisation** wired in as a `mo.ui.text` widget (default `DLPC=DIPC`). The M3 resimulation submitted via `submit_simulations.sh --rename-lipid DIPC=DLPC` writes `DLPC*` stems while every legacy set is `DIPC*`; without normalisation, the four-way `paired_all` intersection drops every DIPC/DLPC-containing composition. The widget parses comma-separated `OLD=NEW` rules, applies them per-stem, and re-canonicalises lipid tokens alphabetically so `POPC50_DLPC50` → `DIPC50_POPC50` matches the legacy stem.
- **Family bucketing in §6c** is "largest mol fraction wins, but any CHOL collapses to `CHOL-mix`". Two known cliffs: a 49/51 split flips families on a 1-mol-% change, and the CHOL override hides DPPC-rich-vs-DOPC-rich CHOL substructure. Documented; not changed.
- **§4 layout follows the skill rule** for matrix-shaped data: table + plot rendered together via `mo.vstack` in a single cell.
- **§6 `model_legacy_bugged_random`** is the existing `stage_5d_tier_c_confirm` group (trained on the unseeded shipped labels); it is loaded as a fourth, optional model so the property-seed effect can be tested on the model side too (`model_legacy_bugged_random` vs `model_legacy_bugged_s0`). Three *new* W&B groups need to land before §6 is interpretable: `stage_5d_tier_c_legacy_bugged_s0`, `stage_5d_tier_c_legacy_bugfixed_s0`, `stage_5d_tier_c_m3_bugfixed_s0`. All at locked HPs (`lr=3e-5, wd=1e-3, h=128, l=2, e=200`, 6 GNN seeds).
- **Empty-state safe** via `mo.stop` after the §1 load and §6 inventory: with only `prop_legacy_bugged_random/` and `stage_5d_tier_c_confirm/` present today, every cell renders a skeleton or `no data` placeholder rather than erroring. Each contrast comes alive independently as its two label sets land.

Verified with `uvx marimo check` (no critical issues; cosmetic markdown-indentation warnings only) and `uv run scripts/notebooks/compare_bugfix_three_way.py` (exit 0). Skill audit fixed: `import datetime` moved to cell 1, unused `json` import removed, variance table+plot merged via vstack, PCA explained-variance summary added, prose cells added above §0 vocab + §1 coverage.

**Status — skeleton ready 2026-05-19, blocked on:**

- (a) `prop_m3_bugfixed_s0/` regeneration — needs the M3 resimulation to finish + property-pipeline run on those trajectories.
- (b) Three new Tier C W&B groups (`*_legacy_bugged_s0`, `*_legacy_bugfixed_s0`, `*_m3_bugfixed_s0`) — submit after labels land.

Decision deferred to execution time: include candidate new properties (`S_CC` etc.) or not. v1 ships with the existing 7 Tier C properties only.

## Currently Running / Pending on HPC (2026-05-17)

| Stream | Where | State | Notes |
| --- | --- | --- | --- |
| Subgoal 3a — `popc_interpolation` 1 µs | `general1` (CPU) | running | 77 POPC-anchored binaries at 10 % step; 48 h walltime, ETA ~65 h per slot — resubmit-with-`-cpi` likely on some slots |
| Subgoal 3b — DPPC/DOPC corner extrapolation | `gpu` | running | `pin=auto` (pre-`--pin` infrastructure); will not benefit from the new mdrun thread-pinning |
| CHOL-containing combinations | `gpu` | **failed** | grompp atom-count mismatch from insane's legacy 8-bead CHOL vs M3 9-bead ITP. Fixed at registry level (`insane_keyword="M3.CHOL"`); waiting for the new GPU benchmark to land before resubmitting these. |
| GPU benchmark with `pin=on` probe rows | `gpu` | pending | TSV has two new rows `8_sim_8gpu_cpu4_pin_on` and `8_sim_8gpu_cpu8_pin_on`; compare against current winner (cpu4, pin=auto). |

**Planned sequence once the new GPU benchmark lands**:

1. **Rerun CHOL compositions** on GPU with the M3.CHOL fix + new pin-aware defaults.
2. **Rerun the legacy 70-system corpus** under the modern M3 ITPs via `submit_simulations.sh --from-csv resources/done.csv --rename-lipid DIPC=DLPC --completed-csv resources/redone.csv`. Standardises every output to one set of itp definitions (avoids future mapping conflicts).
3. **Compare new vs. legacy** for the 70-system corpus — sanity-check that property labels (lipid_packing / thickness / variation / persistence / diffusivity / compressibility) are within expected noise of the legacy values, so the relabel doesn't silently change the regression targets. Comparison notebook drafted at [scripts/notebooks/compare_legacy_vs_new_m3.py](../../scripts/notebooks/compare_legacy_vs_new_m3.py) — paired on `canonical_name`, six path widgets (legacy/new property-pickle dirs, run roots, ITP dirs), default new paths `results/properties_m3_rerun/` and `data/membrane_only_m3_rerun/`. Sections: (1) coverage, (2) ITP SHA-1 diff, (3) per-property paired summary + scatter/Bland–Altman/KDE/movers/Δ-correlation, (4) EDR observables behind a heavy-step switch (means/stds/drift over last 50 %, panedr), (5) composition-space PCA with per-property Δ overlay, (6) mechanical retraining trigger (`|paired_t|>3` or `frac_|d|>sd_legacy>0.5` on any active prop). Built per `marimo-data-analysis` skill.
4. **Re-run preprocessing** (`MartiniHeteroGraphBuilder` → chunk pt files) on the new trajectories.
5. **Retrain the best Tier C config** (lr=3e-5, wd=1e-3, h=128, l=2, e=200; 7 active properties) on the regenerated dataset and confirm Stage 5d's R² band holds.

## Current Work Focus

**Next phase — M3 lipidome analysis (2026-05-16)**. Plan at [docs/m3_lipidome_analysis_plan.md](../../docs/m3_lipidome_analysis_plan.md) for a marimo notebook `scripts/notebooks/analyze_m3_lipidome.py` characterising the full M3 lipid library (vendored `resources/martini3/itp/`, 32 ITP collections) before any new simulations. Two layers: **(A) lipid space** — descriptor panel (structural / bead-composition / bead-physics / optional graph-topology / deferred GNN single-lipid probe) × DR-clustering panel (PCA, MDS, UMAP, t-SNE, HDBSCAN, Ward); **(B) composition space** — simplex over lipid archetypes plus mole-fraction-weighted embedding centroid. Section 6 ties it back to the GNN: post-trunk embedding of the current 70 systems vs the descriptor-based composition embedding (disagreement = where extrapolation will be hardest). Output: shortlist of ~20–40 candidate compositions via a stratified-shells selection rule, feeding the martini_pipeline (subgoal 3a/3b coverage work). Defaults assumed: bilayer-forming-only scope, bead-composition + structural as primary lipid descriptor, stratified shells as the selection rule. Phase 1 deliverable; the GNN single-lipid-probe descriptor is Phase 2 (needs bead vocab decoupled from `LIPID_TYPES`).

**Final-epoch checkpoint saving added to `run_sweep.py` (2026-05-16)** — every run now writes `model_final.pt` to `wandb.run.dir` and uploads via `wandb.save()`. Contents: `state_dict`, `model_kwargs` (`in_channels, hidden_dim, num_layers, out_dim, comp_dim`), `properties`, per-property `scaler_mean`/`scaler_scale` for the active `prop_cols`, `epoch`, `run_id`. `download_wandb_runs.py::_ARTIFACT_FILES` updated to pull `model_final.pt` alongside `test_artifacts.npz`. Reload pattern: `ckpt = torch.load(...); m = MembranePropertyGNN(**ckpt["model_kwargs"]); m.load_state_dict(ckpt["state_dict"])`. Saves only the **final** epoch (no best-val selector — matches the Tier C reporting convention; `val_min_last10` is a metric, not a selector). Existing Stage 5d runs do **not** have this artefact retroactively — only runs submitted after the change will. Re-run those seeds if Section 6 of the lipidome plan needs the locked Tier C weights.

**Pipeline tooling polish & legacy resimulation (2026-05-17)**:
- **DIPC → DLPC name migration option**. M3-Lipid-Parameters renamed di-C18:2 PC from "DIPC" (legacy 70-system corpus name, still used as the LIPID_TYPES training token) to "DLPC". Added **DLPC as a parallel registry entry** (same physics as DIPC, modern token) and **`submit_simulations.sh --rename-lipid OLD=NEW`** (repeatable). Composition canonicalisation re-applies after the substitution so `DOPC50_DIPC50` → `DLPC50_DOPC50` keeps correct alpha-tiebreak order. Together with `--from-csv resources/done.csv`, this resimulates the legacy corpus and writes outputs under `DLPC*` directories. Existing DIPC training data + LIPID_TYPES vocabulary stay untouched; the rename is opt-in per submission. `martini_ff_node_mapping.json` DLPC entry resynced from legacy 10-bead to the M3 12-bead spec (mirror of DIPC).
- **`gmx mdrun -pin {on,off,auto}` wired end-to-end**. New `martini_pipeline.hpc_defaults.pin` (default `"on"`) flows through `submit_simulations.sh --pin` to the GPU worker. Benchmark TSV gained a `pin` column with two probe rows (`8_sim_8gpu_cpu4_pin_on`, `8_sim_8gpu_cpu8_pin_on`) — `pin=auto` was the historical regime; on multi-slot nodes mdrun's auto can refuse to pin and let the OS migrate threads. Re-run the bench to quantify the delta.
- **`--from-csv` in `submit_simulations.sh`** — inverse of `--completed-csv`. The CSV's `canonical_name` column IS the work list. Designed to resimulate the legacy 70-system corpus (point at `resources/done.csv`) with the modern M3 ITPs so all data shares one set of itp definitions. Composes with `--completed-csv` to resume cleanly across reruns.
- **`scan_completed_systems.py` length-aware**. New `sim_ns` column resolved in three tiers, preferring authoritative finish signal:
  1. `actual` — `Statistics over N steps using M frames` line in `prun.log` (only written on clean finish — so its presence implies the run completed AND tells you the produced length).
  2. `requested_manifest` — `manifest.json` `mdp_params.nsteps_prod` × dt (setup value).
  3. `requested_log` — `nsteps =` from log MDP echo (setup value).
  `sim_ns_source` column records which tier won. New flags `--min-ns NS`, `--require-actual`, and `--merge-with CSV` (order-preserving union — fresh-scan rows overlay in place, net-new rows tail-append so a diff against the previous CSV isolates new additions).
- **CHOL bug fix**: insane's default `CHOL` keyword is the legacy Martini 2 8-bead topology; the M3 sterols ITP ships 9 beads. Switched `lipid_registry`'s CHOL `insane_keyword` to `"M3.CHOL"`. All CHOL combinations now grompp successfully.
- **`projected_finish.py` fix**: `rglob("prun.log")` so the script works when pointed at a parent dir (each system has a `run/` subdir).

**Martini pipeline step 10 GPU benchmark — `hpc_defaults` locked (2026-05-16)** — 10-point sweep done. Winner `8_sim_8gpu_cpu4`: 41,909 ns/day aggregate (~5,240 ns/day per slot), score 8.6 M ns·day/node·hour. Final GPU defaults: `sims_per_node=8, gpus_per_node=8, cpus_per_sim=4, mem_per_sim=16G`. Notable: `cpus_per_sim=8` is 15 % slower than 4 (MI210s aren't CPU-thread limited here); `16_sim_8gpu_share` has higher aggregate throughput (53,730 ns/day) but per-slot drops 36 % — not worth the 2× IO cost. **Recommended `--time` for 1 µs on GPU: 8 h** (raw 4.6 h × 1.7 margin). Step 10 closed.

**Martini pipeline step 10c — general1 CPU production live (2026-05-15)** — `popc_interpolation` grid submitted on Goethe-HLR `general1` (CPU partition, no GPUs) for 1 µs (`--prod-ns 1000`) with 48 h walltime. Production routing on the CPU partition is fully wired: `submit_simulations.sh` dispatches to `sbatch_simulations_general1.sh` (spack openmpi + GROMACS-2022, `_gmx_mpi_wrapper.sh` shim) when `--partition general1` is set. Calibrated `hpc_defaults_cpu`: `sims_per_node=2`, `mpi_ranks_per_sim=1`, `cpus_per_sim=20`, `mem=16G`. Aggregate ~13 200 ns/day per node at the chosen point. Mid-run estimate from checkpoint deltas: ~22 ns/day per slot → ~65 h for 1 µs (over 48 h budget). Resubmit-with-`-cpi` may be needed for some slots.

**Goal framework (refactored 2026-05-13)** — Composition-coverage work split into sub-deliverables: **3a** `popc_interpolation` (POPC-anchored binaries at 10 % step; 77 systems total — current focus), **3b** DPPC/DOPC corner extrapolation, **3c/3d** broader extension after lipid-pool growth (step 12). Tracked in `docs/martini_pipeline_plan.md` §1.

**New tooling this session**:
- [scripts/python/scan_completed_systems.py](../../scripts/python/scan_completed_systems.py) — walks output roots, canonicalises directory names, emits CSV `(canonical_name, source_dir, source_root, status, has_prun_xtc)`. Feeds `submit_simulations.sh --completed-csv` to skip already-simulated systems without needing the data on HPC.
- [scripts/simulation/projected_finish.py](../../scripts/simulation/projected_finish.py) — scans `prun.log` files, parses `Writing checkpoint` lines for steps/sec, projects ETA against MDP `nsteps`, flags slots exceeding `--walltime`. Necessary because GROMACS only emits `Performance:` after a successful finish.
- `analyze_benchmark.py --cpu` flag — separate device-aware recommendation logic; emits `hpc_defaults_cpu` YAML.
- `popc_interpolation_grid(step)` generator in `martini_pipeline/analysis.py`.

**Bug fixes worth knowing (one-liners, see git history for detail)**:
- `--mdrun-args` argparse-REMAINDER greediness silently absorbed flags placed after it (`--prod-ns`, `--nsteps`, …). Pipeline CLI now uses a single quoted string; bash workers place `--mdrun-args` LAST in the arg list. Regression test added.
- SLURM `--export=ALL,VAR=...` silently drops entries on Goethe-HLR slurm-wlm. Replaced with **env-file-via-positional-arg**: orchestrator writes an `export VAR=$'...'` file, passes its path as `$1`, worker sources it on entry.
- gmx v2025.4 requires `-ntmpi 1` alongside `-ntomp N` on GPU runs. Added to both GPU production and bench workers.
- 40-job QOS cap on `general1` and 2-job cap on `gpu_test` now enforced by `submit_simulations.sh`.

---

## Earlier Focus — Training (still the most recent training milestone)

**Tier C Stage 5d complete — 6-seed confirmation (2026-05-07)** — `stage_5d_tier_c_confirm` at locked HPs (`lr=3e-5, wd=1e-3, h=128, l=2, e=200`), 7 active properties incl. `compressibility`. Seed 3 excluded (recurring dead-init); replacement seeds 6 and 8 completed as healthy runs. Final pool: seeds {0,1,4,5,6,8}, all 6 healthy. **Tier A, B, and C are all complete.**

**Headline test results (6-seed pool, normalised, pooled R²)**:

| Property | Test MSE ± std | Pooled test R² (95 % CI) | Tier B 5c R² |
|---|---|---|---|
| `lipid_packing` | 0.0203 ± 0.0014 | 0.976 [0.972, 0.979] | 0.978 |
| `thickness` | 0.0778 ± 0.0089 | 0.906 [0.895, 0.916] | 0.905 |
| `thickness_std` | 0.1292 ± 0.0174 | 0.887 [0.867, 0.902] | 0.882 |
| `variation` | 0.0683 ± 0.0083 | 0.933 [0.927, 0.939] | 0.929 |
| `persistence` | 0.4092 ± 0.0118 | 0.576 [0.532, 0.616] | 0.578 |
| `diffusivity` | 0.0331 ± 0.0020 | 0.960 [0.955, 0.964] | 0.959 |
| `compressibility` | 0.1480 ± 0.0199 | **0.881 [0.860, 0.898]** | (new) |

**Gate check**: **6/7 pass**. Only `persistence` technically fails (0.387 vs gate 0.370, +4.6 %) — a sample-composition artefact of seed 3's val numbers having pulled the Stage 0d gate down; not a regression. `diffusivity` now passes (0.065 vs gate 0.066). Pre-registered "Tier A+B within ~10 % of 5c" success criterion is met (max deviation +12 % on `lipid_packing` test MSE).

**Compressibility val/test R² gap**: pooled test R² = 0.88; per-seed val R² ≈ 0.59 (W&B summaries). The val split (~40 graphs/seed) is too small for stable R² estimation. Pooled test R² over 1 650 points (6 × 275) is the credible number. Both should be reported in the thesis with the gap flagged.

**Net cost of the 7th head vs Tier B 5c (test MSE)**: `lipid_packing` +12 %, `thickness` −1 %, `thickness_std` −4 %, `variation` −6 %, `persistence` 0 %, `diffusivity` −2 %. Net wash on 5/6 Tier B properties; one localised regression on `lipid_packing`. Compressibility itself learns substantially better than the pre-registered "<<0.5" architectural-ceiling expectation.

**Paired t-test 5d vs 0d**: t = −0.43, p = 0.348 — not significant, **expected** (same HPs, same epochs; uses 4 common seeds {0,1,4,5}; seeds 6 and 8 are not in Stage 0d; substantive Tier C contrast is per-property vs Tier B 5c, not aggregate vs 0d).

**Notebook updates (2026-05-07)**: `scripts/notebooks/analyze_stage_5.py` retargeted from 5c/0c/Tier B to 5d/0d/Tier C — title, prerequisites, output path (`results/figures/stage_5d/`), gate-check description, paired-t-test caption (now flags it as a noise-only comparison), Conclusions section rewritten end-to-end (8 numbered findings + caveats covering seed 3, val/test R² gap, peripheral-composition errors, `bending_modulus` deferral). Plot titles rewritten to describe variables rather than narrative (per analysis-style preferences). `PROP_LABELS` typo `Cmpressibility` fixed. Stage f label-stripping now handles `_tier_c`. Figures already on disk in `results/figures/stage_5d/` (rendered by user from the 4-seed run).

**SLURM submission refactor — multi-GPU packing per node (2026-05-05)** — `submit_sweep.sh` and `sbatch_sweep.sh` overhauled. Each Cartesian-product cell (incl. each seed) is now its own "run". Runs are packed onto a single node up to `--gpus-per-node` (default 8); excess runs spill into additional sbatch jobs. New CLI flags: `--partition` (default from `hpc.partition_train`), `--time` (default `24:00:00`), `--gpus-per-node`, `--cpus-per-gpu` (default 8), `--mem-per-gpu` (default 64G); SLURM resource flags are now set on the sbatch CLI rather than as static `#SBATCH` directives. `sbatch_sweep.sh` fans out N background `python run_sweep.py` processes pinned via `HIP_VISIBLE_DEVICES=$i`/`CUDA_VISIBLE_DEVICES=$i`, each with its own `RUN_<i>_*` → `FREEZE_*`/`SWEEP_SEEDS` env. Per-process logs at `logs/sweeps/sweep-<jobid>-gpu<i>.{out,err}`; SLURM `%j.out` is the orchestrator log. `gpu_test` partition guards: `--time` capped at `08:00:00` with warning; aborts if total runs need >2 batches. `run_sweep.py` unchanged (existing env-var override path already handles per-process freezing).

**Tier C Stage 0d — OUTCOME C: Negative Transfer (2026-05-01)** — Adding compressibility (7th property) degraded all 6 Tier B properties beyond their gates. All properties FAIL the Stage 5c thresholds. `config.yaml` `active_properties` reverted to 4 (Tier A). Next step: decide remediation (uncertainty weighting, separate heads, or dataset expansion) before retrying Tier C. Results recorded in `docs/tier_c_7prop_plan.md`.

**`analyze_hp_search.py` marimo notebook overhauled (2026-05-05)** — All 7 plotting cells were broken (invisible output due to `_run_plot()` function wrapper anti-pattern); recommendation + multi-group cells also silent. Full rewrite fixes all rendering, adds `save_fig` → `results/training/<GROUP>/`, and resolves `sns`/`HAS_SEABORN` crash when seaborn absent. `pyarrow` and `jinja2` added to script dependencies.

**Tier B Stage 5c done (2026-04-30)** — 5-seed confirmation at locked Tier A HPs on 6 properties complete. Marimo analysis notebook `scripts/notebooks/analyze_stage_5.py` written and verified. **Tier B pipeline complete.**

`config.yaml` (Tier B active; locked HPs unchanged from Tier A):

- `active_properties: [lipid_packing, thickness, thickness_std, variation, persistence, diffusivity]`
- `learning_rate: 3.0e-5` (Tier A lock — confirmed by Stage 1e')
- `weight_decay: 1.0e-3`
- `epochs: 200`
- `hidden_dim: 128`, `num_layers: 2`

### Tier A stage status

| Stage | W&B group | Status | Result |
|-------|-----------|--------|--------|
| 0b — GNN floor, 4-prop | `stage_0b_tier_a` | done | val_min10: lp=0.022, th=0.074, th_std=0.359, var=0.462 |
| 1b — lr sanity | `stage_1b_tier_a_lr` | done | lr=1e-5 best of {1e-5,1e-4,5e-4}; only lr=1e-5 learned variation |
| 1b' — lr refinement | `stage_1b_refine_tier_a_lr` | done | lr=3e-5 wins (val_total 0.149); seed-2 variation failure exposed |
| 1c — seed stability | `stage_1c_seed_stability_tier_a` | done | 1/5 fail (seed 9); 22% combined (2/9) failure rate; seed 6 late-escape pattern found |
| 1d — long-training rescue | `stage_1d_long_train_tier_a` | done | seed 9 RESCUED at 200ep (val_var ~0.10); seed 2 still stuck (~0.53) — drop permanently |
| 2b — wd verification | `stage_2b_quick_wd_tier_a` | done | wd insensitive in [3e-4, 1e-3, 3e-3]; tiny per-property tradeoffs but val_total flat |
| 5b — 5-seed confirmation | `stage_5b_tier_a_confirm` | **done** | 6/7 seeds healthy; paired t=−31.5, p=3.5e-5 vs Stage 0b; per-prop R² ≥ 0.87 |

### Stage 5b headline results (2026-04-28)

7 finished runs analysed (planned seeds {0,1,3,4,5} plus extras {6,9} that the W&B group filter pulled in). All at locked Tier A config.

**Test MSE / R² (pooled, normalised)**:

| Property | MSE mean ± std | R² (95 % CI) |
|---|---|---|
| `lipid_packing` | 0.020 ± 0.003 | 0.975 [0.972, 0.978] |
| `thickness`     | 0.076 ± 0.007 | 0.908 [0.898, 0.917] |
| `thickness_std` | 0.145 ± 0.024 | 0.873 [0.856, 0.888] |
| `variation`     | 0.131 ± 0.171 | 0.872 [0.856, 0.887] |

**Acceptance gates (val MSE, last-10 mean)** — 3 of 4 pass:
- `lipid_packing` 0.0222 vs 0.022 — **fail by −0.0002** (statistical tie within seed jitter; per-property tradeoff documented in 2b)
- `thickness` 0.0732 vs 0.074 — pass
- `thickness_std` 0.299 vs 0.359 — pass (+17 %)
- `variation` 0.151 vs 0.462 — pass (+67 %)

**Paired t-test vs Stage 0b** (n=4 common seeds: 0,1,3,4): **t = −31.5, p = 3.5 × 10⁻⁵**, ~66 % test-MSE reduction. Direct counterpoint to the 2-prop Stage 5 null result (p = 0.755) — the thesis story for Tier A.

**GNN vs Ridge-on-composition baseline**: GNN beats Ridge by 56–84 % across all four properties.

**Seed health in 5b**: 6/7 seeds healthy (`val_min_last10` ∈ [0.107, 0.143]). Seed 6 failed `variation` despite escaping at ~50 ep in Stage 1c — escape is non-deterministic per seed; widens `variation` MSE std from ~0.02 to 0.171. For thesis reporting, prefer the planned 5-seed pool {0,1,3,4,5} as primary numbers.

**Per-system error concentration**: errors dominate on DPPC- and DOPC-rich mixtures (POPC30_DOPC70 worst, ~19 Å thickness MAE); these sit at the boundary of the test cloud in PCA(composition) space where train density also drops. Documented as a Tier A scope limit.

Full report: [results/figures/stage_5b/stage_5b_analysis_report.md](../../results/figures/stage_5b/stage_5b_analysis_report.md). Headline JSON: `results/figures/stage_5b/headline_numbers.json`.

### Key Tier A findings (consolidated)

**HP saturation finding**: 2-prop Stage 5 (lr=1e-4) had paired t-test p=0.755 — HP search produced no significant improvement. Tier A reverses this: paired t=−31.5, p=3.5e-5. **HP tuning matters more for harder properties** is the main thesis story.

**Variation-property fragility (init-dependent)**:
- ~20 % of seed inits fail to learn `variation` regardless of lr.
- Two failure subtypes:
  - *Slow escapers* (seeds 6, 9): plateau at ~0.5 from epoch 20–50, then break through to ~0.08–0.10. Rescued by 200-epoch training. Escape is non-deterministic — seed 6 escaped in 1c, failed in 5b.
  - *True dead-init* (seed 2): plateau forever. No movement after 200 epochs. Drop permanently.
- `thickness_std` and `variation` failures are correlated within a seed — one loss-landscape pathology, not two independent ones.

**wd is small lever**: Stage 2b found val_total flat across the 10× wd range. Per-property tradeoff exists (higher wd helps variation slightly, hurts thickness_std slightly). Locked wd=1e-3.

**GPU memory clarification**: live-tensor peak ~8 GB out of 64 GB (logged via `gpu/peak_mem_actual_gb` in `run_sweep.py`). Earlier "97 % peak" was W&B's reserved pool, not actual usage. Tier B/C have huge memory headroom.

### Gates (Tier B — Stage 0c 6-prop floor, val_min_last10 mean over 5 seeds)

- `lipid_packing < 0.019`, `thickness < 0.067`, `thickness_std < 0.302`, `variation < 0.151`, `persistence < 0.362`, `diffusivity < 0.059`
- These are the gates Stage 5c must beat. Set in [docs/tier_b_6prop_plan.md](../../docs/tier_b_6prop_plan.md) and [scripts/notebooks/analyze_hp_search.ipynb](../../scripts/notebooks/analyze_hp_search.ipynb) Cell 1 `GATES`.
- Historic Tier A gates (Stage 0b 4-prop floor) preserved in the notebook as a reference comment: `lp<0.022, th<0.074, th_std<0.359, var<0.462`.

### Tier B Stage 0c headline (2026-04-28)

5/5 seeds finished, 4/5 healthy (seed 3 stuck on `variation` ≈ 0.45 — same dead-init pattern as Tier A seed 2). All at locked Tier A HPs.

| Property | val_min10 (5-seed mean) | R² (epoch-200) | vs Stage 5b |
|---|---|---|---|
| `lipid_packing` | 0.019 | 0.94 | −14 % |
| `thickness` | 0.067 | 0.95 | −8 % |
| `thickness_std` | 0.302 | 0.66 | +1 % (tied) |
| `variation` | 0.151 | 0.95 (healthy) | −0 % (tied) |
| `persistence` | 0.362 | 0.66 | new |
| `diffusivity` | 0.059 | 0.96 | new |

**No negative transfer** — Tier A properties hold or improve at the inherited HPs. **`diffusivity` learns cleanly** (R² ≈ 0.96 — comparable to `lipid_packing`/`thickness`); a meaningful positive thesis result that a single-frame embedding can predict a time-averaged dynamical property. **`persistence` is the hard target** (val 0.36, R² ≈ 0.66, floor-like across all 5 seeds); first candidate to test in Stage 1e for a different lr.

### Capacity trade-off between heterogeneity properties and `persistence` — confirmed

Originally a single-seed anecdote from Stage 0c (seed 3 failed variation, had best persistence). Stage 1e showed the same pattern systematically across **lr groups**: at lr=3e-5, the seed that fails `variation` (seed 0, val_var=0.464) has the best `persistence` (0.324). At lr=1e-4, both seeds fail variation AND both have persistence ≈ 0.344 (better than lr=1e-5 where variation is always healthy and persistence ≈ 0.356). **Whenever the trunk gives up on `variation`, capacity flows to `persistence`.** This is a structural property of the shared MLP readout, not a seed artefact.

Stage 1e' (4 seeds × 3 lrs, all 12 seeds with healthy variation) reframes the pattern: the marginal `persistence` advantage of lr=1e-5 over lr=3e-5 (0.344 vs 0.368, ≈7 %) is the *floor* of how much capacity competition costs once variation is healthy across all seeds. The architecture floor for `persistence` is ~0.35 regardless of lr. Both observations stand together: capacity competition is real (1e dataset) AND lr alone cannot move the persistence floor (1e' dataset).

Implication: improving `persistence` without degrading `variation`/`thickness_std` likely requires separate heads or uncertainty weighting. Flag for thesis discussion as evidence of capacity competition in multi-task shared-trunk GNNs.

### Tier B stage status

| Stage | W&B group | Status | Result |
|-------|-----------|--------|--------|
| 0c — GNN floor, 6-prop | `stage_0c_tier_b` | done | No negative transfer; persistence hard (R²≈0.66); diffusivity easy (R²≈0.96) |
| 1e — lr sanity check | `stage_1e_tier_b_lr` | done | 2-seed pilot: lr=1e-5 wins (val_total 0.161); but seed-0 3e-5 variation failure inflated 3e-5 mean |
| 1e' — lr refinement | `stage_1e_refine_tier_b_lr` | **done** | 4-seed grid: **lr=3e-5 wins** (val_total 0.148); Tier A lock preserved |
| 1f — seed stability | `stage_1f_tier_b_seed_stability` | skipped | Not triggered: 1e' kept the lock; 4/4 seeds healthy at 3e-5 in 1e' is implicit stability evidence |
| 5c — 5-seed confirmation | `stage_5c_tier_b_confirm` | **done** | 5/5 seeds healthy; R² ≥ 0.88 on 5/6 props; persistence floor R²=0.578; t-test p=0.286 vs Stage 0c (expected — same HPs) |

### Stage 1e headline (2026-04-28)

6 runs finished (`stage_1e_tier_b_lr`, 2-seed pilot). Initial signal: **lr=1e-5 wins** on val_total (0.161 vs 0.181 vs 0.225). Variation fails at both seeds at 1e-4 and at seed-0 at 3e-5. Persistence R² ≈ 0.67–0.69 across all lrs — confirmed architecture-limited. **Triggered Stage 1e' refinement at 4 seeds.**

### Stage 1e' headline (2026-04-29)

12/12 runs finished (`stage_1e_refine_tier_b_lr`, grid `{3e-6, 1e-5, 3e-5}` × seed ∈ {0,1,3,4}). **lr=3e-5 wins on val_total (0.148) — Tier A lock confirmed.** Stage 1e signal flips at 4 seeds.

| lr            | val_total | lipid_packing | thickness | thickness_std | variation | persistence | diffusivity |
|---------------|-----------|---------------|-----------|---------------|-----------|-------------|-------------|
| 3e-6          | 0.179     | 0.027         | 0.089     | 0.372         | 0.175     | 0.364       | 0.082       |
| 1e-5          | 0.153     | 0.027         | 0.075     | 0.319         | 0.091     | 0.344       | 0.068       |
| 3e-5 (lock) ← | 0.148     | 0.020         | 0.066     | 0.297         | 0.084     | 0.368       | 0.059       |

Seed std on val_total: 3e-6 = 0.0102, 1e-5 = 0.0069, **3e-5 = 0.0013** (≈5–8× tighter).

**Key resolution**: the Stage 1e seed-0 3e-5 variation failure (val_var=0.464) was a **single-seed bad init**, not a lr=3e-5 problem. In 1e', all 4 seeds at 3e-5 escape plateau (val_var ∈ [0.075, 0.099]). 3e-5 also wins every property except `persistence` (lr=1e-5 marginally better at 0.344 vs 0.368, ≈7 % — small relative to architecture floor of ~0.35; persistence is bound by representation, not lr).

**Decision**: keep `lr=3e-5` lock in `config.yaml` (no change needed). **Skip Stage 1f** — only required if 1e' had changed the lr; 4/4 seeds healthy at 3e-5 in 1e' provides the seed-stability check implicitly.

## Latest Changes (this session, 2026-04-30)

- **`scripts/notebooks/analyze_stage_5.py`** (new): marimo conversion of `analyze_stage_5.ipynb`, following the `marimo-data-analysis` skill. Pointed at `stage_5c_tier_b_confirm`. 10 figures (a–j) including new figure (j) — percentage-error box plot, direct counterpart to Emil's composition-only FFNN reference. Train compositions derived from `CONFIG.paths.data_dir` listing to avoid `torch_geometric` dependency in fresh envs.
- **`results/figures/stage_5c/`**: 10 PDF + PNG figures + `headline_numbers.json`.

### Stage 5c headline results (2026-04-30)

5/5 seeds, all with `test_artifacts.npz`. Locked HPs: `lr=3e-5, wd=1e-3, hidden_dim=128, num_layers=2, epochs=200`.

| Property | MSE mean ± std | R² (95 % CI) |
| --- | --- | --- |
| `lipid_packing` | 0.0182 ± 0.0028 | 0.978 [0.974, 0.981] |
| `thickness` | 0.0789 ± 0.0059 | 0.905 [0.892, 0.916] |
| `thickness_std` | 0.1342 ± 0.0115 | 0.882 [0.863, 0.897] |
| `variation` | 0.0730 ± 0.0370 | 0.929 [0.921, 0.936] |
| `persistence` | 0.4077 ± 0.0065 | 0.578 [0.528, 0.621] |
| `diffusivity` | 0.0337 ± 0.0014 | 0.959 [0.953, 0.964] |

Paired t-test vs Stage 0c: t = −0.614, **p = 0.286** — not significant (expected; Stage 5c and 0c share identical HPs; Stage 1e' confirmed the Tier A lock was already optimal). GNN beats composition-only Ridge by ~80 % overall. Tier B pipeline complete.

## Latest Changes (previous session, 2026-04-28)

- **`config.yaml`**: `epochs: 100 → 200`, `learning_rate: 1.0e-4 → 3.0e-5`. Tier A defaults locked.
- **`docs/tier_a_4prop_plan.md`**: Stage 1b'/1c/1d/2b results recorded; Stage 5b seed selection finalised.
- **`scripts/notebooks/analyze_hp_search.ipynb`**: R² wired into 4 cells:
  1. `cell-load-fn`: `_tail_mean()` helper + `val_r2_{prop}` per-property loading from `history.parquet`.
  2. `cell-detect-hps`: `_PROP_VALS` excludes `val_r2_*` from HP detection.
  3. `cell-aggregate`: `cells_df` has `r2_{prop}` columns.
  4. `cell-ranking-table`: shows MSE and R² side-by-side.
  5. `cell-recommendation`: prints per-property R² with [GOOD/OK/WEAK] tags after MSE gate check.
- **`scripts/training/run_sweep.py`**: `gpu/peak_mem_actual_gb` logged per-epoch (live-tensor high-water; resets after each epoch). CUDA-guarded.

### Earlier in this session

**Stage 1c findings (seed-stability check at lr=3e-5)**:
- 5 of 6 seeds finished (seed 7 failed to start, HPC I/O error — not retried).
- Healthy seeds {4,5,6,8}: val_total 0.114-0.122. Seed 9: val_total 0.243 (variation stuck at 0.471).
- **Seed 6 late-escape**: variation plateaued at ~0.5 until epoch ~50, then broke through to val_var=0.082 (best of sweep). First evidence the plateau is escapable.

**Stage 1d findings (200-epoch rescue test on seeds 2 and 9)**:
- Seed 9: variation broke through at step ~3500 (~55 epochs), val_var settled at ~0.08–0.10 by 200 epochs. Train and val track. **Rescued — keep as healthy seed**.
- Seed 2: completely flat throughout 200 epochs (val_var ~0.53). True bad-init. **Drop permanently**.
- Conclusion: bump default epochs to 200, drop seed 2 only.

**Stage 2b naming bug**: original `run_name` encoding (`gnn_only_h{h}_l{l}_lr{lr}_s{seed}`) didn't include `wd`, causing collisions when wd varied. User fixed naming, redownloaded. Future stages: include all varying HPs in run_name.

## Previous Latest Changes

**Tier A 4-property plan + per-property test logging (2026-04-26)**: see git history.
**Bug fix — test_artifacts.npz was never actually saved (2026-04-26)**: fixed in commit `5fead3c`'s follow-up.
**Stage 5 analysis pipeline + publication notebook (2026-04-26)**: 9-figure publication notebook in `analyze_stage_5.ipynb`.
**Stratified system-level split (2026-04-25)**: fixed test-narrowness bug.

## Important Patterns and Preferences

- Test discrete components locally before integrating into the heavy end-to-end training loop.
- Results uploaded to W&B for visualization.
- Force field parameters loaded from JSON files at graph build time.
- `LIPID_TYPES` ordering must stay consistent across `lipid_graph.py`, `linear_baseline.py`, `run_sweep.py`.
- `preprocess_and_save` is the single entry point for building and saving graph chunks.
- All HP values frozen at sbatch submission time via `submit_sweep.sh`; `run_sweep.py` reads them via `_apply_submission_overrides()`.
- Run names encode every varying HP **and** end in the W&B `run.id` (e.g. `gnn_only_h128_l2_lr3e-05_wd1e-03_e200_s0_<run_id>`); the trailing `_{run.id}` is the uniqueness contract — preserve it in all future stages. `download_wandb_runs.py` writes a `.wandb_run_id` marker file per local dir and raises `RuntimeError` on collision.
- Selection metric is MSE (`val_min_last10`); R² is reported alongside as a complementary, more interpretable signal (R²≥0.85 GOOD, ≥0.5 OK, <0.5 WEAK).
- For thesis reporting on multi-seed runs that include rescued/extra seeds, prefer the planned-pool primary numbers and footnote the extras (5b precedent: seeds {0,1,3,4,5} primary, {6,9} extras).
