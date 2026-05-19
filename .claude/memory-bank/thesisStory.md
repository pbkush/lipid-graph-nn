# Thesis Story

Narrative of how the project got from initial idea to the Tier A confirmation result. Captures the motivating question, the major architectural and methodological decisions, the findings that drove each pivot, and the headline numbers at every stage. Intended as the "what happened, in order, and why" companion to `progress.md` (which lists current state) and `activeContext.md` (which captures the most recent slice of work).

---

## 0. Starting point and scientific question

The project asks: **can we learn a useful, transferable embedding of membrane systems from coarse-grained MD snapshots, validated through property prediction?** The end goal is a representation that generalises to any membrane (and eventually to protein+membrane), not the property predictions themselves — those are training signal and a sanity check. See `projectbrief.md` and `productContext.md`.

**Inputs**: 70 Martini 3 membrane compositions drawn from a 10-lipid pool, MD frames `[50:667]` from 1 µs equilibrated production runs. 8 target properties pre-computed per composition (`properties.md`); the model predicts a configurable subset.

**Initial architecture**: Heterogeneous GNN with two edge types (bonded topology + spatial cutoff) over CG beads carrying continuous Martini 3 force-field features `[mass, charge, σ, ε]`. `GATv2Conv` per edge type via `HeteroConv`, `GraphNorm`, mean+max pool readout, optional composition-vector concatenation at the readout. Reasoning: physically meaningful node features beat learned vocab embeddings, attention beats SAGE for weighting bond vs. spatial edges, and the `comp_dim=0` / `comp_dim=10` toggle lets us probe how much the topology+geometry alone contributes.

---

## 1. Pre-HP infrastructure decisions (early 2026)

These are the methodological choices that pre-date the HP search proper but materially shape the thesis story:

- **Continuous physics features over learned embeddings** — a deliberate "physics-informed" choice over vocab-style integer IDs.
- **Two edge types** (bonded static, spatial dynamic) with **Gaussian RBF distance encoding** (16 basis functions) — separates chemistry from geometry.
- **Chunked disk streaming** (`MartiniDiskDataset`) — avoids RAM pressure; carried over from Colab era.
- **Stratified system-level split** (`preprocess_graphs.py --split-method stratified`) — replaced random split after discovering the test set's y-range was 4× narrower than train, masking generalisation failures. K-means in z-scored y-space.
- **HPC migration (Goethe MI210 / ROCm)** — Colab notebooks frozen as legacy reference; all training now goes through `submit_sweep.sh` → SLURM. Prevents queue-drift corruption via `FREEZE_*` env vars consumed by `_apply_submission_overrides()`.
- **W&B-only logging + offline parquet pull** — `download_wandb_runs.py` pulls a group to disk; `analyze_hp_search.ipynb` operates on parquet. Decouples analysis from live W&B API.

These together define the *evaluation harness* against which every subsequent number is measured.

---

## 2. The 2-property HP search (Stages 0–5, single-property + 2-prop)

**Properties**: `lipid_packing` + `thickness`. Goal: beat the Colab-era ~0.138 val-MSE baseline before scaling up.

| Stage | What was swept | Best cell |
|---|---|---|
| 0 | GNN floor at `hidden_dim=64, num_layers=2, lr=5e-4, wd=5e-3` | val ≈ 0.138 |
| 1 | learning rate | `lr=1e-4` wins |
| 2 | weight decay | `wd=1e-3` wins (mild) |
| 3 | architecture (`hidden_dim`, `num_layers`) | `hidden_dim=128, num_layers=2` wins |
| 5 | 5-seed confirmation at locked config | val MSE **0.038** |

**Locked 2-prop config**: `hidden_dim=128, num_layers=2, lr=1e-4, wd=1e-3, epochs=100`.

**Pivotal negative result**: paired t-test of Stage 5 (winner) vs. Stage 0 (floor) gave **p = 0.755** — HP tuning produced no statistically significant improvement on the 2-property task. This is *the* finding that motivated everything that followed: it suggested the 2-property task was already saturated at the GNN floor, and that HP tuning would only matter once the property set included properties that were genuinely hard for the architecture.

**Stage 5 publication notebook** (`analyze_stage_5.ipynb`) was built here — 9 figures + bootstrap CIs + paired t-test + per-property R² — and was reused unchanged for Tier A.

---

## 3. The pivot to Tier A (4 properties)

Per the tiered plan in `docs/multi_property_training_plan.md` and `docs/tier_a_4prop_plan.md`, the project switched from 2 properties to 4: `lipid_packing`, `thickness`, `thickness_std`, `variation`. Reasoning:

- The 2-property HP-saturation result implied we needed harder properties to make HP tuning visible.
- `thickness_std` (intra-trajectory variability) and `variation` (Voronoi-area CoV) probe lipid-lipid interactions, not composition averages — a stronger test of the embedding.
- All four are "Tier A" properties (local geometric, high SNR, reachable with the 11 Å spatial cutoff). `bending_modulus` and `compressibility` were deliberately excluded as Tier C because they require longer-range receptive fields the current cutoff cannot provide.

Crucially: **no chunk rebuild needed** — the chunks already store all 8 properties in `y`, and `run_sweep.py` slices the active subset. The Tier A switch is a one-line `config.yaml` edit.

---

## 4. Tier A stages — what was decided and why

### Stage 0b — 4-property GNN floor

5 seeds, locked 2-prop HPs. Establishes the floor any later HP-tuned config must beat.

**Result (val_min_last10, mean over seeds)**: `lp=0.022, th=0.074, th_std=0.359, var=0.462`. These four numbers became the **acceptance gates** for every subsequent stage. `variation` at 0.462 is essentially the "no-skill" floor — the network was not learning the property.

### Stage 1b — lr sweep

`lr ∈ {1e-5, 1e-4, 5e-4} × seed ∈ {0, 1}`. Discovered that `variation` only learns at `lr=1e-5` (val 0.082) and stays at the floor (~0.46) at higher lr. **First evidence that HP tuning matters more for harder properties.** The 2-prop locked `lr=1e-4` was wrong for 4 properties.

### Stage 1b' — lr refinement

`lr ∈ {3e-6, 1e-5, 3e-5} × seed ∈ {0, 1, 2, 3}`. `lr=3e-5` won on every property (val_total mean 0.149); `3e-6` was undertrained at 100 epochs. **Locked `lr=3e-5`.**

Surprise finding: seed 2 stuck at the `variation` floor (0.53) regardless of lr — first hint of a seed-init failure mode.

### Stage 1c — seed stability

`lr=3e-5, seeds ∈ {4, 5, 6, 7, 8, 9}`. 5 of 6 finished (seed 7 HPC I/O failure). Result: 1/5 fail (seed 9), but seed 6 produced the **late-escape pattern**: `variation` plateaued at 0.5 from epoch 20–50, then broke through to 0.082 by epoch 100. Combined with Stage 1b': **2/9 seeds fail, ~22 % init failure rate**. Failure is correlated within a seed across `thickness_std` and `variation` — one loss-landscape pathology, not two.

### Stage 1d — 200-epoch rescue

Re-ran seeds 2 and 9 at 200 epochs to test the slow-escaper hypothesis. **Seed 9 rescued cleanly** (val_var settled at 0.08–0.10 by epoch 200). **Seed 2 still flat** at 200 epochs — true dead-init. Decisions:
- Bump default `epochs: 100 → 200`.
- Drop seed 2 permanently from the healthy pool.
- Two distinct failure modes documented: *slow escapers* (rescuable by 200 epochs) and *true dead-init* (drop the seed).

### Stage 2b — wd verification

`wd ∈ {3e-4, 1e-3, 3e-3} × seeds ∈ {0, 4}` at `lr=3e-5`. val_total flat across the 10× wd range (0.116 vs 0.118). Per-property tradeoff exists — higher wd helps `variation` slightly, hurts `thickness_std` slightly — but nets to noise. **Locked `wd=1e-3`.** Bug fix here: original `run_name` encoding didn't include `wd`, causing download collisions; fixed and recorded as the "always include all varying HPs in run_name" rule.

### Stage 5b — 5-seed confirmation

Locked config (`hidden_dim=128, num_layers=2, lr=3e-5, wd=1e-3, epochs=200`) on seeds {0, 1, 3, 4, 5}. The W&B group also picked up seeds {6, 9} from earlier runs — 7 finished runs analysed.

**Headline test results (pooled, normalised)**:

| Property | MSE mean ± std | R² (95 % CI) |
|---|---|---|
| `lipid_packing` | 0.020 ± 0.003 | 0.975 [0.972, 0.978] |
| `thickness`     | 0.076 ± 0.007 | 0.908 [0.898, 0.917] |
| `thickness_std` | 0.145 ± 0.024 | 0.873 [0.856, 0.888] |
| `variation`     | 0.131 ± 0.171 | 0.872 [0.856, 0.887] |

**Acceptance gates** (val MSE, last-10 mean): 3 of 4 pass cleanly; `lipid_packing` 0.0222 vs gate 0.022 — a tied "fail" (margin −0.0002) within seed jitter, reflecting the per-property tradeoff documented in Stage 2b.

**Paired t-test vs Stage 0b** (n=4 common seeds: 0, 1, 3, 4): **t = −31.5, p = 3.5 × 10⁻⁵**, ~66 % test-MSE reduction. Direct counterpoint to the 2-prop Stage 5 null result.

**GNN vs Ridge-on-composition baseline**: GNN beats Ridge by 56–84 % across all four properties.

**Seed health in 5b**: 6 of 7 seeds healthy (`val_min_last10` ∈ [0.107, 0.143]). Seed 6 failed to escape `variation` despite escaping at epoch ~50 in Stage 1c — confirms the escape is non-deterministic per seed; widens `variation` MSE std from ~0.02 to 0.171.

The full 5b report is in `results/figures/stage_5b/stage_5b_analysis_report.md`.

---

## 5. The thesis story arc, in one paragraph

The 2-property HP search produced a clean negative result — paired t-test p = 0.755, no statistical gain over the GNN floor. This *was* the result that motivated the Tier A pivot. Adding `thickness_std` and `variation` (harder, lipid-lipid-interaction-sensitive properties) immediately surfaced an HP signal: `variation` only learned at `lr=3e-5`, an order of magnitude below the 2-prop optimum, and only with 200-epoch training. The HP-tuned 4-property GNN beats its untuned Stage 0b counterpart by ~66 % test MSE (p = 3.5 × 10⁻⁵) and the linear-composition baseline by 56–84 %. Per-property R² is ≥ 0.87 on every Tier A property. The story is: **HP tuning was a no-op for easy targets and indispensable for hard ones, and the dominant lever was learning rate**. Secondary finding: **~20 % of random inits fail to learn `variation` due to a loss-landscape pathology**, with two failure subtypes (slow escapers, rescuable by 200 epochs; true dead-init, must be dropped). This is documented as a Tier A scope limit.

---

## 6. Tier B Stage 0c — first 6-property measurement (2026-04-28)

`active_properties` extended to 6 by adding `persistence` and `diffusivity` to
the `config.yaml`; locked Tier A HPs unchanged. 5 seeds `{0, 1, 3, 4, 5}` ran at
the inherited config. Headline:

| Property        | val_min10 (5-seed mean) | R² (epoch 200) | Δ vs Stage 5b |
|-----------------|-------------------------|----------------|---------------|
| `lipid_packing` | 0.019                   | 0.94           | −14 %         |
| `thickness`     | 0.067                   | 0.95           | −8 %          |
| `thickness_std` | 0.302                   | 0.66           | +1 %          |
| `variation`     | 0.151                   | 0.95 (healthy) | −0 %          |
| `persistence`   | **0.362**               | 0.66           | new           |
| `diffusivity`   | **0.059**               | 0.96           | new           |

Three findings worth keeping for the thesis:

1. **No negative transfer.** Adding two more targets to the shared trunk did not
   degrade Tier A — every 4-property number holds within seed jitter, with
   `lipid_packing` and `thickness` actually improving by ~10 %. The
   homoscedastic-uncertainty-weighting remedy planned in `tier_b_6prop_plan.md`
   stays deferred.
2. **`diffusivity` is recoverable from a static snapshot.** R² ≈ 0.96 — on par
   with `lipid_packing`. A single MD frame's bead geometry contains enough
   information to predict the time-averaged lateral diffusion coefficient. This
   is a non-trivial structural-vs-dynamical positive result; pre-registered as
   uncertain in the plan.
3. **`persistence` is the new hard target.** val 0.36, R² ≈ 0.66, floor-like
   across all 5 seeds. Same structural-vs-dynamical question as `diffusivity`,
   opposite answer. Most likely candidate for an lr re-tune in Stage 1e —
   echoing the Stage 1b finding where `variation` only learned at lr two
   decades below the 2-prop optimum.

Seed fragility on `variation` recurred (seed 3 stuck ~0.45 — true dead-init,
matching Tier A seed 2). 4/5 healthy is consistent with the documented ~20 %
init-failure rate.

The Stage 0c per-property means become the Tier B gates for Stage 5c, locked
in `tier_b_6prop_plan.md` § "Acceptance gates for Stage 0c" and in
`scripts/notebooks/analyze_hp_search.ipynb` Cell 1 `GATES`.

---

## 7. Tier B Stage 5c — 6-property confirmation (2026-04-30)

5-seed confirmation at the locked Tier A HPs on all six Tier B properties.
Full report in `results/figures/stage_5c/` (10 figures + `headline_numbers.json`).

| Property | MSE mean ± std | R² (95 % CI) |
| --- | --- | --- |
| `lipid_packing` | 0.0182 ± 0.0028 | 0.978 [0.974, 0.981] |
| `thickness` | 0.0789 ± 0.0059 | 0.905 [0.892, 0.916] |
| `thickness_std` | 0.1342 ± 0.0115 | 0.882 [0.863, 0.897] |
| `variation` | 0.0730 ± 0.0370 | 0.929 [0.921, 0.936] |
| `persistence` | 0.4077 ± 0.0065 | 0.578 [0.528, 0.621] |
| `diffusivity` | 0.0337 ± 0.0014 | 0.959 [0.953, 0.964] |

**Paired t-test vs Stage 0c**: t = −0.614, p = 0.286 — not significant. This is *expected*: Stage 1e' confirmed the Tier A lr=3e-5 lock was already optimal, so Stage 5c and Stage 0c differ only in run order, not in configuration. Unlike Tier A (t=−31.5 because lr changed from 1e-4 to 3e-5), the Tier B HP search produced no configuration change. The story is: **HP search confirmed the optimum rather than finding a new one.**

Three headline findings for the thesis:

1. **`persistence` is architecture-limited, not HP-limited.** R² = 0.578, flat across all seeds and all three lrs tested in Stage 1e'. The capacity-competition finding from Stage 1e/1e' (seeds that fail `variation` have better `persistence`) implies the shared MLP trunk cannot simultaneously represent both heterogeneity properties and persistence. Separate heads or uncertainty weighting are the likely remedy.

2. **`diffusivity` confirms static-snapshot → dynamical-property prediction.** R² = 0.959, comparable to `lipid_packing`. A single MD frame's bead geometry carries enough information to predict the time-averaged lateral diffusion coefficient. This is a clean positive result that addresses the "can structure predict dynamics" question directly.

3. **No negative transfer confirmed at the formal confirmation stage.** Tier A R² values hold within seed jitter when two additional properties are added. The homoscedastic weighting remedy from the tier-B plan stays deferred.

A new figure (j) — percentage-error box plot `(pred−true)/true×100` — was added to the analysis notebook as a direct counterpart to Emil's composition-only FFNN reference. The GNN achieves ±1–2 % IQR on `lipid_packing`, `thickness`, `variation`; `diffusivity` ±5 % IQR; `persistence` shows widest spread (+3.5 % median bias).

---

## 8. Tier C Stage 0d → 1g → 1g' → 5d (2026-05-01 to 2026-05-07)

`active_properties` extended to 7 by adding `compressibility` (area
compressibility modulus, Å³/kT). `bending_modulus` was dropped from the target
set — the undulation-spectrum-fit label is too noisy/unreliable to serve as a
trustworthy training signal. Pre-registration: `compressibility` R² « 0.5,
architecture-limited because area-fluctuation statistics integrate over scales
beyond the 11 Å spatial cutoff.

**Stage 0d (5-seed floor at locked Tier B HPs)**: Outcome C — all 6 Tier B
properties degraded 2–24 % vs Tier B 5c gates; `compressibility` R² = 0.55,
already above the pre-registered ceiling. The 24 % `lipid_packing` regression
triggered Stage 1g.

**Stage 1g pilot (2 seeds × 3 lrs)**: lr=1e-5 won on val_ab6 (the Tier-A+B-only
selection metric), 0.160 vs 0.194 vs 0.235. Seed-0 `variation` failure at lr=3e-5
inflated the 3e-5 std — same single-seed-bad-init pattern that flipped Tier B
1e at 2 seeds. Triggered Stage 1g'.

**Stage 1g' refinement (4 seeds × 3 lrs)**: lr=3e-5 wins clearly (val_ab6 =
0.146 vs 0.157 vs 0.190); the 1g pilot signal dissolved at 4 seeds. Tier A/B
lock confirmed; **a single `lr=3e-5` survives three tiers**. Seed-std at lr=3e-5
slightly wider than at lr=1e-5 (0.012 vs 0.005), reverse of the Tier B 1e'
pattern — a flag for 5d.

**Stage 5d (6-seed confirmation, ex seed 3)**: seed 3 reproduced its Tier B 0c
dead-init on `variation` and was excluded; replacement seeds 6 and 8 were
submitted and completed as healthy runs. Final pool {0, 1, 4, 5, 6, 8},
all 6 healthy.

**Headline results (test, pooled, normalised, 6 seeds × 275 = 1 650 points)**:

| Property | Test MSE ± std | Pooled test R² (95 % CI) | Δ vs Tier B 5c |
|---|---|---|---|
| `lipid_packing`   | 0.0203 ± 0.0014 | 0.976 [0.972, 0.979] | +12 % MSE   |
| `thickness`       | 0.0778 ± 0.0089 | 0.906 [0.895, 0.916] | −1 % (tied) |
| `thickness_std`   | 0.1292 ± 0.0174 | 0.887 [0.867, 0.902] | −4 % (tied) |
| `variation`       | 0.0683 ± 0.0083 | 0.933 [0.927, 0.939] | −6 %         |
| `persistence`     | 0.4092 ± 0.0118 | 0.576 [0.532, 0.616] | 0 % (tied)  |
| `diffusivity`     | 0.0331 ± 0.0020 | 0.960 [0.955, 0.964] | −2 % (tied) |
| `compressibility` | 0.1480 ± 0.0199 | **0.881 [0.860, 0.898]** | (new)   |

**Five Tier C findings for the thesis**:

1. **The Tier A/B lock survives a third tier.** A single
   `lr=3e-5, wd=1e-3, h=128, l=2, e=200` carries from 4 → 6 → 7 properties
   without a single change. The Tier B story (HP search confirmed the optimum)
   replays identically in Tier C: paired t-test 5d vs 0d gives p = 0.348 — not
   significant, expected, because the configurations are identical.

2. **`compressibility` learns substantially better than the receptive-field
   argument predicted.** Pooled test R² ≈ 0.88, far above the «<<0.5» prior.
   Two values worth reporting together:
   - W&B per-seed `val/r2_compressibility` ≈ 0.59 (mean of last-10 epochs)
   - pooled test R² ≈ 0.88 (over 6 seeds × 275 graphs = 1 650 points)
   The val split (~40 graphs/seed) is too small for stable R² estimation on a
   property whose targets span ~3× their mean; the pooled test number is the
   credible estimate. Interpretation: the local 11 Å lipid-packing geometry
   encodes a strong proxy for whole-bilayer area-fluctuation density — local
   packing density ≈ local area-fluctuation density, and that correlation
   extends further than the receptive-field upper bound predicted. The
   architectural argument for EFA-style long-wavelength receptive fields is
   not falsified — but `bending_modulus`, the analogous undulation-spectrum
   target, has been dropped on label-quality grounds, so this question will
   not be tested in this project.

3. **`persistence` is architecture-bound across all three tiers.**
   R² ≈ 0.57 (Tier C) vs 0.58 (Tier B 5c) vs 0.66 (Stage 0c).
   Flat across all lrs in 1e' and 1g'; flat across all training durations.
   The shared MLP trunk + 11 Å spatial cutoff is the binding constraint.
   Capacity-competition with the heterogeneity properties (`variation`,
   `thickness_std`) — same shared-trunk pathology as Tier B. Separate heads
   or uncertainty weighting are the candidate remedies.

4. **The cost of the 7th head is one localised regression** (`lipid_packing`
   test MSE +14 %); the other five Tier B properties are tied within seed
   jitter on the test set. Net wash, with `compressibility` itself learning a
   real signal. The 7-property shared-trunk model is the right trade.

5. **Seed-3 dead-init reproduced for a third time**, matching Tier A's seed 2
   and Tier B 0c's seed 3. ~20 % init-failure rate on `variation` is now
   confirmed across three independent sweeps. Cross-tier scope limit, not a
   Tier C-specific issue.

**Gate check (val_min10 vs Stage 0d 7-prop floor)**: 6/7 pass. `persistence`
0.387 vs 0.370 (+4.6 %) technically fails within seed jitter — a
sample-composition artefact of seed 3's per-property val numbers having pulled
the Stage 0d gate down. `diffusivity` now passes (0.065 vs gate 0.066) with
the 6-seed pool. Not regressions; pre-registered "Tier A+B within ~10 % of 5c"
success criterion is met (max test deviation +12 % on `lipid_packing`).

Full notebook: `scripts/notebooks/analyze_stage_5.py` retargeted for Tier C;
figures and `headline_numbers.json` in `results/figures/stage_5d/`.

---

## 9. Open questions and next phases

- **Tier B Stage 1e (next)** — `lr ∈ {1e-5, 3e-5, 1e-4}` × 2 seeds. Watch `val/loss_persistence` specifically. If `persistence` learns at a lower lr, the 4 → 6 property pivot will replay the Stage 1b lr-saturation discovery.
- **`bending_modulus`** — permanently dropped from the target set. The undulation-spectrum-fit label is too noisy/unreliable to serve as a trustworthy training signal; the 7-property Tier C set (Tier A/B + `compressibility`) is final.
- **Martini 3 lipid simulation pipeline (long-term)** — a general-purpose Martini 3 membrane simulation pipeline, parameterised in lipid types and system parameters, capable in principle of simulating the entire Martini 3 lipidome. Stands as a research deliverable in its own right; newly simulated systems are not necessarily training data. Sequenced subgoals: (1) build the dynamic creation pipeline; (2) fill the existing 10-lipid composition space (in particular the DPPC- and DOPC-rich corners flagged by the Stage 5b MAE concentration — POPC30_DOPC70 worst, ~19 Å thickness MAE); (3) extend the lipid pool beyond the current 10. The Stage 5b train-coverage gap is the most concrete near-term motivation for subgoal (2); if the pipeline is not built, the gap is documented as a Tier A scope limit instead.
- **Seed fragility** — 20 % init-failure rate is acknowledged in the thesis but not yet remedied. Possible levers: warm-up + cosine schedule; init-conditioned learning rate; gradient clipping at the property head.
- **Embedding evaluation, not just property prediction** — the long-term scientific question is the quality of the membrane embedding. Once Tier A/B/C land, the embedding should be probed directly (e.g. clustering, interpretability, transfer to held-out compositions or to protein+membrane systems).
- **Protein+membrane extension design space (2026-05-18)** — working notes in [docs/protein_membrane_embedding_thoughts.md](../../docs/protein_membrane_embedding_thoughts.md). Recommended starting pair WALP + β2AR (one minimal hydrophobic-mismatch probe, one M3-validated GPCR with documented CHOL fingerprint); compositions kept inside the 70-system training corpus so every protein+membrane system has a matched pure-bilayer reference; ~1 GPU-node-day total cost for a 6-system factorial. Phasing: Phase 0 = extend bead vocabulary to M3 protein beads in `lipid_graph.py` / FF JSON / `MembranePropertyGNN`; Phase 1 = WALP/POPC100 inference-only smoke test on the locked Tier C model; Phase 2 = 6-system factorial inference; Phase 3 (optional) = scenario-(B) fine-tune with a local label such as radially-binned thickness or `S(q_min)`.
- **EFA status reopened (2026-05-18)** — [docs/efa_spatial_layer_future.md](../../docs/efa_spatial_layer_future.md) predates Tier C and its motivating-target list is stale: `bending_modulus` is dropped on label-quality grounds, and `compressibility` lands at pooled test R² 0.88 — the receptive-field upper-bound argument was falsified empirically. EFA's strongest motivation in the project is now the **protein extension** (oriented inhomogeneous perturbation source → long-wavelength bilayer response → textbook global-mixing regime). Test order unchanged ((f) deeper MP → (c) readout-only EFA → (b) per-layer parallel), but the acceptance criterion has to be redefined — preferred candidate is an `S(q_min)` architectural-probe label that doubles as a candidate protein-side scenario-(B) target.

---

## 10. Vendored resources

External code or data copied into the repository for reproducibility. Each entry records: what, where, source URL, version, license, and why vendored.

**`insane.py`** — `resources/martini3/insane.py` (GPLv2). Origin: Tsjerk A. Wassenaar's 2014-06-03 build (`previous = "20140603.11.TAW"`) with Helgi I. Ingolfsson lipid-template additions and Emil customisations; sourced from `lipid_gnn/functions_emil/insane.py` in this repo. Converted from Python 2 to Python 3 via `2to3` on 2026-05-07 (mechanical patch only — see `resources/martini3/INSANE_PROVENANCE.md`). Used by `system_builder.py` via `subprocess`. Vendored to lock the specific build that produced the 70 training systems.

When adding entries here, also update `lipid_gnn/martini_pipeline/manifest.py` so the per-system JSON manifest records the vendored version actually used. Plan and progress for the simulation pipeline live in [`docs/martini_pipeline_plan.md`](../../docs/martini_pipeline_plan.md).

Parity check (2026-05-07): rebuilding POPC100 with the vendored Python-3 insane produces 10162 atoms vs 10125 in the legacy build — a difference of +37 solvent atoms from Python 2→3 RNG / dict-iteration differences. Lipid count (392 POPC × 12 = 4704 membrane beads) is identical. Divergence is accepted and documented in `INSANE_PROVENANCE.md`.

---

*Last updated: 2026-05-07, after Step 5 (vendor insane.py) complete; § 10 "Vendored resources" populated.*
