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
- **Stratified system-level split** (`prepare_colab_subset.py --split-method stratified`) — replaced random split after discovering the test set's y-range was 4× narrower than train, masking generalisation failures. K-means in z-scored y-space.
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

## 6. Open questions and next phases

- **Tier B (+`persistence`, +`diffusivity`)** — same architecture, locally dynamical properties. Watch for negative transfer through the shared MLP trunk; remedy if needed is homoscedastic uncertainty weighting (Kendall & Gal 2017).
- **Tier C (+`compressibility`, +`bending_modulus`)** — these need long-range receptive fields the 11 Å cutoff cannot provide. Likely floor-bound until the spatial channel is extended (`docs/efa_spatial_layer_future.md` proposes Euclidean Fast Attention as the eventual remedy; deferred until simpler levers exhaust).
- **Train-coverage gap** — Stage 5b per-system MAE concentrates on DPPC- and DOPC-rich mixtures (POPC30_DOPC70 worst, ~19 Å thickness MAE). Augmenting train coverage in the PC1 < 0 region of composition space is the most direct remediation; alternative is to document as a Tier A scope limit.
- **Seed fragility** — 20 % init-failure rate is acknowledged in the thesis but not yet remedied. Possible levers: warm-up + cosine schedule; init-conditioned learning rate; gradient clipping at the property head.
- **Embedding evaluation, not just property prediction** — the long-term scientific question is the quality of the membrane embedding. Once Tier A/B/C land, the embedding should be probed directly (e.g. clustering, interpretability, transfer to held-out compositions or to protein+membrane systems).

---

*Last updated: 2026-04-28, after Stage 5b confirmation.*
