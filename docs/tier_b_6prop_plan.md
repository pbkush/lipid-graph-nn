# Tier B — 6-Property Training Plan

**Properties**: `lipid_packing`, `thickness`, `thickness_std`, `variation`, `persistence`, `diffusivity`

**Inherited locked HPs** (from Tier A Stage 5b confirmation):
`hidden_dim=128`, `num_layers=2`, `lr=3.0e-5`, `wd=1.0e-3`, `epochs=200`

---

## Scientific motivation

Tier A validated that the GNN embeds local geometric membrane state (packing density,
bilayer thickness, its variability, Voronoi heterogeneity) with R² ≥ 0.87 on all four
properties. Tier B asks whether the same architecture can also capture **dynamical**
properties:

- `persistence` — mean lipid-lipid contact persistence over `lag=50` frames
  (~75 ns; dimensionless, POPC100 ref ≈ 0.078). Probes neighbourhood stability.
- `diffusivity` — mean lateral diffusivity over `lag=10` frames (~15 ns; units Å²,
  POPC100 ref ≈ 757 Å²). Probes lipid mobility.

Both are computed from the *same* single-frame snapshot graph as the Tier A properties
(the GNN sees one MD frame, not a trajectory). The question is whether time-averaged
dynamical properties can be predicted from a static-snapshot embedding — a structurally
interesting hypothesis that distinguishes easy from hard targets.

**Negative-transfer risk**: `persistence` and `diffusivity` have different noise
characteristics and magnitude scales from the geometric properties. Mixing them in the
shared MLP readout trunk may degrade the already-converged Tier A predictions. The
stage chain is designed to detect this early (Stage 0c) before any HP re-tuning.

**Remedy if negative transfer is detected**: homoscedastic uncertainty weighting
(Kendall & Gal 2017) — replace the equal-weight MSE sum with learned per-task
log-variance weights. Implementation in `run_sweep.py` is ~20 lines; defer until
Stage 0c result justifies it.

---

## Config change to activate Tier B

Edit `config.yaml` before any of the stages below:

```yaml
vocab:
  active_properties: [lipid_packing, thickness, thickness_std, variation, persistence, diffusivity]
```

No chunk rebuild needed — all 8 properties are stored in `y` at preprocessing time;
`run_sweep.py` slices the relevant columns via `prop_cols`.

Verify the column order in `run_sweep.py` after the change:

```bash
python -c "
from lipid_gnn.config import CONFIG
print(CONFIG.vocab.active_properties)
"
```

---

## Stage 0c — GNN floor on 6 properties

Establishes the naive GNN baseline on all 6 Tier B properties before any HP tuning.
Required for:
1. The paired t-test in Stage 5c (analogous to Stage 0b → 5b).
2. Checking for negative transfer — do `persistence` and `diffusivity` degrade the
   Tier A properties at the inherited HPs?

**Grid**: `seed ∈ {0, 1, 3, 4, 5}` = 5 runs (same 5-seed pool as Stage 5b; seed 2
and 6 excluded — dead-init and non-deterministic escaper respectively)
**W&B group**: `stage_0c_tier_b`
**Epochs**: 200 (Tier A default)

Also run the linear baseline for 6 properties:

```bash
python scripts/training/linear_baseline.py --stratified
```

The baseline script reads `active_properties` from `config.yaml`, so update the
config first.

**Submit command**:

```bash
bash scripts/bash/submit_sweep.sh --group stage_0c_tier_b \
    --lr "3e-5" \
    --seeds "0" --seeds "1" --seeds "3" --seeds "4" --seeds "5"
```

**Decision rule after Stage 0c**:

| Outcome | Condition | Action |
|---------|-----------|--------|
| A — clean floor | `persistence` and `diffusivity` learn (val MSE clearly below naive floor), Tier A properties hold within ~10 % of Stage 5b | Proceed to Stage 1e (lr check for new properties). Tier A HPs are a good starting point. |
| B — new properties don't learn | `persistence` or `diffusivity` val MSE stuck near normalised 1.0 at 200 epochs | Run Stage 1e regardless — the harder properties may just need a different lr, repeating the Stage 1b → 1b' pattern from Tier A. |
| C — negative transfer | Tier A properties degrade > 20 % vs Stage 5b at same HPs | Implement per-task loss weighting before proceeding (see § Negative-transfer remedy below). |
| D — all properties learn but noisily | Wide seed variance or partial failures on `persistence`/`diffusivity` | Run Stage 1c-analog (seed stability check) before Stage 5c. |

**Acceptance gates for Stage 0c** (val_min_last10 mean, 5 seeds — set the Tier B floor):

Stage 0c finished 2026-04-28 with 5/5 seeds (`{0, 1, 3, 4, 5}`). Decision matrix
outcome: **A — clean floor**. No negative transfer; Tier A properties hold or
improve vs Stage 5b at the inherited locked HPs.

| Property      | Stage 0c floor | Δ vs Stage 5b (4-prop) | Notes                                                                                              |
|---------------|----------------|------------------------|----------------------------------------------------------------------------------------------------|
| lipid_packing | **0.019**      | −14 %                  | Cleanly learnt; R² ≈ 0.94                                                                          |
| thickness     | **0.067**      | −8 %                   | Cleanly learnt; R² ≈ 0.95                                                                          |
| thickness_std | **0.302**      | +1 %                   | Holds within seed jitter; R² ≈ 0.66                                                                |
| variation     | **0.151**      | −0 %                   | Init-fragile recurred (seed 3 stuck ~0.45); 4/5 health                                             |
| persistence   | **0.362**      | new                    | Hard target. Stuck high across all seeds; R² ≈ 0.66. Floor-like, candidate for Stage 1e re-tune.   |
| diffusivity   | **0.059**      | new                    | Cleanly learnt; R² ≈ 0.96. Single-frame embedding *can* predict time-averaged lateral mobility.    |

These are the gates for Stage 5c.

---

## Stage 1e — lr sanity check for 6 properties

Same purpose as Stage 1b in Tier A: verify that the inherited `lr=3e-5` is still
near-optimal when `persistence` and `diffusivity` are added to the loss. The harder
dynamical properties may require a lower lr (as `variation` did when moving from 2
to 4 properties in Stage 1b).

**Grid**: `lr ∈ {1e-5, 3e-5, 1e-4}` × `seed ∈ {0, 1}` = 6 runs
**W&B group**: `stage_1e_tier_b_lr`
**Decision rule**: pick lowest `val_min_last10` total averaged over seeds.

- If `lr=3e-5` wins → lock and proceed to Stage 5c.
- If a different lr wins → run Stage 1e' (refinement, same half-decade spacing as
  Stage 1b' in Tier A).

**Watch**: `val/loss_persistence` and `val/loss_diffusivity` specifically. If either
fails to decrease past ~epoch 20, the property is noise-limited at the current lr;
try the full lr range before concluding it is architecture-limited.

### Stage 1e results (2026-04-28)

6/6 runs finished. W&B group: `stage_1e_tier_b_lr`. **Decision: `lr=1e-5` wins → run Stage 1e'.**

**Per-run val_min_last10 (normalised MSE)**:

| lr | seed | lipid_packing | thickness | thickness_std | variation | persistence | diffusivity | **val_total** |
|----|------|---------------|-----------|---------------|-----------|-------------|-------------|---------------|
| 1e-5 | 0 | 0.030 | 0.085 | 0.349 | 0.101 | 0.349 | 0.071 | 0.164 |
| 1e-5 | 1 | 0.029 | 0.068 | 0.319 | 0.104 | 0.363 | 0.061 | 0.157 |
| 3e-5 | 0 | 0.031 | 0.085 | 0.382 | **0.464** | 0.324 | 0.061 | 0.225 |
| 3e-5 | 1 | 0.018 | 0.059 | 0.272 | 0.076 | 0.342 | 0.054 | 0.137 |
| 1e-4 | 0 | 0.024 | 0.076 | 0.383 | **0.462** | 0.342 | 0.061 | 0.224 |
| 1e-4 | 1 | 0.026 | 0.074 | 0.382 | **0.461** | 0.345 | 0.062 | 0.225 |

**Mean by lr (n=2 seeds)**:

| lr | val_total | lipid_packing | thickness | thickness_std | variation | persistence | diffusivity |
|----|-----------|---------------|-----------|---------------|-----------|-------------|-------------|
| **1e-5** | **0.161** | 0.029 | 0.076 | 0.334 | **0.102** | 0.356 | 0.066 |
| 3e-5 (Tier A lock) | 0.181 | 0.025 | 0.072 | 0.327 | 0.270 | **0.333** | **0.058** |
| 1e-4 | 0.225 | 0.025 | 0.075 | 0.383 | 0.462 | 0.344 | 0.061 |

**Findings**:

1. **lr=1e-5 wins on val_total (0.161)** — driven by `variation` stability: both seeds
   escape the plateau at 1e-5 (0.101, 0.104), recapitulating the Tier A Stage 1b pattern
   ("only lr=1e-5 learned variation"). At 3e-5, seed 0 fails variation (0.464, same
   dead-init plateau); at 1e-4, both seeds fail variation (0.46).

2. **`persistence` is NOT fixed by lr.** Best mean is 0.333 at lr=3e-5 — only ~8 %
   below the Stage 0c floor (0.362). R² stays ≈ 0.67–0.69 across all three lrs.
   `persistence` is architecture/representation-limited, not lr-limited. This is a
   negative result for Stage 1e's secondary goal: the floor of ~0.35 cannot be moved
   by lr alone.

3. **Capacity trade-off confirmed, not anecdotal.** At lr=3e-5, the seed that *fails*
   `variation` (seed 0, val_var=0.464) also has the *best* `persistence` (0.324). The
   lr=1e-4 group shows the same pattern: both seeds fail variation AND have persistence
   ≈ 0.344 (better than the variation-healthy 1e-5 runs at 0.356). This repeats across
   Stages 0c and 1e: whenever the shared trunk gives up on `variation`, capacity frees
   for `persistence`. The hypothesis is real and architecture-level, not a seed anecdote.

4. **lr=1e-5 slightly regresses Tier A properties** vs the 3e-5 lock: `lipid_packing`
   0.029 vs 0.025, `thickness` 0.076 vs 0.072. Within seed jitter; 1e' at 4 seeds will
   give a cleaner signal.

5. **lr=1e-4 hard upper bound**: both seeds fail variation and `thickness_std` worsens.
   1e' grid should be `{3e-6, 1e-5, 3e-5}` only — no point testing above 3e-5.

**Decision**: lr=1e-5 ≠ 3e-5 → **trigger Stage 1e' refinement** and **Stage 1f seed
stability** (lr change mandates fragility re-check per plan).

---

## Stage 1e' — lr refinement (conditional on Stage 1e)

Only run if Stage 1e selects a lr other than `3e-5`.

**Grid**: half-decade triplet centred on the Stage 1e winner × `seed ∈ {0, 1, 3, 4}` = 12 runs
**W&B group**: `stage_1e_refine_tier_b_lr`
**Decision rule**: same as Stage 1b'.

Example: if Stage 1e selects `lr=1e-5`, the refinement grid would be
`{3e-6, 1e-5, 3e-5}`.

### Stage 1e' results (2026-04-29)

12/12 runs finished (`stage_1e_refine_tier_b_lr`, grid `{3e-6, 1e-5, 3e-5}` × `seed ∈ {0, 1, 3, 4}`).
**Decision: `lr=3e-5` wins → keep Tier A lock. Skip Stage 1f, proceed to Stage 5c.**

**Mean by lr (4 seeds, val_min_last10 normalised MSE)** — winner row marked `←`:

| lr            | val_total | lipid_packing | thickness | thickness_std | variation | persistence | diffusivity |
|---------------|-----------|---------------|-----------|---------------|-----------|-------------|-------------|
| 3e-6          | 0.179     | 0.027         | 0.089     | 0.372         | 0.175     | 0.364       | 0.082       |
| 1e-5          | 0.153     | 0.027         | 0.075     | 0.319         | 0.091     | 0.344       | 0.068       |
| 3e-5 (lock) ← | 0.148     | 0.020         | 0.066     | 0.297         | 0.084     | 0.368       | 0.059       |

**Seed std on val_total**: 3e-6 = 0.0102, 1e-5 = 0.0069, **3e-5 = 0.0013** (≈5–8× tighter
than the lower lrs).

**Findings**:

1. **lr=3e-5 wins on val_total (0.148)** and on every property except `persistence`.
   The Stage 1e signal (1e-5 wins) was driven entirely by a single seed-0 variation
   failure at 3e-5; with 4 seeds, **all 4 seeds at 3e-5 escape the variation plateau**
   (val_var ∈ [0.075, 0.099]). The seed-0 failure was a one-seed bad init, not a
   systematic lr=3e-5 problem.

2. **3e-5 is the most stable lr** (seed std ≈ 0.001 on val_total). Lower lrs are
   noisier seed-to-seed; higher lrs (Stage 1e: 1e-4) fail variation outright.

3. **Persistence is flat across all 3 lrs** (0.344–0.368, R² 0.66–0.69). Confirms
   the Stage 1e finding: persistence is architecture-limited, not lr-limited. The
   marginal lr=1e-5 advantage on persistence (0.344 vs 0.368, ≈7 %) does not justify
   sacrificing the clean Tier A wins.

4. **lr=3e-6 is too low**: every property worsens vs 1e-5/3e-5; variation suffers
   most (0.175 mean), and no seed achieves the variation health seen at 3e-5.

**Decision rationale**:

- val_total winner: 3e-5
- Tier A property winner: 3e-5
- Variation health: 4/4 seeds healthy at 3e-5 (vs 3/4 at 1e-5; the Stage 1e seed-0
  3e-5 failure was a single-seed event not replicated in 1e')
- Seed stability: 3e-5 is ≈5× tighter than alternatives
- Only loss: persistence (0.368 vs 0.344) — small, and persistence is architecture-bound

**Action**: Keep `learning_rate: 3.0e-5` in `config.yaml`. **Skip Stage 1f** (only
required if 1e' changed the locked lr). **Proceed to Stage 5c**.

---

## Stage 1f — seed stability check (conditional)

Only run if Stage 0c reveals significant seed fragility on `persistence` or
`diffusivity` (outcome D above), or if Stage 1e' changes the locked lr.

**Status (2026-04-29): SKIPPED.** Stage 1e' confirmed `lr=3e-5` lock; 4/4 seeds
healthy at the locked lr in 1e' provides the seed-stability check implicitly.

**Grid**: locked lr × `seed ∈ {4, 5, 6, 8, 9}` = 5 runs (or a fresh 5-seed pool
if different fragility pattern emerges)
**W&B group**: `stage_1f_tier_b_seed_stability`
**Failure criterion**: `val_min10_{prop} > 0.3` for the new dynamical properties
(analogue of the Tier A `variation` threshold; adjust if the scale differs).

---

## Negative-transfer remedy (if Stage 0c outcome C)

If the Tier A properties degrade > 20 % vs Stage 5b:

1. **Homoscedastic uncertainty weighting** (Kendall & Gal 2017): replace the
   sum of per-property MSEs with a learned-weight sum:
   ```
   L = Σ_i  (1 / (2 σ_i²)) * MSE_i  +  Σ_i  log σ_i
   ```
   where `σ_i` are learnable log-variance parameters, one per property.
   Implementation: add a `nn.Parameter` vector of length `n_props` to
   `MembranePropertyGNN`; modify the loss computation in `run_sweep.py`.

2. **Gradient surgery** (optional, more complex): project conflicting gradients
   to remove inter-task interference. Only consider if weighting alone fails.

3. **Separate heads**: split `persistence`/`diffusivity` into a second MLP head
   with its own parameters while sharing the GNN body. Adds ~1 % parameter overhead.
   Only consider if the shared trunk is confirmed as the bottleneck.

---

## Stage 5c — 5-seed confirmation

Run 5 seeds at the Tier B locked HP. Produces `test_artifacts.npz` for analysis.

**Grid**: `seed ∈ {0, 1, 3, 4, 5}` = 5 runs
**W&B group**: `stage_5c_tier_b_confirm`

**Gate to pass** (normalized val MSE per property, last-10-epoch mean over seeds):
Locked from Stage 0c results (all 5 seeds, including the variation-failure seed 3,
following the Stage 0b convention).

| Property      | Gate (Stage 0c floor) | Notes                                            |
|---------------|-----------------------|--------------------------------------------------|
| lipid_packing | < 0.019               | Should remain near Stage 5b 0.022                |
| thickness     | < 0.067               | Should remain near Stage 5b 0.073                |
| thickness_std | < 0.302               | Holds within seed jitter at locked HPs           |
| variation     | < 0.151               | Pool reflects ~20 % init-failure rate            |
| persistence   | < 0.362               | First Tier B gate; floor-like at locked lr=3e-5  |
| diffusivity   | < 0.059               | First Tier B gate; learnt cleanly at locked HPs  |

**Success criterion for thesis**: all Tier B properties show statistically
significant improvement over Stage 0c (paired t-test p < 0.05 on common seeds)
AND Tier A properties are maintained within ~10 % of Stage 5b.

---

## Stage chain summary

| Stage | W&B group | Condition | What it answers |
|-------|-----------|-----------|-----------------|
| 0c — 6-prop GNN floor | `stage_0c_tier_b` | always | Baseline + negative-transfer check |
| 1e — lr sanity check | `stage_1e_tier_b_lr` | always | Is lr=3e-5 still optimal? |
| 1e' — lr refinement | `stage_1e_refine_tier_b_lr` | if 1e changes lr | What is the better lr? |
| 1f — seed stability | `stage_1f_tier_b_seed_stability` | if fragility found | What is the Tier B init-failure rate? |
| 5c — 5-seed confirmation | `stage_5c_tier_b_confirm` | always | Final Tier B result |

---

## Reporting

Use `analyze_stage_5.ipynb` with:

```python
GROUP          = "stage_5c_tier_b_confirm"
BASELINE_GROUP = "stage_0c_tier_b"
```

The notebook is already parameterised for any number of active properties —
`headline_numbers.json` and all figures will extend naturally to 6 rows.

Update `GROUPS_PROG` in the HP-progression figure to include the full Tier B stage
chain once the runs are complete.

---

## Tier B scope limits (pre-registered)

- The GNN sees a **single static frame** per composition, not a trajectory. Dynamical
  properties (`persistence`, `diffusivity`) are time-averaged over 50/10 frames
  respectively. If the GNN cannot predict them, the most likely explanation is that
  a single-frame snapshot does not contain enough information — an architecture
  limitation, not a HP limitation.
- `diffusivity` scale (POPC100 ref ~757 Å²) is two orders of magnitude larger than
  the dimensionless properties. z-scoring in `run_sweep.py` normalises this, but
  outlier compositions with anomalously high diffusivity could dominate the loss.
  Monitor `val/loss_diffusivity` vs. other properties to detect scale-driven
  gradient imbalance.
- `persistence` ref value (~0.078) is close to zero. z-scoring will amplify noise on
  near-zero compositions; watch for high `thickness_std`/`variation`-style seed
  fragility on this property.
