# Tier C — 7-Property Training Plan

**Properties**: `lipid_packing`, `thickness`, `thickness_std`, `variation`, `persistence`,
`diffusivity`, `compressibility`

**Inherited locked HPs** (from Tier B Stage 5c confirmation):
`hidden_dim=128`, `num_layers=2`, `lr=3.0e-5`, `wd=1.0e-3`, `epochs=200`

---

## Scientific motivation

Tier B confirmed that dynamical properties (diffusivity R²=0.96, persistence R²=0.58) are
learnable from a single-frame snapshot embedding, with `persistence` being architecture-limited
(flat across all lrs). Tier C adds **`compressibility`** — the area compressibility modulus
(units Å³/kT; POPC100 ref ≈ 4.9) — a thermodynamic/mechanical property that couples to
area fluctuations at the whole-bilayer scale.

**Expected difficulty**: `compressibility` is derived from area-fluctuation statistics, which
have a longer correlation length than the 11 Å spatial cutoff of the current GNN. Expect
R² to be architecture-limited (similar to `persistence`), not HP-limited. The question is
whether the single-frame local geometry still carries a usable signal, and whether adding a
noisy long-wavelength target degrades the already-good Tier B properties.

**`bending_modulus` excluded**: fit from undulation-spectrum regression, carries higher label
noise, and is even more strongly long-wavelength than compressibility. It remains deferred
to a future EFA (equivariant fast-attention spatial layer) experiment
([docs/efa_spatial_layer_future.md](efa_spatial_layer_future.md)).

**Primary HP criterion**: `val_total` over the **6 Tier A+B properties only**.
`compressibility` R² is reported and tracked but excluded from the HP selection metric —
it cannot be improved by lr/wd/capacity changes without an architectural change to the
spatial channel.

---

## Config change to activate Tier C

Edit `config.yaml` before any of the stages below:

```yaml
vocab:
  active_properties: [lipid_packing, thickness, thickness_std, variation, persistence, diffusivity, compressibility]
```

No chunk rebuild needed — `compressibility` is column 3 in the preprocessed `y` vector
(`y` stores all 8 properties; `run_sweep.py` slices `prop_cols` at runtime).

Verify the column order:

```bash
python -c "
from lipid_gnn.config import CONFIG
print(CONFIG.vocab.active_properties)
"
```

---

## Stage 0d — GNN floor on 7 properties

Establishes the Tier C baseline and checks for negative transfer from adding
`compressibility` to the shared loss.

**Grid**: `seed ∈ {0, 1, 3, 4, 5}` = 5 runs (same pool as Tiers A and B)
**W&B group**: `stage_0d_tier_c`
**HPs**: locked Tier B config (no changes)

Also run the linear baseline for 7 properties:

```bash
python scripts/training/linear_baseline.py --stratified
```

**Submit command**:

```bash
bash scripts/bash/submit_sweep.sh --group stage_0d_tier_c \
    --lr "3e-5" \
    --seeds "0" --seeds "1" --seeds "3" --seeds "4" --seeds "5"
```

**Decision rule after Stage 0d**:

| Outcome | Condition | Action |
|---------|-----------|--------|
| A — clean floor | `compressibility` shows any learning (val MSE < 1.0), Tier A+B props hold within ~10 % of Stage 5c | Proceed directly to Stage 5d. No HP re-tune needed — compressibility is architecture-bound. |
| B — compressibility stuck | `compressibility` val MSE ≈ 1.0 throughout (no learning signal) | Proceed to Stage 5d regardless. Document as architecture-limited. Do not run Stage 1g. |
| C — negative transfer | Tier A+B properties degrade > 20 % vs Stage 5c at same HPs | Run Stage 1g (lr check with compressibility in loss). Consider uncertainty weighting. |
| D — broad seed fragility | Wide variation in Tier A+B val MSE across seeds, or new properties triggering variation failures | Run Stage 1g-analog seed stability check. |

**Acceptance gates** (Tier B Stage 5c floors — locked):

| Property      | Gate (Tier B floor)  |
|---------------|----------------------|
| lipid_packing | < 0.019              |
| thickness     | < 0.067              |
| thickness_std | < 0.302              |
| variation     | < 0.151              |
| persistence   | < 0.362              |
| diffusivity   | < 0.059              |
| compressibility | *new — floor set by Stage 0d result* |

### Stage 0d Results (2026-05-01) — **OUTCOME C: Negative Transfer**

**5-seed mean val MSE** (W&B group `stage_0d_tier_c`, seeds {0,1,3,4,5}, 200 epochs, `lr=3e-5`):

| Property        | val MSE (5-seed mean) | Stage 5c gate | Δ vs gate  | Status |
|-----------------|----------------------|---------------|-----------|--------|
| lipid_packing   | 0.0236               | 0.019         | +24.2 %   | ❌ FAIL |
| thickness       | 0.0733               | 0.067         | +9.4 %    | ❌ FAIL |
| thickness_std   | 0.3241               | 0.302         | +7.3 %    | ❌ FAIL |
| variation       | 0.1728               | 0.151         | +14.4 %   | ❌ FAIL |
| persistence     | 0.3701               | 0.362         | +2.2 %    | ❌ FAIL |
| diffusivity     | 0.0655               | 0.059         | +11.0 %   | ❌ FAIL |
| compressibility | 0.3931               | 1.0 (new)     | —         | ✅ PASS |

**Per-property R² (last-10 epoch mean, 5-seed mean)**:

| Property        | R²    | Rating |
|-----------------|-------|--------|
| lipid_packing   | 0.932 | GOOD   |
| thickness       | 0.942 | GOOD   |
| thickness_std   | 0.608 | OK     |
| variation       | 0.884 | GOOD   |
| persistence     | 0.654 | OK     |
| diffusivity     | 0.951 | GOOD   |
| compressibility | 0.549 | OK     |

**Summary totals**: val_mean=0.2033 ± 0.0415, test_mean=0.1478, |test−val| gap=0.056

**Key findings**:
- `compressibility` shows a clear learning signal (R²=0.549), **exceeding** the pre-registered
  expectation of R² << 0.5. The 11 Å local geometry carries a non-trivial signal for this
  whole-bilayer property. Hypothesis: local lipid packing geometry is a partial proxy for
  area fluctuation density.
- `lipid_packing` degraded by **24.2 %** vs Stage 5c, exceeding the 20 % threshold → **Outcome C**.
  This is the primary trigger for Stage 1g.
- Remaining Tier A+B properties degraded by 2–15 %, consistent with loss dilution from the
  `compressibility` gradient. Convergence slow-down is the likely mechanism (not instability).
- Seed-to-seed variation is not abnormal (std=0.041 on val_mean); no Outcome D signal.

---

## Stage 1g — lr sanity check (conditional)

Only run if Stage 0d shows outcome C (negative transfer > 20 % on Tier A+B props).
`compressibility` being architecture-limited means lr changes are unlikely to improve it;
Stage 1g's goal is to protect the Tier A+B result, not to rescue compressibility.

**Grid**: `lr ∈ {1e-5, 3e-5, 1e-4}` × `seed ∈ {0, 1}` = 6 runs
**W&B group**: `stage_1g_tier_c_lr`
**HP selection metric**: `val_total` computed over the **6 Tier A+B properties only**
(exclude `compressibility` from the total when comparing lrs)

- If `lr=3e-5` wins → keep lock, proceed to Stage 5d.
- If a different lr wins → run Stage 1g' (half-decade refinement, 4 seeds ×
  3 lrs = 12 runs), following the Stage 1e' pattern.

### Stage 1g Results (2026-05-05) — **OUTCOME: lr=1e-5 wins pilot; triggers Stage 1g'**

**6/6 runs finished** (`stage_1g_tier_c_lr`, `lr ∈ {1e-5, 3e-5, 1e-4}` × seeds {0, 1}):

**Selection metric**: val_mean over **6 Tier A+B properties** (excludes compressibility):

| lr     | val_ab6 (mean) | val_ab6 (std) | val_total7 (mean) | val_total7 (std) |
|--------|---------------|--------------|------------------|-----------------|
| 1e-5 ← | **0.1601**    | 0.0034       | 0.1902           | 0.0064          |
| 3e-5   | 0.1938        | 0.0566       | 0.2235           | 0.0573          |
| 1e-4   | 0.2348        | 0.0073       | 0.2675           | 0.0058          |

**Per-property val MSE (seed-mean, min-last-10)**:

| Property        | lr=1e-5 | lr=3e-5 | lr=1e-4 | Gate   |
|-----------------|---------|---------|---------|--------|
| lipid_packing   | 0.0285  | 0.0251  | 0.0230  | 0.019  |
| thickness       | 0.0815  | 0.0760  | 0.0797  | 0.067  |
| thickness_std   | 0.2962  | 0.3329  | 0.3827  | 0.302  |
| variation       | 0.1010  | 0.2745  | 0.4596  | 0.151  |
| persistence     | 0.3853  | 0.3923  | 0.3947  | 0.362  |
| diffusivity     | 0.0678  | 0.0619  | 0.0691  | 0.059  |
| compressibility | 0.3656  | 0.4014  | 0.4640  | (new)  |

**Per-property R² (seed-mean, last-10 mean)**:

| Property        | lr=1e-5 | lr=3e-5 | lr=1e-4 |
|-----------------|---------|---------|---------|
| lipid_packing   | 0.917   | 0.928   | 0.934   |
| thickness       | 0.936   | 0.940   | 0.937   |
| thickness_std   | 0.638   | 0.598   | 0.538   |
| variation       | 0.932   | 0.816   | 0.692   |
| persistence     | 0.638   | 0.633   | 0.631   |
| diffusivity     | 0.949   | 0.954   | 0.949   |
| compressibility | 0.576   | 0.540   | 0.468   |

**Key findings**:
- **lr=1e-5 wins on val_ab6** (0.160 vs 0.194 vs 0.235) → triggers Stage 1g' per the plan.
- lr=3e-5 seed-0 suffered a **variation failure** (val_var≈0.454; seed-1 has val_var=0.095),
  inflating the lr=3e-5 std (0.057) and mean artificially. This is the same pilot-level
  noise pattern seen in Stage 1e vs 1e': at 2 seeds, a single bad-init distorts the winner.
- lr=1e-4 clearly last: high variation MSE on both seeds (val_var=0.46) and highest
  thickness_std (0.38). Eliminated.
- Gate pass/fail at 2-seed level is not the decision criterion — pilot seed counts are
  too small to assess gate compliance reliably.
- **Stage 1e' prior**: the identical {3e-6, 1e-5, 3e-5} triplet at 4 seeds had lr=3e-5
  winning (0.148 vs 0.153 vs 0.179) with all 4 seeds healthy. Stage 1g' will determine
  whether lr=3e-5 recovers once the single bad-init is diluted by more seeds.

**Decision**: run **Stage 1g'** with triplet `{3e-6, 1e-5, 3e-5}` × seeds {0, 1, 3, 4} = 12 runs.

---

## Stage 1g' — lr refinement (conditional on Stage 1g)

Only run if Stage 1g selects a lr other than `3e-5`.

**Grid**: `lr ∈ {3e-6, 1e-5, 3e-5}` × `seed ∈ {0, 1, 3, 4}` = 12 runs
**W&B group**: `stage_1g_refine_tier_c_lr`
**HP selection metric**: val_mean over **6 Tier A+B properties** (same as Stage 1g)

**Submit command**:

```bash
bash scripts/bash/submit_sweep.sh --group stage_1g_refine_tier_c_lr \
    --lr "3e-6" --lr "1e-5" --lr "3e-5" \
    --seeds "0" --seeds "1" --seeds "3" --seeds "4"
```

- If `lr=3e-5` wins → restore lock, proceed to Stage 5d.
- If `lr=1e-5` wins → update lock to `1e-5` in `config.yaml`, proceed to Stage 5d.
- If `lr=3e-6` wins → unexpected; re-examine per-property breakdown before deciding.

---

## Stage 5d — 5-seed confirmation

Run 5 seeds with 500 epochs at the Tier C locked HP. Produces `test_artifacts.npz` for analysis.

**Grid**: `seed ∈ {0, 1, 3, 4, 5}` = 5 runs
**W&B group**: `stage_5d_tier_c_confirm`

**Submit command** (run after Stage 0d decision — assuming outcome A or B):

```bash
bash scripts/bash/submit_sweep.sh --group stage_5d_tier_c_confirm \
    --lr "3e-5" \
    --seeds "0" --seeds "1" --seeds "3" --seeds "4" --seeds "5"
```

**Gates** (set from Stage 0d 5-seed mean):

| Property        | Gate (< Stage 0d mean) |
|-----------------|------------------------|
| lipid_packing   | < 0.0236               |
| thickness       | < 0.0733               |
| thickness_std   | < 0.3241               |
| variation       | < 0.1728               |
| persistence     | < 0.3701               |
| diffusivity     | < 0.0655               |
| compressibility | < 0.3931               |

**Success criterion for thesis**: Tier A+B properties maintained within ~10 % of Stage 5c
(paired t-test vs Stage 0d on common seeds). `compressibility` R² reported as exploratory;
low R² is the expected and interpretable result (architecture-limited, documents a known
scope limit of the 11 Å spatial cutoff).

---

## Stage chain summary

| Stage | W&B group | Status | What it answers |
|-------|-----------|--------|-----------------|
| 0d — 7-prop GNN floor | `stage_0d_tier_c` | **done** — Outcome C | Baseline + negative-transfer check |
| 1g — lr sanity check | `stage_1g_tier_c_lr` | **done** — lr=1e-5 wins pilot | Does lr=3e-5 still protect Tier A+B? |
| 1g' — lr refinement | `stage_1g_refine_tier_c_lr` | **next** | What is the better lr at 4 seeds? |
| 5d — 5-seed confirmation | `stage_5d_tier_c_confirm` | pending | Final Tier C result |

---

## Reporting

Use `analyze_stage_5.py` with:

```python
GROUP          = "stage_5d_tier_c_confirm"
BASELINE_GROUP = "stage_0d_tier_c"
```

The notebook is parameterised for any number of active properties — all figures extend
naturally to 7 rows. `headline_numbers.json` will include `compressibility`.

For the thesis, present `compressibility` R² as a secondary result alongside the note that
it is architecture-limited (long-wavelength property beyond the 11 Å spatial cutoff).
Contrast with `diffusivity` (R²=0.96, also a non-geometric property but local in scale)
to tell the story about which properties a single-frame GNN can and cannot learn.

---

## Tier C scope limits (pre-registered)

- **Architecture ceiling on `compressibility`**: area fluctuation statistics require
  long-wavelength receptive fields. The 11 Å spatial cutoff samples ~4 nearest neighbours;
  the compressibility modulus integrates area fluctuations at box scale. Pre-registered
  expectation was R² << 0.5, but Stage 0d achieved R²=0.549 — the local geometry carries
  a partial proxy signal for whole-bilayer area fluctuations. This is interpretable
  (local packing density ≈ local area fluctuation density) and does not change the
  architectural conclusion: long-wavelength EFA would still be needed for full accuracy.
  Flag for EFA future work.
- **Loss dilution**: adding a high-noise compressibility gradient to the shared trunk may
  slightly dampen updates on Tier A+B properties. Monitor per-property val curves in Stage 0d
  from epoch 0 — if Tier A+B properties converge slower, that is the dilution signal.
- **`bending_modulus` deferred**: excluded due to higher label noise from undulation-spectrum
  regression. If Tier C results motivate a future architectural extension (EFA spatial layer),
  `bending_modulus` would be the natural 8th property to add at that point.
