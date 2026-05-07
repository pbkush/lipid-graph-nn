# Persistence Improvement Plan

**Target property**: `persistence` — mean lipid-lipid contact persistence after
`lag=50` frames (~75 ns; dimensionless, POPC100 ref ≈ 0.078).

**Current ceiling**: R² ≈ 0.57–0.66, flat across three tiers (Tier A/B/C),
four learning rates (3e-6 to 1e-4), and 18+ seeds. The floor does not move.

**Baseline HPs inherited from Tier C**:
`hidden_dim=128`, `num_layers=2`, `lr=3.0e-5`, `wd=1.0e-3`, `epochs=200`

---

## Diagnosis

Three causes are active in the experimental record. Each maps to a different
class of intervention.

### Cause 1 — Capacity competition (strongest evidence)

The clearest signal is in the Stage 1e data. Across all runs, seeds that
fail to learn `variation` achieve better `persistence`:

| lr | seed | variation | persistence | note |
|----|------|-----------|-------------|------|
| 3e-5 | 0 | 0.464 (stuck) | **0.324** | best persistence in the sweep |
| 1e-5 | 0 | 0.101 (healthy) | 0.349 | persistence retreats |

When the shared trunk dedicates capacity to `variation` and `thickness_std`
(both heterogeneity properties with competing gradient signals), less
representational capacity flows to `persistence`. This is structural to the
shared-trunk architecture, not a tunable parameter.

**Targeted interventions**: separate property heads (Stage P1),
uncertainty-weighted loss (Stage P2).

### Cause 2 — Depth was never varied with `persistence` active

`num_layers` was optimised during Tier A (properties: `lipid_packing`,
`thickness`, `thickness_std`, `variation`). Tier B and Tier C inherited
`num_layers=2` without re-sweeping it once `persistence` became part of the
loss. `persistence` measures neighbourhood stability over ~75 ns; a 2-layer
GNN with 11 Å cutoff reaches beads within ~22 Å. A 3–4 layer GNN extends
this to ~33–44 Å, potentially covering the relevant patch scale.

**Targeted intervention**: depth ablation (Stage P0) — zero code change,
directly tests the receptive-field hypothesis for `persistence`.

### Cause 3 — Representation ceiling for a dynamical property from a static frame

`persistence` counts how many of a lipid's frame-t neighbours are still
present at frame t+50. This is a *trajectory* property reduced to a
composition-average scalar. The GNN sees only geometry at one frame; it must
infer neighbourhood stability from the static configuration. Mean+max pooling
averages over all beads equally, which may lose the local stability signal
that a handful of "sticky" lipid environments carry.

**Targeted intervention**: attention-based global pooling (Stage P3), which
learns to up-weight the most persistence-informative bead environments.

---

## Stage overview

| Stage | Name | Code change | Key question |
|-------|------|-------------|--------------|
| P0 | Depth ablation | none | Does deeper MP move the persistence floor? |
| P1 | Separate heads | model | Does removing gradient conflict fix capacity competition? |
| P2 | Uncertainty weighting | loss | Does rebalancing the loss further improve P1? |
| P3 | Attention pooling | model | Does learned pooling recover stability signal missed by mean+max? |

Each stage has an explicit go/stop criterion. Stop as soon as `persistence`
test R² exceeds 0.75 (GOOD band entry) or the intervention produces no
improvement over its predecessor.

---

## Stage P0 — Depth ablation

**What it tests**: whether the representational ceiling for `persistence` is
primarily a receptive-field problem (fixable by stacking more MP layers) or a
trunk-sharing problem (requiring Stage P1/P2).

**Grid**: `num_layers ∈ {2, 3, 4}` × `seed ∈ {0, 4, 5}` = 9 runs.
All other HPs locked at Tier C values. `active_properties` unchanged (all 7).

**W&B group**: `stage_p0_depth_ablation`

**Submit**:
```bash
bash scripts/bash/submit_sweep.sh --group stage_p0_depth_ablation \
    --lr "3e-5" --num-layers "2" \
    --seeds "0" --seeds "4" --seeds "5"

bash scripts/bash/submit_sweep.sh --group stage_p0_depth_ablation \
    --lr "3e-5" --num-layers "3" \
    --seeds "0" --seeds "4" --seeds "5"

bash scripts/bash/submit_sweep.sh --group stage_p0_depth_ablation \
    --lr "3e-5" --num-layers "4" \
    --seeds "0" --seeds "4" --seeds "5"
```

**Watch**: `val/loss_persistence` and `val/r2_persistence` per seed.
Compare the mean and std over the 3 seeds at each depth.

**Decision rule**:

| Outcome | Condition | Action |
|---------|-----------|--------|
| Depth helps | `persistence` val R² at `l=3` or `l=4` exceeds `l=2` by > 0.05, with other properties held | Proceed with the winning depth into Stage P1. |
| Depth neutral | Difference < 0.05 or inconsistent across seeds | Depth is not the bottleneck. Proceed to P1 with `num_layers=2`. |
| Depth hurts | Other properties degrade meaningfully (> 15 % MSE) | `persistence` may trade off against the Tier A properties at greater depth. Document; proceed to P1 with `num_layers=2`. |

---

## Stage P1 — Separate property heads

**What it tests**: whether removing the gradient interference between
`persistence` and the heterogeneity properties (`variation`, `thickness_std`)
moves the `persistence` floor.

### Architecture change

Replace the single shared output MLP with independent per-property decoders
applied after the global pool. The GNN message-passing layers and the pool
remain shared; only the final decode is split.

**Current** (`MembranePropertyGNN`):
```
node features
  → GATv2Conv × L (shared, bonded + spatial)
  → GraphNorm
  → mean+max pool → z (global embedding)
  → MLP_shared(z) → [ŷ₁, ŷ₂, …, ŷ_P]
```

**After P1**:
```
node features
  → GATv2Conv × L (shared)
  → GraphNorm
  → mean+max pool → z
  → MLP_1(z) → ŷ₁        (one MLP per active property)
  → MLP_2(z) → ŷ₂
  → …
```

Each per-property MLP has the same depth and width as the current shared MLP
(typically 2 layers, `hidden_dim` → `hidden_dim/2` → 1). Total parameter
count increases by roughly (P−1) × (size of one output MLP), which is
negligible relative to the GNN body.

**Implementation note**: in `lipid_gnn/model.py`, replace the single
`self.output_mlp` `nn.Sequential` with `nn.ModuleList` of P `nn.Sequential`
instances, one per property. The forward pass loops over them and concatenates
outputs. `P = len(active_properties)` is already available at construction
time.

**Grid**: `seed ∈ {0, 4, 5, 6, 8}` = 5 runs (full 5-seed pool from the
healthy Stage 5d seeds). `num_layers` = best from P0 (or 2 if P0 is neutral).
All other HPs at Tier C locked values.

**W&B group**: `stage_p1_sep_heads`

**Submit** (after implementing the arch change):
```bash
bash scripts/bash/submit_sweep.sh --group stage_p1_sep_heads \
    --lr "3e-5" \
    --seeds "0" --seeds "4" --seeds "5" --seeds "6" --seeds "8"
```

**Primary metric**: `val/r2_persistence` (W&B summary, last-10 mean).
**Secondary metrics**: all other `val/r2_*` — confirm no regression on the
six well-learned properties.

**Decision rule**:

| Outcome | Condition | Action |
|---------|-----------|--------|
| Persistence improves | Mean `val_r2_persistence` > Tier C 5d by > 0.05 | Proceed to P2; P1 is the new baseline. |
| Persistence unchanged | Δ < 0.05 | Capacity competition is not the dominant cause; still proceed to P2 (uncertainty weighting may handle the optimisation side). |
| Regression on other properties | Any Tier A/B property MSE increases > 15 % | Check gradient flow; the split may have reduced the shared-trunk's training signal. Document; report as trade-off. |

---

## Stage P2 — Uncertainty-weighted loss

**What it tests**: whether dynamically rebalancing the per-property loss
weights during training further improves `persistence` over Stage P1 (or
over the Tier C baseline if P1 was neutral).

### Loss change

Replace the uniform sum `L = Σ_i MSE_i` with homoscedastic uncertainty
weighting (Kendall & Gal, NeurIPS 2017):

```
L = Σ_i  [1 / (2 σ_i²)] · MSE_i  +  log σ_i
```

where `σ_i > 0` is a learnable scalar per active property. The log term
prevents `σ_i → ∞` (trivial minimisation). At convergence, `σ_i` is large
for easy properties (down-weighted) and small for hard ones (up-weighted) —
the optimiser automatically increases the effective weight on `persistence`
relative to, e.g., `lipid_packing`.

**Implementation note**: in `run_sweep.py`, add `self.log_vars =
nn.Parameter(torch.zeros(P))` to the model (or as a standalone module
alongside the GNN), and change the loss computation from
`F.mse_loss(pred, target)` to the weighted sum above.
Log `sigma_i` values to W&B per epoch as `train/log_sigma_{prop}` so
convergence of the uncertainty parameters can be monitored.

**Grid**: `seed ∈ {0, 4, 5, 6, 8}` = 5 runs. Architecture = best from P1
(separate heads + depth from P0). All other HPs at Tier C locked values.

**W&B group**: `stage_p2_unc_weight`

**Decision rule**:

| Outcome | Condition | Action |
|---------|-----------|--------|
| Persistence improves over P1 | Δ `val_r2_persistence` > 0.03 | P2 is additive; confirm σ_persistence is the smallest (most up-weighted). |
| No improvement | Δ < 0.03 | The capacity competition is already resolved by P1; uncertainty weighting is redundant. Keep P1 architecture, discard log-var parameters. |
| Instability | Loss diverges or σ_i collapses | Add a small L2 penalty on `log_sigma` (or clamp `σ_i` to [0.1, 5]); re-run. |

---

## Stage P3 — Attention-based global pooling

**What it tests**: whether learned pooling, which can up-weight bead
environments that are most informative about neighbourhood stability,
recovers signal that the uniform mean+max pool discards.

### Architecture change

Replace the mean+max concatenation with `GlobalAttention` (Jakovac et al.):

```python
# current
z = torch.cat([global_mean_pool(h, batch), global_max_pool(h, batch)], dim=-1)

# after P3
from torch_geometric.nn import GlobalAttention
gate_nn = nn.Linear(hidden_dim, 1)
attn_pool = GlobalAttention(gate_nn)
z = attn_pool(h, batch)   # (n_graphs, hidden_dim)
```

`z` is now `hidden_dim` rather than `2 × hidden_dim`, so the per-property
heads (from P1) must be re-initialised. Alternatively, keep mean+max alongside
the attention output and concatenate all three (more expressive, higher dim).
Test the plain attention variant first.

**When to run**: only if the best of {P1, P2} leaves `persistence` test R²
below 0.70. If P1+P2 already reaches ≥ 0.70, Stage P3 is optional.

**Grid**: `seed ∈ {0, 4, 5, 6, 8}` = 5 runs. Architecture = best from P1+P2.

**W&B group**: `stage_p3_attn_pool`

**Decision rule**: same pattern as P2 — proceed only if Δ > 0.05 over the
P1+P2 baseline.

---

## EFA spatial layer — deferred

The Euclidean Fast Attention option documented in
[docs/efa_spatial_layer_future.md](efa_spatial_layer_future.md) expects
**low** gain for `persistence` specifically (the expected-gain table ranks
it as a local-scale property, unlike `bending_modulus` and `compressibility`
which are listed as the acid test). Run all of P0–P3 before considering EFA
for persistence. If P0–P3 all fail, the most likely interpretation is that
`persistence` requires trajectory information that a static-frame GNN cannot
reconstruct from any architecture, in which case EFA would not help either.

---

## Acceptance gates

Run all stages against the Tier C 5d baseline (`val/r2_persistence` mean over
5 seeds ≈ 0.63, pooled test R² ≈ 0.58).

| Target | Condition | Label |
|--------|-----------|-------|
| Partial improvement | Test R² ≥ 0.65 | Improvement worth reporting as a thesis finding |
| GOOD band entry | Test R² ≥ 0.75 | Primary success criterion; `persistence` joins the other six properties |
| Architecture floor confirmed | No stage moves test R² by > 0.05 | Document as a static-frame limitation; flag trajectory-aware models as future work |

---

## Non-regression requirement

Every stage must additionally confirm that the six well-learned Tier C
properties (`lipid_packing`, `thickness`, `thickness_std`, `variation`,
`diffusivity`, `compressibility`) do not regress more than **10 %** in test
MSE relative to Stage 5d. The `persistence` improvement is not worth the
trade if it degrades the rest of the model.

---

## Relationship to the thesis narrative

The flat `persistence` R² across three tiers is already documented as an
architecture-floor finding. This plan tests whether that floor is real (a
fundamental static-frame limitation) or artefactual (a capacity competition
or receptive-field problem that can be fixed without changing the data or the
fundamental model family). Either outcome is a thesis contribution:

- **Floor is fixable** (P0–P3 succeed): the shared-trunk multi-task
  architecture was under-serving `persistence`; the fix is separate heads or
  deeper message-passing, not new data or a new architecture family.
- **Floor is real** (P0–P3 all fail): static-frame GNNs cannot recover
  75 ns neighbourhood stability from a single snapshot, even with optimal
  pooling and loss weighting. The experiment proves a representational limit
  and motivates trajectory-aware models as a future direction.
