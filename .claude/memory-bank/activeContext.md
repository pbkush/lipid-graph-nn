# Active Context

## Current Work Focus

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
