# Active Context

## Current Work Focus

**Tier A — Stage 5b ready to submit (2026-04-27)**

`config.yaml` finalized for Tier A:
- `active_properties: [lipid_packing, thickness, thickness_std, variation]`
- `learning_rate: 3.0e-5` (locked from Stage 1b'/1c, verified by Stage 2b)
- `weight_decay: 1.0e-3` (verified insensitive in [3e-4, 3e-3] by Stage 2b)
- `epochs: 200` (bumped from 100 per Stage 1d — slow-converger seeds need >100 epochs)
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
| 5b — 5-seed confirmation | `stage_5b_tier_a_confirm` | **next to submit** | Seeds {0,1,3,4,5} at lr=3e-5, 200 epochs |

### Stage 5b submit command

```bash
bash scripts/bash/submit_sweep.sh --group stage_5b_tier_a_confirm \
    --lr "3e-5" \
    --seeds "0" --seeds "1" --seeds "3" --seeds "4" --seeds "5"
```

5 SLURM jobs, ~2.8h each, all parallel. Skips seed 2 (confirmed bad-init). Available healthy pool: {0,1,3,4,5,6,8,9}.

### Key Tier A findings

**Variation-property fragility (init-dependent failure mode)**:
- ~20% of seed inits fail to learn `variation` regardless of lr.
- Two failure subtypes identified:
  - "Slow escapers" (e.g. seed 6, 9): plateau at ~0.5 from epoch 20-50, then break through to ~0.08-0.10. Rescued by 200-epoch training.
  - "True dead-init" (e.g. seed 2): plateau forever. No movement after 200 epochs. Only fix: drop the seed.
- `thickness_std` and `variation` failures are correlated within a seed — same loss-landscape pathology. Suggests one bottleneck, not two independent ones.

**HP saturation finding**: 2-prop Stage 5 (already done, lr=1e-4) showed paired t-test p=0.755 vs Stage 0 baseline — HP search produced NO significant improvement. Tier A's hope is that HP tuning matters more for harder properties; Stage 1b' confirmed this for variation (only learns at lr=3e-5/1e-5, not lr=1e-4).

**GPU memory clarification**: live-tensor peak ~8 GB out of 64 GB (logged via `gpu/peak_mem_actual_gb` in `run_sweep.py`). Earlier "97% peak" was W&B's reserved pool, not actual usage. Tier B/C have huge memory headroom.

**wd is small lever**: Stage 2b (4 runs at wd∈{3e-4, 3e-3} × seeds {0,4}) found val_total flat across the 10× wd range (0.116 vs 0.118). Per-property: higher wd improves variation slightly (val 0.090 → 0.079), hurts thickness_std slightly (0.272 → 0.287). Net wash. Locked wd=1e-3.

### Gates (Stage 0b 4-prop baseline, val_min_last10 mean)
- `lipid_packing < 0.022`, `thickness < 0.074`, `thickness_std < 0.359`, `variation < 0.462`
- Set in [docs/tier_a_4prop_plan.md](../../docs/tier_a_4prop_plan.md) and [scripts/notebooks/analyze_hp_search.ipynb](../../scripts/notebooks/analyze_hp_search.ipynb) Cell 1 `GATES`.

## Latest Changes (this session, 2026-04-27)

- **`config.yaml`**: `epochs: 100 → 200`, `learning_rate: 1.0e-4 → 3.0e-5`. Tier A defaults locked.
- **`docs/tier_a_4prop_plan.md`**: Stage 1b'/1c/1d/2b results recorded; Stage 5b seed selection finalized.
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
- Seed 9: variation broke through at step ~3500 (~55 epochs), val_var settled at ~0.08-0.10 by 200 epochs. Train and val track. **Rescued — keep as healthy seed**.
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
- Run names must include all varying HPs to avoid download collisions.
- Selection metric is MSE (`val_min_last10`); R² is reported alongside as a complementary, more interpretable signal (R²≥0.85 GOOD, ≥0.5 OK, <0.5 WEAK).
