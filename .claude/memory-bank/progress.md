# Progress

## What Works

- **Graph construction pipeline**: `MartiniHeteroGraphBuilder` successfully converts MD trajectories to `HeteroData` with continuous physics features, bonded + spatial edges
- **Chunked dataset loading**: `MartiniDiskDataset` streams data from disk without OOM
- **Model forward pass**: `MembranePropertyGNN` runs over the heterogeneous graph (composition-blind by construction)
- **Force field parsing**: `ff_parser.py` extracts parameters from Martini 3 `.itp` files into JSON maps
- **Training infrastructure**: Local `run_sweep.py` (chunk-based + W&B + AMP, mirrors the Colab notebook), linear baseline, smoke tests, result summarization all functional
- **HP analysis tooling**: `scripts/python/download_wandb_runs.py` pulls W&B groups to `logs/training/`; `scripts/notebooks/analyze_hp_search.ipynb` aggregates over seeds, ranks HP cells, and produces 7 visualizations (loss curves, heatmap, training stats, system metrics).
- **HP search (single-property, 2-prop) complete**: Stages 0–5 done. Winner: `hidden_dim=128, num_layers=2`, `lr=1e-4`, `wd=1e-3`. Val MSE: **0.038** (2-prop, stratified chunks, 5 seeds). Paired t-test p=0.755 — HP search produced no significant gain on the 2-property task.
- **HP search (4-property Tier A) complete**: Stages 0b–5b done. Winner: `hidden_dim=128, num_layers=2`, `lr=3e-5`, `wd=1e-3`, `epochs=200`. Paired t=−31.5, p=3.5e-5 vs Stage 0b — significant ~66 % test-MSE reduction. Per-property R² ≥ 0.87 on all four properties.
- **Stratified system-level split**: `preprocess_graphs.py` defaults to `--split-method stratified` (k-means in y-space). Fixes test-narrowness bug from random split.
- **Stage 5 / 5b analysis pipeline**: `dataset.py` tags graphs with `composition` + `system_idx`; `run_sweep.py` saves `test_artifacts.npz` per run and uploads via `wandb.save()`; `download_wandb_runs.py` fetches it via `run.files()` (basename matching); `linear_baseline.py` has `--stratified` mode. `scripts/notebooks/analyze_stage_5.ipynb` produces 9 publication-grade figures + `headline_numbers.json` with bootstrap CIs and paired t-test. Re-pointed at `stage_5b_tier_a_confirm` for the Tier A run; outputs in `results/figures/stage_5b/`.
- **Per-property test MSE**: `run_sweep.py` logs `test/mse_{prop}` for each active property to W&B summary.
- **SLURM queue-drift fix + per-node GPU packing**: `scripts/bash/submit_sweep.sh` freezes all HP values at submission time as `RUN_<i>_*` env vars (one slot per Cartesian-product cell, including each individual seed). `sbatch_sweep.sh` then backgrounds N parallel `python run_sweep.py` processes on a single node — one per GPU, pinned via `HIP_VISIBLE_DEVICES`/`CUDA_VISIBLE_DEVICES` — translating `RUN_<i>_*` → `FREEZE_*`/`SWEEP_SEEDS` per slot. Default packs up to 8 runs/node; excess runs spill into additional sbatch jobs. New CLI flags: `--partition`, `--time`, `--gpus-per-node`, `--cpus-per-gpu`, `--mem-per-gpu`. `gpu_test` partition has built-in guards (8h max, 2 jobs max). Per-process logs at `logs/sweeps/sweep-<jobid>-gpu<i>.{out,err}`. (Previous: 1 sbatch job = 1 GPU, 7 GPUs idle.)
- **Tier A plan**: `docs/tier_a_4prop_plan.md` — Stage 0b → 1b → 1b' → 1c → 1d → 2b → 5b. All complete.
- **Test suite**: 9 test files; core suite 63 tests (21 in `test_properties.py` after the 2026-05-18 second-pass audit), 508 in `martini_pipeline` — full repo green at 571 passed / 7 skipped.
- **Bug-fixed property pipeline** ([lipid_gnn/properties.py](../../lipid_gnn/properties.py), 2026-05-18): all 8 properties under bug-fixed implementations (`legacy=False` default). Cleanup-plan §2 bugs #1–#12 plus a second-pass audit covering (a) bending-modulus normalisation — returns physical κ in kBT, not the legacy `κ_fit · 1000` "kT/Å³" magnification; (b) FFT axis labelling and grid-step spacing for arbitrary rectangular grids; (c) NaN-padded per-frame thickness series via `frame_mask`; (d) Voronoi CV returns NaN on degenerate input (no silent zero bias). `legacy=True` reproduces every historical bug bit-for-bit for label-regeneration parity. Production labels in `results/properties/prop_legacy_bugfixed_s0/`.
- **Central config**: `config.yaml` + `lipid_gnn/config.py`. All runtime callers read defaults from `CONFIG`.
- **Final-epoch model checkpointing**: `run_sweep.py` saves `model_final.pt` (state_dict + model_kwargs + properties + per-property scaler + epoch + run_id) to `wandb.run.dir` and uploads via `wandb.save()`. `download_wandb_runs.py` pulls it alongside `test_artifacts.npz`. Enables offline reload of any trained model for downstream analysis (e.g. M3 lipidome embedding probe). Only available for runs submitted after 2026-05-16; Stage 5d Tier C weights would need re-running to get the artefact.
- **Legacy vs new M3 paired-comparison notebook** ([scripts/notebooks/compare_legacy_vs_new_m3.py](../../scripts/notebooks/compare_legacy_vs_new_m3.py), 2026-05-17): marimo notebook that pairs the 70 legacy systems against their M3-rerun on `canonical_name`. Six path widgets (legacy/new property-pickle dirs, run roots, ITP dirs); reads pickle-encoded `<COMP>.h5` mean-dicts. Sections: (1) coverage, (2) ITP SHA-1 diff, (3) per-property paired summary + scatter / Bland–Altman / KDE / top-5 movers / Δ-correlation (heatmap + numeric table), (4) EDR observables via `panedr` behind a heavy-step switch (means/stds/drift over last 50 %, plus area-per-lipid panel), (5) composition-space PCA with per-property Δ overlay, (6) mechanical retraining trigger (`|paired_t|>3` or `frac_|d|>sd_legacy>0.5` on any active prop). Empty-state safe (renders skeleton when new-side is absent). Built per the `marimo-data-analysis` skill.
- **Three-way bugfix comparison notebook** ([scripts/notebooks/compare_bugfix_three_way.py](../../scripts/notebooks/compare_bugfix_three_way.py), 2026-05-19): marimo notebook over the new `results/properties/prop_<traj>_<method>_<seed>/` layout. Four label sets (`legacy_bugged_random`, `legacy_bugged_s0`, `legacy_bugfixed_s0`, `m3_bugfixed_s0`); four contrasts (seed / bug-fix / FF / total). Plan at [docs/compare_bugfix_three_way_plan.md](../../docs/compare_bugfix_three_way_plan.md). DIPC↔DLPC rename rules wired via a `mo.ui.text` widget (default `DLPC=DIPC`) that normalises stems and re-canonicalises lipid-token order before pairing. Sections: §0 vocab, §1 paths + rename + coverage, §2 ITP SHA-1 diff + code mtime, §3a seed-only sanity panel, §3b paired summary across four contrasts, §3c 7×4 quadruple-scatter, §3d 7×4 Bland–Altman, §3e four-curve KDE overlay, §3f top-5 movers per (property, contrast), §4 variance decomposition (six terms: 3 var + 3 covar, signed stacked bar + table via `mo.vstack`) + auto-callout naming the dominant component, §5 PCA-Δ overlay with contrast dropdown, §6 model-level comparison across `stage_5d_tier_c_confirm` + three pending Tier C W&B groups (test MSE / pooled R² table, paired t-tests, per-system residual scatter with `largest-mol-fraction / CHOL-mix-override` family bucketing, cross-model prediction-vs-residual spread), §7 four-branch verdict callout. Empty-state safe — each contrast renders independently as its two label sets land.

## Tier A Status (4 properties: lipid_packing, thickness, thickness_std, variation)

| Stage | Status | Key result |
| --- | --- | --- |
| Stage 0b — 4-prop GNN baseline | done | val_min10: lp=0.022, th=0.074, th_std=0.359, var=0.462 |
| Stage 1b — lr sweep {1e-5,1e-4,5e-4} × 2 seeds | done | lr=1e-5 wins; variation only learns at 1e-5 |
| Stage 1b' — lr refinement {3e-6,1e-5,3e-5} × 4 seeds | done | lr=3e-5 wins (val_total 0.149); seed-2 variation failure exposed |
| Stage 1c — seed stability at lr=3e-5 | done | 1/5 fail (seed 9); 22% combined failure rate; seed-6 late-escape pattern |
| Stage 1d — long-training rescue at 200 ep | done | seed 9 rescued (val_var ~0.10); seed 2 stuck (~0.53) — drop permanently |
| Stage 2b — wd verification at lr=3e-5 | done | wd insensitive in [3e-4, 1e-3, 3e-3] (val_total 0.116-0.118) |
| Stage 5b — 5-seed confirmation | **done** | 6/7 seeds healthy; paired t=−31.5, p=3.5e-5 vs Stage 0b; per-prop R² ≥ 0.87; GNN beats Ridge by 56–84 % |

**Locked HPs (final, in config.yaml)**: `hidden_dim=128`, `num_layers=2`, `lr=3.0e-5`, `wd=1.0e-3`, `epochs=200`.

## Stage 5b headline numbers (test, pooled, normalised)

| Property | MSE mean ± std | R² (95 % CI) | Gate (val) |
| --- | --- | --- | --- |
| `lipid_packing` | 0.020 ± 0.003 | 0.975 [0.972, 0.978] | 0.0222 vs 0.022 — tied |
| `thickness` | 0.076 ± 0.007 | 0.908 [0.898, 0.917] | 0.0732 vs 0.074 — pass |
| `thickness_std` | 0.145 ± 0.024 | 0.873 [0.856, 0.888] | 0.299 vs 0.359 — pass (+17 %) |
| `variation` | 0.131 ± 0.171 | 0.872 [0.856, 0.887] | 0.151 vs 0.462 — pass (+67 %) |

The wide MSE std on `variation` is driven by seed 6 (failed to escape; widens std from ~0.02 to 0.171). For thesis numbers, prefer the planned 5-seed pool {0,1,3,4,5}.

Per-system errors concentrate on DPPC- and DOPC-rich mixtures (POPC30_DOPC70 worst, ~19 Å thickness MAE) — these sit at the boundary of the test cloud in PCA(composition) space where train density also drops. Documented as a Tier A scope limit.

Full report: [results/figures/stage_5b/stage_5b_analysis_report.md](../../results/figures/stage_5b/stage_5b_analysis_report.md).

## Variation seed-fragility — two failure modes

- *Slow escapers* (seeds 6, 9): variation plateau at ~0.5 from epoch 20–50, then breakthrough to ~0.08–0.10. Rescued by 200-epoch training. **Escape is non-deterministic** — seed 6 escaped in 1c, failed in 5b.
- *True dead-init* (seed 2): plateau forever, no movement after 200 epochs. Drop permanently.
- `thickness_std` and `variation` failures are correlated within a seed → single loss-landscape pathology, not two independent ones.
- ~20 % population failure rate (2/9 seeds across Stages 1b' + 1c, plus the recurring seed-6 jitter in 5b). Documented as a Tier A limitation for the thesis.

## Headline thesis story

**HP saturation finding**: 2-prop Stage 5 (lr=1e-4) had paired t-test p=0.755 — HP search produced no significant gain. Tier A reverses this: paired t=−31.5, p=3.5e-5. **HP tuning matters more for harder properties** is the main thesis story; lr was the dominant lever (variation only learns at lr=3e-5/1e-5, not lr=1e-4).

**GATES** (Stage 0b 4-prop baseline, 5-seed val_min10 mean — in plan doc and notebook):
`lipid_packing < 0.022`, `thickness < 0.074`, `thickness_std < 0.359`, `variation < 0.462`

**wd is small lever**: Stage 2b confirmed wd=1e-3 is roughly optimal in [3e-4, 3e-3] range. Per-property tradeoff exists, but val_total flat. Locked wd=1e-3.

**GPU memory clarification**: earlier "97 % peak = OOM danger" was a misread of W&B's `memoryAllocated` (reserved pool, not live tensors). Real proxy `torch.cuda.max_memory_allocated()` added to `run_sweep.py` as `gpu/peak_mem_actual_gb` (per-epoch reset). Live peak ~8 GB out of 64 GB. Tier B/C have huge memory headroom.

**Run-name schema (collision-proof)**: schema is `h{h}_l{l}_lr{lr:.0e}_wd{wd:.0e}_e{e}_s{seed}_{run_id}`. W&B `run.id` suffix (8 chars, globally unique) guarantees no collision even for same-(HPs, seed) retries. `download_wandb_runs.py` writes `.wandb_run_id` marker file and raises `RuntimeError` on mismatch (defence-in-depth). Preserve trailing `_{run.id}` in all future stages. (Historical runs through 2026-05 carried an extra `gnn_only_` prefix; the prefix was dropped together with the composition-mode toggle.)

**R² added to analyze_hp_search.ipynb**: complementary reporting metric (selection still MSE-driven). 4 cells modified: `cell-load-fn`, `cell-detect-hps`, `cell-aggregate`, `cell-ranking-table`, `cell-recommendation`. R² uses `_tail_mean()` (not `_tail_min()`) to avoid amplifying favourable noise spikes on the small val set.

## Tier B Status (6 properties: + persistence, + diffusivity)

| Stage | Status | Key result |
| --- | --- | --- |
| Stage 0c — 6-prop GNN floor at locked Tier A HPs | done | val_min10: lp=0.019, th=0.067, th_std=0.302, var=0.151, persistence=0.362, diffusivity=0.059. No negative transfer; `diffusivity` learns cleanly (R² ≈ 0.96), `persistence` floor-like (R² ≈ 0.66). 4/5 seeds healthy (seed 3 stuck on `variation`). Decision matrix outcome **A** with caveat: `persistence` may need lr re-tune. |
| Stage 1e — lr sanity check (2 seeds) | done | Initial signal: lr=1e-5 wins (val_total 0.161); seed-0 variation failure at 3e-5 inflated 3e-5 mean. Triggered 1e'. |
| Stage 1e' — lr refinement (4 seeds × 3 lrs) | done | **lr=3e-5 wins (val_total 0.148)** — Tier A lock confirmed; signal from 1e was a single-seed bad-init artefact. 4/4 seeds healthy at 3e-5, ≈5× tighter seed-std than alternatives. Persistence floor (~0.35) flat across all lrs — architecture-limited. |
| Stage 1f — seed stability | skipped | Not triggered; 1e' kept the lock and 4/4 seeds at 3e-5 are implicit stability evidence. |
| Stage 5c — 5-seed confirmation | **done** | R² ≥ 0.88 on 5/6 props; persistence R²=0.578 (architecture floor); t-test p=0.286 vs Stage 0c (expected — same HPs). |

**Tier B GATES (Stage 0c 6-prop floor, 5-seed val_min10 mean — locked in `tier_b_6prop_plan.md` and `analyze_hp_search.ipynb` Cell 1)**:
`lipid_packing < 0.019`, `thickness < 0.067`, `thickness_std < 0.302`, `variation < 0.151`, `persistence < 0.362`, `diffusivity < 0.059`.

## Tier C Status (7 properties: + compressibility)

| Stage | Status | Key result |
| --- | --- | --- |
| Stage 0d — 7-prop GNN floor at locked Tier B HPs | done | Outcome C: all 6 Tier A+B props miss 5c gates by 2–24%; compressibility R²=0.55 (above pre-registered "<<0.5"). Triggered Stage 1g. |
| Stage 1g — lr sanity (2 seeds × 3 lrs) | done | Pilot signal: lr=1e-5 wins on val_ab6 (0.160 vs 0.194 vs 0.235); seed-0 3e-5 variation failure inflated 3e-5 mean. Triggered Stage 1g'. |
| Stage 1g' — lr refinement (4 seeds × 3 lrs) | done | **lr=3e-5 wins (val_ab6=0.146)** — Tier A/B lock confirmed, 1g signal flips at 4 seeds. Same pattern as Tier B 1e → 1e'. |
| Stage 5d — 6-seed confirmation | **done** | 6/7 Stage 0d val gates pass (`persistence` fails by +4.6 % within seed jitter; `diffusivity` now passes); 5/6 Tier B 5c test-MSE numbers tied; `lipid_packing` test +12 %; **compressibility pooled test R² = 0.88** (per-seed val R² ≈ 0.59 — small-val-split artefact). Seeds {0,1,4,5,6,8} all healthy. |

**Tier C locked HPs**: identical to Tier A/B (`hidden_dim=128, num_layers=2, lr=3e-5, wd=1e-3, epochs=200`). Single lr lock survives all three tiers.

## Stage 5d headline numbers (6 seeds {0,1,4,5,6,8}, test, pooled, normalised)

| Property | test MSE mean ± std | Pooled test R² (95 % CI) | Per-seed val R² | vs Tier B 5c test MSE |
| --- | --- | --- | --- | --- |
| `lipid_packing` | 0.0203 ± 0.0014 | 0.976 [0.972, 0.979] | 0.93 | +12 % |
| `thickness` | 0.0778 ± 0.0089 | 0.906 [0.895, 0.916] | 0.94 | tied (−1 %) |
| `thickness_std` | 0.1292 ± 0.0174 | 0.887 [0.867, 0.902] | 0.65 | tied (−4 %) |
| `variation` | 0.0683 ± 0.0083 | 0.933 [0.927, 0.939] | 0.94 | tied (−6 %) |
| `persistence` | 0.4092 ± 0.0118 | 0.576 [0.532, 0.616] | 0.63 | tied (+0 %) |
| `diffusivity` | 0.0331 ± 0.0020 | 0.960 [0.955, 0.964] | 0.95 | tied (−2 %) |
| `compressibility` | 0.1480 ± 0.0199 | **0.881 [0.860, 0.898]** | 0.59 | new (above pre-reg) |

Pooled test R² is computed across 6 seeds × 275 graphs = 1 650 points and is the more stable estimate. Per-seed val R² is the W&B summary value (last-10-epoch mean over the small ~40-graph val split per seed); the gap on `compressibility` (pooled 0.88 vs val 0.59) reflects val-split-size variance, not a generalisation failure — the per-graph % errors in figure (j) cluster near `diffusivity`'s width, which is consistent with the pooled R²=0.88. Both numbers should be reported in the thesis.

**Gate check vs Stage 0d val_min10 means**: 6/7 pass. `persistence` (0.387 vs 0.370, +4.6 %) technically fails within seed jitter — a sample-composition artefact of seed 3 having lucky per-property val numbers in Stage 0d. Not a regression; the pre-registered "Tier A+B within ~10 % of 5c" success criterion is met (max deviation +12 % on `lipid_packing` test MSE; everything else within 7 %). `diffusivity` now passes (0.065 vs gate 0.066).

**Paired t-test 5d vs 0d**: t = −0.43, p = 0.348 — not significant, **as expected** (same HPs, same epochs; uses 4 common seeds {0,1,4,5} since Stage 0d does not have seeds 6 or 8; the substantive Tier C contrast is per-property vs Tier B 5c, not aggregate vs 0d).

Seed 3 was excluded — recurring dead-init on `variation` (same as Tier A's seed 2 and Tier B 0c's seed 3). Replacement seeds 6 and 8 both completed as healthy runs. Documented as a cross-tier seed-fragility limitation. Adding compressibility to the shared head costs ~12 % on `lipid_packing` test MSE and tied/improves on the other five Tier B properties — net wash, with `compressibility` itself learning a stronger-than-prior signal.

## Martini Pipeline Status (simulation deliverable)

| Step | Status | Notes |
| --- | --- | --- |
| 1–9 — pipeline build, MDP audit, sanity checks | done | DIPC100 smoke + POPC100 sanity passed earlier |
| 10a — GPU production routing (`gpu`/`gpu_test`) | done | `sbatch_simulations.sh`; 8 sims/node, HIP pinning |
| 10 — GPU benchmark sweep | done (2026-05-16) | Winner `8_sim_8gpu_cpu4` (41,909 ns/day agg, ~5,240/slot). `hpc_defaults`: sims=8, gpus=8, cpus_per_sim=4, mem=16G. cpus_per_sim=8 is 15 % slower (not CPU-thread limited); 16 sims/8 GPUs gives higher aggregate but −36 % per slot. **Recommended 1 µs walltime: 8 h.** |
| 10c — general1 CPU production routing | done | `sbatch_simulations_general1.sh` + `_gmx_mpi_wrapper.sh`; spack openmpi/GROMACS-2022; calibrated `hpc_defaults_cpu` (sims=2, ranks=1, cpus=20, mem=16G; ~13 200 ns/day/node) |
| 11a — subgoal 3a popc_interpolation, 1 µs | **done (CHOL-free)** (2026-05-19) | CHOL-free cells finished; CHOL-containing cells still missing |
| 11b — subgoal 3b DPPC/DOPC corners | **done (CHOL-free)** (2026-05-19) | DPPC and DOPC corners finished for non-CHOL partners; CHOL partner column still missing |
| 11d — CHOL-containing cells across 3a/3b grids | **unblocked** (2026-05-19) | Blocker was insane KeyError on `M3.CHOL` (see CHOL fix part 2 below); `DOPC60_CHOL40` smoke build on general1 now succeeds. CHOL partner of `popc_interpolation` + `dppc_corner` + `dopc_corner` (capped 40 % per Decision 35) still to run at full 1 µs — resubmit with the same three `--missing-from-grid` calls, the CSV gate skips done work. |
| 11e — resimulate legacy 70-system corpus with M3 ITPs | **pending** | `submit_simulations.sh --from-csv resources/simulation_tables/done.csv --rename-lipid DIPC=DLPC`; standardises every system to one ITP set |
| 12 — extend lipid pool beyond current 10 | future | Unlocks 3c/3d |

**Pipeline tooling**:

- `scan_completed_systems.py` — CSV scanner; canonicalises legacy non-canonical dir names; resolves `sim_ns` via three-tier lookup (`actual` from `Statistics over N steps` line, then `requested_manifest`, then `requested_log`); flags `--min-ns`, `--require-actual`, `--merge-with` (order-preserving union for diff-friendly updates).
- `submit_simulations.sh`: `--completed-csv` (skip already-done), `--from-csv` (use CSV as work list, e.g. for resimulating legacy data with new ITPs), `--pin {on,off,auto}` (gmx mdrun thread pinning).
- `projected_finish.py` — mid-run ETA from `Writing checkpoint` lines (GROMACS only writes final `Performance:` after success). Recurses with `rglob` so it works on parent dirs.
- `analyze_benchmark.py --cpu` for device-aware recommendation + `hpc_defaults_cpu` YAML emit; `pin` column carried through to recommended YAML.
- Partition dispatch + QOS caps (general1=40 jobs, gpu_test=2 jobs) in `submit_simulations.sh`.
- Env propagation refactored to env-file-via-positional-arg (SLURM `--export` silently drops entries on Goethe-HLR).
- **CHOL fix (2026-05-17)**: registry `insane_keyword="M3.CHOL"` instead of `"CHOL"` (insane's default is the legacy 8-bead M2 topology; M3 ITP has 9 beads → grompp atom-count mismatch on every CHOL-containing system).
- **CHOL fix part 2 — insane source pin (2026-05-19)**: the PyPI `insane-1.2.0` wheel is a stale `1.1-dev` snapshot (in-code `__version__='1.1-dev'`, no `M3.CHOL` in `lipids.dat`, no `-dat` option) — so the 05-17 registry change landed in the source tree but the running insane never knew about `M3.CHOL` and threw `KeyError: 'M3.CHOL'` on every CHOL system. `requirements.txt` now installs `insane @ git+https://github.com/Tsjerk/Insane.git@11de0c7f1abac2f5296ce1ab74285f904afc2125`, which is the upstream commit that actually ships those features. `system_builder.py::build_command` also unconditionally passes `-dat lipid_gnn/martini_pipeline/templates/insane_extra_lipids.dat` (a bundled `[ sterols ]` block carrying `M3.CHOL`) as a belt-and-suspenders guard, and `_parse_molecule_counts` strips the `; Defined in '...'` inline comment insane appends to `[ molecules ]` lines for `-dat`-loaded lipids (otherwise `n_membrane_beads` came out as 0). `build_system` error path now includes the last 1000 chars of stdout in the `RuntimeError` — insane prints lipid-lookup errors there, not on stderr. **Reinstall recipe on existing envs**: `pip install --force-reinstall --no-deps "insane @ git+https://github.com/Tsjerk/Insane.git@<commit>"`; `--upgrade` alone won't refetch a git-pinned spec.
- **DIPC → DLPC migration (2026-05-17)**: DLPC added as parallel registry entry (same physics as DIPC; modern M3 token). `submit_simulations.sh --rename-lipid DIPC=DLPC` (repeatable, `OLD=NEW`) rewrites composition tokens in-flight; recanonicalises so alpha-tiebreak order stays right. Pair with `--from-csv resources/done.csv` to migrate legacy 70-system corpus to `DLPC*` output dirs. LIPID_TYPES + existing training data unchanged.

## What's Left to Build

- **Tier C complete**: Stage 5d confirmed on 6 seeds {0,1,4,5,6,8}. Report at `docs/stage_5d_analysis_report.md`.
- **`bending_modulus` (8th property)**: dropped permanently. Label is too noisy/unreliable (undulation-spectrum fit) to serve as a trustworthy training signal. The 7-property Tier C set is the final target set.
- **Martini 3 lipid simulation pipeline (long-term)**: build a general-purpose Martini 3 membrane simulation pipeline — not solely for training data. Newly simulated systems may or may not be used as training data; the pipeline is a separate research deliverable. Subtasks (rough order):
  1. **Dynamic membrane creation pipeline** — parameterised in number of lipid types, per-lipid mol fractions, and other system parameters (box size, temperature, ions, water level, simulation length). Should compose with the existing `data/membrane_only/` layout so downstream graph-construction code keeps working unchanged.
  2. **Pipeline goal — entire Martini 3 lipidome**: the pipeline should be capable, in principle, of simulating any lipid in the Martini 3 force field, not just the current 10-lipid pool.
  3. **Early subgoal — fill composition coverage with the current 10 lipids**: simulate the missing/sparse regions of the 10-lipid composition space (in particular the DPPC- and DOPC-rich corners flagged by Stage 5b per-system MAE concentration). Doubles as a pipeline shake-out and as candidate training-coverage augmentation.
  4. **Later — extend the lipid pool**: introduce additional Martini 3 lipids beyond the current 10, expanding the composition space (and the `LIPID_TYPES` vocabulary) for future training rounds.
- **Embedding evaluation, not just property prediction**: the long-term scientific question is the quality of the membrane embedding. Once Tier A/B/C land, probe the embedding directly (clustering, interpretability, transfer to held-out compositions or to protein+membrane systems).
- Explore transfer to protein+membrane systems (long-term research goal).

## Current Status

### Phase: Tier C complete (Stage 5d done, 6 seeds); Tier A, B, C all complete

`config.yaml` `active_properties` is set to 7-property Tier C. Stage 5d confirmed on seeds {0,1,4,5,6,8} — all 6 healthy. Seed 3 excluded (recurring dead-init on `variation`); replacement seeds 6 and 8 completed successfully.

## Known Issues

1. **LIPID_TYPES consistency**: The 10-element lipid list must be identical across `lipid_graph.py`, `linear_baseline.py`, and `run_sweep.py` — maintained manually.
2. **Per-property test MSE missing in Stage 0b runs**: `test/mse_{prop}` logging was added after Stage 0b ran; only `test/mse_total` is in those summaries. Val-only analysis for Stage 0b.
3. **Seed-6 jitter in Stage 5b**: seed 6 escaped `variation` in Stage 1c but failed in 5b at the same config. Escape is non-deterministic per seed — running the same seed twice can produce different outcomes. For thesis reporting, prefer the planned 5-seed pool {0,1,3,4,5}.

## Deferred Ideas (Not Active Tasks)

- **Euclidean Fast Attention (EFA) block on the spatial channel.** Linear-cost, SE(3)-equivariant attention (Frank et al., Nat. Mach. Intell. 2026). Reconsider only after all 8 targets are implemented and simpler levers are exhausted. Full plan at [docs/efa_spatial_layer_future.md](../../docs/efa_spatial_layer_future.md).

## Evolution of Project Decisions

1. **Integer vocab → continuous physics features**: Switched to continuous `[mass, charge, sigma, epsilon]` from Martini 3 FF.
2. **Single graph type → heterogeneous graph**: Moved to `HeteroData` with bonded and spatial edge types.
3. **Full in-memory loading → chunked disk streaming**: Added `MartiniDiskDataset`.
4. **Random split → stratified split**: Fixed test-narrowness bug (test std 4× narrower than train).
5. **Live config at execution → frozen env vars at submission**: `submit_sweep.sh` + `_apply_submission_overrides()` to prevent queue-drift corruption.
6. **2-prop lr=1e-4 → 4-prop lr=3e-5**: `variation` property only learns at lower lr; grid spacing too coarse — refinement sweep needed (Stage 1b').
7. **100 → 200 epochs (Tier A default)**: Stage 1d found slow-escaper seeds need >100 epochs to break through `variation` plateau.
