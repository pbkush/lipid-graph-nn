# Dataset Expansion Plan

Based on the dataset analysis of the 70 Martini 3 membrane compositions, the following hybrid approach is proposed for expanding the simulation dataset. This plan aims to address current model limitations (MAE spikes on extreme mixtures) and test generalization to more realistic biological compositions.

## Coverage Analysis (quantitative basis for prioritisation)

A KDE coverage analysis was run on the existing 70 compositions to convert qualitative gap arguments into ranked candidates (`scripts/notebooks/analyze_dataset.ipynb` §9–10; outputs in `results/dataset_analysis/`).

**Key findings:**

- **All 70 compositions are POPC-binary or pure.** Every binary mixture contains POPC as one component. PE, PS, and CHOL lipids have never been seen without POPC. This is the dominant coverage gap and the root cause of MAE spikes.
- **Stage 5b worst-MAE compositions sit in low-density composition-PCA regions.** POPC30_DOPC70 (rank 15/70), POPC30_DPPC70 (rank 16/70), POPC40_DIPC60 (rank 5/70) are all confirmed below the 25th-percentile density threshold. The error is coverage-driven.
- **The mathematical top-gap is the high-PS space** (DOPS/POPS appearing only at ≤10% in training). This region is compositionally isolated but biologically low-priority.
- **The biologically motivated top-gap (PS ≤ 15 mol% filter)** is the non-POPC PC binary space: DIPC+DOPC, DIPC+DPPC, DOPC+DIPC binaries rank 1–5, followed by three-PC ternaries and CHOL+non-POPC ternaries. This directly confirms the qualitative critique below.

The full ranked list is in `results/dataset_analysis/gap_candidates_bio.csv` (physiological filter applied: total PS ≤ 15 mol%).

---

## Phase 1: Non-POPC Binary Mixtures (Highest Priority)

**Goal:** Break the POPC-centricity and force the model to learn universal lipid interaction embeddings.

All 70 current compositions are POPC + X. Adding POPC-free binaries is the highest-leverage intervention: it directly addresses the structural gap identified by the KDE analysis and is relatively cheap (1 µs per binary, fast equilibration for fluid-phase systems).

**Specific targets** (ranked by KDE gap score, all absent from current dataset):

| Composition | Biological rationale | Martini 3 status |
| --- | --- | --- |
| DPPC + DOPC (40–60%) | Canonical phase-separating PC binary; tail-mismatch without CHOL; prerequisite for raft ternary interpretation | Excellent — among the most validated M3 lipids |
| DIPC + DOPC (45–55%) | Both unsaturated PCs; tests whether the model learned "unsaturation" as a generalizable concept | Excellent |
| DIPC + DPPC (40–60%) | Di-unsat + saturated PC; completes the PC×PC matrix | Excellent |
| DPPC + CHOL (30–50% CHOL) | Lo phase endpoint; canonical condensing-effect test without POPC | Good — M3 CHOL condensing effect well calibrated |
| DOPC + CHOL (20–40% CHOL) | Ld phase endpoint; CHOL outside its current POPC context | Good |

Drop `DOPC + DOPE` from the original plan — both unsaturated, no phase separation, weak scientific signal.

**Note on extreme-fraction sampling (original Phase 1.1):** the KDE analysis shows the Stage 5b MAE spikes at POPC30_DOPC70 are at a *density* failure, not purely a fraction-extreme failure (rank 15/70, not rank 1). Adding non-POPC binaries addresses the root cause more directly. DPPC/DOPC at 85/95% with POPC is useful but lower priority than POPC-free binaries.

---

## Phase 2: Ternary Mixtures (Medium Priority)

**Goal:** Test zero-shot/few-shot transferability of learned embeddings to multi-component systems.

The KDE analysis identified DOPC/DIPC/DPPC ternaries and DIPC/DPPC/CHOL as the next-tier gaps after the non-POPC binaries. Phase 2 should include at least one non-phase-separating and one phase-separating ternary to disentangle "does the GNN generalise to ternary mixing" from "does the GNN detect phase separation."

**Specific targets:**

1. **DOPC / DPPC / CHOL (1:1:1, ~33/33/33%)** — canonical Lo/Ld phase-separating raft. Primary transferability test.  
   ⚠️ **Simulation time**: raft demixing in Martini 3 requires **5–10 µs** minimum (Marrink group benchmarks). The current pipeline uses `[50:667]` frames × 1.5 ns ≈ 925 ns — insufficient for a fully demixed raft. Either run at 10 µs and redefine the frame window, or accept a metastable-state measurement and document it as such in the thesis.

2. **POPC / POPE / CHOL (70/20/10%)** — non-phase-separating ternary; plasma-membrane-like composition. Tests ternary mixing without the phase-separation complication. Standard 1 µs is sufficient.

3. **DIPC / DPPC / CHOL (40/50/10%)** — identified by the KDE analysis as the highest-density-gap ternary within the physiological filter.

Drop `POPC / POPS / CHOL` from the original plan — POPS+POPC+CHOL does not phase separate the same way as PC/PC/CHOL raft systems; it would test different physics (charged-lipid clustering) that is not the thesis focus.

**Single-frame embedding caveat:** The GNN processes one MD frame. For phase-separated ternaries, a single frame shows a *spatially heterogeneous* patch that the model has never trained on. With an 11 Å spatial cutoff, the receptive field cannot span a raft boundary. Treating the non-phase-separating ternary as the primary generalization test and the phase-separating one as an exploratory stress test is recommended.

---

## Phase 3: Filling Structural Gaps in POPC Binaries (Lower Priority)

Where the Stage 5b error analysis specifically calls out sparse regions (DPPC-rich, DOPC-rich at 70% partner):

- POPC20_DPPC80 (equivalent to DPPC at 80%) — currently in val split, not well-represented in train
- POPC20_DOPC80 — same
- If DIPC is included: POPC20_DIPC80

These are straightforward 1 µs binary runs on the existing POPC+X template. Lower effort, lower payoff than Phase 1 — defer until Phase 1 results are in.

---

## Future Directions (Low Priority / Exploratory)

- **Sphingomyelin (DPSM, POSM):** Required for biologically realistic plasma-membrane raft models (PC / PSM / CHOL). Martini 3 params are available (~2022–2024) but raft demixing is slow — budget 5–10 µs. Only worthwhile after Phase 1 DPPC/DOPC/CHOL ternary is established.
- **Tail-length sweep (DLPC, DSPC):** Tests whether the GNN's 11 Å spatial cutoff is the limiting factor for `thickness` and `compressibility`. Controlled axis: add only tail length, keep headgroup (PC) fixed.
- **Temperature titration (T = 298K, 310K, 323K):** Introduces temperature as a global graph feature. ⚠️ Martini 3 coarse-graining smooths gel/fluid transition temperatures and reduces temperature sensitivity relative to experiment — any trained T-dependence would reflect Martini's model, not universal membrane physics. Include a thesis caveat if this is pursued.
- **PIP2:** No consensus Martini 3 parameterisation as of early 2026. Defer until an official release. Do not use community-contributed parameters without explicit validation.
- **Asymmetric leaflets:** Real plasma membranes have PS only on the inner leaflet. Simulable in Martini, but flip-flop rates are unphysically fast. Out of scope for current thesis.

---

## Martini 3 Force-Field Calibration Notes

Relevant to simulation planning and thesis claims:

| Family | Calibration | Caveats |
| --- | --- | --- |
| PC (POPC, DPPC, DOPC, DIPC) | Excellent; APL/thickness/Kₐ match experiment | DPPC Tₘ offset (~298–310 K in M3, broader than experiment); gel state slightly under-ordered |
| CHOL | Major M3 rework (Souza 2021); Lo/Ld separation reproduces | Demixing kinetics slow — µs-scale required |
| PE (POPE, DOPE, DPPE) | Well parameterised; NH₃ H-bond bead | No major caveats |
| PS (POPS, DOPS) | OK with Na⁺/Cl⁻ | Ca²⁺/Mg²⁺ bridging not captured (non-polarisable potential); high-PS compositions are non-physiological anyway |
| Sphingomyelin | Reasonable (~2022–2024 params) | Demixing slow |
| PIP2 | No official M3 release | Out of scope |
