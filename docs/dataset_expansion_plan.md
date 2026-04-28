# Dataset Expansion Plan

Based on the dataset analysis of the 70 Martini 3 membrane compositions, the following hybrid approach is proposed for expanding the simulation dataset. This plan aims to address current model limitations (MAE spikes on extreme mixtures) and test generalization to more realistic biological compositions.

## Phase 1: Patching Binary Gaps (High Priority)

**Goal:** Reduce the GNN's over-reliance on POPC as a "universal solvent" and smooth out the learning curve near pure states.

1. **Dense Sampling at Extremes:** 
   * Simulate high-fraction DPPC and DOPC mixtures (e.g., 85% and 95% fractions).
   * *Reasoning:* This addresses the localized Mean Absolute Error (MAE) spikes observed in Stage 5b near pure states.

2. **Non-POPC Binary Mixtures:** 
   * Simulate combinations of non-POPC lipids (e.g., `DPPC + CHOL`, `DOPC + DOPE`, `POPE + POPS`).
   * *Reasoning:* Forces the model to learn universal interaction embeddings rather than just learning how a single lipid perturbs a POPC membrane.

## Phase 2: Ternary Mixtures (Medium Priority)

**Goal:** Test the zero-shot/few-shot transferability of the GNN's learned embeddings on complex, realistic biological compositions using the existing 10-lipid vocabulary.

1. **Canonical Raft Mixtures:** 
   * Simulate 3-5 ternary mixtures, such as `DOPC / DPPC / CHOL` or `POPC / POPS / CHOL`.
   * *Reasoning:* Successfully predicting properties (like `diffusivity` and `persistence`) for phase-separating ternary mixtures will strongly validate the model's physical relevance and generalization capabilities.

## Future Directions (Low Priority / Exploratory)

* **New Lipid Species:** Introduce Sphingomyelin (e.g., DPSM, POSM) to model eukaryotic plasma membranes, explicitly shorter/longer tail PCs (DLPC, DSPC) to test spatial receptive fields for thickness/compressibility, or highly charged lipids (PIP2) for future protein-membrane modeling.
* **Thermodynamic Titration:** Simulate existing compositions at varied temperatures (e.g., T=298K, 310K, 323K) to introduce temperature as a global graph feature, making the GNN a true thermodynamic predictor.