# Representation-learning objective for lipid-graph-nn — second opinion

## Context

The thesis north star is a **transferable CG-membrane embedding** that eventually extends to protein+membrane. Property regression on two scalars is a training signal, not the goal. A preliminary analysis recommended **frame-pair VICReg/BYOL + fragment-masked bead-type prediction** on the current GATv2 backbone, with equivariant denoising as an upgrade path. This plan pushes back where warranted, raises options not yet surfaced, and commits to a concrete plan that runs **in parallel to** the current property-regression track without interfering with it.

## Current codebase — what constrains the choice

Grounded in reading the actual code, not the preliminary analysis:

- `MembranePropertyGNN.forward` ([membrane_prop_gnn.py](../colab_lipid_gnn_subset/lipid_gnn/membrane_prop_gnn.py)): `x_dict, edge_index_dict, batch_dict, edge_attr_dict, comp_vec` → pooled `graph_repr` (line 94) → MLP head. Per-bead embeddings `bead_h` are accessible at line 85, so node-level heads are cheap.
- Positions are **not** stored on graphs; the chunked `.pt` files carry `data['bead'].x` [N,4], bonded `edge_attr` [B,2], spatial `edge_attr` [S,16] (RBF only — **raw distance is discarded**), `comp_vec` [10], `y` [1,num_props]. No system id, no frame index.
- `.xtc` and `.tpr` for every system are on disk, frames currently subsampled at `linspace(num_frames=25)`. Pair/triplet frame sampling is feasible but requires reprocessing.
- **No forces on disk** (`nstfout=0` in production `.mdp`). No SSL, augmentation, or masking code of any kind exists yet.
- **Scope**: focus is the `gnn_only` model (`comp_dim=0`). `comp_vec` is carried on the graph but not consumed. Keeping the model blind to composition at every stage is a hard requirement for transfer to unseen lipid types and to protein+membrane, and it is what every recommendation below assumes.

## Critique of the preliminary recommendation

**Frame-pair contrastive (VICReg/BYOL) is weaker than it looks for this data.**

1. *Composition-identity shortcut, even without `comp_vec`.* In the gnn_only setting `comp_vec` is already ignored, but composition is still near-fully recoverable from the node-feature multiset: with 10 lipid types and ~20 Martini bead types, a mean-pooled histogram over `[mass, charge, σ, ε]` is essentially a composition classifier. Frame pairs from the same trajectory share that statistic exactly; frames from a different composition differ by a nearly-deterministic fingerprint. An invariant encoder can solve frame-pair alignment by behaving as a composition classifier and ignoring the structural signal entirely — encoding *which membrane* rather than *what the membrane is doing*, which is the opposite of what transfer to unseen lipid types and protein+membrane requires. There is no single feature to gate; the fix is to pick a different objective.
2. *~70 compositions is a small and non-uniform "class" space.* Many compositions are near-neighbors on a simplex (POPC90/DOPC10 vs POPC85/DOPC15). Negatives in-batch are often near-positives in physics. VICReg's variance/covariance prevents collapse but does not fix label noise in the contrast signal. What you end up embedding is "which of 70 clusters" — a distance that is not the distance you want.
3. *What the equilibrium contrast even learns.* Two frames from the same trajectory separated by ≥ τ_autocorr are *independent draws from the equilibrium distribution.* Alignment therefore targets the invariants of that distribution — composition, phase, leaflet symmetry. Those are exactly the features that *don't* transfer to protein+membrane (proteins break all of them). Frame-pair contrastive optimizes the wrong invariants for the north star.
4. *Autocorrelation-time assumption.* Martini 3 lateral-diffusion autocorrelations are long (μs in real time); across 25 linspace-sampled frames of a single run the pairs will either all be correlated (short run) or all independent (long run). There is no tunable τ — you get one regime.

**Masked bead-type prediction is near-trivial.** With continuous `[mass, charge, σ, ε]` inputs and bonded topology visible, the masked bead's type is recoverable from its neighbors' types in a Martini bilayer. Fragment-masking delays but does not fix this — the target cardinality is ~20 bead types, and local chemistry is the signal. Low representational pressure.

**Invariant denoising via RBF reconstruction discards angular physics.** Martini 3 bilayer thermodynamics (tail tilt, chain packing, bending) are dihedral- and angle-driven. Reconstructing pairwise-distance RBFs is blind to all of that — the output is a mostly-scalar reconstruction of a scalar input, exercising the attention machinery on the wrong signal. This is a weaker pretext than the denoising framing suggests.

## Objectives not yet discussed that are genuinely better

A. **Force regression with per-bead local-frame projection.** The existing production runs did not save forces (`nstfout=0`), so forces must be **regenerated via `mdrun -rerun`** against the existing `.xtc` frames with a new `.mdp` that sets `nstfout=nstxout`. `-rerun` does not integrate dynamics; it recomputes energies/forces/pressures on each supplied frame, which on Martini 3 runs at roughly the speed of writing the trajectory back out — minutes per system on CPU, not hours. Cost for the full 70-system set is bounded but non-trivial; budget one short HPC job per system (embarrassingly parallel, fits one `sbatch_rerun.sh` analogous to the existing `sbatch_preprocess.sh`). Construct a per-bead local frame from its bonded neighbors (every CG bead in a Martini lipid has deterministic bonded neighbors → well-defined orthonormal frame). Project the force onto that local frame: `(f_parallel, f_perp1, f_perp2)` — three invariant scalars per bead. Predict them as a node-level regression target with the **current GATv2 backbone, no equivariance required.** This is effectively what 3D-EMGP and NequIP do internally, adapted to an invariant setting via the Frad-style local-frame trick. Per-bead forces are the most information-dense target available per frame (3N scalars vs. 2 scalar properties), and the mapping is exactly the CG force-field locally — the most transferable physics there is, because the force a bead feels depends on its local environment (lipids, water, eventually protein), not on the global system type.

B. **Time-lagged predictive objective in embedding space (VAMP/JEPA-style).** Given frame at `t`, predict embedding of frame at `t+τ`. Captures the slow collective modes of the membrane, which are universal (bending, thickness fluctuations, leaflet coupling). Compatible with invariant backbone; needs a reprocessing pass that emits `(frame_t, frame_{t+τ})` pairs with `system_id, frame_idx` metadata stored. Stronger than equilibrium contrastive because the target is a *conditional* rather than *marginal* — encodes dynamics, not just which system.

C. **Per-lipid scalar order-parameter regression.** Compute S_CD, tail tilt, local thickness, local APL per lipid with MDAnalysis from existing `.xtc`s. These are scalar per-lipid targets, directly invariant, interpretable, and the exact observables lipidologists recognize. Predict them from per-lipid pooled embeddings (grouped by residue index). A strong physical grounding and a natural intermediate evaluation that makes the thesis chapter readable to a structural-biology committee.

D. **Lipid-environment contrastive, not frame contrastive.** Positives: "same lipid (same type + similar local neighborhood) in a different bilayer" or "same lipid at `t+τ`". Negatives: "different lipid type / very different local neighborhood." Run on per-lipid embeddings, not graph-level. This avoids the composition shortcut and targets transferability directly: the invariant being learned is "lipid-in-context," which is exactly the object that persists when a protein is introduced.

Other ideas considered and deprioritized:

- Equivariant-pretrain → invariant-finetune: the engineering cost of swapping in NequIP/MACE for pretraining only, then distilling to the GATv2 encoder, doesn't fit a thesis timeline and adds a distillation gap that muddies ablations.
- Boltzmann generators / flow matching on CG frames: enormous infrastructure commitment for marginal thesis payoff.
- Teacher–student from a simple baseline: the baseline already regresses the two scalars; no headroom.
- Multi-view bonded↔spatial alignment (GraphMVP): pleasant sanity-check auxiliary, but not a primary signal — bonded and spatial views of a CG bilayer are too tightly coupled to drive rich representations.

## Recommendation

**Primary objective: Force regression with local-frame projection (A).** It is the only pretext that (i) reuses the existing invariant GATv2 backbone without architectural changes, (ii) targets a per-environment physical quantity whose transferability to protein+membrane is immediate (a bead's local force depends on its local environment, not on system class), (iii) has information density high enough to actually shape the encoder (3N scalars per frame vs. 2 scalar properties), and (iv) is intrinsically a per-bead target, so it cannot be solved by a composition classifier pooling the node-feature multiset.

**Auxiliary: Per-lipid order-parameter regression (C), at a small weight.** Free from existing trajectories, grounds the embedding in recognizable physics, adds a second eval axis for the thesis chapter.

**Deprecated from the preliminary plan:** frame-pair contrastive (composition-identity shortcut + wrong invariant), masked bead-type prediction (too easy), RBF reconstruction (angle-blind).

**Parallel tracks, not joint training.** This representation-learning work runs **alongside** the existing property-regression track, not on top of it. The current `MembranePropertyGNN` + `run_sweep.py` pipeline keeps producing property-regression numbers unchanged; the repr-learning track lives in its own model script, its own training script, and its own W&B project. No shared optimizer, no shared loss, no change to the existing supervised training. Cross-track interaction happens only at evaluation time: the repr-learning encoder is evaluated by a linear probe on the same held-out splits used by the property-regression baseline, so the two tracks produce directly comparable numbers for the thesis chapter.

**New model script.** Create `colab_lipid_gnn_subset/lipid_gnn/representation_gnn.py` (sibling of `membrane_prop_gnn.py`). It owns the SSL-specific forward path: the same HeteroConv + GATv2Conv + GraphNorm encoder stack (duplicated initially — do not factor out a shared encoder until both scripts are working, to avoid coupling churn), plus a `force_head: MLP(hidden_dim → 3)` on per-bead embeddings and an `order_head: MLP(hidden_dim → K_order)` on per-lipid pooled embeddings. No supervised property head. Always runs in the `comp_dim=0` configuration — the encoder must never see composition, which is what keeps the "generalizes to unseen lipid types and eventually to protein+membrane" claim defensible.

**Training regime within the repr-learning track.** Joint multi-task over the SSL heads only: `L = λ_f · L_force + λ_o · L_order`, weights `(1.0, 0.3)`. No pretrain→finetune split initially; with only 70 systems, joint training over the SSL heads gives one clean run per ablation. The fine-tuned property baseline stays with the other track.

**Invariants: existing pipeline must not regress.** `MembranePropertyGNN`, `run_sweep.py`, `train_colab_rev.ipynb`, and the existing 25-test suite keep passing unchanged. Any shared code path (`MartiniHeteroGraphBuilder`, `dataset.preprocess_and_save`) that this track extends must do so behind additive, opt-in flags with the current default preserved.

## Implementation plan (4–6 weeks, single student)

### Phase 1 — data plumbing (week 1)

1. `mdrun -rerun` on one representative system (e.g. POPC100). Take the existing production `.mdp`, set `nstfout = nstxout` (default was `0`, which is why no forces are on disk), re-`grompp` to get a `.tpr` with force output enabled, then `gmx mdrun -s rerun.tpr -rerun prun.xtc -o rerun.trr`. Confirm per-bead forces land in `rerun.trr`, validate magnitudes against expectation (~kJ/mol/nm scale for Martini 3). Expected wall time: a few minutes per system on CPU.
2. Scale Phase 1.1 across the full dataset via a new `scripts/bash/sbatch_rerun.sh` (analogous to `sbatch_preprocess.sh`) that loops over systems; total compute should fit well within the `gpu_test` CPU budget and is embarrassingly parallel. Store `rerun.trr` alongside `prun.xtc` under each `data/membrane_only/<COMP>/run/` directory.
3. Extend `MartiniHeteroGraphBuilder` to **optionally** load `rerun.trr` alongside `prun.xtc` — new kwarg `forces_path=None` defaulting off, so the existing pipeline is unchanged. When enabled, projected force targets are stored on the graph; positions stay out.
4. Add a bonded-local-frame helper: given a bead, pick its two lowest-index bonded neighbors (deterministic), Gram–Schmidt to an orthonormal frame. For beads with <2 bonded neighbors, fall back to a chain-index convention or mark as "project onto bond axis only" (one scalar target, masked in loss for the other two components).
5. Extend `dataset.preprocess_and_save` with additive opt-in args — new kwargs `with_forces: bool = False`, `with_order: bool = False`. When either is True, writes additional tensors (`force_local` [N,3], `force_mask` [N,3], `lipid_order` [num_lipids, K_order], `bead_to_lipid` [N]) to a **separate subdirectory** (e.g. `processed_ssl/train|val|test/`), keeping the existing `processed/train|val|test/` chunks untouched. Also store `data.system_id` (int) and `data.frame_idx` (int) in the SSL chunks (useful later for frame-pair objectives; do not touch the existing chunks).
6. Precompute per-lipid order parameters (S_CD per tail, tilt angle) once per `.xtc` via MDAnalysis — cache as an `.npz` per system; read in the builder when `with_order=True`.

### Phase 2 — new model and training script (week 2)

7. Create `colab_lipid_gnn_subset/lipid_gnn/representation_gnn.py`. Encoder mirrors `MembranePropertyGNN`'s HeteroConv+GATv2+GraphNorm stack (initially duplicated — shared factoring is a later cleanup once both tracks are green). Two heads: `force_head: MLP(hidden_dim → 3)` on per-bead `bead_h`, `order_head: MLP(hidden_dim → K_order)` on per-lipid mean-pooled `bead_h` grouped by `bead_to_lipid`. No supervised property head. `comp_dim` not accepted as a kwarg — the class is intentionally composition-blind.
8. Create `scripts/training/run_representation.py` (sibling of `run_sweep.py`). Reads SSL chunks from `processed_ssl/`, trains `RepresentationGNN`, logs to a **separate W&B project** (e.g. `lipid-gnn-repr`) so dashboards don't collide with the property-regression runs.
9. Loss: `L = λ_f · L_force_masked_mse + λ_o · L_order_mse`, weights `(1.0, 0.3)`. Force loss reduced over unmasked components only.
10. Do **not** modify [membrane_prop_gnn.py](../colab_lipid_gnn_subset/lipid_gnn/membrane_prop_gnn.py), [run_sweep.py](../scripts/training/run_sweep.py), or [train_colab_rev.ipynb](../scripts/colab/train_colab_rev.ipynb). Verify the existing 25-test suite still passes at the end of Phase 2.

Critical files to touch: **new** `colab_lipid_gnn_subset/lipid_gnn/representation_gnn.py`, **new** `scripts/training/run_representation.py`, **new** `scripts/bash/sbatch_rerun.sh`, [lipid_graph.py](../colab_lipid_gnn_subset/lipid_gnn/lipid_graph.py) (additive force/order kwargs, default off), [dataset.py](../colab_lipid_gnn_subset/lipid_gnn/dataset.py) (additive `with_forces`/`with_order`, separate output directory, defaults preserve current behavior).

### Phase 3 — smoke + ablations (weeks 3–4)

11. **Smoke**: 1–2 systems, 5 epochs, repr-learning script only. Confirm force loss drops well below the naive "predict zero force" baseline; order-parameter loss drops below "predict dataset mean."
12. **Composition-shortcut probe (frame contrastive)**: as a separate repr-learning run, train a frame-pair VICReg baseline on the same `RepresentationGNN` encoder. Show that a linear composition classifier trained on the resulting graph embeddings reaches ~perfect accuracy — evidence that frame contrastive on this dataset collapses into a composition classifier even in the gnn_only setting. Contrast with the same probe on force-pretrained embeddings, which should be substantially worse at composition classification while better at the leave-one-lipid-out probe. One-slide thesis figure.
13. **Leave-one-lipid-out (cross-track eval)**: pretrain `RepresentationGNN` on 9 of 10 lipid types, then linear-probe thickness / lipid_packing on the held-out lipid. Compare head-to-head against the property-regression track's own held-out-lipid number using the identical split — this is where the two parallel tracks meet.
14. **Objective ablation**: within the repr-learning track only: (i) +force alone, (ii) +force+order, (iii) +force+order+frame-pair contrastive (to confirm contrastive adds little). Cross-reference all three against the property-regression baseline on the same linear-probe protocol.
15. **Local-frame ablation**: replace local-frame projection with naive ||force|| magnitude (rotation-invariant scalar). Demonstrates that directional information within the local frame is what drives the gains.

### Phase 4 — transferability story (week 5+)

16. Even without running protein+membrane, show:
    - Per-bead force prediction R² holds up on held-out lipid types (force-field locality claim).
    - Per-lipid embeddings from `RepresentationGNN` cluster by bead-environment, not by parent system (UMAP on held-out compositions).
    - Linear probe from `RepresentationGNN` embeddings on held-out composition scalars outperforms the property-regression baseline on those same scalars.
   These three together make a defensible argument for "this embedding is per-environment, not per-system, and therefore admits protein-membrane extension" without requiring the protein-membrane experiment.

## Verification

- **Existing pipeline regression test**: the full property-regression track (25-test suite + a smoke run of `run_sweep.py` with default flags) keeps passing unchanged after every Phase of this plan. Run the suite at the end of Phase 1 and again at the end of Phase 2 as an explicit gate.
- Force rerun correctness: compare `-rerun` forces against a fresh `-nstfout` run on the same 10 frames; RMSD should be numerically zero.
- Local-frame determinism: unit test that permuting bonded-neighbor input ordering does not change the projected force (ordering by bonded-partner index).
- Composition blindness: `RepresentationGNN` does not accept `comp_dim>0`; add a test that asserts perturbing `comp_vec` on the graph leaves the output bit-identical.
- Metadata preservation: `test_train_val_test_splits_are_disjoint` should be extended to the SSL chunk directory so no `(system_id, frame_idx)` appears in more than one split there either.
- Linear probe protocol: freeze the trained `RepresentationGNN` encoder, train a linear head on `graph_repr`, report MSE on held-out compositions AND on the leave-one-lipid-out system. Use the **same splits as the property-regression track** so the two tracks' numbers are directly comparable in the thesis chapter.

## Key risks

- **Forces are not on disk** (`nstfout=0` in the production `.mdp`). Regeneration via `mdrun -rerun` with `nstfout=nstxout` is the standard workflow for this exact situation and does not require re-equilibration, but it is a real compute step (minutes per system × 70 systems, plus storage for `rerun.trr`). Budget half a day to validate the pipeline end-to-end on one system before scaling.
- Bonded local frame is undefined for beads with <2 bonded partners (rare in lipids, but the terminal tail bead C4 has only one). Masking those components is the cleanest fix; losing ~1/12 of force components per lipid is acceptable.
- Joint training may require tuning `λ_f, λ_o` — start at (1.0, 0.3) and sweep if force loss dominates.
- Per-lipid order parameters for non-standard lipids (cholesterol) require a bespoke definition. Drop CHOL from the order-param target if the definition is contentious; it is still in the force target.

## Fallback plan if rerun is blocked

If `mdrun -rerun` turns out to be impractical (missing `.tpr`s, mismatched GROMACS/Martini versions, compute quota), degrade the plan without abandoning it:

- **Promote per-lipid order-parameter regression (C) to primary.** Free from existing `.xtc`s via MDAnalysis, and directly targets the physics the thesis argues about. Reuses the existing invariant backbone with a per-lipid head.
- **Add time-lagged predictive objective (B) as auxiliary.** Pairs of frames `(t, t+τ)` are recoverable from existing `.xtc`s by reprocessing with a pair-aware loader; predict the future embedding from the current one (JEPA-style stop-gradient on the target branch). Encodes dynamics without needing forces.
- Expected ceiling of this fallback is lower than the force-regression route (3N scalar targets per frame vs. ~5 order-parameter targets per lipid), but it is strictly better than the deprecated preliminary recommendation and fully within the student's existing tooling.
