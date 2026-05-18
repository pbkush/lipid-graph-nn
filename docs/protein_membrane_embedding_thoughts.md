# Embedding protein + membrane: design-space notes

Working notes for the next scientific step in this thesis — extending the
membrane-only GNN embedding to membrane-with-protein systems. The intent of
this document is **not** to commit to a plan, but to lay out the choices, the
tradeoffs, and the questions that need to be answered before any simulation is
submitted. References are listed inline; everything in §8 is a follow-up
question for Phillip.

Cross-references: project goal in
[projectbrief.md](../.claude/memory-bank/projectbrief.md);
current corpus and pipeline in
[activeContext.md](../.claude/memory-bank/activeContext.md);
properties in [properties.md](../.claude/memory-bank/properties.md);
M3 lipidome plan in [m3_lipidome_analysis_plan.md](m3_lipidome_analysis_plan.md).

---

## 1. What is the actual scientific question?

The thesis goal is to learn a **transferable membrane embedding** — property
prediction is the training signal, not the deliverable. So for the
protein+membrane extension, the question is *not* "can the model predict
properties of protein-containing systems?" (the properties as currently defined
— APL, thickness, Voronoi-area variation, lateral diffusivity, contact
persistence, compressibility, thickness_std — are not even meaningfully defined
in the presence of an embedded protein, since the protein occupies area and
deforms the bilayer locally).

The question is one of three things, and we should pick one before designing
the simulations:

- **(A) Embedding generalisation under domain shift.** Take the locked
  membrane-only model, run it on protein+membrane snapshots, and ask whether
  the post-trunk embedding places these systems in a *sensible* region of the
  embedding space relative to the pure-bilayer reference. "Sensible" = the
  protein-containing system clusters near the pure bilayer of the same lipid
  composition, with structured deviations attributable to the protein. This
  is the easiest experiment — no retraining, no new labels.
- **(B) Embedding extension via fine-tuning / multi-task.** Add a small
  amount of protein+membrane training data and ask whether the embedding can
  *learn* the protein perturbation without degrading the membrane-only
  R² band. Requires new labels — most plausibly **local** properties (e.g.,
  thickness at a given radial bin from the protein) rather than the current
  whole-bilayer scalars.
- **(C) Pure structural probe.** Use the protein+membrane systems as a
  qualitative test bed (PCA, UMAP of embeddings, similarity to nearest
  bilayer) with no quantitative training objective. Lowest-cost, lowest-claim.

The rest of this doc assumes **(A) is the right minimal first step**, with (B)
as a stretch goal once (A) has shown signal. (C) is the fallback if (A) reveals
that the trunk is too lipid-specific to embed proteins at all — which is itself
a reportable result.

---

## 2. Membrane-protein candidates

The selection axes are: **architecture** (helix bundle vs β-barrel vs
peripheral), **size** (impacts box size and cost), **lipid sensitivity** (does
the protein behave the same in every bilayer? — important for the embedding
test), and **Martini-3 validation status**.

### 2.1 Single-TM helices and peptides

The cheapest entry point. Used in every M3 protein paper as a sanity case.

- **WALP / KALP**. Synthetic α-helical peptides (~23 residues). The
  canonical hydrophobic-mismatch system: vary the bilayer thickness and the
  peptide tilts or the bilayer locally deforms. Cheap, well-characterised in
  Martini 2 and 3, useful precisely because **the response is known**. Many
  references — Killian's group, the Marrink benchmark series.
- **LL-37 / magainin / melittin**. Antimicrobial peptides. Surface-binding,
  pore-forming. More interesting biology but the pore-formation dynamics
  bring sampling concerns.
- **Glycophorin A TM dimer (GpA)**. Classical TM helix-helix association
  benchmark; M3 association free-energy reproduced by Souza et al. 2021.

WALP is the right *smoke test* protein: minimal cost, known response,
and we can compare the embedding-deviation magnitude across a thickness
sweep (DLPC → POPC → DPPC).

### 2.2 GPCRs and other 7TM helix bundles

- **β2-adrenergic receptor (β2AR)** — Souza et al. 2021 *Nat. Methods*
  Martini-3 benchmark. Heavy lipid-fingerprinting literature; PIP2 / CHOL
  binding sites are documented.
- **A2A adenosine receptor** — Marrink-lab Martini-3 paper.
- **Rhodopsin** — extensive M2/M3 history; well-characterised tilt and
  membrane-curvature coupling.

A GPCR is more interesting scientifically than a WALP — the protein actively
recruits specific lipids, so the embedding has a chance of distinguishing
"POPC with β2AR" from "POPC alone" *and* from "POPC+CHOL with β2AR" in a
non-trivial way. Cost is moderate (~12 × 12 nm box).

### 2.3 β-barrels (bacterial outer membrane)

- **OmpA / OmpF / OmpLA** — small to medium β-barrels. OmpF is a trimer in
  vivo; OmpA monomer is the simpler benchmark.

β-barrels probe a different region of the protein structural space and have
distinct lipid-environment requirements (bacterial outer membrane, LPS). Less
well aligned with our current PC/PE/PG/CHOL lipid corpus.

### 2.4 Larger complexes (out of scope for v1)

ATP synthase, photosystems, large transporters (Sec, LacY). Cost grows fast.
Defer.

### 2.5 Recommendation

For the first protein+membrane experiment, pick **two contrasting proteins**:

- **WALP** as a minimal, known-response probe. Embedding deviation should
  scale with hydrophobic mismatch — a falsifiable prediction.
- **β2AR** as a biologically realistic, M3-validated probe with established
  lipid-fingerprint behaviour.

One protein alone cannot distinguish "the embedding learned this specific
protein" from "the embedding learned proteins in general" — at least two are
needed to claim any kind of transfer. Three is better but the cost grows
linearly and WALP+β2AR already span the size and architecture spread we need
for a thesis-scope statement.

---

## 3. Membrane composition for the protein systems

Two pulls in opposite directions:

- **Stay inside the training distribution.** The 70-composition corpus is
  built from the 10-lipid PC/PE/PG/PS/SM/CHOL/DI*/DO* pool. Any protein
  embedded in a *new* lipid composition (e.g. LPS, cardiolipin) confounds
  protein-domain-shift with lipid-domain-shift.
- **Test where the protein actually lives.** β2AR's native environment is
  CHOL-rich PC; bacterial proteins want PE/PG. Picking a "physiological"
  bilayer is more defensible biologically.

The cleanest design uses **compositions that already exist in the membrane-only
training corpus**, so that the membrane-only reference for each
protein-containing system is a real training point, not an extrapolation:

| Protein | Bilayer (in training corpus?) | Why |
|---|---|---|
| WALP | POPC100 (yes), DLPC100 (yes), DPPC100 (yes) | Thickness sweep — tests hydrophobic-mismatch response in embedding |
| β2AR | POPC100 (yes), POPC70_CHOL30 (yes if scanned) | CHOL recruitment is the documented signal |

This makes the embedding test crisp: **for every protein+lipid system there
exists a matched pure-bilayer training point**, so the embedding-distance
signal is directly attributable to the protein.

A useful extension: pick a fourth bilayer that is *just outside* the training
distribution (e.g. one new PE-rich mix) to probe lipid-extrapolation and
protein-presence as orthogonal axes.

---

## 4. Atomistic → coarse-grained: how hard?

Routine for monomeric, structurally well-resolved proteins; the tooling is
mature.

### 4.1 Tooling

- **`martinize2`** (vermouth-martini, GitHub: `marrink-lab/vermouth-martinize`)
  — current standard, replaces the old `martinize.py`. Handles M3 mapping,
  DSSP-derived secondary structure, elastic-network / Go-model attachment,
  multimeric chains. *This is the entry point*; everything else falls out of
  flag choices.
- **`insane.py`** (vendored already at `resources/martini3/insane.py`)
  for protein insertion + lipid placement. Supports `-with-protein
  protein.gro` to centre the protein and pack lipids around it.
- **DSSP** for the secondary-structure assignment martinize2 needs.

### 4.2 Decisions you have to make

1. **Backbone restraint method**:
   - *Elastic network (ElNeDyn)* — pairwise harmonic springs between
     backbone beads within a cutoff (default 0.9 nm) with a tunable force
     constant (500–1000 kJ/mol/nm²). Robust, but freezes large-scale
     hinge motions.
   - *Go model* — springs only between native-contact pairs. Preserves
     conformational substates better; preferred for proteins where domain
     motion matters. M3-compatible Go implementations exist (Poma et al.;
     Souza et al. 2021).
   - For first systems (WALP, β2AR), **ElNeDyn is fine** — these proteins
     are not undergoing large conformational changes on µs timescales.
2. **Charges / pH state** — defaults at pH 7. Histidines need attention.
3. **Disulfides** — declare via martinize2 flags.
4. **Multimer handling** — chain-by-chain martinize2, then concatenate;
   place under one ElNeDyn or independent per-chain.

### 4.3 What's genuinely hard

- **Intrinsically disordered regions** — elastic networks will artificially
  rigidify them. M3 has a tunable "IDP-friendly" recipe (Thomasen et al.
  2022, *JCTC*) but it's a separate workflow.
- **Quaternary assemblies that exchange subunits** — out of scope.
- **Membrane-embedded oligomerisation as a dynamical observable** — needs
  long sampling; not for v1.
- **Protein-bead vocabulary**. The current graph builder maps M3 bead types
  for lipids only. Protein backbone and side-chain beads (BB, SC1–SC4 of
  the M3 protein force field) will need to be added to the node feature
  table. This is mostly bookkeeping but it is **a non-trivial change to
  `lipid_graph.py` and the FF JSON files** and should be planned as its
  own subgoal, not as part of the simulation work.

In short: producing valid CG protein topologies for v1 is a 1–2 day task per
protein for someone fluent in the toolchain. Wiring the resulting bead types
into the GNN node-feature table is the larger lift.

References for the CG protein workflow:
- Souza, P. C. T., Alessandri, R., Barnoud, J. et al., *Nat. Methods* **18**,
  382–388 (2021). Martini 3 release; protein FF benchmarks.
- Kroon, P. C., Grünewald, F. et al. *vermouth/martinize2* docs.
- Thomasen, F. E. et al., *JCTC* **18**, 2033–2041 (2022). IDP-friendly M3.
- Poma, A. B. et al., *JCTC* **13**, 1366–1374 (2017). Go-model for M3-style.
- Marrink, S. J. & Tieleman, D. P., *Chem. Soc. Rev.* **42**, 6801 (2013).
  Original Martini protein review (still useful).
- Pezeshkian, W. et al., *Nat. Commun.* **11**, 2296 (2020). Whole-organelle
  M3 modelling — for scale calibration.

---

## 5. One protein or many?

Both extremes have problems:

- **Single protein, single composition** — cannot distinguish "the embedding
  generalises to protein+membrane" from "the embedding happens to place this
  one outlier somewhere not-completely-wrong".
- **Many proteins, many compositions** — combinatorial cost, dilutes the
  evidence per system, and we already know the labels are not well defined
  for protein systems.

The right middle is a small **factorial sweep**: 2 proteins × 3 compositions =
6 systems, with each system having a **matched pure-bilayer reference already
in the training set**. This gives:

- 2 protein-presence levels (absent / present) — tests "embedding sees the
  protein".
- 3 lipid axes — tests "embedding still responds to lipid composition with
  the protein in place".
- 2 protein architectures — tests "the embedding distinguishes proteins from
  each other".

If we can afford it, add a third protein (an OmpA-style β-barrel) for one
composition to break the helix-only confound. So the table is:

| System | Box (nm) | Lipids | Reference bilayer in train? |
|---|---|---|---|
| WALP / POPC | ~8 × 8 | ~200 | yes (POPC100) |
| WALP / DLPC | ~8 × 8 | ~200 | yes (DLPC100) |
| WALP / DPPC | ~8 × 8 | ~200 | yes (DPPC100) |
| β2AR / POPC | ~12 × 12 | ~400 | yes (POPC100) |
| β2AR / POPC+CHOL | ~12 × 12 | ~400 | yes (POPC70_CHOL30 if in corpus) |
| β2AR / DOPC | ~12 × 12 | ~400 | yes (DOPC100) |
| (opt.) OmpA / POPC | ~12 × 12 | ~400 | yes |

Seven systems is tractable on the GPU defaults (8 sims/node, ~5240 ns/day
per slot) — see [activeContext.md](../.claude/memory-bank/activeContext.md)
for the locked HPC defaults.

---

## 6. Well-researched in general vs. well-researched in Martini 3?

Strong preference for **well-validated in Martini 3 specifically**, for three
reasons:

1. **Parameter trust**. The M3 protein force field is still evolving (Souza
   et al. 2021, with revisions in 2022–2024 for IDPs, glycosylated proteins,
   and protein-lipid interaction strengths). Picking a system that has
   already been benchmarked in M3 means the published reference observables
   (tilt angle, lipid annulus composition, oligomerisation kinetics) are
   available as a sanity check.
2. **Topology availability**. M3-published systems often have their
   `martinize2` invocation in the paper SI; we get a known-good CG topology
   for free.
3. **Confounding control**. If the embedding test fails, we want to be able
   to say "the embedding is wrong" — not "we don't know if the embedding or
   the protein FF is wrong".

Of the candidates in §2: WALP (every M3 paper), β2AR (Souza et al. 2021),
A2A (Souza et al. 2021), OmpA (Marrink lab benchmark series) all qualify.
Avoid for v1: any IDP, anything that requires Go-model tuning, glycosylated
proteins, large complexes.

---

## 7. Simulation size and length

### 7.1 Box and lipid count

Current membrane-only systems (per `data/membrane_only/`) use ~390 lipids
total at ~12 × 12 nm box footprint (e.g. POPC100 has 392 lipids ≈ 196/leaflet
→ box edge ~11.5 nm in PC). For protein systems:

- **Minimum lipid annulus**: ≥ 3 lipid shells around the protein to avoid
  PBC self-interaction of the protein's perturbation field. WALP needs ~2 nm
  of lipid in each direction; β2AR ~3 nm.
- **Minimum box edge**: `2 × annulus + protein_diameter`. WALP ≈ 1.5 + 4 +
  1.5 = 7 nm → round up to 8 × 8. β2AR ≈ 3 + 4 + 3 = 10 → 12 × 12 to be
  safe. OmpA monomer similar to β2AR.

So WALP systems are *smaller* than the current membrane-only corpus
(~200 lipids vs ~390), and β2AR systems are *the same size* (~400 lipids).
**Total bead count is comparable.** No re-benchmarking of HPC defaults
needed.

### 7.2 Trajectory length

Two timescales matter:

- **Lipid annulus equilibration around the protein**: 100–500 ns. The
  first lipid shell composition (especially for CHOL+β2AR) is the slow
  variable.
- **Protein-internal relaxation under the elastic network**: tens of ns.
  Fast.

For embedding-only use ("dump frames, encode, look at clusters"), the
relevant criterion is independent-frame count. Lipid annulus autocorrelation
time is ~50–100 ns in M3 (Sejdiu & Tieleman 2020, *Biophys. J.* — "Lipid
contact-annulus dynamics around a GPCR"). To get ~50 independent frames
per system, we need ~5 µs of post-equilibration trajectory.

That is **5× longer than the current 1 µs membrane-only production**. But:
the current 1 µs runs use frames [50:667] at 1.5 ns spacing → ~617 frames.
If 50 truly independent frames suffices for embedding statistics, 1 µs is
already enough — *for the membrane*. The protein adds a slower timescale.

Recommendation for v1:

- **2 µs production** per protein+membrane system, frames 100–end at 1.5 ns
  spacing → ~1250 frames, ~25–40 independent w.r.t. lipid annulus.
- Reserve longer (5 µs) for one or two of the systems if v1 results indicate
  the embedding response is small and we need more statistical power.

### 7.3 Cost estimate

At ~5240 ns/day per slot (current GPU defaults):

- 6 systems × 2 µs = 12 µs total trajectory.
- 12 000 / 5240 ≈ 2.3 slot-days = 7 hours on 8-way packed node = **one
  GPU-node day**, comfortably inside one walltime block.

This is **negligible cost** relative to the existing membrane-only corpus.
The expensive step is the preprocessing/topology setup, not the wall clock.

---

## 8. Architecture: would Euclidean Fast Attention (EFA) change this?

Existing memo: [efa_spatial_layer_future.md](efa_spatial_layer_future.md).
That document treats EFA as a *membrane-only* upgrade aimed at
`bending_modulus` and `compressibility`. Both motivating targets have changed
status since it was written, and the protein extension introduces new EFA-
shaped targets — so the question is worth reopening, in two parts.

### 8.1 Has the membrane-only case for EFA weakened?

Yes, partly. The original argument (efa doc §2 and the impact table) leaned on
two long-wavelength targets:

- **`bending_modulus`** — now **permanently dropped** on label-quality grounds
  (undulation-spectrum fit too noisy; see
  [properties.md](../.claude/memory-bank/properties.md) note). The acid-test
  target is gone.
- **`compressibility`** — was pre-registered with R² « 0.5 on receptive-field
  grounds; **Tier C 5d lands at pooled test R² 0.88** (see
  [activeContext.md](../.claude/memory-bank/activeContext.md)). The receptive-
  field upper bound argument was falsified empirically — local 11 Å packing
  geometry already correlates strongly with whole-bilayer area-fluctuation
  density. There is no obvious headroom for EFA to claim here.
- **`persistence`** R² ≈ 0.58 is the remaining architecture-bound target, but
  it sits in the *medium-scale* row of the efa-doc impact table — not the
  high-expected-gain row.

So the membrane-only EFA test now has **no failing target to rescue**. That
does not mean EFA is useless on membrane-only — it might still tighten the
embedding (sharper clusters, smaller seed variance, smaller per-system error
on the DPPC/DOPC corners that Stage 5b flagged) — but the acceptance
criterion has to change. The original "moves bending_modulus and
compressibility" gate is no longer available; a replacement would have to be
spelled out before running. Plausible replacements:

- *Embedding-quality gates*: post-trunk-embedding intra-composition
  variance, inter-composition separation, or transfer-R² to a held-out
  lipid type (e.g. CHOL-containing systems trained without CHOL exposure).
- *Per-system error-tail gates*: percentile-95 error on the
  DPPC/DOPC-rich corner that currently dominates the Tier B/C residual
  budget (POPC30_DOPC70 ~19 Å thickness MAE, per Stage 5b).
- *An architectural-probe label not used for thesis claims*: e.g.
  `S(q_min)` (undulation spectral amplitude at the smallest accessible
  wave-vector) — same scale as bending_modulus but read off the bilayer
  directly rather than via a fit, so the label noise issue is smaller.
  This is essentially "bending_modulus done right for the architecture
  test" and could be computed retroactively from existing trajectories.

### 8.2 Would EFA matter for the protein case?

This is where EFA's case *strengthens*. Three reasons:

1. **The protein introduces a localised, oriented perturbation source.**
   The bilayer responds with long-wavelength deformations: curvature field
   around the inclusion, undulation suppression in the protein vicinity,
   radial lipid-recruitment profile, area redistribution across the box.
   These are exactly the modes a `sinc(ω·r)` global-mixing kernel
   represents well and that an 11 Å cutoff cannot reach. If scenario (B)
   from §1 is pursued, the natural labels (curvature(r), tilt, annulus
   composition spectrum) are EFA-shaped by construction.

2. **SE(3) equivariance becomes salient.** A pure bilayer has an
   approximate up/down symmetry plus in-plane isotropy — a scalar model
   ignores orientation cheaply. A protein-in-bilayer system has a
   well-defined protein axis, a bilayer normal, and a relative tilt
   between them — three vector quantities. A scalar (`ℓ=0`) GNN has no
   dedicated channel for that information; an `ℓ ≥ 1` equivariant model
   has one for free. The empirical question is whether the trunk already
   recovers orientation implicitly from the bead geometry; the structural
   argument says it should not, especially with the elastic-network
   constraints that fix the protein backbone.

3. **Inhomogeneity is the regime where global-mixing wins.** Pure
   bilayers are statistically homogeneous — local geometry is locally
   representative, so the cutoff-MP wins easily (this is *why*
   `compressibility` did not need EFA after all). Protein systems break
   homogeneity by construction. The information at lipid `i`'s annulus
   shell *is* different from the information far from the protein, and
   needs to be communicated farther than the cutoff to characterise the
   protein-bilayer coupling. This is the textbook EFA setting.

### 8.3 But the engineering stack now grows

For the protein extension *with* EFA, three independent lifts compound:

1. Bead-vocabulary extension to M3 protein beads (Phase 0 in §9 already).
2. Position tensors restored to saved chunks (efa-doc caveat 2) — chunks
   regenerated.
3. PBC-correct EFA implementation (efa-doc caveat 1) — symmetrisation sum
   over reciprocal-lattice directions, not the SO(3) integral; not a
   drop-in of the paper's reference code. Plus ℓ≥1 equivariant feature
   plumbing if orientation is to be exploited (point 8.2.2).

Each of these is feasible; their *product* is a substantial reorganisation.
None of it should be attempted until the membrane-only EFA experiment in
§8.4 below has produced a positive signal.

### 8.4 Suggested order, integrated with the protein roadmap

Your suggestion is correct — **test EFA on membrane-only first**, with the
explicit acknowledgement that the acceptance criterion has to change from the
original efa-doc. Concretely:

1. **Define the membrane-only EFA acceptance criterion first** (probably
   one of the three options in §8.1 — embedding-quality, per-system error
   tails, or an `S(q_min)` architectural-probe label).
2. **Run efa-doc variant (f)**: deeper MP only, no EFA. If a few more
   layers of plain MP move the chosen metric, EFA is unnecessary —
   the cutoff was just too short for too few layers, and the protein
   case can be addressed by deeper MP plus the bead-vocab extension.
3. **If (f) does not move it**, run efa-doc variant (c): readout-only
   EFA on membrane-only. PBC variant, scalar (`ℓ=0`) only. Cheapest EFA
   experiment that exists.
4. **If (c) moves it on membrane-only**, *then* graduate to the protein
   case — but at that point, run efa-doc variant (b) (per-layer parallel
   EFA, with ℓ=1 features if the protein-axis orientation argument from
   8.2.2 is to be tested).
5. **If (f) suffices on membrane-only**, the protein case probably also
   doesn't need EFA on day one. Re-evaluate after seeing the Phase 1
   inference test from §9.

Net: EFA stays deferred *and* the original deferred-EFA doc gets a status
update — the membrane-only motivating target list has changed. The protein
extension is the strongest *new* motivation for EFA in this project, but
that motivation is conditional on a positive membrane-only signal under a
new, embedding-quality-based acceptance criterion.

A useful side effect: §8.1 option iii (`S(q_min)` as an architectural-probe
label) is also a candidate *protein-side* training target in scenario (B) —
the protein-induced suppression of `S(q_min)` is a textbook membrane-elastic
observable. So the membrane-only EFA test and the protein-extension labels
can share infrastructure.

---

## 9. Open questions for Phillip

These are the calls I'd want to make together before submitting anything:

1. **Are we committing to scenario (A), (B), or (C) from §1?** The
   downstream design changes substantially.
2. **Is WALP+β2AR the right pair, or do you have an existing collaborator
   protein system you want to use?** (A protein already in someone's hands
   in the group is worth more than a textbook benchmark — the topology is
   already validated and you can compare directly.)
3. **Do we need protein-side labels at all?** If yes, what — radially-binned
   thickness? lipid-contact-occupancy spectra? tilt angle? — these would
   become the new training targets in scenario (B). I'd argue **no labels
   for v1**: do (A) qualitatively first, decide on labels only after seeing
   what the embedding does.
4. **Bead-vocabulary extension**: when do we plan the
   `lipid_graph.py` / `ff_node_mapping.json` / `MembranePropertyGNN`
   changes for protein beads? This is the biggest engineering lift and
   should probably precede the first protein simulation, not follow it.
5. **Should the protein systems also feed the M3 lipidome composition
   shortlist** (cf. [m3_lipidome_analysis_plan.md](m3_lipidome_analysis_plan.md)
   §2 stratified shells), or are they a parallel, independent track?
6. **EFA acceptance criterion** (cf. §8): which of the three replacement
   gates from §8.1 do we adopt before running variant (f)/(c) on
   membrane-only? My weak preference is the `S(q_min)` architectural-probe
   label — it is the cleanest "long-wavelength target without bending-fit
   noise" we can construct, and the same observable transfers to the
   protein case as a candidate scenario-(B) label.

---

## 10. Suggested phasing (one-paragraph summary)

**Phase 0 (engineering)**: extend the bead vocabulary and node-feature
table to cover M3 protein beads; add a protein-bead-to-protein-bead edge
type to the heterograph (bonded + spatial as for lipids). No simulation yet.
**Phase 1 (smoke test)**: WALP in POPC100, 2 µs, single replicate; run the
locked Tier C model in inference mode and check whether the post-trunk
embedding lands near the POPC100 training point (it should) with a
structured offset attributable to the peptide. **Phase 2 (factorial)**: the
6-system table in §5, same inference protocol, with a small report on
embedding distances vs. pure-bilayer references and on inter-protein
distinguishability. **Phase 3 (optional)**: scenario (B) — define a local
radial property, generate labels for the 6 systems, fine-tune the trunk,
test whether membrane-only R²s hold.

The defensible thesis claim from Phase 1+2 alone is: *"the membrane
embedding learned from pure-bilayer data places protein+membrane systems
in a structured, lipid-composition-respecting region of the embedding
space"* — which is exactly the generalisation claim the
[projectbrief.md](../.claude/memory-bank/projectbrief.md) makes a stretch
goal. Phase 3 is upside, not required.

---

*Draft 2026-05-18. Followups in §8 are open.*
