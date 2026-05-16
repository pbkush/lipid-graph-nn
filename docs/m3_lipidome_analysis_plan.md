# M3 Lipidome Analysis Plan

Plan for a marimo notebook (`scripts/notebooks/analyze_m3_lipidome.py`) that
characterises the structure of the Martini 3 lipidome before any new
simulations are run. The analysis is the first scientific question of the next
thesis phase, not a confirmation stage. It frames the question as:

> Does a low-dimensional structure of the M3 lipidome exist that can serve as
> (i) a sampling backbone for choosing which compositions to simulate, and
> (ii) a probe for how far the GNN's learned embedding generalises across the
> M3 lipidome?

The notebook is exploratory and follows the `marimo-data-analysis` skill:
reactive, prose-first, evidence-led, plot titles describe variables,
`_fig,` returns (no `plt.show()`), `mo.stop` guards after data loads,
`mo.callout` only where there is a finding.

The analysis has two layers — **(A) lipid space** (each lipid is a point) and
**(B) composition space** (each membrane is a point). The user's outline
commits to composition space as the target; lipid space is the necessary
precursor — without lipid features, every composition embedding either
collapses to one-hot-on-the-pool or has to invent ad-hoc lipid descriptors.

The full M3 lipid library is already vendored at `resources/martini3/itp/`
(32 ITP collections — PC/PE/PS/PA/PG/PI/SM, CL, ether/plasmalogen variants,
sterols, ceramides, di/monoglycerides, DAG, DOTAP, fatty acids, hydrocarbons).
Material is in place; this plan is about how to use it.

---

## Section 0 — Setup & scope

- Parse `resources/martini3/itp/*.itp` to enumerate every `[moleculetype]`
  and harvest its `@INSANE` line + `[atoms]` + `[bonds]`. Add sterols from the
  separate sterols ITP.
- Define the M3 lipidome scope explicitly. **Default**: bilayer-forming
  lipids only (PA/PC/PE/PG/PI/PS/SM/CL/sterols + ether/plasmalogen variants);
  exclude solvents, ions, hydrocarbons, fatty acids unless explicitly toggled.
  Tag the broader set so the toggle is one line.
- Cross-reference with the current 10-lipid training pool (`LIPID_TYPES`) so
  every downstream plot can mark which points are already simulated.
- Report counts per headgroup family, per tail-unsaturation class, per
  linker class (ester/ether/plasmalogen). The inventory itself is a figure:
  **Figure 1 — "What is in the M3 lipidome?"**

## Section 1 — Lipid feature representations (multiple, compared)

Build several lipid descriptor vectors and analyse them side-by-side. The
point is not to pick a "best" one but to expose how much the downstream
structure depends on the feature choice.

1. **Structural descriptors** — parsed from ITP/INSANE: headgroup class
   (one-hot), linker class, per-tail length, per-tail double-bond count and
   positions, total bead count, net charge.
2. **Bead-composition vector** — count of each Martini 3 bead type
   (Q1, Q5, P1, C1, C4h, …) per lipid. Lipid as a bag-of-beads.
3. **Bead-physics vector** — sum/mean of `[mass, charge, σ, ε]` from
   `martini_ff_params.json` over each lipid's beads. Same physics features
   the GNN consumes.
4. **Graph-derived descriptors** — cheap topological stats on the lipid's
   bonded graph (diameter, branching, head/tail bead-count split). Optional.
5. **GNN-embedding of a single-lipid bilayer (probe)** — for each lipid the
   model can already encode, run the trained Tier C model on a synthetic
   pure-lipid graph and read out the post-trunk embedding. The only
   descriptor the **model itself** produces. Most interesting for the
   "how far does the embedding generalise" question, but only defined for
   the 10 lipids the model has seen unless the bead vocabulary is extended.
   **Phase 2 of this analysis** — needs `LIPID_TYPES` decoupled from
   training; structural/bead/physics descriptors come first.

For each descriptor: z-score, then compute the pairwise distance matrix
(Euclidean for continuous, Hamming/Jaccard for categorical, Gower for mixed).
The distance matrix is the substrate for Section 2.

## Section 2 — Lipid-space dimensionality reduction & clustering

Panel of methods on each descriptor, reported side-by-side:

- **Linear**: PCA, with scree + loadings. The loadings are where
  headgroup vs. tail-saturation as the dominant axes either does or doesn't
  fall out.
- **Distance-preserving non-linear**: MDS (sanity check vs PCA),
  UMAP with `n_neighbors ∈ {5, 15, 50}` (local vs global trade-off).
- **Local non-linear**: t-SNE with a perplexity sweep — kept mainly because
  reviewers expect it; UMAP is load-bearing.
- **Clustering**: HDBSCAN (no `k` to pick), Ward hierarchical (gives a
  dendrogram — useful narrative figure), k-means (only as reference for
  "if we forced k clusters"). Report cluster-quality metrics
  (silhouette, Davies-Bouldin) as descriptors, not objective functions.

**Headline figure**: 2D embedding of the M3 lipidome, coloured by headgroup
family, overlaid with the 10 current training lipids highlighted. Read
directly: are training lipids covering the lipidome or are they a corner?

**Robustness check**: stability under bootstrap resampling of features.
Don't oversell clusters that vanish under a different subset.

## Section 3 — Composition-space construction

The M3 lipidome is a finite set of points (~hundreds); membranes are
*mixtures*. A composition is a point on the simplex over lipids. Naive
simplex = ~hundreds-dimensional; intractable.

Two complementary representations, built and compared:

1. **Simplex over lipid clusters**, not lipids. Use the Section-2 lipid
   clustering with k from the dendrogram. Each composition becomes a
   probability vector over ≈ 5–15 "lipid archetypes" — instead of
   "70 % POPC, 30 % DOPC" we represent it as "70 % saturated-PC archetype,
   30 % unsaturated-PC archetype". Much lower-dimensional, but loses
   identity within a cluster.
2. **Embedding-space centroid**: a composition is encoded as the
   mole-fraction-weighted average of its lipids' coordinates in the
   Section-2 lipid-feature space. Continuous vector of fixed length
   (= embedding dim) regardless of how many lipids the membrane contains.
   **This is the natural composition-space representation for arbitrary M3
   mixtures** and is what makes the iterative "centre → periphery"
   generalisation test possible.

For both: also compute "physics" composition descriptors (mean lipid charge,
fraction unsaturated tails, fraction sterol, headgroup-class fractions) as
interpretable axes any reduced embedding should correlate with — sanity check
that the embedding isn't picking up an artefact.

## Section 4 — Composition-space exploration

- Generate a candidate set of compositions: (i) every pure lipid;
  (ii) all binary 50/50 mixtures within and across clusters;
  (iii) a Latin-hypercube or Dirichlet sample of mixtures over the cluster
  simplex. Tens of thousands of points are cheap — no simulation needed yet.
- Reduce (PCA + UMAP) the composition embedding, cluster (HDBSCAN/Ward).
- Mark the 70 already-simulated compositions and the held-out test
  compositions. **Read directly**: do current sims live in one cluster,
  span multiple, or hug a boundary? Where are the empty regions?
- Compute a coverage / leverage statistic per cluster: how many simulated
  points fall in each, and the distance from each cluster centroid to its
  nearest simulated neighbour. This gives a quantitative "simulation gap"
  map — directly actionable for picking subsets per cluster.

## Section 5 — Selection scheme for simulation candidates (preview, not a decision)

Three candidate selection rules to compare in the notebook. No simulations
are triggered here — this is still analysis.

1. **Cluster-centroid picks**: nearest existing-or-candidate composition to
   each cluster centroid. ~k systems.
2. **Maximin / farthest-point sampling** over the composition embedding,
   restricted to candidate compositions reachable by `insane`.
3. **Stratified shells**: per cluster, pick rings at distances
   `r₁ < r₂ < r₃` from the centroid. This is the user's "iteratively farther
   away" design made concrete — literally the experimental knob for measuring
   embedding extrapolation.

Output: a recommended **shortlist** (≈ 20–40 compositions) with cluster ID,
distance-to-centroid, mole fractions, and a note on whether `insane` can
build them with the current pipeline. This becomes the input to the
simulation pipeline (Step (2)/(3) in `progress.md` → `martini_pipeline_plan.md`).

## Section 6 — Probing the current model on the current pool (cheap, do now)

Before any new simulations: run the trained Tier C model on the existing
70 systems, pull the post-trunk embedding per system, and project it through
the same PCA/UMAP basis built on the composition descriptors. **The
question**: does the *model's* internal geometry align with the
*descriptor-based* composition geometry?

Disagreement is informative — it tells us where the model has
compressed/expanded the composition space and which directions are likely to
extrapolate well vs. badly. This is the only section that ties the lipidome
analysis back to the actual GNN before any new compute is spent.

---

## Notebook contract

- All imports in cell 1; `mo.stop` guards after each data load;
  `mo.callout` only where there is a finding, not as a section header.
- Figures saved to `results/figures/m3_lipidome/` (PNG + PDF), one per
  analysis, descriptive filenames.
- No conclusions section that summarises what the plots already show. End
  with a short "open questions" cell listing what is needed to actually
  answer the embedding-generalisation question (Section 6's disagreement
  plot is the bridge).
- `uvx marimo check` + `uv run` clean before declaring done.

## Defaults assumed in absence of clarification

- **Scope**: bilayer-forming M3 lipids only (toggle for broader set).
- **Primary lipid descriptor**: bead-composition + structural features;
  physics descriptor as secondary.
- **Selection rule for the next thesis step**: stratified shells — it
  operationalises the user's stated experiment.
- **Section 6 is a deliverable of this notebook** (not deferred), because
  it costs nothing and is the only piece that ties the analysis back to
  the GNN before new simulations.

## Open methodological flags

- **Lipid space first vs composition space first**: composition-space
  clustering on top of a flat lipid space inherits the flatness. Doing
  Sections 1–2 first is cheap and lets Sections 3–5 land on solid ground.
- **Hint for embedding quality without trusting M3 parametrisation**:
  Section 6's agreement between model embedding and descriptor embedding on
  the current pool is the cleanest free signal. New simulations will give a
  stronger signal but cost compute.
- **GNN single-lipid probe (descriptor 5)**: most scientifically
  interesting but requires bead-vocabulary extension beyond `LIPID_TYPES`.
  Flagged as Phase 2 of this analysis, not Phase 1.
