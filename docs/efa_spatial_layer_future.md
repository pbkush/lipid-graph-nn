# Future idea: Euclidean Fast Attention for the spatial channel

**Status:** Deferred. Not an active task. To be reconsidered **after** all 8 target properties are implemented and the obvious alternative solutions (deeper MP, more data, better pooling, better features) have been exhausted.

**Paper:** Frank, Chmiela, Müller, Unke — *Euclidean Fast Attention* (Nature Machine Intelligence 2026, arXiv:2412.08541).

**Primary goal reminder.** The goal of this project is the best possible embedding for *membrane-only* systems. Transfer to protein+membrane is a long-term aim, but the layers do **not** have to be identical across the two settings. That frees EFA on the membrane side from any "must generalise to protein+membrane" constraint.

---

## TL;DR

Consider adding an **Euclidean Fast Attention (EFA) block on the spatial channel only** — bonded edges stay as `GATv2Conv`. EFA is a linear-cost, SE(3)-equivariant, globally-connected attention mechanism that replaces a hard distance cutoff with a soft `sinc(ω·r)` weighting. Its theoretical strength lines up with the properties in this project that a cutoff-limited local MP has the hardest time reaching: **bending modulus** and **compressibility**, both of which are collective / long-wavelength observables.

The paper's own usage pattern is EFA as an **augmentation** to local message passing, not a replacement — which maps cleanly onto this project's hetero-graph design (keep bonded as local MP; augment the spatial channel with EFA).

---

## Why EFA fits the spatial channel and not the bonded channel

### Bonded edges — poor fit
- Encode chemical topology (a discrete graph with force-constant / equilibrium-length edge features).
- EFA ingests positions only; it has no edge list. The "chemistry vs. proximity" distinction that motivates the hetero-graph would collapse.
- Bonded interactions are 1–2 bonds — local by construction. EFA's long-range benefit offers no leverage.

### Spatial edges — strong fit
- Membrane properties are collective. Bending modulus fits `⟨|h(q)|²⟩` at low `q`; compressibility is an area-fluctuation statistic; persistence and diffusivity couple to long-range correlations. The cutoff-based spatial graph (even at 11 Å) caps the receptive field per layer.
- No arbitrary cutoff. Current project has oscillated between 7.5 / 9.0 / 11.0 Å, each with a failure mode (isolated beads, doubled memory). EFA's `sinc(ω·r)` kernel is soft and cutoff-free.
- SE(3) equivariance "for free". The current model is not equivariant.
- Linear `O(N·G)` scaling with Lebedev grid size `G` (≈ 50–200). Could reduce per-graph memory by removing the `edge_index` / RBF `edge_attr` tensors for the spatial channel.

---

## Three candidate integrations (ordered cheap → expensive)

### (f) Baseline first: deeper MP, no EFA

- Raise `num_layers` (e.g. 3 → 5 or 6) before touching EFA.
- Zero new concepts, no positions re-added, no new hyperparameters beyond depth.
- If this closes the gap on the long-wavelength targets, EFA is unnecessary. If it does not, the null hypothesis is ruled out and EFA experiments are justified.

### (c) Readout-only EFA

- Leave all MP layers untouched. Insert a single EFA block between the final MP layer and the global mean+max pool.
- One shot of global mixing after local features are built — captures low-wavenumber moments of the embedded field.
- Lowest architectural risk EFA variant. Only one new hyperparameter cluster (`G`, `ℓ_max`, `{ω_k}`, frequencies).

### (b) Parallel augmentation — per-layer EFA

- Keep bonded GAT as-is. Shrink the spatial cutoff back to something cheap (e.g. 7.5 Å) for the local spatial MP. Add an EFA block in parallel inside each layer; fuse outputs additively (or by a learned gate).
- Matches the paper's own recipe exactly. Highest capacity.
- Largest implementation surface. Needs equivariant-feature plumbing if `ℓ_max ≥ 1` is used.

---

## Expected impact per target property

| Target | Dominant scale | Expected gain |
| --- | --- | --- |
| `thickness_std` | local | low |
| `persistence` | local | low |
| `lipid_packing` | local, globally averaged | low–medium |
| `thickness` | local normal-direction | low–medium |
| `variation` | medium (patch-scale) | medium |
| `diffusivity` | medium (hydrodynamic coupling) | medium |
| `compressibility` | long-wavelength (⟨ΔA²⟩) | **high** |
| `bending_modulus` | inherently long-wavelength | **highest** |

`bending_modulus` and `compressibility` are the acid test. If EFA does not help these two, it is not the right tool for this problem.

---

## Critical caveats before any implementation

1. **PBC adjustment required.** Martini membrane simulations run under periodic boundary conditions. The paper's default EFA integrates over all directions on `S²` assuming full SO(3) symmetry (isolated system). Under PBC the unit-cell SO(3) symmetry is broken. The paper's prescribed fix (SI section *Other Symmetrisation Operations*) replaces the `S²` integral with a **discrete sum over special directions `u` chosen from the lattice** (reciprocal-lattice-compatible; related to an Ewald-like plane-wave expansion). The membrane box is orthorhombic with known per-frame lattice vectors, so this is tractable — but the implementation is **not** a drop-in of the paper's reference code: the symmetrisation sum, the frequency set `{ω_k}`, and the quadrature weights all have to be rederived for the PBC variant. Any future implementation must start from the PBC form, not the SO(3) form.
2. **Positions are no longer in the saved graphs.** The 2026-04-18 memory optimisation explicitly removed `data['bead'].pos` from the saved `.pt` chunks. EFA requires atomic positions as input, so this optimisation would be reversed. Chunks would have to be regenerated.
3. **Transferability of the paper's claim is assumed, not proven.** The paper's wins are on force/energy regression for small molecules and reactions, not on coarse-grained graph-level scalar regression. The "long-range helps" story is physically reasonable here but not empirically demonstrated in this regime.
4. **Equivariance helps only if irreps with `ℓ ≥ 1` are used.** Current node features are all scalars (mass, charge, σ, ε). A scalar-only (`ℓ=0`) EFA is still useful but leaves the orientation channel empty unless ℓ=1 features are designed (e.g. a local tail-vector).
5. **New hyperparameter burden.** Lebedev grid size `G`, maximum degree `ℓ_max`, frequency set `{ω_k}`, fusion strategy (sum vs. concat vs. gate). Under PBC, also the lattice-compatible direction set.
6. **Martini's own interactions are short-range.** Martini LJ cuts off at 1.1 nm. Any "long-range" signal in the MD is collective correlation, not direct force — so the EFA benefit is subtler than in all-atom MLFF settings. It captures emergent physics, not missing direct interactions.

---

## Suggested order when this is picked up

1. Implement all 8 target properties in the current architecture.
2. Exhaust simpler improvements: batch size / AMP / grad clip, hyperparameter sweeps, better pooling, richer node features.
3. Run **(f)** — deeper MP — as the null hypothesis.
4. If (f) does not close the gap on bending modulus / compressibility, implement **(c)** — readout-only EFA, using the **PBC** variant of ERoPE from the start.
5. If (c) moves the needle on the two long-wavelength targets, graduate to **(b)** — per-layer parallel augmentation.

This ordering avoids paying the full cost of (b) (positions back in chunks, equivariant feature plumbing, per-layer tensor-product irreps, PBC symmetrisation) until the cheaper variants prove long-range information is actually what is missing.
