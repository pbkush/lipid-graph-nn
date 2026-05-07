# Product Context

## Why This Project Exists

The core goal is to find a meaningful **embedding of membrane systems** using a physics-informed GNN. Property prediction (e.g., Area Per Lipid) serves as a training signal and validation, but is not the end goal itself — simpler models could predict properties. The real aim is to learn the underlying physical rules governing membrane behavior. A successful embedding would generalize to any kind of membrane system, including protein+membrane complexes.

The 7 active training targets (lipid_packing, thickness, thickness_std, compressibility, persistence, diffusivity, variation) are documented in [properties.md](properties.md). An 8th computed property, `bending_modulus`, is dropped permanently — its undulation-spectrum-fit label is too noisy to be a trustworthy training signal.

## Problems It Solves

1. **Lack of general membrane representations**: No existing embedding captures the physics of arbitrary membrane compositions in a transferable way
2. **Generalization across compositions**: Learning from diverse lipid mixtures (70 compositions of 10 lipid types) to build representations that work for unseen systems
3. **Physics-informed representation**: Encoding Martini 3 force field parameters directly as node features gives the model physically meaningful inputs rather than arbitrary learned embeddings

## How It Should Work

1. Take a coarse-grained MD snapshot (`.tpr` topology + `.xtc`/`.trr` trajectory)
2. Build a heterogeneous graph where nodes are CG beads and edges encode bonded topology + spatial proximity
3. The GNN processes this graph, learning an internal embedding of the membrane system
4. Property prediction validates that the embedding captures meaningful physics
5. The embedding should generalize — the long-term goal is applicability to any membrane system, including protein+membrane

## User Experience Goals

This is a thesis research project. The "users" are:
- **The researcher (Phillip)**: needs reproducible training runs, clear experiment tracking, and the ability to iterate on model architecture and hyperparameters
- **Future researchers**: the codebase should be modular enough that components (graph builder, dataset, model) can be reused or extended independently
