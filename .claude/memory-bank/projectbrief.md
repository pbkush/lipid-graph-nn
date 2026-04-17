# Project Overview

## Scientific question

Find an embedding for membrane systems from which its physical characteristica can be learned. This will not be tested in this project, but leave the possibillity open that this same embedding model should also be able to, to learn an embedding for protein plus membrane.

## Idea

Use a physics-informed Heterogeneous Graph Neural Network (HeteroGNN). The input will come from coarse-grained Martini 3 MD snapshots. The frames are taken from production runs of membrane systems, that were minimized and equilibriated beforehand. Nodes are CG beads with force-field parameters as features; two edge types encode bonded topology and spatial proximity.

