import fnmatch
from pathlib import Path

import json
import MDAnalysis as mda
import torch
import numpy as np
from torch_geometric.data import HeteroData
from MDAnalysis.lib.distances import capped_distance

# Fixed-order lipid vocabulary for composition fraction vectors.
# Must match the LIPID_TYPES list used in linear_baseline.py and run_sweep.py.
LIPID_TYPES = ['POPC', 'DOPC', 'DIPC', 'DPPC', 'POPE', 'DOPE', 'DPPE', 'DOPS', 'POPS', 'CHOL']
LIPID_COMP_DIM = len(LIPID_TYPES)  # 10


def create_global_encoder(*args, **kwargs):
    raise DeprecationWarning("create_global_encoder is deprecated! The model now uses physical continuous force field parameters loaded via JSON maps instead of integer vocabulary encoders.")

class GaussianExpansion(torch.nn.Module):
    def __init__(self, start=0.0, stop=12.0, num_gaussians=16):
        super().__init__()
        offset = torch.linspace(start, stop, num_gaussians)
        self.coeff = -0.5 / (offset[1] - offset[0]).item()**2
        self.register_buffer('offset', offset)

    def forward(self, dist):
        return torch.exp(self.coeff * (dist.view(-1, 1) - self.offset.view(1, -1))**2)

class MartiniHeteroGraphBuilder:
    """
    An optimized, stateful builder for converting Martini MD trajectories 
    into PyTorch Geometric HeteroData objects.
    
    Caches static topology (bonds, node types) to make processing 
    multi-frame trajectories highly efficient.
    """
    def __init__(self, tpr_file, trajectory_file, selection="not (resname W or name NA or name CL)", spatial_cutoff=7.5, ff_params_path=None, ff_edge_params_path=None, ff_node_mapping_path=None):
        print("Initializing MartiniGraphBuilder...")

        # 1. Validate topology format — .tpr is required for full bonded topology + atom types
        if not str(tpr_file).lower().endswith(".tpr"):
            raise ValueError(
                f"tpr_file must be a .tpr GROMACS run-input file (got: {tpr_file}). "
                "Other topology formats (e.g. .gro, .pdb) lack the bonded topology "
                "and atom-type information required by MartiniHeteroGraphBuilder."
            )

        # 2. Load Universe (.tpr for topology, .xtc/.trr/.gro for trajectory frames)
        self.u = mda.Universe(tpr_file, trajectory_file)
        self.selection_string = selection
        self.spatial_cutoff = spatial_cutoff
        
        # 2. Isolate Selection
        self.lipids = self.u.select_atoms(self.selection_string)
        self.n_nodes = len(self.lipids)
        print(f"Tracking {self.n_nodes} beads out of {self.u.atoms.n_atoms} total.")
        
        # 2.5 RBF Encoder for spatial distances
        self.rbf = GaussianExpansion(start=0.0, stop=self.spatial_cutoff, num_gaussians=16)
        
        # 3. Cache Node Features (Static continuous physics variables)
        if ff_params_path is None or ff_node_mapping_path is None:
            raise ValueError("ff_params_path and ff_node_mapping_path must be provided to map categorical beads to continuous physics vectors.")
            
        with open(ff_params_path, 'r') as f:
            ff_dict = json.load(f)
            
        with open(ff_node_mapping_path, 'r') as f:
            node_map = json.load(f)
            
        node_features = []
        for mol_name, name in zip(self.lipids.resnames, self.lipids.names):
            bead_type = None
            if mol_name in node_map and name in node_map[mol_name]:
                bead_type = node_map[mol_name][name]
                
            if bead_type and bead_type in ff_dict:
                params = ff_dict[bead_type]
                node_features.append([
                    params.get('mass', 0.0),
                    params.get('charge', 0.0),
                    params.get('sigma', 0.0),
                    params.get('epsilon', 0.0)
                ])
            else:
                print(f"WARNING: Molecule '{mol_name}' Atom '{name}' not mapped to force field params! Defaulting to baseline zeroes.")
                node_features.append([0.0, 0.0, 0.0, 0.0])
                
        self.node_x = torch.tensor(node_features, dtype=torch.float32)
        
        # 3.5 Cache Edge Features Dictionary
        if ff_edge_params_path is not None:
            with open(ff_edge_params_path, 'r') as f:
                self.ff_edge_params = json.load(f)
        else:
            self.ff_edge_params = None

        # 3.6 Compute and cache composition fraction vector
        self.composition_vec = self._compute_composition_vector()
        
        # 4. Cache Index Mapping & Bonded Topology (Static)
        self._cache_topology()

    def _compute_composition_vector(self) -> torch.Tensor:
        """
        Computes the molar fraction of each lipid type present in the membrane.

        Returns a fixed-length float32 tensor of shape (LIPID_COMP_DIM,) where
        each element is the fraction of residues of that lipid type.
        The lipid order matches LIPID_TYPES (e.g. POPC, DOPC, DIPC, ...).

        Residue types not in LIPID_TYPES are silently ignored (e.g. water,
        ions are already filtered out by the selection string).
        """
        resnames = self.lipids.resnames
        unique_res, counts = np.unique(resnames, return_counts=True)
        total_residues = counts.sum()

        vec = np.zeros(LIPID_COMP_DIM, dtype=np.float32)
        for resname, count in zip(unique_res, counts):
            if resname in LIPID_TYPES:
                vec[LIPID_TYPES.index(resname)] = count / total_residues

        return torch.tensor(vec, dtype=torch.float32)

    def _cache_topology(self):
        """Pre-computes and caches the bonded edge index."""
        # Create global-to-local map
        global_to_local = np.full(self.u.atoms.n_atoms, -1, dtype=int)
        global_to_local[self.lipids.indices] = np.arange(self.n_nodes)
        
        # Get global bonds and map to local
        global_bonds = self.lipids.bonds.to_indices()
        local_bonds = global_to_local[global_bonds]
        
        # Filter valid bonds and create bidirectional tensor
        mask = (local_bonds[:, 0] != -1) & (local_bonds[:, 1] != -1)
        self.valid_local_bonds = local_bonds[mask] # Cache for subtraction later
        valid_global_bonds = global_bonds[mask]
        
        bond_index = torch.tensor(self.valid_local_bonds.T, dtype=torch.long)
        self.bond_index = torch.cat([bond_index, bond_index.flip(0)], dim=1)
        
        # Build Edge Features
        if self.ff_edge_params is not None:
            bond_features = []
            for global_idx_pair in valid_global_bonds:
                u_idx, v_idx = global_idx_pair
                u_atom = self.u.atoms[u_idx]
                v_atom = self.u.atoms[v_idx]
                
                mol_name = u_atom.resname
                sorted_names = sorted([u_atom.name, v_atom.name])
                bond_key = f"{sorted_names[0]}-{sorted_names[1]}"
                
                if mol_name in self.ff_edge_params and bond_key in self.ff_edge_params[mol_name]:
                    props = self.ff_edge_params[mol_name][bond_key]
                    bond_features.append([props['length'], props['force_constant']])
                else:
                    raise ValueError(f"Molecule {mol_name} or bond {bond_key} missing from ff_edge_params. Cannot proceed without physical edges.")
                    
            edge_features_tensor = torch.tensor(bond_features, dtype=torch.float32)
            self.bond_edge_attr = torch.cat([edge_features_tensor, edge_features_tensor], dim=0)
        else:
            self.bond_edge_attr = None
        
        # Cache the set of bonded pairs for fast subtraction during frame processing
        bond_set = set(map(tuple, self.valid_local_bonds))
        reversed_bond_set = {(b, a) for a, b in bond_set}
        self.all_bonded_pairs_set = bond_set.union(reversed_bond_set)
        
        print(f"Cached {self.bond_index.shape[1]} directed bonded edges.")

    def process_frame(self, frame_idx=0):
        """
        Generates a HeteroData graph for a specific frame in the trajectory.
        Only calculates spatial edges and updates positions.
        """
        # Move universe to the requested frame
        self.u.trajectory[frame_idx]
        
        # 1. Get current positions
        current_pos = torch.tensor(self.lipids.positions, dtype=torch.float)
        
        # 2. Calculate Spatial Edges based on current coordinates
        pairs, dists = capped_distance(
            self.lipids.positions, 
            self.lipids.positions, 
            max_cutoff=self.spatial_cutoff, 
            box=self.u.dimensions,
            return_distances=True
        )
        
        # 3. Filter Self-loops and Bonds (Vectorized for Speed)
        # Using NumPy arrays instead of Python loops reduces graph build time 
        # from minutes to seconds, especially critical on CPU-only sessions.
        if len(pairs) > 0:
            # Mask self-loops
            loop_mask = (pairs[:, 0] != pairs[:, 1])
            pairs = pairs[loop_mask]
            dists = dists[loop_mask]
            
            # Mask bonded pairs
            if len(pairs) > 0:
                # Efficiently find pairs that are already in the bonded topology
                # Map (i, j) to a unique integer scalar for vectorized search
                pair_scalars = pairs[:, 0].astype(np.int64) * self.n_nodes + pairs[:, 1]
                
                bonded_arr = np.array(list(self.all_bonded_pairs_set), dtype=np.int64)
                if len(bonded_arr) > 0:
                    bonded_scalars = bonded_arr[:, 0] * self.n_nodes + bonded_arr[:, 1]
                    bonded_mask = np.isin(pair_scalars, bonded_scalars)
                    
                    # Apply inverse mask to keep only non-bonded spatial neighbors
                    spatial_pairs = pairs[~bonded_mask]
                    spatial_distances = dists[~bonded_mask]
                else:
                    spatial_pairs = pairs
                    spatial_distances = dists
            else:
                spatial_pairs = np.empty((0, 2), dtype=np.int64)
                spatial_distances = np.empty((0,), dtype=np.float32)
        else:
            spatial_pairs = np.empty((0, 2), dtype=np.int64)
            spatial_distances = np.empty((0,), dtype=np.float32)
        
        # 4. Create Spatial Edge Tensor and Attributes
        if len(spatial_pairs) > 0:
            spatial_index = torch.tensor(spatial_pairs.T, dtype=torch.long)
            # Apply RBF encoding to the distances
            spatial_attr = self.rbf(torch.tensor(spatial_distances, dtype=torch.float32))
        else:
            spatial_index = torch.empty((2, 0), dtype=torch.long)
            spatial_attr = torch.empty((0, 16), dtype=torch.float32)
            
        # 5. Construct HeteroData Object
        data = HeteroData()
        
        # Assign Cached Static Data
        data['bead'].x = self.node_x
        data['bead'].num_nodes = self.n_nodes
        data['bead', 'bonded', 'bead'].edge_index = self.bond_index
        if self.bond_edge_attr is not None:
            data['bead', 'bonded', 'bead'].edge_attr = self.bond_edge_attr
        
        # Assign Dynamic Data
        data['bead', 'spatial', 'bead'].edge_index = spatial_index
        data['bead', 'spatial', 'bead'].edge_attr = spatial_attr

        # Composition descriptor (graph-level feature, same for all frames)
        data.comp_vec = self.composition_vec  # shape: (LIPID_COMP_DIM,)
        
        return data

def main():
    pass


if __name__ == "__main__":
    # Example Usage
    # Ensure you have your .tpr and .xtc files
    # builder = MartiniHeteroGraphBuilder(tpr_file="topol.tpr", trajectory_file="traj.xtc", spatial_cutoff=11.0)
    
    # Mocking data structure for demonstration
    main()
