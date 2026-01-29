
import MDAnalysis as mda
import torch
import numpy as np
from torch_geometric.data import HeteroData
from MDAnalysis.lib.distances import capped_distance
from sklearn.preprocessing import LabelEncoder


def create_hetero_lipid_graph(structure_file, topology_file, spatial_cutoff=11.0):
    """
    Creates a Heterogeneous Graph from a Martini system.
    
    Nodes: 
        - 'bead' (All lipid beads)
    
    Edges:
        - ('bead', 'bonded', 'bead'): Topology connections from .tpr
        - ('bead', 'spatial', 'bead'): Radius interactions (excluding bonds)
    
    Args:
        structure_file (str): Path to .gro
        topology_file (str): Path to .tpr (preferred) or .top
        spatial_cutoff (float): Cutoff in Angstroms (default 11.0 A = 1.1 nm)
    """
    
    # 1. Load System
    print(f"--- Loading Universe ---")
    u = mda.Universe(topology_file, structure_file)
    
    # Select Lipids (customize string for your specific lipids)
    lipids = u.select_atoms('not resname W PW ion')
    n_nodes = len(lipids)
    print(f"Selected {n_nodes} beads out of {u.atoms.n_atoms} total atoms.")

    # 2. Global -> Local Index Mapping
    # Create an array where index=GlobalAtomID, value=LocalGraphID
    # Initialize with -1 (meaning "atom not in graph")
    global_to_local = np.full(u.atoms.n_atoms, -1, dtype=int)
    global_to_local[lipids.indices] = np.arange(n_nodes)

    # 3. Process BONDED Edges (Topology)
    print("--- Processing Topology Bonds ---")
    # Get global indices of bonded pairs
    global_bonds = lipids.bonds.to_indices()
    
    # Map to local indices
    local_bonds = global_to_local[global_bonds]
    
    # Filter bonds where one atom might be outside selection (safety check)
    mask = (local_bonds[:, 0] != -1) & (local_bonds[:, 1] != -1)
    bond_index = torch.tensor(local_bonds[mask].T, dtype=torch.long)
    
    # Make bonds bidirectional for the GNN (A-B and B-A)
    bond_index = torch.cat([bond_index, bond_index.flip(0)], dim=1)
    
    # 4. Process SPATIAL Edges (Radius Search with PBC)
    print(f"--- Processing Spatial Neighbors (r={spatial_cutoff}A) ---")
    # capped_distance handles PBC automatically using u.dimensions
    # returns pairs (i, j) where i < j
    pairs, dists = capped_distance(lipids.positions, lipids.positions, 
                                   max_cutoff=spatial_cutoff, 
                                   box=u.dimensions,
                                   return_distances=True)
    
    # These indices are already local (0 to n_nodes) because we passed lipids.positions
    # However, we must filter out Self-loops and Bonds
    
    # Convert arrays to sets of tuples for fast set subtraction
    # Note: spatial pairs from MDAnalysis are generally i < j
    spatial_set = set(map(tuple, pairs))
    
    # Remove Self-loops (i==i)
    spatial_set = {p for p in spatial_set if p[0] != p[1]}
    
    # Create set of bonded pairs (we need to check both i-j and j-i directions
    # because 'spatial_set' is sorted i<j, but bonds might not be)
    bond_set = set(map(tuple, local_bonds[mask]))
    reversed_bond_set = {(b, a) for a, b in bond_set}
    all_bonded_pairs = bond_set.union(reversed_bond_set)
    
    # SUBTRACTION: Pure Spatial = All Spatial - Bonded
    pure_spatial_set = spatial_set - all_bonded_pairs
    
    # Convert back to tensor
    if len(pure_spatial_set) > 0:
        spatial_edges = np.array(list(pure_spatial_set))
        spatial_index = torch.tensor(spatial_edges.T, dtype=torch.long)
        
        # Make bidirectional
        spatial_index = torch.cat([spatial_index, spatial_index.flip(0)], dim=1)
    else:
        spatial_index = torch.empty((2, 0), dtype=torch.long)

    print(f"  Bonded Edges: {bond_index.shape[1]}")
    print(f"  Spatial Edges: {spatial_index.shape[1]}")

    # 5. Node Features
    # Feature 1: Bead Type Encoding
    le = LabelEncoder()
    bead_types = le.fit_transform(lipids.names)
    x = torch.tensor(bead_types, dtype=torch.long).unsqueeze(1)
    
    # Feature 2: Position
    pos = torch.tensor(lipids.positions, dtype=torch.float)

    # 6. Build HeteroData Object
    data = HeteroData()
    
    # Add Nodes
    data['bead'].x = x
    data['bead'].pos = pos
    data['bead'].num_nodes = n_nodes
    
    # Add Bonded Edges
    data['bead', 'bonded', 'bead'].edge_index = bond_index
    
    # Add Spatial Edges
    data['bead', 'spatial', 'bead'].edge_index = spatial_index
    
    # Optional: Add distances as edge attributes
    # You would need to recalculate distances for specific indices if needed
    
    return data, le


def main():
    pass


if __name__ == "__main__":
    # Example Usage
    # Ensure you have your .tpr and .gro files
    # data, encoder = create_hetero_lipid_graph("box.gro", "topol.tpr", spatial_cutoff=11.0)
    
    # Mocking data structure for demonstration
    main()
