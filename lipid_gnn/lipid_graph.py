import fnmatch
from pathlib import Path

import MDAnalysis as mda
import torch
import numpy as np
from torch_geometric.data import HeteroData
from MDAnalysis.lib.distances import capped_distance
from sklearn.preprocessing import LabelEncoder


def create_global_encoder(base_dir, topology_pattern="*.tpr", selection="not (resname W or name NA or name CL)", exclude_patterns=['*min*', '*eq*', '*nvt*', '*npt*']):
    """
    Scans a directory recursively for topology files to build a global vocabulary
    of all possible bead types across multiple simulations.
    
    Args:
        base_dir (str): Root directory containing simulation subdirectories.
        topology_pattern (str): Glob pattern to find topology files (e.g., "*.tpr" or "topol.tpr").
        selection (str): MDAnalysis selection string to isolate relevant beads.
        exclude_patterns (list): Unix wildcard patterns to skip intermediate files. 
                                 Defaults to ignoring common MD intermediate steps.
                                 min, eq, nvt, npt
        
    Returns:
        LabelEncoder: A fitted scikit-learn LabelEncoder containing the global vocabulary.
    """
    print(f"Scanning '{base_dir}' for topologies matching '{topology_pattern}'...")
    all_bead_names = set()
    topology_files = list(Path(base_dir).rglob(topology_pattern))
    
    if not topology_files:
        raise ValueError(f"No files matching '{topology_pattern}' found in {base_dir}")
        
        
    filtered_topologies = []
    for top_file in topology_files:
        path_str = str(top_file).lower()
        # Only keep the file if it DOES NOT match any of the exclude patterns
        if not any(fnmatch.fnmatch(path_str, pat.lower()) for pat in exclude_patterns):
            filtered_topologies.append(top_file)
            
    # Safety fallback: If your prod runs also happen to be named 'eq' or similar,
    # and everything gets filtered out, we revert to parsing all files.
    if not filtered_topologies:
        print("WARNING: Exclude patterns filtered out ALL files. Reverting to parsing all files.")
        filtered_topologies = topology_files
        
    print(f"Filtered down to {len(filtered_topologies)} topology files to parse."  \
          " Started with {len(topology_files)}.")
        
    for top_file in filtered_topologies:
        try:
            # We only need the topology file to read bead names, no trajectory needed
            u = mda.Universe(str(top_file))
            lipids = u.select_atoms(selection)
            all_bead_names.update(lipids.names)
        except Exception as e:
            print(f"Skipping {top_file.name} due to read error: {e}")
            
    # Sorting ensures the encoding (0, 1, 2...) is perfectly deterministic
    # across different machines or operating systems.
    unique_beads = sorted(list(all_bead_names))
    print(f"Found {len(unique_beads)} unique bead types: {unique_beads}")
    
    global_encoder = LabelEncoder()
    global_encoder.fit(unique_beads)
    
    return global_encoder

class MartiniHeteroGraphBuilder:
    """
    An optimized, stateful builder for converting Martini MD trajectories 
    into PyTorch Geometric HeteroData objects.
    
    Caches static topology (bonds, node types) to make processing 
    multi-frame trajectories highly efficient.
    """
    def __init__(self, topology_file, trajectory_file, selection="not (resname W or name NA or name CL)", spatial_cutoff=11.0, encoder=None):
        print("Initializing MartiniGraphBuilder...")
        
        # 1. Load Universe (handles both static .gro or dynamic .xtc/.trr)
        self.u = mda.Universe(topology_file, trajectory_file)
        self.selection_string = selection
        self.spatial_cutoff = spatial_cutoff
        
        # 2. Isolate Selection
        self.lipids = self.u.select_atoms(self.selection_string)
        self.n_nodes = len(self.lipids)
        print(f"Tracking {self.n_nodes} beads out of {self.u.atoms.n_atoms} total.")
        
        # 3. Cache Node Features (Static)
        if encoder is None:
            print("WARNING: No global encoder provided. Fitting locally. This may break multi-simulation inference.")
            self.le = LabelEncoder()
            bead_types = self.le.fit_transform(self.lipids.names)
        else:
            self.le = encoder
            # Using transform() guarantees that bead ID '0' means the exact same 
            # physical bead type across all simulation graphs.
            try:
                bead_types = self.le.transform(self.lipids.names)
            except ValueError as e:
                raise ValueError(f"Encountered a bead type not present in the global encoder! {e}")
                
        self.node_x = torch.tensor(bead_types, dtype=torch.long).unsqueeze(1)
        
        # 4. Cache Index Mapping & Bonded Topology (Static)
        self._cache_topology()

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
        
        bond_index = torch.tensor(self.valid_local_bonds.T, dtype=torch.long)
        self.bond_index = torch.cat([bond_index, bond_index.flip(0)], dim=1)
        
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
        
        # 3. Filter Self-loops and Bonds
        spatial_set = set(map(tuple, pairs))
        spatial_set = {p for p in spatial_set if p[0] != p[1]} # Remove self loops
        pure_spatial_set = spatial_set - self.all_bonded_pairs_set # Remove bonds
        
        # 4. Create Spatial Edge Tensor
        if len(pure_spatial_set) > 0:
            spatial_edges = np.array(list(pure_spatial_set))
            spatial_index = torch.tensor(spatial_edges.T, dtype=torch.long)
            spatial_index = torch.cat([spatial_index, spatial_index.flip(0)], dim=1)
        else:
            spatial_index = torch.empty((2, 0), dtype=torch.long)
            
        # 5. Construct HeteroData Object
        data = HeteroData()
        
        # Assign Cached Static Data
        data['bead'].x = self.node_x
        data['bead'].num_nodes = self.n_nodes
        data['bead', 'bonded', 'bead'].edge_index = self.bond_index
        
        # Assign Dynamic Data
        data['bead'].pos = current_pos
        data['bead', 'spatial', 'bead'].edge_index = spatial_index
        
        return data

def main():
    pass


if __name__ == "__main__":
    # Example Usage
    # Ensure you have your .tpr and .gro files
    # data, encoder = create_hetero_lipid_graph("box.gro", "topol.tpr", spatial_cutoff=11.0)
    
    # Mocking data structure for demonstration
    main()
