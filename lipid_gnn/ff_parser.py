import os
import json
from pathlib import Path

def parse_martini_itp(itp_file_path):
    """
    Parses a Martini 3 .itp file to extract:
    - Mass
    - Charge
    - Self-interaction Sigma
    - Self-interaction Epsilon
    for each unique bead type.
    
    Returns:
        dict: A mapping from bead_name (str) to physical parameters (dict).
    """
    ff_params = {}
    
    in_atomtypes = False
    in_nonbond = False
    
    with open(itp_file_path, 'r') as f:
        for line in f:
            # Clean up inline comments and whitespace
            line = line.split(';')[0].strip()
            if not line:
                continue
                
            # Detect section changes
            if line.startswith('['):
                section = line.replace('[', '').replace(']', '').strip()
                in_atomtypes = (section == 'atomtypes')
                in_nonbond = (section == 'nonbond_params')
                continue
                
            if in_atomtypes:
                parts = line.split()
                if len(parts) >= 3:
                    bead_type = parts[0]
                    try:
                        mass = float(parts[1])
                        charge = float(parts[2])
                        ff_params[bead_type] = {
                            'mass': mass,
                            'charge': charge,
                            'sigma': 0.0,
                            'epsilon': 0.0
                        }
                    except ValueError:
                        pass
                        
            elif in_nonbond:
                parts = line.split()
                if len(parts) >= 5:
                    b1 = parts[0]
                    b2 = parts[1]
                    # We are only interested in self-interactions (b1 == b2) for baseline properties
                    if b1 == b2 and b1 in ff_params:
                        try:
                            sigma = float(parts[3])
                            epsilon = float(parts[4])
                            ff_params[b1]['sigma'] = sigma
                            ff_params[b1]['epsilon'] = epsilon
                        except ValueError:
                            pass
                            
    return ff_params

def parse_molecule_itp(itp_file_path):
    """
    Parses a Martini 3 molecule .itp file to extract bond topology.
    Returns:
        tuple: (mol_edge_dict, mol_node_dict) 
               mol_edge_dict maps molecule_name -> bonds dict.
               mol_node_dict maps molecule_name -> atom_name -> bead_type dict.
    """
    mol_dict = {}
    node_dict = {}
    current_mol = None
    
    in_moleculetype = False
    in_atoms = False
    in_bonds = False
    
    atom_id_to_name = {}
    
    with open(itp_file_path, 'r') as f:
        for line in f:
            line = line.split(';')[0].strip()
            if not line:
                continue
                
            if line.startswith('['):
                section = line.replace('[', '').replace(']', '').strip()
                in_moleculetype = (section == 'moleculetype')
                in_atoms = (section == 'atoms')
                in_bonds = (section == 'bonds')
                
                # Turn off other section triggers for safety if [angles] etc follow
                if section not in ['moleculetype', 'atoms', 'bonds']:
                    in_moleculetype = in_atoms = in_bonds = False
                continue
                
            if in_moleculetype:
                parts = line.split()
                if len(parts) >= 1:
                    current_mol = parts[0]
                    mol_dict[current_mol] = {}
                    node_dict[current_mol] = {}
                    atom_id_to_name = {} # Reset for new molecule
                    in_moleculetype = False # Only read the first line after [moleculetype]
                    
            elif in_atoms:
                parts = line.split()
                if len(parts) >= 5 and current_mol:
                    atom_id = parts[0]
                    bead_type = parts[1]
                    atom_name = parts[4]
                    atom_id_to_name[atom_id] = atom_name
                    node_dict[current_mol][atom_name] = bead_type
                    
            elif in_bonds:
                parts = line.split()
                if len(parts) >= 5 and current_mol:
                    b1_id = parts[0]
                    b2_id = parts[1]
                    try:
                        length = float(parts[3])
                        force = float(parts[4])
                        
                        b1_name = atom_id_to_name.get(b1_id)
                        b2_name = atom_id_to_name.get(b2_id)
                        
                        if b1_name and b2_name:
                            # Standardize by alphabetical sort to ensure edge orientation agnosticism
                            sorted_names = sorted([b1_name, b2_name])
                            bond_key = f"{sorted_names[0]}-{sorted_names[1]}"
                            mol_dict[current_mol][bond_key] = {
                                'length': length,
                                'force_constant': force
                            }
                    except ValueError:
                        pass
                        
    return mol_dict, node_dict

def create_ff_mapping(data_dir, output_json_path):
    """
    Finds a martini_v3.0.0.itp file in the dataset, parses it, and writes the mapping
    to a structured JSON file.
    """
    # Assuming standard project structure: we'll just pick the first valid DIPC100 folder's itp
    itp_file = Path(data_dir) / 'DIPC100/toppar/martini_v3.0.0.itp'
    
    if not itp_file.exists():
        raise FileNotFoundError(f"Could not find Martini topology file at: {itp_file}")
        
    print(f"Parsing force field parameters from: {itp_file}")
    ff_dict = parse_martini_itp(itp_file)
    
    with open(output_json_path, 'w') as f:
        json.dump(ff_dict, f, indent=4)
        
    print(f"Successfully extracted parameters for {len(ff_dict)} unique Martini bead types.")
    print(f"Saved physical bead mapping to: {output_json_path}")
    

def create_ff_edge_node_mapping(data_dir, output_edge_json_path, output_node_json_path):
    """
    Finds all molecule .itp files in the dataset, parses them, and writes both
    molecular bond physical edge properties and node bead-type mappings to JSON files.
    """
    itp_files = list(Path(data_dir).rglob("*.itp"))
    
    master_edge_dict = {}
    master_node_dict = {}
    
    for itp_file in itp_files:
        # Skip the absolute base file as it lacks [moleculetype] definitions
        if "martini_v3.0.0.itp" in itp_file.name:
            continue
            
        mol_dict, node_dict = parse_molecule_itp(itp_file)
        
        # Merge individual molecules into the master mapping databases
        for mol_name, bonds in mol_dict.items():
            if mol_name not in master_edge_dict:
                master_edge_dict[mol_name] = bonds
            else:
                master_edge_dict[mol_name].update(bonds)
                
        for mol_name, nodes in node_dict.items():
            if mol_name not in master_node_dict:
                master_node_dict[mol_name] = nodes
            else:
                master_node_dict[mol_name].update(nodes)
                
    with open(output_edge_json_path, 'w') as f:
        json.dump(master_edge_dict, f, indent=4)
        
    with open(output_node_json_path, 'w') as f:
        json.dump(master_node_dict, f, indent=4)
        
    print(f"Successfully extracted parameters for {len(master_edge_dict)} molecules.")
    print(f"Saved physical explicit molecule node/edge mappings relative to topological inputs.")
    
if __name__ == "__main__":
    from lipid_gnn.config import CONFIG
    data_dir = CONFIG.paths.data_dir
    out_file = CONFIG.paths.ff_params_file
    create_ff_mapping(data_dir, out_file)

    out_edge_file = CONFIG.paths.ff_edge_params_file
    out_node_file = CONFIG.paths.ff_node_mapping_file
    create_ff_edge_node_mapping(data_dir, out_edge_file, out_node_file)
