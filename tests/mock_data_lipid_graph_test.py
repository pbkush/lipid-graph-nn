
import os
import shutil

import torch
import pytest

from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder


@pytest.fixture(scope="session")
def gen_mock_data(tmp_path_factory):
    base_dir = "mock_data"
    base_path = tmp_path_factory.mktemp(base_dir)

    prod_dir_path = base_path / "prod"
    eq_dir_path = base_path / "eq"

    prod_dir_path.mkdir()
    eq_dir_path.mkdir()

    for dir_path in [prod_dir_path, eq_dir_path]:
        gro_path = dir_path / "mock.gro"
        pdb_path = dir_path / "mock.pdb"

        # Write .gro file (10x10x10 nm box)
        with open(gro_path, "w", encoding="utf-8") as f:
            f.write("Mock System for Testing\n")
            f.write("    5\n")
            # Lipid 1 (Near origin)
            f.write("    1LIP      A    1   0.100   0.100   0.100\n")
            f.write("    1LIP      B    2   0.100   0.100   0.500\n") # Distance 0.4nm from A
            # Lipid 2 (Near opposite edge to test PBC)
            f.write("    2LIP      A    3   9.900   9.900   9.900\n") 
            f.write("    2LIP      B    4   9.900   9.900   9.500\n") # Distance 0.4nm from A
            # Water (Should be filtered out)
            f.write("    3W        W    5   5.000   5.000   5.000\n")
            f.write("  10.00000  10.00000  10.00000\n")

        # Write self-contained .pdb file for mock topology
        with open(pdb_path, "w") as f:
            f.write("ATOM      1  A   LIP     1       1.000   1.000   1.000\n")
            f.write("ATOM      2  B   LIP     1       1.000   1.000   5.000\n")
            f.write("ATOM      3  A   LIP     2      99.000  99.000  99.000\n")
            f.write("ATOM      4  B   LIP     2      99.000  99.000  95.000\n")
            f.write("ATOM      5  W   W       3      50.000  50.000  50.000\n")
            f.write("CONECT    1    2\n")
            f.write("CONECT    3    4\n")
            f.write("END\n")

    return base_path, prod_dir_path, eq_dir_path


@pytest.fixture(scope="session")
def prod_gro_and_pdb_file_paths(gen_mock_data):
    gro_path = gen_mock_data[1] / "mock.gro"
    pdb_path = gen_mock_data[1] / "mock.pdb"
    return gro_path, pdb_path


@pytest.fixture(scope="session")
def ff_params_file(tmp_path_factory):
    import json
    path = tmp_path_factory.mktemp("mock_ff") / "mock_ff.json"
    with open(path, "w") as f:
        json.dump({"A": {"mass": 1.0, "charge": 0.0, "sigma": 0.0, "epsilon": 0.0},
                   "B": {"mass": 2.0, "charge": 0.0, "sigma": 0.0, "epsilon": 0.0},
                   "W": {"mass": 3.0, "charge": 0.0, "sigma": 0.0, "epsilon": 0.0}}, f)
    return str(path)


@pytest.fixture(scope="session")
def ff_edge_params_file(tmp_path_factory):
    import json
    path = tmp_path_factory.mktemp("mock_ff_edge") / "mock_ff_edge.json"
    with open(path, "w") as f:
        # PDB has molecule LIP containing A and B atom ties
        json.dump({"LIP": {"A-B": {"length": 0.45, "force_constant": 5000.0}}}, f)
    return str(path)

@pytest.fixture(scope="session")
def ff_node_mapping_file(tmp_path_factory):
    import json
    path = tmp_path_factory.mktemp("mock_ff_node") / "mock_ff_node.json"
    with open(path, "w") as f:
        # PDB has molecule LIP with atom names A, B. Map them securely back to physics params matching keys "A" and "B"
        json.dump({"LIP": {"A": "A", "B": "B"}}, f)
    return str(path)

def test_ff_params(ff_params_file):
    assert "mock_ff.json" in str(ff_params_file)


@pytest.fixture(scope="session")
def hetero_graph(prod_gro_and_pdb_file_paths, ff_params_file, ff_edge_params_file, ff_node_mapping_file):
    # Cutoff 5.0A to catch the PBC interaction between Lipids 1 and 2, but ignore water
    cutoff_angstroms = 5.0 
    builder = MartiniHeteroGraphBuilder(
        topology_file=prod_gro_and_pdb_file_paths[1],
        trajectory_file=prod_gro_and_pdb_file_paths[0],
        spatial_cutoff=cutoff_angstroms,
        ff_params_path=ff_params_file,
        ff_edge_params_path=ff_edge_params_file,
        ff_node_mapping_path=ff_node_mapping_file
    )
    data = builder.process_frame(frame_idx=0)
    return data


@pytest.fixture(scope="session")
def edges(hetero_graph):
    """Returns the bond and spatial edges of the hetero graph.

    :param hetero_graph: The hetero graph with bond and spatial edge types.
    :return: Bond and spatial edge indices.
    """
    bond_edges = hetero_graph["bead", "bonded", "bead"].edge_index
    spatial_edges = hetero_graph["bead", "spatial", "bead"].edge_index
    return bond_edges, spatial_edges


# Test graph invariants
def test_water_filter(hetero_graph):
    # Water should be filtered out. If not number of nodes will be 5.
    assert hetero_graph['bead'].num_nodes == 4


def test_bond_edges(edges):
    # Test if number of bonded edges are 4 (bidirectional).
    assert edges[0].shape[1] == 4


def test_edge_attr(hetero_graph):
    # Validate the continuous physics feature structure assigned to edge bounds
    edge_attr = hetero_graph["bead", "bonded", "bead"].edge_attr
    assert edge_attr is not None
    # 4 Bidirectional bonds arraying structure [E, 2]
    assert edge_attr.shape == (4, 2)
    # Check physics bounds are correctly retrieved
    assert edge_attr[0, 0] == 0.45   # target equilibrium length
    assert edge_attr[0, 1] == 5000.0 # harmonic force constraint

def test_spatial_edge_attr(hetero_graph):
    # Validate the RBF-encoded distances for spatial interactions
    edge_attr = hetero_graph["bead", "spatial", "bead"].edge_attr
    assert edge_attr is not None
    # Check shape [E, num_gaussians (16)]
    num_edges = hetero_graph["bead", "spatial", "bead"].edge_index.shape[1]
    assert edge_attr.shape == (num_edges, 16)
    # Ensure values are within standard RBF range [0, 1]
    assert (edge_attr >= 0.0).all() and (edge_attr <= 1.0).all()
    assert not torch.isnan(edge_attr).any()


def test_self_loops(edges):
    # There should not be any self loops.
    assert not (edges[0][0] == edges[0][1]).any()
    assert not (edges[1][0] == edges[1][1]).any()


@pytest.fixture(scope="session")
def edges_sets(edges):
    b_set = set(map(tuple, edges[0].T.tolist()))
    s_set = set(map(tuple, edges[1].T.tolist()))
    return b_set, s_set

def test_mutual_exclusion(edges_sets):
    # No edge should be in both sets of edges.
    intersection = edges_sets[0].intersection(edges_sets[1])
    assert len(intersection) == 0


def test_bidirectional_integrity(edges_sets):
    for edge_set in edges_sets:
        for u, v in edge_set:
            assert (v, u) in edge_set


def test_pbc(edges_sets):
    # Test the periodic boundary condition (PBC), i.e. the sim box has copies of itself around itself.
    # Test this by checking if atoms on both sides near edge see each other.
    assert len(edges_sets[1]) > 0


def test_ff_application(hetero_graph):
    # Check shape represents [N, in_channels (4)]
    assert hetero_graph["bead"].x.shape == (4, 4)



def test_real_membrane_system(real_data_dir="real_test_data"):
    """
    Integration test for a real Martini membrane (e.g., generated by insane.py).
    This tests parser compatibility, real-world edge densities, and performance.
    """
    if not os.path.exists(real_data_dir):
        print(f"\n⏭️  Skipping Real System Test: Directory '{real_data_dir}' not found.")
        print(f"   (To run this, create '{real_data_dir}/' and place a real .tpr and .gro file inside).")
        return

    print(f"\n--- Running Real Membrane System Test ({real_data_dir}) ---")
    try:
        import glob
        tpr_files = glob.glob(os.path.join(real_data_dir, "**", "*.tpr"), recursive=True)
        gro_files = glob.glob(os.path.join(real_data_dir, "**", "*.gro"), recursive=True)
        
        if not tpr_files or not gro_files:
            print("❌ Could not find both .tpr and .gro files in the real_test_data directory.")
            return
            
        test_tpr = tpr_files[0]
        test_gro = gro_files[0]

        ff_params_file = "/home/phillip/Goethe/Thesis/lipid-graph-nn/resources/martini_ff_params.json"
        ff_edge_params_file = "/home/phillip/Goethe/Thesis/lipid-graph-nn/resources/martini_ff_edge_params.json"
        ff_node_mapping_file = "/home/phillip/Goethe/Thesis/lipid-graph-nn/resources/martini_ff_node_mapping.json"
        
        # 3. Build Graph
        # Use full parameters including the new explicit edge mappings
        cutoff_angstroms = 11.0 # Standard Martini non-bonded cutoff (1.1 nm)
        builder = MartiniHeteroGraphBuilder(
            topology_file=test_tpr, 
            trajectory_file=test_gro, 
            spatial_cutoff=cutoff_angstroms,
            ff_params_path=ff_params_file,
            ff_edge_params_path=ff_edge_params_file,
            ff_node_mapping_path=ff_node_mapping_file
        )
        
        data = builder.process_frame(frame_idx=0)
        
        # 4. Real System Sanity Checks
        assert data['bead'].x.shape[1] == 4, "Node features should have 4 dimensions (mass, charge, sigma, epsilon)."
        
        # Check edge existence (a real membrane must have edges)
        b_edges = data['bead', 'bonded', 'bead'].edge_index
        s_edges = data['bead', 'spatial', 'bead'].edge_index
        assert b_edges.shape[1] > 0, "No bonded edges found in real system!"
        assert s_edges.shape[1] > 0, "No spatial edges found in real system!"
        
        # Check mutual exclusion on real data
        b_set = set(map(tuple, b_edges.T.tolist()))
        s_set = set(map(tuple, s_edges.T.tolist()))
        intersection = b_set.intersection(s_set)
        assert len(intersection) == 0, f"Overlap detected in real system! {len(intersection)} edges double-counted."
        
        print(f"✅ Real System Graph Built Successfully!")
        print(f"   Nodes: {data['bead'].num_nodes}")
        print(f"   Bonded Edges (Bidirectional): {b_edges.shape[1]}")
        print(f"   Spatial Edges (Bidirectional): {s_edges.shape[1]}")
        print(f"   Average Spatial Degree: {s_edges.shape[1] / data['bead'].num_nodes:.2f} neighbors/bead")

    except Exception as e:
        print(f"❌ Real System Test Failed: {e}")
        raise e


def main():
    pass


if __name__ == "__main__":
    main()
