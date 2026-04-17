
from pathlib import Path

import pytest

from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder


@pytest.fixture(scope="session")
def data_dir_path():
    return Path("/home/phillip/Goethe/Thesis/lipid-graph-nn/data/membrane_only/POPC100")


@pytest.fixture(scope="session")
def ff_params_file():
    return Path("/home/phillip/Goethe/Thesis/lipid-graph-nn/resources/martini_ff_params.json")


@pytest.fixture(scope="session")
def ff_edge_params_file():
    return Path("/home/phillip/Goethe/Thesis/lipid-graph-nn/resources/martini_ff_edge_params.json")

@pytest.fixture(scope="session")
def ff_node_mapping_file():
    return Path("/home/phillip/Goethe/Thesis/lipid-graph-nn/resources/martini_ff_node_mapping.json")


@pytest.fixture(scope="session")
def test_file_paths(data_dir_path):
    run_dir_path = data_dir_path / "run"
    tpr_file_path = run_dir_path / "prun.tpr"
    traj_file_path = run_dir_path / "prun.xtc"
    return tpr_file_path, traj_file_path


@pytest.fixture(scope="session")
def hetero_graph(test_file_paths, ff_params_file, ff_edge_params_file, ff_node_mapping_file):
    cutoff_angstroms = 11.0
    builder = MartiniHeteroGraphBuilder(
        topology_file=test_file_paths[0],
        trajectory_file=test_file_paths[1],
        spatial_cutoff=cutoff_angstroms,
        ff_params_path=str(ff_params_file),
        ff_edge_params_path=str(ff_edge_params_file),
        ff_node_mapping_path=str(ff_node_mapping_file)
    )
    data = builder.process_frame(frame_idx=1)
    return data


@pytest.fixture(scope="session")
def edges(hetero_graph):
    b_edges = hetero_graph['bead', 'bonded', 'bead'].edge_index
    s_edges = hetero_graph['bead', 'spatial', 'bead'].edge_index
    return b_edges, s_edges


@pytest.fixture(scope="session")
def edges_sets(edges):
    b_set = set(map(tuple, edges[0].T.tolist()))
    s_set = set(map(tuple, edges[1].T.tolist()))
    return b_set, s_set

def test_node_features_shape(hetero_graph):
    # Node features should have 4 dimensions (Mass, Charge, Sigma, Epsilon)
    assert hetero_graph["bead"].x.shape[1] == 4


def test_edge_exist(edges):
    # Check if edges exist in the system.
    assert edges[0].shape[1] > 0
    assert edges[1].shape[1] > 0


def test_mutual_exclusion(edges_sets):
    intersection = edges_sets[0].intersection(edges_sets[1])
    assert len(intersection) == 0



def main():
    pass


if __name__ == "__main__":
    main()
