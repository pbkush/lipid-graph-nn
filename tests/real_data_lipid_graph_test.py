
from pathlib import Path

import pytest

from lipid_gnn.config import CONFIG
from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder


@pytest.fixture(scope="session")
def data_dir_path():
    return CONFIG.paths.data_dir / CONFIG.dataset.reference_system


@pytest.fixture(scope="session")
def ff_params_file():
    return CONFIG.paths.ff_params_file


@pytest.fixture(scope="session")
def ff_edge_params_file():
    return CONFIG.paths.ff_edge_params_file


@pytest.fixture(scope="session")
def ff_node_mapping_file():
    return CONFIG.paths.ff_node_mapping_file


@pytest.fixture(scope="session")
def test_file_paths(data_dir_path):
    run_dir_path = data_dir_path / CONFIG.paths.trajectory_subdir
    tpr_file_path = run_dir_path / CONFIG.paths.topology_filename
    traj_file_path = run_dir_path / CONFIG.paths.trajectory_filename
    return tpr_file_path, traj_file_path


@pytest.fixture(scope="session")
def hetero_graph(test_file_paths, ff_params_file, ff_edge_params_file, ff_node_mapping_file):
    builder = MartiniHeteroGraphBuilder(
        tpr_file=test_file_paths[0],
        trajectory_file=test_file_paths[1],
        spatial_cutoff=CONFIG.dataset.spatial_cutoff,
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
    # Node features should match the continuous physics parameters in the config.
    assert hetero_graph["bead"].x.shape[1] == CONFIG.model.in_channels


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
