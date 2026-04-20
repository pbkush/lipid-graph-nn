import os
import numpy as np
import torch
from unittest.mock import MagicMock, patch

from lipid_gnn.dataset import preprocess_and_save


@patch('lipid_gnn.dataset.MartiniHeteroGraphBuilder')
@patch('lipid_gnn.dataset.pkl_load')
def test_preprocess_and_save_frame_sampling(mock_pkl_load, mock_builder_class, tmp_path):
    """
    Verifies that preprocess_and_save samples the correct frame indices from each
    system using linspace and calls process_frame accordingly.
    """
    mock_pkl_load.return_value = ({'lipid_packing': 0.5}, None)

    mock_builder = MagicMock()
    mock_builder.u.trajectory.n_frames = 100
    mock_builder_class.return_value = mock_builder

    from torch_geometric.data import HeteroData
    mock_data = HeteroData()
    mock_data['bead'].x = torch.randn(1, 4)
    mock_builder.process_frame.return_value = mock_data

    num_frames = 10
    sim_tuples = [
        ("sys1/topol.tpr", "sys1/traj.xtc", "sys1/props.pkl"),
        ("sys2/topol.tpr", "sys2/traj.xtc", "sys2/props.pkl"),
    ]

    saved = preprocess_and_save(
        sim_tuples=sim_tuples,
        processed_dir=str(tmp_path),
        target_properties=['lipid_packing'],
        num_frames=num_frames,
        chunk_size=100,
    )

    # 2 systems × 10 frames = 20 graphs total, all in one chunk
    assert len(saved) == 1
    graphs = torch.load(saved[0], weights_only=False)
    assert len(graphs) == 20

    # Verify process_frame was called with the correct linspace indices for each system
    expected_indices = np.linspace(0, 99, num_frames, dtype=int)
    called_indices = [call.kwargs['frame_idx'] for call in mock_builder.process_frame.call_args_list]

    for idx in expected_indices:
        assert called_indices.count(int(idx)) == 2, (
            f"Frame {idx} should be sampled once per system (2 times total)"
        )


@patch('lipid_gnn.dataset.MartiniHeteroGraphBuilder')
@patch('lipid_gnn.dataset.pkl_load')
def test_preprocess_and_save_interleaves_systems(mock_pkl_load, mock_builder_class, tmp_path):
    """
    With interleave=True, early chunks must contain graphs from more than one system.
    Guards against the regression where each chunk held only one system's (identical-y)
    frames, which caused per-batch target variance to collapse and training MSE to
    plateau at the dataset mean.
    """
    def _pkl_side_effect(path, verbose=False):
        return (({'lipid_packing': 0.1}, None)
                if 'sys1' in str(path)
                else ({'lipid_packing': 0.9}, None))
    mock_pkl_load.side_effect = _pkl_side_effect

    mock_builder = MagicMock()
    mock_builder.u.trajectory.n_frames = 100
    mock_builder_class.return_value = mock_builder

    from torch_geometric.data import HeteroData
    def _fresh_graph(frame_idx):
        d = HeteroData()
        d['bead'].x = torch.randn(1, 4)
        return d
    mock_builder.process_frame.side_effect = _fresh_graph

    sim_tuples = [
        ("sys1/topol.tpr", "sys1/traj.xtc", "sys1/props.pkl"),
        ("sys2/topol.tpr", "sys2/traj.xtc", "sys2/props.pkl"),
    ]

    saved = preprocess_and_save(
        sim_tuples=sim_tuples,
        processed_dir=str(tmp_path),
        target_properties=['lipid_packing'],
        num_frames=10,
        chunk_size=5,         # small chunk → forces the interleaving to show up per chunk
        interleave=True,
        shuffle_seed=0,
    )

    assert len(saved) >= 1
    first = torch.load(saved[0], weights_only=False)
    y_values = {float(g.y.item()) for g in first}
    assert len(y_values) > 1, (
        f"Expected mixed systems in first chunk (targets 0.1 and 0.9), got {y_values}. "
        "Chunks appear system-homogeneous; interleaving did not take effect."
    )


if __name__ == "__main__":
    try:
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as tmp:
            test_preprocess_and_save_frame_sampling(
                MagicMock(), MagicMock(), pathlib.Path(tmp)
            )
        print("Test Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
