import os
import random as _random
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


@patch('lipid_gnn.dataset.MartiniHeteroGraphBuilder')
@patch('lipid_gnn.dataset.pkl_load')
def test_train_val_test_splits_are_disjoint(mock_pkl_load, mock_builder_class, tmp_path):
    """
    After a system-level train/val/test split, no membrane composition (identified
    by its unique y value) should appear in more than one split directory.
    Mirrors the split logic used by prepare_colab_subset.prepare_colab_subset().
    """
    n_systems = 6
    # Each system has a unique target so any overlap is detectable.
    targets = {f'sys{i}': round(i * 0.1, 1) for i in range(n_systems)}

    def _pkl_side_effect(path, verbose=False):
        for key, val in targets.items():
            if key in str(path):
                return ({'lipid_packing': val}, None)
        return ({'lipid_packing': 0.0}, None)
    mock_pkl_load.side_effect = _pkl_side_effect

    mock_builder = MagicMock()
    mock_builder.u.trajectory.n_frames = 10
    mock_builder_class.return_value = mock_builder

    from torch_geometric.data import HeteroData
    def _fresh_graph(frame_idx):
        d = HeteroData()
        d['bead'].x = torch.randn(1, 4)
        return d
    mock_builder.process_frame.side_effect = _fresh_graph

    all_sims = [
        (f"sys{i}/topol.tpr", f"sys{i}/traj.xtc", f"sys{i}/props.pkl")
        for i in range(n_systems)
    ]

    # Replicate the split logic from prepare_colab_subset (val_frac=0.15, test_frac=0.15)
    val_frac = test_frac = 0.15
    rng = _random.Random(0)
    shuffled = list(all_sims)
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_frac))
    n_val  = max(1, round(len(shuffled) * val_frac))
    test_sims  = shuffled[:n_test]
    val_sims   = shuffled[n_test:n_test + n_val]
    train_sims = shuffled[n_test + n_val:]

    assert len(train_sims) + len(val_sims) + len(test_sims) == n_systems

    for split_name, sims in [("train", train_sims), ("val", val_sims), ("test", test_sims)]:
        split_dir = tmp_path / split_name
        split_dir.mkdir()
        preprocess_and_save(
            sim_tuples=sims,
            processed_dir=str(split_dir),
            target_properties=['lipid_packing'],
            num_frames=5,
            chunk_size=50,
        )

    def _y_set(split_name):
        chunks = sorted((tmp_path / split_name).glob('chunk_*.pt'))
        ys = set()
        for c in chunks:
            for g in torch.load(c, weights_only=False):
                ys.add(round(float(g.y.item()), 2))
        return ys

    train_y = _y_set('train')
    val_y   = _y_set('val')
    test_y  = _y_set('test')

    assert train_y & val_y  == set(), f"Train/val overlap in y values: {train_y & val_y}"
    assert train_y & test_y == set(), f"Train/test overlap in y values: {train_y & test_y}"
    assert val_y   & test_y == set(), f"Val/test overlap in y values: {val_y & test_y}"


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
