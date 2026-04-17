import os
import torch
import numpy as np
from unittest.mock import MagicMock, patch
from scripts.training.run_sweep import load_data

@patch('scripts.training.run_sweep.os.listdir')
@patch('scripts.training.run_sweep.os.path.isdir')
@patch('scripts.training.run_sweep.os.path.exists')
@patch('scripts.training.run_sweep.MartiniHeteroGraphBuilder')
@patch('scripts.training.run_sweep.pkl_load')
def test_load_data_multiplicity(mock_pkl_load, mock_builder_class, mock_exists, mock_isdir, mock_listdir):
    """Verifies that load_data returns the correct number of graphs when using multi-frame augmentation."""
    
    # Setup mocks
    mock_listdir.return_value = ['comp1', 'comp2']
    mock_isdir.return_value = True
    mock_exists.return_value = True
    
    # Mock property loading
    mock_pkl_load.return_value = ({'lipid_packing': 0.5}, None)
    
    # Mock Builder and Trajectory
    mock_builder = MagicMock()
    mock_builder.u.trajectory.n_frames = 100
    mock_builder_class.return_value = mock_builder
    
    # Mock process_frame to return a simple HeteroData object
    from torch_geometric.data import HeteroData
    mock_data = HeteroData()
    mock_data['bead'].x = torch.randn(1, 4)
    mock_builder.process_frame.return_value = mock_data
    
    # Run load_data with 10 frames per comp
    num_frames = 10
    train_graphs, test_graphs = load_data(
        target_properties=['lipid_packing'],
        num_frames_per_comp=num_frames,
        test_size=0.5 # 1 comp in train, 1 in test
    )
    
    # total graphs should be 2 * 10 = 20
    assert len(train_graphs) + len(test_graphs) == 20
    assert len(train_graphs) == 10
    assert len(test_graphs) == 10
    
    # Verify that process_frame was called with correct indices
    # linspace(0, 99, 10) -> [0, 11, 22, 33, 44, 55, 66, 77, 88, 99]
    expected_indices = np.linspace(0, 99, num_frames, dtype=int)
    
    for idx in expected_indices:
        # Check if process_frame was called with this frame_idx
        found = False
        for call in mock_builder.process_frame.call_args_list:
            if call.kwargs.get('frame_idx') == int(idx):
                found = True
                break
        assert found, f"Frame {idx} was not sampled"

if __name__ == "__main__":
    # If run as script, use simple assertions
    try:
        test_load_data_multiplicity()
        print("Test Passed!")
    except Exception as e:
        print(f"Test Failed: {e}")
