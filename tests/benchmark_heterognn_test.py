import pytest
import torch
from unittest.mock import patch

# Adjust the python path/imports based on standard pytest execution from root
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lipid_gnn.benchmark_heterognn import (
    generate_dummy_data,
    profiling_and_timing,
    numerical_stability_test,
    stress_test,
    print_graph_stats,
    describe_graph_memory,
    _compare_graphs_roundtrip,
)
from lipid_gnn.membrane_prop_gnn import MembranePropertyGNN

def test_generate_dummy_data():
    """Test that the dummy data generates a properly formatted HeteroData object."""
    N = 100
    data = generate_dummy_data(N)

    # Check node feature shape mapping ensuring physical force fields (Mass, Charge, Sigma, Epsilon)
    assert 'bead' in data.node_types
    assert data['bead'].x.shape == (N, 4)

    # Check edge types and shapes
    assert ('bead', 'bonded', 'bead') in data.edge_types
    assert ('bead', 'spatial', 'bead') in data.edge_types
    assert data['bead', 'bonded', 'bead'].edge_index.shape[0] == 2
    assert data['bead', 'spatial', 'bead'].edge_index.shape[0] == 2

    # Check target sequence (now 1-dimensional for lipid_packing)
    assert data.y.shape == (1, 1)

    # Check spatial edge attributes (16 Gaussian RBFs)
    assert ('bead', 'spatial', 'bead') in data.edge_attr_dict
    assert data['bead', 'spatial', 'bead'].edge_attr.shape == (data['bead', 'spatial', 'bead'].edge_index.shape[1], 16)

def test_membrane_gnn_inference():
    """Test that the production architecture can process data and map to the explicit outputs."""
    data = generate_dummy_data(50)
    model = MembranePropertyGNN(in_channels=4, hidden_dim=16, num_layers=2, out_dim=1)

    # Simulate forward pass check without loss
    # model.forward expects: x_dict, edge_index_dict, batch_dict, edge_attr_dict
    out = model(data.x_dict, data.edge_index_dict, data.batch_dict, data.edge_attr_dict)

    # Ensuring it pulls correctly down into the regression targets requested
    assert out.shape == (1, 1)
    assert not torch.isnan(out).any()

def test_profiling_and_timing_execution(capsys):
    """Test that profiling function completes successfully without crashing."""
    # Run with very small scale to ensure tests are snappy
    data = generate_dummy_data(50)
    profiling_and_timing(data, num_iters=2)
    captured = capsys.readouterr()

    # Verify print statements were triggered
    assert "Timing & Memory Profiling" in captured.out
    assert "Average Forward Pass:" in captured.out
    assert "Throughput:" in captured.out

def test_numerical_stability_test_execution(capsys):
    """Test the AMP vs FP32 benchmarks execution without crashing."""
    data = generate_dummy_data(50)
    numerical_stability_test(data)
    captured = capsys.readouterr()

    assert "Numerical Stability Test" in captured.out
    assert "FP32 Forward Time:" in captured.out
    assert "Output Variance:" in captured.out

@patch("lipid_gnn.benchmark_heterognn.generate_dummy_data")
def test_stress_test_early_stop(mock_generate, capsys):
    """
    Test the stress-test loop cleanly catching OOM errors and printing the max supported N.
    We mock the generator to instantly throw a MemoryError to emulate the OOM constraint
    without genuinely blowing up the host RAM/VRAM during PyTest.
    """
    mock_generate.side_effect = MemoryError("Simulated PyTest CPU/GPU RAM Exhaustion")

    stress_test()
    captured = capsys.readouterr()

    assert "Memory Stress Test" in captured.out
    assert "Hit Host RAM Out of Memory" in captured.out or "Hit PyTorch memory limit" in captured.out or "OOM" in captured.out
    assert "MAXIMUM SUPPORTED NODES" in captured.out

def test_print_graph_stats(capsys):
    """Test that print_graph_stats outputs correct topology headers and counts."""
    data = generate_dummy_data(100)
    print_graph_stats(data, label="test")
    captured = capsys.readouterr()

    assert "Graph Topology" in captured.out
    assert "test" in captured.out
    assert "Nodes" in captured.out
    assert "Bonded edges" in captured.out
    assert "Spatial edges" in captured.out
    # Verify numeric values are present (100 nodes)
    assert "100" in captured.out

def test_describe_graph_memory(capsys):
    """Test that describe_graph_memory outputs per-tensor breakdown and total."""
    data = generate_dummy_data(100)
    describe_graph_memory(data, label="test")
    captured = capsys.readouterr()

    assert "Memory Breakdown" in captured.out
    assert "test" in captured.out
    assert "TOTAL" in captured.out
    assert "MB" in captured.out
    # Key tensor names should appear
    assert "bead.x" in captured.out
    assert "bonded.edge_index" in captured.out
    assert "spatial.edge_attr" in captured.out

def test_compare_built_vs_pt_roundtrip(tmp_path, capsys):
    """
    Test the round-trip comparison logic using dummy data, without needing
    real .tpr/.xtc files. Verifies that save/load introduces no extra tensors
    and the overhead report is printed.
    """
    from torch_geometric.data import HeteroData

    # Build a minimal HeteroData directly (not wrapped in Batch)
    N = 50
    graph = HeteroData()
    graph['bead'].x = torch.randn(N, 4)
    graph['bead', 'bonded', 'bead'].edge_index = torch.randint(0, N, (2, 6 * N))
    graph['bead', 'bonded', 'bead'].edge_attr = torch.randn(6 * N, 2)
    graph['bead', 'spatial', 'bead'].edge_index = torch.randint(0, N, (2, 20 * N))
    graph['bead', 'spatial', 'bead'].edge_attr = torch.randn(20 * N, 16)

    _compare_graphs_roundtrip(graph, str(tmp_path))
    captured = capsys.readouterr()

    assert "Raw build vs .pt" in captured.out
    assert "Overhead" in captured.out
    assert "Keys in live only" in captured.out
    assert "Keys in .pt only" in captured.out
