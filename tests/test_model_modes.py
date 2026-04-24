import torch
import pytest
from torch_geometric.data import HeteroData, Batch

from lipid_gnn.config import CONFIG
from lipid_gnn.membrane_prop_gnn import MembranePropertyGNN

def create_mock_hetero_data():
    """Creates a single mock HeteroData object for testing."""
    data = HeteroData()
    # 4 nodes, in_channels features each
    data['bead'].x = torch.randn(4, CONFIG.model.in_channels)
    data['bead'].num_nodes = 4

    # Bonded edges (bidirectional)
    data['bead', 'bonded', 'bead'].edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    data['bead', 'bonded', 'bead'].edge_attr = torch.randn(4, CONFIG.model.bonded_edge_attr_dim)

    # Spatial edges (subset)
    data['bead', 'spatial', 'bead'].edge_index = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
    data['bead', 'spatial', 'bead'].edge_attr = torch.randn(2, CONFIG.model.spatial_edge_attr_dim)

    # Composition vector (all lipid types in the config vocabulary)
    data.comp_vec = torch.randn(CONFIG.vocab.lipid_comp_dim)
    
    # Target
    data.y = torch.tensor([[0.5]], dtype=torch.float)
    
    return data

@pytest.mark.parametrize("comp_mode, batch_size", [
    ("gnn_only", 1),
    ("gnn_only", 2),
    ("gnn_plus_comp", 1),
    ("gnn_plus_comp", 2),
    ("comp_only", 1),
    ("comp_only", 2),
])
def test_model_forward_modes(comp_mode, batch_size):
    """Verifies that the model forward pass works for all three Phase 1 modes."""
    # Setup model parameters
    hidden_dim = 32
    comp_dim = CONFIG.vocab.lipid_comp_dim if comp_mode in ["gnn_plus_comp", "comp_only"] else 0

    model = MembranePropertyGNN(
        in_channels=CONFIG.model.in_channels,
        hidden_dim=hidden_dim,
        num_layers=2,
        out_dim=1,
        comp_dim=comp_dim
    )
    
    # Create batch
    data_list = [create_mock_hetero_data() for _ in range(batch_size)]
    batch = Batch.from_data_list(data_list)
    
    # Prepare inputs for model
    x_dict = batch.x_dict
    edge_index_dict = batch.edge_index_dict
    batch_dict = batch.batch_dict
    edge_attr_dict = batch.edge_attr_dict
    
    # Handlers for mode differences
    if comp_mode == "gnn_only":
        comp_vec = None
    else:
        # PyG Batching stacks global attributes automatically into [BatchSize, Dim]
        comp_vec = batch.comp_vec
        
    # Forward Pass
    try:
        out = model(x_dict, edge_index_dict, batch_dict, edge_attr_dict, comp_vec=comp_vec)
    except RuntimeError as e:
        pytest.fail(f"Model failed in {comp_mode} mode with batch_size={batch_size}: {e}")
        
    # Assertions
    assert out.shape == (batch_size, 1)
    assert not torch.isnan(out).any()
    
    # Backprop test
    loss = out.mean()
    loss.backward()
    
    # Ensure gradients reached the model
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient not found for {name}"
