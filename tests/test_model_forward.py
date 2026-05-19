import pytest
import torch
from torch_geometric.data import HeteroData, Batch

from lipid_gnn.config import CONFIG
from lipid_gnn.membrane_prop_gnn import MembranePropertyGNN


def create_mock_hetero_data():
    """Creates a single mock HeteroData object for testing."""
    data = HeteroData()
    data['bead'].x = torch.randn(4, CONFIG.model.in_channels)
    data['bead'].num_nodes = 4

    data['bead', 'bonded', 'bead'].edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)
    data['bead', 'bonded', 'bead'].edge_attr = torch.randn(4, CONFIG.model.bonded_edge_attr_dim)

    data['bead', 'spatial', 'bead'].edge_index = torch.tensor([[0, 2], [2, 0]], dtype=torch.long)
    data['bead', 'spatial', 'bead'].edge_attr = torch.randn(2, CONFIG.model.spatial_edge_attr_dim)

    data.y = torch.tensor([[0.5]], dtype=torch.float)
    return data


@pytest.mark.parametrize("batch_size", [1, 2])
def test_model_forward(batch_size):
    """Forward + backward smoke test on a batched HeteroData."""
    model = MembranePropertyGNN(
        in_channels=CONFIG.model.in_channels,
        hidden_dim=32,
        num_layers=2,
        out_dim=1,
    )

    batch = Batch.from_data_list([create_mock_hetero_data() for _ in range(batch_size)])

    out = model(
        batch.x_dict,
        batch.edge_index_dict,
        batch.batch_dict,
        batch.edge_attr_dict,
    )

    assert out.shape == (batch_size, 1)
    assert not torch.isnan(out).any()

    out.mean().backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Gradient not found for {name}"
