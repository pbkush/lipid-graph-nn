import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv, GATv2Conv, global_mean_pool, global_max_pool, GraphNorm

class MembranePropertyGNN(torch.nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, num_layers=3, out_dim=1, heads=4, comp_dim=0):
        """
        Args:
            in_channels (int): Number of continuous physical parameters per bead (e.g. 4 for mass, charge, sigma, epsilon).
            hidden_dim (int): Dimensionality of the node embeddings inside the network.
            num_layers (int): Number of message passing layers.
            out_dim (int): Number of target properties to predict (e.g., 1 for Area Per Lipid).
            heads (int): Number of parallel attention mechanisms for bonded interactions. 1 maintains VRAM scaling exactly identical to SAGEConv topologies.
            comp_dim (int): Size of the optional composition fraction vector to concatenate before the MLP head.
                            Set to 0 (default) for GNN-only mode. Set to LIPID_COMP_DIM (10) for GNN+comp mode.
        """
        super().__init__()
        self.comp_dim = comp_dim
        
        # 1. Node Embedding: Converts continuous physical vectors into dense hidden representations
        self.bead_embedding = nn.Linear(in_channels, hidden_dim)
        
        # 2. Heterogeneous Message Passing Layers & Normalization
        self.convs = nn.ModuleList()
        
        # GraphNorm normalizes node features over the graph with a learnable shift.
        # It is generally better than LayerNorm for graph-level prediction tasks because it explicitly considers graph size/structure.
        self.norms = nn.ModuleList()
        
        for _ in range(num_layers):
            conv = HeteroConv({
                # GATv2Conv enables natively computing multidimensional continuous parameters directly across attention mapping metrics
                ('bead', 'bonded', 'bead'): GATv2Conv((-1, -1), hidden_dim, edge_dim=2, heads=heads, concat=False, add_self_loops=False),
                # Spatial interactions now also use GATv2Conv to process RBF-encoded distances for density/packing awareness
                ('bead', 'spatial', 'bead'): GATv2Conv((-1, -1), hidden_dim, edge_dim=16, heads=heads, concat=False, add_self_loops=False),
            }, aggr='sum') # 'sum' aggregates the messages from both edge types
            self.convs.append(conv)
            self.norms.append(GraphNorm(hidden_dim))
            
        # 3. Readout / Prediction Head
        # MLP input = mean+max pooled graph repr + optional composition vector
        mlp_in = hidden_dim * 2 + comp_dim
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim)
        )

    def forward(self, x_dict, edge_index_dict, batch_dict, edge_attr_dict=None, comp_vec=None):
        """
        Forward pass for a batch of HeteroData graphs.
        
        Args:
            x_dict: Dictionary containing node features.
            edge_index_dict: Dictionary containing edge indices.
            batch_dict: Dictionary mapping nodes to their respective graphs in a batch.
            edge_attr_dict: Dictionary containing mapping variables evaluating continuous topological parameters natively!
            comp_vec: Optional composition fraction vector of shape [batch_size, comp_dim].
                      When provided, it is concatenated to the pooled graph representation
                      before the MLP head. Enables the GNN+comp comparison mode.
        """
        # Project physical node parameters [num_nodes, in_channels] -> [num_nodes, hidden_dim]
        x = self.bead_embedding(x_dict['bead'])
        
        # We re-pack it into a dictionary for HeteroConv
        h_dict = {'bead': x}
        
        # Message Passing
        for conv, norm in zip(self.convs, self.norms):
            # Pass edge attributes for both bonded and spatial if available
            if edge_attr_dict is not None:
                h_dict = conv(h_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
            else:
                h_dict = conv(h_dict, edge_index_dict)
                
            # Apply GraphNorm to stabilize continuous physics features
            h_dict['bead'] = norm(h_dict['bead'], batch_dict['bead'])
            h_dict['bead'] = F.relu(h_dict['bead'])
            
        # Extract the updated bead embeddings
        bead_h = h_dict['bead']
        bead_batch = batch_dict['bead'] # Tells PyG which nodes belong to which graph in the batch
        
        # Global Pooling (Graph-Level Readout)
        # Aggregates node features into a single vector per graph
        pool_mean = global_mean_pool(bead_h, bead_batch)
        pool_max = global_max_pool(bead_h, bead_batch)
        
        # Concatenate [Batch_size, hidden_dim] -> [Batch_size, hidden_dim * 2]
        graph_repr = torch.cat([pool_mean, pool_max], dim=1)

        # Optionally append composition fraction vector
        if comp_vec is not None and self.comp_dim > 0:
            # Reshape ensures [BatchSize, comp_dim] even if input was 1D [comp_dim]
            comp_vec = comp_vec.view(-1, self.comp_dim)
            graph_repr = torch.cat([graph_repr, comp_vec], dim=1)
        
        # Final Regression Prediction
        out = self.mlp(graph_repr)
        
        return out

# --- Example Usage ---
if __name__ == "__main__":
    from torch_geometric.data import Batch
    from lipid_graph import MartiniHeteroGraphBuilder
    
    # 1. Assume we built a few graphs using your builder
    # builder = MartiniHeteroGraphBuilder("topol.tpr", "traj.xtc")
    # graph1 = builder.process_frame(0)
    # graph2 = builder.process_frame(100)
    
    # Let's mock a batch of 2 heterogeneous graphs for testing the architecture
    from torch_geometric.data import HeteroData
    import numpy as np
    
    # Mock Graph 1
    g1 = HeteroData()
    g1['bead'].x = torch.randint(0, 10, (100, 1)) # 100 beads, 10 possible types
    g1['bead', 'bonded', 'bead'].edge_index = torch.randint(0, 100, (2, 50))
    g1['bead', 'bonded', 'bead'].edge_attr = torch.randn((50, 2), dtype=torch.float32)
    g1['bead', 'spatial', 'bead'].edge_index = torch.randint(0, 100, (2, 300))
    
    # Mock Graph 2
    g2 = HeteroData()
    g2['bead'].x = torch.randint(0, 10, (120, 1)) # 120 beads
    g2['bead', 'bonded', 'bead'].edge_index = torch.randint(0, 120, (2, 60))
    g2['bead', 'bonded', 'bead'].edge_attr = torch.randn((60, 2), dtype=torch.float32)
    g2['bead', 'spatial', 'bead'].edge_index = torch.randint(0, 120, (2, 350))
    
    # PyG Batching groups them into a single disconnected super-graph
    batch = Batch.from_data_list([g1, g2])
    
    # 2. Initialize Model
    # 10 unique bead types in our mock data
    model = MembranePropertyGNN(num_bead_types=10, hidden_dim=64, num_layers=3, out_dim=1)
    
    # 3. Forward Pass containing structural physical traits!
    edge_attributes = batch.edge_attr_dict if hasattr(batch, 'edge_attr_dict') else None
    predictions = model(batch.x_dict, batch.edge_index_dict, batch.batch_dict, edge_attributes)
    
    print(f"Batch contains {batch.num_graphs} graphs.")
    print(f"Output shape: {predictions.shape}") # Should be [2, 1] for 2 graphs, 1 property
    print(f"Predictions:\n{predictions.detach().numpy()}")