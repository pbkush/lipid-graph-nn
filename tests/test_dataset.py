import os
import shutil
import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from lipid_gnn.dataset import MartiniDiskDataset

# Helper to create synthetic HeteroData
def create_synthetic_graph(node_val):
    data = HeteroData()
    data['bead'].x = torch.tensor([[node_val]], dtype=torch.float)
    data.y = torch.tensor([[node_val * 2]], dtype=torch.float)
    return data

@pytest.fixture
def dummy_processed_dir(tmpdir):
    proc_dir = os.path.join(tmpdir, "processed")
    os.makedirs(proc_dir)
    return proc_dir

def test_martini_disk_dataset_no_shuffle(dummy_processed_dir):
    """Test sequential retrieval of chunked graphs."""
    # Create 3 chunks with 2 graphs each (total 6 graphs)
    chunk_files = []
    val = 0
    for chunk_idx in range(3):
        chunk_graphs = []
        for _ in range(2):
            chunk_graphs.append(create_synthetic_graph(val))
            val += 1
        path = os.path.join(dummy_processed_dir, f"chunk_{chunk_idx}.pt")
        torch.save(chunk_graphs, path)
        chunk_files.append(path)
        
    # Initialize dataset
    dataset = MartiniDiskDataset(chunk_files, shuffle=False)
    
    # Check that iteration yields graphs sequentially
    collected_vals = []
    for graph in dataset:
        collected_vals.append(graph['bead'].x.item())
        
    assert collected_vals == [0, 1, 2, 3, 4, 5], "Graphs were not yielded in exact sequential order"

def test_martini_disk_dataset_shuffle(dummy_processed_dir):
    """Test that shuffle=True randomizes chunk order and intra-chunk items."""
    # Create 10 chunks with 10 graphs each
    chunk_files = []
    for chunk_idx in range(10):
        # We assign a unique ID per graph to test intra-chunk shuffling
        chunk_graphs = [create_synthetic_graph(chunk_idx * 10 + i) for i in range(10)]
        path = os.path.join(dummy_processed_dir, f"chunk_{chunk_idx}.pt")
        torch.save(chunk_graphs, path)
        chunk_files.append(path)
        
    dataset = MartiniDiskDataset(chunk_files, shuffle=True)
    
    # Check chunk shuffling
    first_vals = []
    for _ in range(5):
        dataset_inst = MartiniDiskDataset(chunk_files, shuffle=True)
        it = iter(dataset_inst)
        first_graph = next(it)
        first_vals.append(first_graph['bead'].x.item() // 10) # Get chunk index
    
    # It's highly unlikely that 5 consecutive randomized dataset instantiations begin with chunk 0
    assert not all(v == 0 for v in first_vals), "Shuffle=True did not randomize chunk order"

    # Check intra-chunk shuffling
    first_chunk_graphs = None
    for chunk_file in dataset.chunk_files:
        if "chunk_0.pt" in chunk_file:
            break
            
    # We can just iterate once and check if sequentially generated items 
    # (e.g. 0 to 9) appear out of order within a chunk.
    collected_all = []
    for graph in dataset:
        collected_all.append(graph['bead'].x.item())
        
    # Check the first 10 items (which correspond to whatever random chunk was loaded first)
    first_10 = collected_all[:10]
    expected_sorted = sorted(first_10)
    assert first_10 != expected_sorted, "Intra-chunk graphs were not shuffled"

def test_martini_disk_dataset_dataloader_multiworker(dummy_processed_dir):
    """Test multi-processing compatibility with PyG DataLoader."""
    chunk_files = []
    val = 0
    for chunk_idx in range(4):
        chunk_graphs = []
        for _ in range(5):
            chunk_graphs.append(create_synthetic_graph(val))
            val += 1
        path = os.path.join(dummy_processed_dir, f"chunk_{chunk_idx}.pt")
        torch.save(chunk_graphs, path)
        chunk_files.append(path)
        
    dataset = MartiniDiskDataset(chunk_files, shuffle=False)
    
    # PyG DataLoader
    loader = DataLoader(dataset, batch_size=5, num_workers=2)
    
    collected_vals = []
    for batch in loader:
        collected_vals.extend(batch['bead'].x.flatten().tolist())
        
    # Since num_workers > 0, PyTorch DataLoader launches workers that fetch subsets of files.
    # All items should eventually be fetched.
    collected_vals.sort()
    assert collected_vals == list(range(20)), f"Multi-worker DataLoader fetched incorrectly: {collected_vals}"
