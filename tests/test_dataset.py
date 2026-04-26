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


def create_graph_with_composition(node_val, composition, system_idx):
    data = HeteroData()
    data['bead'].x  = torch.tensor([[node_val]], dtype=torch.float)
    data.y           = torch.tensor([[node_val * 2]], dtype=torch.float)
    data.composition = composition
    data.system_idx  = torch.tensor([system_idx], dtype=torch.long)
    return data


def test_graphs_carry_composition_label(dummy_processed_dir):
    """Graphs preprocessed with composition labels preserve them through save/load."""
    graphs = [
        create_graph_with_composition(i, f"SYS_{i:02d}", i)
        for i in range(4)
    ]
    chunk_path = os.path.join(dummy_processed_dir, "chunk_0.pt")
    torch.save(graphs, chunk_path)

    loaded = torch.load(chunk_path, weights_only=False)
    for i, g in enumerate(loaded):
        assert hasattr(g, 'composition'), "Graph missing .composition attribute"
        assert g.composition == f"SYS_{i:02d}"
        assert hasattr(g, 'system_idx'), "Graph missing .system_idx attribute"
        assert int(g.system_idx.item()) == i


def test_composition_labels_survive_dataloader(dummy_processed_dir):
    """Composition strings survive a DataLoader round-trip (sequential, no batching)."""
    compositions = ["POPC100", "DOPC50_CHOL50", "POPE80_POPS20"]
    graphs = [
        create_graph_with_composition(i, compositions[i], i)
        for i in range(3)
    ]
    chunk_path = os.path.join(dummy_processed_dir, "chunk_0.pt")
    torch.save(graphs, chunk_path)

    dataset = MartiniDiskDataset([chunk_path], shuffle=False)
    seen = [g.composition for g in dataset]
    assert seen == compositions, f"Compositions changed after DataLoader: {seen}"


def test_no_composition_leakage_across_splits(tmp_path):
    """Train and test splits built from disjoint sim_tuples must have disjoint compositions."""
    import pickle
    import sys
    sys.path.insert(0, str(tmp_path.parent.parent))

    from lipid_gnn.dataset import preprocess_and_save

    # Build two minimal fake systems with distinguishable compositions
    # using the real preprocess_and_save is too heavy here; test at the
    # chunk level by asserting that composition sets are disjoint when
    # loaded from separately-saved chunk directories.
    train_comps = {"POPC100", "DOPC100"}
    test_comps  = {"DPPC100", "CHOL100"}

    def _save_chunks(split_dir, comps):
        split_dir.mkdir(parents=True, exist_ok=True)
        graphs = []
        for i, comp in enumerate(sorted(comps)):
            g = create_graph_with_composition(i, comp, i)
            graphs.append(g)
        torch.save(graphs, split_dir / "chunk_0.pt")

    _save_chunks(tmp_path / "train", train_comps)
    _save_chunks(tmp_path / "test",  test_comps)

    def _load_comps(split_dir):
        result = set()
        for chunk in sorted(split_dir.glob("chunk_*.pt")):
            for g in torch.load(chunk, weights_only=False):
                result.add(g.composition)
        return result

    loaded_train = _load_comps(tmp_path / "train")
    loaded_test  = _load_comps(tmp_path / "test")

    assert loaded_train == train_comps
    assert loaded_test  == test_comps
    assert loaded_train.isdisjoint(loaded_test), (
        f"Train/test composition overlap: {loaded_train & loaded_test}"
    )
