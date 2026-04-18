import os
import glob
import math
import gc
import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import IterableDataset

try:
    from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder
except ImportError:
    pass

try:
    from lipid_gnn.functions_emil.functions import pkl_load
except ImportError:
    pass


class MartiniDiskDataset(IterableDataset):
    """
    An IterableDataset that loads chunks of PyTorch Geometric HeteroData objects from disk.
    If shuffle=True, it randomizes the exact chunk loading order and randomizes graphs within
    the loaded chunk, seamlessly allowing PyTorch DataLoader to scale prefetching without OOMs.
    """
    def __init__(self, chunk_files, shuffle=False):
        super().__init__()
        self.chunk_files = list(chunk_files)
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        
        if worker_info is None:
            files_to_process = list(self.chunk_files)
        else:
            # Split chunk files automatically for multi-worker prefetching
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
            # Calculate exactly how many items each worker gets
            # Better distribution math:
            per_worker = int(math.ceil(len(self.chunk_files) / float(num_workers)))
            files_to_process = self.chunk_files[worker_id * per_worker : min((worker_id + 1) * per_worker, len(self.chunk_files))]
            
        if self.shuffle:
            random.shuffle(files_to_process)
            
        for chunk_file in files_to_process:
            graphs = torch.load(chunk_file, weights_only=False)
            
            # Graphs within the chunk should be randomized if shuffle is enabled
            # so we don't return sequentially homogeneous data chunks
            if self.shuffle:
                random.shuffle(graphs)
                
            for graph in graphs:
                yield graph


def preprocess_and_save(sim_tuples,
                        processed_dir,
                        target_properties,
                        num_frames=10,
                        chunk_size=50,
                        spatial_cutoff=9.0,
                        ff_params_path=None,
                        ff_edge_params_path=None,
                        ff_node_mapping_path=None):
    """
    Builds HeteroData graphs from Martini trajectories and saves them as chunked .pt files.

    Args:
        sim_tuples: List of (tpr_path, xtc_path, props_h5_path) for each system.
        processed_dir: Directory where chunk_*.pt files are written.
        target_properties: List of property names to store as graph.y.
            Available properties: 'lipid_packing', 'thickness', 'thickness_std',
            'compressibility', 'persistence', 'diffusivity'.
        num_frames: Number of evenly-spaced frames to sample per system.
        chunk_size: Number of graphs per .pt chunk file.
        spatial_cutoff: Distance cutoff (Angstrom) for spatial edges.
        ff_params_path: Path to martini_ff_params.json.
        ff_edge_params_path: Path to martini_ff_edge_params.json.
        ff_node_mapping_path: Path to martini_ff_node_mapping.json.

    Returns:
        List of paths to saved chunk files.
    """
    if not sim_tuples:
        print("Warning: sim_tuples is empty, nothing to process.")
        return []

    os.makedirs(processed_dir, exist_ok=True)

    stale_chunks = sorted(glob.glob(os.path.join(processed_dir, "chunk_*.pt")))
    if stale_chunks:
        print(f"Removing {len(stale_chunks)} stale chunk file(s) from {processed_dir}")
        for path in stale_chunks:
            os.remove(path)

    # Validate target_properties against the first available props file
    first_props = sim_tuples[0][2]
    mean_dict_probe, _ = pkl_load(first_props, verbose=False)
    missing = [p for p in target_properties if p not in mean_dict_probe]
    if missing:
        available = sorted(mean_dict_probe.keys())
        raise ValueError(
            f"Requested properties {missing} not found in {first_props}. "
            f"Available: {available}"
        )

    current_chunk = []
    chunk_index = 0
    total_graphs = 0
    saved_chunks = []

    for tpr_path, xtc_path, props_path in tqdm(sim_tuples, desc="Preprocessing to disk"):
        tpr_path, xtc_path, props_path = Path(tpr_path), Path(xtc_path), Path(props_path)

        builder = MartiniHeteroGraphBuilder(
            tpr_file=str(tpr_path),
            trajectory_file=str(xtc_path),
            spatial_cutoff=spatial_cutoff,
            ff_params_path=ff_params_path,
            ff_edge_params_path=ff_edge_params_path,
            ff_node_mapping_path=ff_node_mapping_path
        )

        n_frames = builder.u.trajectory.n_frames
        if n_frames <= num_frames:
            sampled_indices = range(n_frames)
        else:
            sampled_indices = np.linspace(0, n_frames - 1, num_frames, dtype=int)

        try:
            mean_dict, _ = pkl_load(props_path, verbose=False)
            target_vec = [mean_dict[prop] for prop in target_properties]

            for f_idx in sampled_indices:
                hetero_data = builder.process_frame(frame_idx=int(f_idx))
                hetero_data.y = torch.tensor([target_vec], dtype=torch.float)
                current_chunk.append(hetero_data)
                total_graphs += 1

                if len(current_chunk) >= chunk_size:
                    out_path = os.path.join(processed_dir, f"chunk_{chunk_index}.pt")
                    torch.save(current_chunk, out_path)
                    saved_chunks.append(out_path)
                    current_chunk = []
                    chunk_index += 1

            del builder
            gc.collect()
        except Exception as e:
            print(f"Error processing {tpr_path.name}: {e}")

    if len(current_chunk) > 0:
        out_path = os.path.join(processed_dir, f"chunk_{chunk_index}.pt")
        torch.save(current_chunk, out_path)
        saved_chunks.append(out_path)
        chunk_index += 1

    print(f"Saved {total_graphs} graphs across {chunk_index} chunk files in {processed_dir}")
    return saved_chunks
