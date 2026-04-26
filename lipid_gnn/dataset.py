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

from lipid_gnn.config import CONFIG

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
                        num_frames=None,
                        chunk_size=None,
                        spatial_cutoff=None,
                        interleave=None,
                        shuffle_seed=None,
                        ff_params_path=None,
                        ff_edge_params_path=None,
                        ff_node_mapping_path=None):
    """
    Builds HeteroData graphs from Martini trajectories and saves them as chunked .pt files.

    When `interleave=True` (default), frames from all systems are shuffled together before
    writing so each chunk contains graphs from many systems in random order. This gives
    heterogeneous per-batch targets at training time, which is required for the model to
    learn per-sample signal (otherwise MSE collapses to the dataset mean). When False,
    the old per-system-sequential ordering is used — retained for tests and debugging.

    Args:
        sim_tuples: List of (tpr_path, xtc_path, props_h5_path) for each system.
        processed_dir: Directory where chunk_*.pt files are written.
        target_properties: List of property names to store as graph.y.
            Available properties: 'lipid_packing', 'thickness', 'thickness_std',
            'compressibility', 'persistence', 'diffusivity'.
        num_frames: Number of evenly-spaced frames to sample per system.
        chunk_size: Number of graphs per .pt chunk file.
        spatial_cutoff: Distance cutoff (Angstrom) for spatial edges.
        interleave: If True, shuffle (system, frame) pairs globally so chunks mix systems.
        shuffle_seed: RNG seed used by the interleaving shuffle (deterministic output).
        ff_params_path: Path to martini_ff_params.json.
        ff_edge_params_path: Path to martini_ff_edge_params.json.
        ff_node_mapping_path: Path to martini_ff_node_mapping.json.

    Returns:
        List of paths to saved chunk files.
    """
    if not sim_tuples:
        print("Warning: sim_tuples is empty, nothing to process.")
        return []

    if num_frames is None:
        num_frames = CONFIG.dataset.num_frames
    if chunk_size is None:
        chunk_size = CONFIG.dataset.chunk_size
    if spatial_cutoff is None:
        spatial_cutoff = CONFIG.dataset.spatial_cutoff
    if interleave is None:
        interleave = CONFIG.dataset.interleave
    if shuffle_seed is None:
        shuffle_seed = CONFIG.dataset.shuffle_seed

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

    # Phase 1 — open all builders, pre-compute per-system frame indices and target vectors.
    builders = {}
    target_vecs = {}
    schedule = []  # flat list of (system_idx, frame_idx)

    for s_idx, (tpr_path, xtc_path, props_path) in enumerate(
        tqdm(sim_tuples, desc="Opening builders")
    ):
        tpr_path, xtc_path, props_path = Path(tpr_path), Path(xtc_path), Path(props_path)
        try:
            builder = MartiniHeteroGraphBuilder(
                tpr_file=str(tpr_path),
                trajectory_file=str(xtc_path),
                spatial_cutoff=spatial_cutoff,
                ff_params_path=ff_params_path,
                ff_edge_params_path=ff_edge_params_path,
                ff_node_mapping_path=ff_node_mapping_path,
            )
            n_frames = builder.u.trajectory.n_frames
            if n_frames <= num_frames:
                sampled_indices = list(range(n_frames))
            else:
                sampled_indices = [int(i) for i in np.linspace(0, n_frames - 1, num_frames, dtype=int)]

            mean_dict, _ = pkl_load(props_path, verbose=False)
            target_vec = [mean_dict[prop] for prop in target_properties]
        except Exception as e:
            print(f"Error opening {tpr_path.name}: {e}")
            continue

        builders[s_idx] = builder
        target_vecs[s_idx] = target_vec
        schedule.extend((s_idx, f) for f in sampled_indices)

    # Phase 2 — optionally shuffle the schedule so chunks mix systems.
    if interleave:
        rng = random.Random(shuffle_seed)
        rng.shuffle(schedule)

    # Phase 3 — stream-process schedule, writing chunk_*.pt as we go.
    current_chunk = []
    chunk_index = 0
    total_graphs = 0
    saved_chunks = []

    for s_idx, f_idx in tqdm(schedule, desc="Preprocessing to disk"):
        try:
            hetero_data = builders[s_idx].process_frame(frame_idx=f_idx)
        except Exception as e:
            print(f"Error processing system {s_idx} frame {f_idx}: {e}")
            continue

        hetero_data.y = torch.tensor([target_vecs[s_idx]], dtype=torch.float)
        # Composition label for post-hoc per-system error analysis.
        # Derived from the parent directory of the .tpr file:
        # data/membrane_only/POPC95_CHOL5/run/prun.tpr → "POPC95_CHOL5"
        tpr_path = sim_tuples[s_idx][0]
        hetero_data.composition = Path(tpr_path).parents[1].name
        hetero_data.system_idx  = torch.tensor([s_idx], dtype=torch.long)
        current_chunk.append(hetero_data)
        total_graphs += 1

        if len(current_chunk) >= chunk_size:
            out_path = os.path.join(processed_dir, f"chunk_{chunk_index}.pt")
            torch.save(current_chunk, out_path)
            saved_chunks.append(out_path)
            current_chunk = []
            chunk_index += 1

    if len(current_chunk) > 0:
        out_path = os.path.join(processed_dir, f"chunk_{chunk_index}.pt")
        torch.save(current_chunk, out_path)
        saved_chunks.append(out_path)
        chunk_index += 1

    # Release builders (and the underlying MDAnalysis Universes).
    builders.clear()
    gc.collect()

    print(f"Saved {total_graphs} graphs across {chunk_index} chunk files in {processed_dir}")
    return saved_chunks
