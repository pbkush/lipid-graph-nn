import gc
import os
import shutil
import time
import zipfile
from pathlib import Path

from lipid_gnn.dataset import preprocess_and_save
from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder

# Properties to embed as graph.y in every preprocessed graph.
# Available: 'lipid_packing', 'thickness', 'thickness_std',
#            'compressibility', 'persistence', 'diffusivity'
TARGET_PROPERTIES = ['lipid_packing', 'thickness']

NUM_FRAMES   = 100
CHUNK_SIZE   = 50
SPATIAL_CUTOFF = 10.0


def prepare_colab_subset(subset_name="colab_lipid_gnn_subset"):
    """
    Preprocesses Martini trajectories into chunked .pt graph files and bundles
    them with the lipid_gnn library into a zip for upload to Google Colab.

    Raw .tpr/.xtc files are NOT included in the output — all graph features and
    target properties are baked in at preprocessing time.
    """
    root_dir     = Path(__file__).resolve().parent.parent.parent
    data_dir     = root_dir / 'data/membrane_only'
    props_dir    = root_dir / 'results/properties'
    resources_dir = root_dir / 'resources'

    ff_params_path       = resources_dir / 'martini_ff_params.json'
    ff_edge_params_path  = resources_dir / 'martini_ff_edge_params.json'
    ff_node_mapping_path = resources_dir / 'martini_ff_node_mapping.json'

    dest_dir  = root_dir / subset_name
    proc_dest = dest_dir / 'processed'
    lib_dest  = dest_dir / 'lipid_gnn'

    for d in [proc_dest, lib_dest]:
        d.mkdir(parents=True, exist_ok=True)

    # --- Collect sim tuples -----------------------------------------------
    compositions = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(data_dir / d)
    )

    sim_tuples = []
    for comp in compositions:
        tpr = data_dir / comp / 'run/prun.tpr'
        xtc = data_dir / comp / 'run/prun.xtc'
        h5  = props_dir / f'{comp}.h5'
        if tpr.exists() and xtc.exists() and h5.exists():
            sim_tuples.append((tpr, xtc, h5))
        else:
            missing = [str(p) for p in [tpr, xtc, h5] if not p.exists()]
            print(f"Skipping {comp}: missing {missing}")

    if not sim_tuples:
        raise RuntimeError(f"No complete systems found in {data_dir}")

    print(f"\nFound {len(sim_tuples)} complete systems.")
    print(f"Target properties : {TARGET_PROPERTIES}")
    print(f"Frames per system : {NUM_FRAMES}")
    print(f"Chunk size        : {CHUNK_SIZE} graphs/chunk")
    print(f"Spatial cutoff    : {SPATIAL_CUTOFF} Å")

    # --- Time probe: one frame from the first system ----------------------
    print("\nProbing first frame to estimate total runtime...")
    tpr0, xtc0, _ = sim_tuples[0]
    builder_probe = MartiniHeteroGraphBuilder(
        topology_file=str(tpr0),
        trajectory_file=str(xtc0),
        spatial_cutoff=SPATIAL_CUTOFF,
        ff_params_path=ff_params_path,
        ff_edge_params_path=ff_edge_params_path,
        ff_node_mapping_path=ff_node_mapping_path,
    )
    t0 = time.time()
    builder_probe.process_frame(0)
    t_per_frame = time.time() - t0
    del builder_probe
    gc.collect()

    total_frames = len(sim_tuples) * NUM_FRAMES
    est_minutes  = (t_per_frame * total_frames) / 60
    print(f"Probe result      : {t_per_frame:.2f} sec/frame")
    print(f"Total graphs      : {total_frames}  "
          f"({len(sim_tuples)} systems × {NUM_FRAMES} frames)")
    print(f"Estimated runtime : ~{est_minutes:.0f} min\n")

    # --- Preprocess -------------------------------------------------------
    preprocess_and_save(
        sim_tuples=sim_tuples,
        processed_dir=proc_dest,
        target_properties=TARGET_PROPERTIES,
        num_frames=NUM_FRAMES,
        chunk_size=CHUNK_SIZE,
        spatial_cutoff=SPATIAL_CUTOFF,
        ff_params_path=ff_params_path,
        ff_edge_params_path=ff_edge_params_path,
        ff_node_mapping_path=ff_node_mapping_path,
    )

    # --- Bundle library ---------------------------------------------------
    print("Bundling lipid_gnn library...")
    if lib_dest.exists():
        shutil.rmtree(lib_dest)
    shutil.copytree(root_dir / 'lipid_gnn', lib_dest)

    # --- Zip (processed/ + lipid_gnn/ only) -------------------------------
    zip_path = root_dir / f"{subset_name}.zip"
    print(f"Creating archive: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for subdir in [proc_dest, lib_dest]:
            for root, _, files in os.walk(subdir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, root_dir)
                    zipf.write(file_path, arcname)

    print(f"Done! Upload {zip_path.name} to Google Colab.")


if __name__ == "__main__":
    prepare_colab_subset()
