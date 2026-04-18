import argparse
import gc
import os
import shutil
import time
import zipfile
from pathlib import Path

from lipid_gnn.dataset import preprocess_and_save
from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder

AVAILABLE_PROPERTIES = [
    'lipid_packing', 'thickness', 'thickness_std', 'compressibility',
    'bending_modulus', 'persistence', 'diffusivity', 'variation'
]


def prepare_colab_subset(
    target_properties,
    num_frames,
    chunk_size,
    spatial_cutoff,
    subset_name="colab_lipid_gnn_subset",
):
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
    print(f"Target properties : {target_properties}")
    print(f"Frames per system : {num_frames}")
    print(f"Chunk size        : {chunk_size} graphs/chunk")
    print(f"Spatial cutoff    : {spatial_cutoff} Å")

    # --- Time probe: one frame from the first system ----------------------
    print("\nProbing first frame to estimate total runtime...")
    tpr0, xtc0, _ = sim_tuples[0]
    builder_probe = MartiniHeteroGraphBuilder(
        tpr_file=str(tpr0),
        trajectory_file=str(xtc0),
        spatial_cutoff=spatial_cutoff,
        ff_params_path=ff_params_path,
        ff_edge_params_path=ff_edge_params_path,
        ff_node_mapping_path=ff_node_mapping_path,
    )
    t0 = time.time()
    builder_probe.process_frame(0)
    t_per_frame = time.time() - t0
    del builder_probe
    gc.collect()

    total_frames = len(sim_tuples) * num_frames
    est_minutes  = (t_per_frame * total_frames) / 60
    print(f"Probe result      : {t_per_frame:.2f} sec/frame")
    print(f"Total graphs      : {total_frames}  "
          f"({len(sim_tuples)} systems × {num_frames} frames)")
    print(f"Estimated runtime : ~{est_minutes:.0f} min\n")

    # --- Preprocess -------------------------------------------------------
    preprocess_and_save(
        sim_tuples=sim_tuples,
        processed_dir=proc_dest,
        target_properties=target_properties,
        num_frames=num_frames,
        chunk_size=chunk_size,
        spatial_cutoff=spatial_cutoff,
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


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Martini trajectories into chunked .pt graph files and "
            "bundle them with the lipid_gnn library into a zip for Colab."
        )
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=["lipid_packing", "thickness"],
        choices=AVAILABLE_PROPERTIES,
        metavar="PROP",
        help=(
            "Properties to embed as graph.y "
            f"(default: lipid_packing thickness). "
            f"Available: {', '.join(AVAILABLE_PROPERTIES)}."
        ),
    )
    parser.add_argument(
        "--num-frames", type=int, default=50,
        help="Evenly-spaced frames sampled per system (default: 50).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50,
        help="Graphs per .pt chunk file (default: 50).",
    )
    parser.add_argument(
        "--spatial-cutoff", type=float, default=7.5,
        help="Spatial edge cutoff in angstrom (default: 7.5).",
    )
    parser.add_argument(
        "--subset-name", default="colab_lipid_gnn_subset",
        help="Output directory and zip name (default: colab_lipid_gnn_subset).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare_colab_subset(
        target_properties=args.properties,
        num_frames=args.num_frames,
        chunk_size=args.chunk_size,
        spatial_cutoff=args.spatial_cutoff,
        subset_name=args.subset_name,
    )
