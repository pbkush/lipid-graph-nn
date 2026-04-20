import argparse
import gc
import os
import random
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
    shuffle_seed=42,
    val_frac=0.15,
    test_frac=0.15,
    split_seed=0,
    subset_name="colab_lipid_gnn_subset",
    sims_dir=None,
    props_dir=None,
    out_dir=None,
    no_zip=False,
):
    """
    Preprocesses Martini trajectories into chunked .pt graph files and (by
    default) bundles them with the lipid_gnn library into a zip for upload
    to Google Colab.

    Systems are split into train/val/test at the system level (before
    preprocessing) using split_seed. Each split gets its own subdirectory
    of interleaved chunks: processed/train/, processed/val/, processed/test/.
    This guarantees no membrane composition appears in more than one split.

    Raw .tpr/.xtc files are NOT included in the output — all graph features and
    target properties are baked in at preprocessing time.

    When `no_zip=True`, only the processed chunks are written (to `out_dir` if
    given, else `<root>/<subset_name>/processed`). No library bundling, no zip.
    This is the HPC / remote-training entry point where code is deployed via
    git and chunks live on a separate filesystem from `$HOME`.
    """
    root_dir      = Path(__file__).resolve().parent.parent.parent
    data_dir      = Path(sims_dir)  if sims_dir  else root_dir / 'data/membrane_only'
    props_dir     = Path(props_dir) if props_dir else root_dir / 'results/properties'
    resources_dir = root_dir / 'resources'

    ff_params_path       = resources_dir / 'martini_ff_params.json'
    ff_edge_params_path  = resources_dir / 'martini_ff_edge_params.json'
    ff_node_mapping_path = resources_dir / 'martini_ff_node_mapping.json'

    if no_zip:
        proc_dest = Path(out_dir) if out_dir else root_dir / subset_name / 'processed'
        lib_dest  = None
    else:
        dest_dir  = root_dir / subset_name
        proc_dest = dest_dir / 'processed'
        lib_dest  = dest_dir / 'lipid_gnn'
        if lib_dest.exists():
            shutil.rmtree(lib_dest)

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

    # --- System-level train / val / test split ----------------------------
    rng = random.Random(split_seed)
    shuffled = list(sim_tuples)
    rng.shuffle(shuffled)
    n_test = max(1, round(len(shuffled) * test_frac))
    n_val  = max(1, round(len(shuffled) * val_frac))
    test_sims  = shuffled[:n_test]
    val_sims   = shuffled[n_test:n_test + n_val]
    train_sims = shuffled[n_test + n_val:]

    print(f"\nFound {len(sim_tuples)} complete systems.")
    print(f"Split (seed={split_seed}): train={len(train_sims)}, val={len(val_sims)}, test={len(test_sims)}")
    print(f"Target properties : {target_properties}")
    print(f"Frames per system : {num_frames}")
    print(f"Chunk size        : {chunk_size} graphs/chunk")
    print(f"Spatial cutoff    : {spatial_cutoff} Å")
    print(f"Shuffle seed      : {shuffle_seed} (cross-system interleave within each split)")

    # --- Time probe: one frame from the first system ----------------------
    print("\nProbing first frame to estimate total runtime...")
    tpr0, xtc0, _ = train_sims[0]
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

    # --- Preprocess each split into its own subdirectory -----------------
    for split_name, sims in [("train", train_sims), ("val", val_sims), ("test", test_sims)]:
        split_dir = proc_dest / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        print(f"--- Preprocessing {split_name} ({len(sims)} systems) ---")
        preprocess_and_save(
            sim_tuples=sims,
            processed_dir=split_dir,
            target_properties=target_properties,
            num_frames=num_frames,
            chunk_size=chunk_size,
            spatial_cutoff=spatial_cutoff,
            shuffle_seed=shuffle_seed,
            ff_params_path=ff_params_path,
            ff_edge_params_path=ff_edge_params_path,
            ff_node_mapping_path=ff_node_mapping_path,
        )

    if no_zip:
        print(f"Done! Chunks written to {proc_dest}/{{train,val,test}}/.")
        return

    # --- Bundle library ---------------------------------------------------
    assert lib_dest is not None
    print("Bundling lipid_gnn library...")
    shutil.copytree(root_dir / 'lipid_gnn', lib_dest)

    # --- Zip (processed/{train,val,test}/ + lipid_gnn/ only) --------------
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
        "--num-frames", type=int, default=25,
        help="Evenly-spaced frames sampled per system (default: 25).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=50,
        help="Graphs per .pt chunk file (default: 50).",
    )
    parser.add_argument(
        "--spatial-cutoff", type=float, default=11.0,
        help="Spatial edge cutoff in angstrom (default: 11.0).",
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=42,
        help="RNG seed for cross-system frame interleaving within each split (default: 42).",
    )
    parser.add_argument(
        "--val-frac", type=float, default=0.15,
        help="Fraction of systems held out for validation (default: 0.15).",
    )
    parser.add_argument(
        "--test-frac", type=float, default=0.15,
        help="Fraction of systems held out for test (default: 0.15).",
    )
    parser.add_argument(
        "--split-seed", type=int, default=0,
        help="RNG seed for train/val/test system assignment (default: 0).",
    )
    parser.add_argument(
        "--subset-name", default="colab_lipid_gnn_subset",
        help="Output directory and zip name (default: colab_lipid_gnn_subset).",
    )
    parser.add_argument(
        "--sims-dir", default=None,
        help="Override directory holding <COMP>/run/prun.{tpr,xtc} (default: <repo>/data/membrane_only).",
    )
    parser.add_argument(
        "--props-dir", default=None,
        help="Override directory holding <COMP>.h5 property files (default: <repo>/results/properties).",
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Output directory for chunks when --no-zip (default: <repo>/<subset-name>/processed).",
    )
    parser.add_argument(
        "--no-zip", action="store_true",
        help="Write chunks only; skip lipid_gnn bundling and zip creation. Use for HPC/remote training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    prepare_colab_subset(
        target_properties=args.properties,
        num_frames=args.num_frames,
        chunk_size=args.chunk_size,
        spatial_cutoff=args.spatial_cutoff,
        shuffle_seed=args.shuffle_seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        subset_name=args.subset_name,
        sims_dir=args.sims_dir,
        props_dir=args.props_dir,
        out_dir=args.out_dir,
        no_zip=args.no_zip,
    )
