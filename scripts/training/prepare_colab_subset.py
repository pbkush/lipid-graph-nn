import argparse
import gc
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np

from lipid_gnn.config import CONFIG
from lipid_gnn.dataset import preprocess_and_save
from lipid_gnn.functions_emil.functions import pkl_load
from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder

AVAILABLE_PROPERTIES = CONFIG.vocab.all_properties


def _stratified_split_systems(
    sim_tuples,
    stratify_on,
    val_frac,
    test_frac,
    split_seed,
    n_clusters=10,
):
    """
    Split systems into train/val/test such that each split spans the y-range
    of `stratify_on`, using k-means cluster IDs in standardized y-space as
    stratification labels. Robust to property scale differences (z-scores
    each property first) and correlations between properties (clustering
    handles multi-D structure naturally, no per-property tercile bins).

    Returns: (train_sims, val_sims, test_sims).
    """
    from sklearn.cluster import KMeans
    from sklearn.model_selection import train_test_split

    y_per_system = []
    for _, _, props_path in sim_tuples:
        mean_dict, _ = pkl_load(props_path, verbose=False)
        y_per_system.append([float(mean_dict[p]) for p in stratify_on])
    Y = np.asarray(y_per_system, dtype=np.float64)

    Yz = (Y - Y.mean(0)) / (Y.std(0) + 1e-12)

    k = min(n_clusters, max(2, len(sim_tuples) // 7))
    labels = KMeans(n_clusters=k, random_state=split_seed, n_init=10).fit_predict(Yz)

    idx = np.arange(len(sim_tuples))
    trainval_idx, test_idx = train_test_split(
        idx, test_size=test_frac, random_state=split_seed, stratify=labels,
    )
    val_relative = val_frac / (1.0 - test_frac)
    train_idx, val_idx = train_test_split(
        trainval_idx, test_size=val_relative, random_state=split_seed,
        stratify=labels[trainval_idx],
    )

    train_sims = [sim_tuples[i] for i in train_idx]
    val_sims   = [sim_tuples[i] for i in val_idx]
    test_sims  = [sim_tuples[i] for i in test_idx]

    print(f"\nStratified split (k={k} clusters in {len(stratify_on)}-D y-space):")
    for split_name, sidx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
        print(f"  {split_name:<6} (n={len(sidx)}):")
        for j, p in enumerate(stratify_on):
            v = Y[sidx, j]
            print(f"    {p:<20} mean={v.mean():>8.4f} std={v.std():>8.4f} "
                  f"range=[{v.min():>8.4f}, {v.max():>8.4f}]")

    return train_sims, val_sims, test_sims


def prepare_colab_subset(
    target_properties,
    num_frames,
    chunk_size,
    spatial_cutoff,
    shuffle_seed=42,
    val_frac=0.15,
    test_frac=0.15,
    split_seed=0,
    split_method="stratified",
    stratify_on=None,
    subset_name=None,
    sims_dir=None,
    props_dir=None,
    out_dir=None,
    no_zip=False,
):
    """
    Preprocesses Martini trajectories into chunked .pt graph files.

    Systems are split into train/val/test at the system level (before
    preprocessing) using split_seed. Each split gets its own subdirectory
    of interleaved chunks: processed/train/, processed/val/, processed/test/.
    This guarantees no membrane composition appears in more than one split.

    Raw .tpr/.xtc files are NOT included in the output — all graph features and
    target properties are baked in at preprocessing time.

    By default a zip of the processed chunks is created for easy transfer.
    Code is NOT bundled — training is HPC-only and code is synced via git.

    When `no_zip=True`, only the processed chunks are written (to `out_dir` if
    given, else `<root>/<subset_name>/processed`). This is the HPC entry point.
    """
    root_dir      = Path(__file__).resolve().parent.parent.parent
    data_dir      = Path(sims_dir)  if sims_dir  else CONFIG.paths.data_dir
    props_dir     = Path(props_dir) if props_dir else CONFIG.paths.props_dir
    resources_dir = CONFIG.paths.resources_dir

    ff_params_path       = CONFIG.paths.ff_params_file
    ff_edge_params_path  = CONFIG.paths.ff_edge_params_file
    ff_node_mapping_path = CONFIG.paths.ff_node_mapping_file

    if subset_name is None:
        subset_dir = CONFIG.paths.subset_bundle_dir
    else:
        subset_dir = root_dir / subset_name

    proc_dest = Path(out_dir) if out_dir else subset_dir / 'processed'

    # --- Collect sim tuples -----------------------------------------------
    compositions = sorted(
        d for d in os.listdir(data_dir)
        if os.path.isdir(data_dir / d)
    )

    sim_tuples = []
    for comp in compositions:
        tpr = data_dir / comp / CONFIG.paths.trajectory_subdir / CONFIG.paths.topology_filename
        xtc = data_dir / comp / CONFIG.paths.trajectory_subdir / CONFIG.paths.trajectory_filename
        h5  = props_dir / f'{comp}.h5'
        if tpr.exists() and xtc.exists() and h5.exists():
            sim_tuples.append((tpr, xtc, h5))
        else:
            missing = [str(p) for p in [tpr, xtc, h5] if not p.exists()]
            print(f"Skipping {comp}: missing {missing}")

    if not sim_tuples:
        raise RuntimeError(f"No complete systems found in {data_dir}")

    # --- System-level train / val / test split ----------------------------
    print(f"\nFound {len(sim_tuples)} complete systems.")

    if split_method == "stratified":
        if stratify_on is None:
            stratify_on = list(CONFIG.vocab.active_properties)
        missing = [p for p in stratify_on if p not in target_properties]
        if missing:
            raise ValueError(
                f"--stratify-on contains properties not in --properties: {missing}. "
                f"Stratification properties must be a subset of the saved y columns."
            )
        train_sims, val_sims, test_sims = _stratified_split_systems(
            sim_tuples, stratify_on, val_frac, test_frac, split_seed,
        )
    elif split_method == "random":
        rng = random.Random(split_seed)
        shuffled = list(sim_tuples)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_frac))
        n_val  = max(1, round(len(shuffled) * val_frac))
        test_sims  = shuffled[:n_test]
        val_sims   = shuffled[n_test:n_test + n_val]
        train_sims = shuffled[n_test + n_val:]
    else:
        raise ValueError(f"Unknown split_method: {split_method!r}")

    print(f"Split (method={split_method}, seed={split_seed}): "
          f"train={len(train_sims)}, val={len(val_sims)}, test={len(test_sims)}")
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

    # --- Zip chunks only (no library — code is synced via git on HPC) -----
    zip_path = root_dir / f"{subset_dir.name}.zip"
    print(f"Creating archive: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(proc_dest):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, root_dir)
                zipf.write(file_path, arcname)

    print(f"Done! Archive at {zip_path.name} (chunks only — transfer to HPC and extract).")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Martini trajectories into chunked .pt graph files. "
            "Writes train/val/test subdirectories under the output directory. "
            "Use --no-zip to skip archiving (HPC mode). "
            "Without --no-zip, creates a zip of chunks only for transfer."
        )
    )
    parser.add_argument(
        "--properties",
        nargs="+",
        default=list(AVAILABLE_PROPERTIES),
        choices=AVAILABLE_PROPERTIES,
        metavar="PROP",
        help=(
            "Properties to embed as graph.y "
            f"(default: {' '.join(AVAILABLE_PROPERTIES)}). "
            f"Available: {', '.join(AVAILABLE_PROPERTIES)}."
        ),
    )
    parser.add_argument(
        "--num-frames", type=int, default=CONFIG.dataset.num_frames,
        help=f"Evenly-spaced frames sampled per system (default: {CONFIG.dataset.num_frames}).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=CONFIG.dataset.chunk_size,
        help=f"Graphs per .pt chunk file (default: {CONFIG.dataset.chunk_size}).",
    )
    parser.add_argument(
        "--spatial-cutoff", type=float, default=CONFIG.dataset.spatial_cutoff,
        help=f"Spatial edge cutoff in angstrom (default: {CONFIG.dataset.spatial_cutoff}).",
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=CONFIG.dataset.shuffle_seed,
        help=f"RNG seed for cross-system frame interleaving within each split (default: {CONFIG.dataset.shuffle_seed}).",
    )
    parser.add_argument(
        "--val-frac", type=float, default=CONFIG.dataset.val_frac,
        help=f"Fraction of systems held out for validation (default: {CONFIG.dataset.val_frac}).",
    )
    parser.add_argument(
        "--test-frac", type=float, default=CONFIG.dataset.test_frac,
        help=f"Fraction of systems held out for test (default: {CONFIG.dataset.test_frac}).",
    )
    parser.add_argument(
        "--split-seed", type=int, default=CONFIG.dataset.split_seed,
        help=f"RNG seed for train/val/test system assignment (default: {CONFIG.dataset.split_seed}).",
    )
    parser.add_argument(
        "--split-method", choices=["stratified", "random"], default="stratified",
        help=(
            "How to assign systems to train/val/test. 'stratified' clusters systems "
            "in standardized y-space (k-means) so each split spans the y-range of "
            "--stratify-on. 'random' shuffles uniformly (legacy; can produce splits "
            "with very narrow y-range, especially in small holdouts). Default: stratified."
        ),
    )
    parser.add_argument(
        "--stratify-on", nargs="+", default=None,
        choices=AVAILABLE_PROPERTIES, metavar="PROP",
        help=(
            "Properties used as stratification basis when --split-method=stratified. "
            f"Default: active_properties from config ({list(CONFIG.vocab.active_properties)}). "
            "Should match the property set of the upcoming experiment so each split spans "
            "the y-range of the metrics you'll evaluate. Must be a subset of --properties."
        ),
    )
    parser.add_argument(
        "--subset-name", default=None,
        help=f"Output directory and zip name (default: {CONFIG.paths.subset_bundle_dir.name}).",
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
        split_method=args.split_method,
        stratify_on=args.stratify_on,
        subset_name=args.subset_name,
        sims_dir=args.sims_dir,
        props_dir=args.props_dir,
        out_dir=args.out_dir,
        no_zip=args.no_zip,
    )
