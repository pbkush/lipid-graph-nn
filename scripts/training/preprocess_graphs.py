"""Preprocess Martini trajectories into chunked .pt graph datasets for training.

Outputs go to ``<preprocessed_graphs_dir>/<run-name>/{train,val,test}/``;
zip archives (when not ``--no-zip``) go to
``<preprocessed_graphs_dir>/archives/<run-name>.zip``. The run name defaults
to the ``--props-set`` value so successive preprocessing runs over different
property sets never overwrite each other.

Training is HPC-only — the zip is for HPC transfer; on the HPC use
``--no-zip`` and point ``--out-dir`` at ``/work/...``.
"""
import argparse
import gc
import json
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np

from lipid_gnn.config import CONFIG
from lipid_gnn.dataset import preprocess_and_save
from lipid_gnn.io import pkl_load
from lipid_gnn.lipid_graph import MartiniHeteroGraphBuilder


def _composition_of(sim_tuple):
    """Extract the composition (canonical_name) from a (tpr, xtc, h5) tuple."""
    tpr_path, _, _ = sim_tuple
    return Path(tpr_path).parent.parent.name


def _split_from_json(sim_tuples, json_path):
    """Look up each sim tuple's composition in a {train,val,test} JSON and
    route it. Compositions in the JSON that aren't in sim_tuples are warned
    about; compositions in sim_tuples missing from the JSON raise."""
    with open(json_path) as f:
        spec = json.load(f)

    missing = [s for s in ("train", "val", "test") if s not in spec]
    if missing:
        raise ValueError(f"split JSON {json_path} missing keys: {missing}")

    membership = {}
    for split_name in ("train", "val", "test"):
        for comp in spec[split_name]:
            if comp in membership:
                raise ValueError(
                    f"composition {comp!r} appears in multiple splits in {json_path}"
                )
            membership[comp] = split_name

    routed = {"train": [], "val": [], "test": []}
    unassigned = []
    for sim in sim_tuples:
        comp = _composition_of(sim)
        if comp in membership:
            routed[membership[comp]].append(sim)
        else:
            unassigned.append(comp)

    if unassigned:
        raise ValueError(
            f"{len(unassigned)} composition(s) in --sims-dir have no split "
            f"assignment in {json_path}: {unassigned[:5]}{'...' if len(unassigned) > 5 else ''}"
        )

    found_comps = {_composition_of(s) for s in sim_tuples}
    extras = [c for c in membership if c not in found_comps]
    if extras:
        print(
            f"WARNING: {len(extras)} composition(s) in {json_path} not found "
            f"in --sims-dir (skipped): {extras[:5]}{'...' if len(extras) > 5 else ''}"
        )

    print(
        f"\nSplit loaded from {json_path} (source: "
        f"{spec.get('source_run', 'unspecified')}): "
        f"train={len(routed['train'])}, val={len(routed['val'])}, "
        f"test={len(routed['test'])}"
    )
    return routed["train"], routed["val"], routed["test"]


def _write_split_json(train_sims, val_sims, test_sims, json_path, source_run):
    """Persist a {train,val,test} composition-name split for later reuse via
    ``--split-from-json``."""
    spec = {
        "source_run": source_run,
        "train": sorted(_composition_of(s) for s in train_sims),
        "val": sorted(_composition_of(s) for s in val_sims),
        "test": sorted(_composition_of(s) for s in test_sims),
    }
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(spec, f, indent=2)
    print(f"Wrote split spec to {json_path}")

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


def preprocess_graphs(
    target_properties,
    num_frames,
    chunk_size,
    spatial_cutoff,
    props_set,
    shuffle_seed=42,
    val_frac=0.15,
    test_frac=0.15,
    split_seed=0,
    split_method="stratified",
    stratify_on=None,
    split_from_json=None,
    write_split_json=None,
    run_name=None,
    sims_dir=None,
    props_base=None,
    parent_dir=None,
    out_dir=None,
    no_zip=False,
):
    """
    Preprocesses Martini trajectories into chunked .pt graph files.

    Systems are split into train/val/test at the system level (before
    preprocessing) using split_seed. Each split gets its own subdirectory
    of interleaved chunks: <run>/train/, <run>/val/, <run>/test/.
    This guarantees no membrane composition appears in more than one split.

    Raw .tpr/.xtc files are NOT included in the output — all graph features and
    target properties are baked in at preprocessing time.

    By default a zip of the processed chunks is created for easy transfer.
    The zip lives in ``<parent_dir>/archives/<run-name>.zip``. Code is NOT
    bundled — training is HPC-only and code is synced via git.

    When ``no_zip=True``, only the processed chunks are written (to ``out_dir``
    if given, else ``<parent_dir>/<run-name>``). This is the HPC entry point.
    """
    data_dir   = Path(sims_dir)   if sims_dir   else CONFIG.paths.data_dir
    props_base = Path(props_base) if props_base else CONFIG.paths.props_dir
    parent_dir = Path(parent_dir) if parent_dir else CONFIG.paths.preprocessed_graphs_dir

    props_dir = props_base / props_set
    if not props_dir.is_dir():
        raise FileNotFoundError(
            f"Property folder not found: {props_dir}. "
            f"Pass --props-set <subfolder of {props_base}>."
        )

    ff_params_path       = CONFIG.paths.ff_params_file
    ff_edge_params_path  = CONFIG.paths.ff_edge_params_file
    ff_node_mapping_path = CONFIG.paths.ff_node_mapping_file

    if run_name is None:
        run_name = props_set
    run_dir = parent_dir / run_name

    proc_dest = Path(out_dir) if out_dir else run_dir

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
    print(f"Property set      : {props_set}  ({props_dir})")
    print(f"Run name          : {run_name}")
    print(f"Output dir        : {proc_dest}")

    if split_from_json is not None:
        train_sims, val_sims, test_sims = _split_from_json(sim_tuples, split_from_json)
        effective_method = f"from_json:{Path(split_from_json).name}"
    elif split_method == "stratified":
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
        effective_method = "stratified"
    elif split_method == "random":
        rng = random.Random(split_seed)
        shuffled = list(sim_tuples)
        rng.shuffle(shuffled)
        n_test = max(1, round(len(shuffled) * test_frac))
        n_val  = max(1, round(len(shuffled) * val_frac))
        test_sims  = shuffled[:n_test]
        val_sims   = shuffled[n_test:n_test + n_val]
        train_sims = shuffled[n_test + n_val:]
        effective_method = "random"
    else:
        raise ValueError(f"Unknown split_method: {split_method!r}")

    if write_split_json is not None:
        _write_split_json(train_sims, val_sims, test_sims, write_split_json, run_name)

    print(f"Split (method={effective_method}, seed={split_seed}): "
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
    archives_dir = parent_dir / "archives"
    archives_dir.mkdir(parents=True, exist_ok=True)
    zip_path = archives_dir / f"{run_name}.zip"
    print(f"Creating archive: {zip_path} ...")
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(proc_dest):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, proc_dest.parent)
                zipf.write(file_path, arcname)

    print(f"Done! Archive at {zip_path} (chunks only — transfer to HPC and extract).")


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Martini trajectories into chunked .pt graph files. "
            "Writes train/val/test subdirectories under "
            "<preprocessed_graphs_dir>/<run-name>/. Use --no-zip to skip "
            "archiving (HPC mode). Without --no-zip, a zip is written to "
            "<preprocessed_graphs_dir>/archives/<run-name>.zip."
        )
    )
    parser.add_argument(
        "--props-set", required=True,
        help=(
            "Subfolder of --props-base holding the per-system .h5 property "
            "files for this run (e.g. 'prop_legacy_bugfixed_s0'). Also the "
            "default --run-name."
        ),
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
        "--split-from-json", default=None,
        help=(
            "Path to a {train,val,test} JSON spec (as written by "
            "--write-split-json). When given, overrides --split-method/--split-seed/"
            "--stratify-on and routes each composition by JSON membership. "
            "Use this to reuse one canonical split across multiple property "
            "sets so per-system errors stay paired across runs (e.g. for the "
            "three-way bugfix comparison)."
        ),
    )
    parser.add_argument(
        "--write-split-json", default=None,
        help=(
            "Path to write the resulting {train,val,test} composition lists as "
            "JSON for later reuse via --split-from-json. Composes with all three "
            "split methods (stratified, random, from_json)."
        ),
    )
    parser.add_argument(
        "--run-name", default=None,
        help=(
            "Output subfolder name under --parent-dir (default: --props-set). "
            "Use this to tag variants (e.g. different num-frames or cutoff) "
            "without overwriting earlier runs."
        ),
    )
    parser.add_argument(
        "--sims-dir", default=None,
        help="Override directory holding <COMP>/run/prun.{tpr,xtc} (default: <repo>/data/membrane_only).",
    )
    parser.add_argument(
        "--props-base", default=None,
        help="Base directory containing per-set property folders (default: <repo>/results/properties).",
    )
    parser.add_argument(
        "--parent-dir", default=None,
        help=(
            "Parent directory under which each preprocessing run gets its own "
            "subfolder and the 'archives/' zip folder lives "
            "(default: <repo>/data/preprocessed_graphs)."
        ),
    )
    parser.add_argument(
        "--out-dir", default=None,
        help="Override output directory for chunks (default: <parent-dir>/<run-name>).",
    )
    parser.add_argument(
        "--no-zip", action="store_true",
        help="Write chunks only; skip zip creation. Use for HPC/remote training.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    preprocess_graphs(
        target_properties=args.properties,
        num_frames=args.num_frames,
        chunk_size=args.chunk_size,
        spatial_cutoff=args.spatial_cutoff,
        props_set=args.props_set,
        shuffle_seed=args.shuffle_seed,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        split_seed=args.split_seed,
        split_method=args.split_method,
        stratify_on=args.stratify_on,
        split_from_json=args.split_from_json,
        write_split_json=args.write_split_json,
        run_name=args.run_name,
        sims_dir=args.sims_dir,
        props_base=args.props_base,
        parent_dir=args.parent_dir,
        out_dir=args.out_dir,
        no_zip=args.no_zip,
    )
