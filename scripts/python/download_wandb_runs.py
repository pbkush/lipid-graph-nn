#!/usr/bin/env python3
"""Download W&B training run groups to disk for offline analysis.

Saves per-run config, summary, loss history, and system metrics to
logs/training/<group>/<run_name>/. Creates a per-group runs_index.json for
fast loading by scripts/notebooks/analyze_hp_search.ipynb.

Usage:
    python scripts/python/download_wandb_runs.py --group stage_3_arch
    python scripts/python/download_wandb_runs.py --group stage_1_lr stage_2_wd stage_3_arch
    python scripts/python/download_wandb_runs.py --group stage_3_arch --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import wandb

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lipid_gnn.config import CONFIG


def _derive_project(properties: list[str]) -> str:
    return f"{CONFIG.wandb.project_prefix}_" + "_".join(properties)


def _filter_summary(summary) -> dict:
    """Keep only JSON-serializable scalar values; drop W&B media objects."""
    result = {}
    for k, v in summary.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            result[k] = v
    return result


def _val_min_last10(history_df: pd.DataFrame) -> float | None:
    """Return min val/loss_total over the last 10 epochs (the plan's selection metric)."""
    if history_df.empty or "val/loss_total" not in history_df.columns:
        return None
    series = history_df["val/loss_total"].dropna()
    return float(series.tail(10).min()) if len(series) else None


def _download_run(run, run_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Download and save a single run's data. Returns (history_df, summary_dict)."""
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)
    cfg = dict(run.config)

    # config.json
    with open(run_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    # summary.json — scalars only; no media refs
    summary = _filter_summary(run.summary)
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # history.parquet — per-epoch loss metrics
    properties = cfg.get("properties", [])
    history_keys = ["epoch", "learning_rate", "train/loss_total", "val/loss_total"]
    for prop in properties:
        history_keys += [
            f"train/loss_{prop}",
            f"val/loss_{prop}",
            f"val/r2_{prop}",
        ]
    history_df = pd.DataFrame()
    try:
        history_df = run.history(keys=history_keys, pandas=True)
        history_df.to_parquet(run_dir / "history.parquet", index=False)
    except Exception as exc:
        print(f"    WARNING: could not fetch history: {exc}")

    # system.parquet — W&B auto-logged GPU/CPU metrics (~15 s sample rate)
    try:
        sys_df = run.history(stream="system", pandas=True)
        if not sys_df.empty:
            sys_df.to_parquet(run_dir / "system.parquet", index=False)
    except Exception as exc:
        print(f"    WARNING: could not fetch system metrics: {exc}")

    # File artifacts uploaded via wandb.save() — e.g. test_artifacts.npz
    _ARTIFACT_FILES = {"test_artifacts.npz", "model_final.pt", "model_best.pt"}
    try:
        found = []
        for wf in run.files():
            if Path(wf.name).name in _ARTIFACT_FILES:
                wf.download(root=str(run_dir), replace=True)
                # Flatten: move from run_dir/wf.name to run_dir/basename if nested
                src = run_dir / wf.name
                dst = run_dir / Path(wf.name).name
                if src.exists() and src != dst:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    src.rename(dst)
                found.append(Path(wf.name).name)
        if not found:
            all_names = [wf.name for wf in run.files()]
            print(f"    WARNING: no artifact files found. Run contains: {all_names}")
        else:
            print(f"    Artifacts: {found}")
    except Exception as exc:
        print(f"    WARNING: could not fetch file artifacts: {exc}")

    return history_df, summary


def download_group(
    group: str,
    project: str,
    entity: str | None,
    out_dir: Path,
    include_crashed: bool,
    force: bool,
) -> None:
    api = wandb.Api()
    entity_project = f"{entity}/{project}" if entity else project

    filters: dict = {"group": group}
    if not include_crashed:
        filters["state"] = "finished"

    print(f"\n[{group}] fetching from {entity_project} …")
    try:
        runs = list(api.runs(entity_project, filters=filters))
    except Exception as exc:
        print(f"  ERROR: {exc}", file=sys.stderr)
        print(
            "  Hint: check --entity and --project; verify WANDB_GROUP was set "
            "at sbatch submission time.",
            file=sys.stderr,
        )
        return

    if not runs:
        msg = "finished " if not include_crashed else ""
        print(
            f"  WARNING: no {msg}runs found for group={group!r} in {entity_project!r}."
        )
        if not include_crashed:
            print("  Try --include-crashed to include non-finished runs.")
        return

    group_dir = out_dir / group
    group_dir.mkdir(parents=True, exist_ok=True)

    # Load existing index for cache checking
    index_path = group_dir / "runs_index.json"
    if index_path.exists() and not force:
        with open(index_path) as f:
            index: list[dict] = json.load(f)
        cached_ids = {e["id"] for e in index}
    else:
        index = []
        cached_ids = set()

    print(f"  {len(runs)} runs found, {len(cached_ids)} already cached.")

    updated: dict[str, dict] = {e["id"]: e for e in index}

    for run in runs:
        if run.id in cached_ids and not force:
            vml10 = updated[run.id].get("val_min_last10")
            vstr = f"{vml10:.4f}" if vml10 is not None else "N/A"
            print(f"  SKIP  {run.name:<45}  val_min_last10={vstr}")
            continue

        print(f"  DL    {run.name:<45}", end="  ", flush=True)
        run_dir = group_dir / run.name
        marker = run_dir / ".wandb_run_id"
        if marker.exists() and marker.read_text().strip() != run.id:
            raise RuntimeError(
                f"Local dir {run_dir} already holds W&B run "
                f"{marker.read_text().strip()!r}; refusing to overwrite with {run.id!r}. "
                "Delete or rename the directory before re-downloading."
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(run.id)
        history_df, summary = _download_run(run, run_dir)

        runtime = summary.get("_runtime")
        vml10 = _val_min_last10(history_df)
        vstr = f"{vml10:.4f}" if vml10 is not None else "N/A"
        runtime_str = f"  {runtime / 3600:.1f}h" if runtime else ""
        print(f"state={run.state}  val_min_last10={vstr}{runtime_str}")

        updated[run.id] = {
            "id": run.id,
            "name": run.name,
            "state": run.state,
            "config": dict(run.config),
            "runtime_seconds": runtime,
            "val_min_last10": vml10,
        }

    index = list(updated.values())
    with open(index_path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"\n  Index saved: {index_path}  ({len(index)} runs total)")


def _parse_args() -> argparse.Namespace:
    default_props = list(CONFIG.vocab.active_properties)

    p = argparse.ArgumentParser(
        description="Download W&B training run groups to disk for offline analysis.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--group",
        nargs="+",
        required=True,
        metavar="GROUP",
        help="W&B group name(s) to download, e.g. stage_3_arch stage_2_wd",
    )
    p.add_argument(
        "--properties",
        nargs="+",
        default=default_props,
        metavar="PROP",
        help=f"Property names used to derive the W&B project name. "
        f"Available: {', '.join(default_props)}",
    )
    p.add_argument(
        "--project",
        default=None,
        metavar="PROJECT",
        help="W&B project name. Defaults to {project_prefix}_{joined_properties}.",
    )
    p.add_argument(
        "--entity",
        default=CONFIG.wandb.entity,
        metavar="ENTITY",
        help="W&B entity (username or team). Defaults to CONFIG.wandb.entity.",
    )
    p.add_argument(
        "--out-dir",
        default=str(REPO_ROOT / "logs" / "training"),
        metavar="DIR",
        help="Root output directory. Groups saved as <out-dir>/<group>/.",
    )
    p.add_argument(
        "--include-crashed",
        action="store_true",
        help="Include non-finished runs (crashed, running, etc.).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download all runs even if already cached.",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    project = args.project or _derive_project(args.properties)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Project : {project}")
    print(f"Entity  : {args.entity or '(default)'}")
    print(f"Out dir : {out_dir}")
    print(f"Groups  : {args.group}")

    for group in args.group:
        download_group(
            group=group,
            project=project,
            entity=args.entity,
            out_dir=out_dir,
            include_crashed=args.include_crashed,
            force=args.force,
        )

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
