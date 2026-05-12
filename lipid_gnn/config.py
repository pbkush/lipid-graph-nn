"""Central config loader for the lipid-graph-nn project.

Loads ``config.yaml`` from the repo root, resolves relative paths against
``REPO_ROOT``, applies environment overrides, and returns a frozen ``Config``
dataclass. A module-level singleton ``CONFIG`` is exposed for convenience.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"


def _resolve_path(value: str) -> Path:
    """Resolve a (possibly relative) path string against REPO_ROOT."""
    p = Path(value)
    return p if p.is_absolute() else (REPO_ROOT / p)


@dataclass(frozen=True)
class PathsConfig:
    data_dir: Path
    props_dir: Path
    resources_dir: Path
    ff_params_file: Path
    ff_edge_params_file: Path
    ff_node_mapping_file: Path
    chunks_dir: Path
    subset_bundle_dir: Path
    results_dir: Path
    training_results_dir: Path
    logs_dir: Path
    wandb_dir: Path
    topology_filename: str
    trajectory_filename: str
    trajectory_subdir: str


@dataclass(frozen=True)
class DatasetConfig:
    spatial_cutoff: float
    num_frames: int
    chunk_size: int
    interleave: bool
    shuffle_seed: int
    val_frac: float
    test_frac: float
    split_seed: int
    atom_selection: str
    rbf_start: float
    rbf_num_gaussians: int
    reference_system: str

    @property
    def rbf_stop(self) -> float:
        return self.spatial_cutoff


@dataclass(frozen=True)
class VocabConfig:
    lipid_types: list
    all_properties: list
    active_properties: list

    @property
    def lipid_comp_dim(self) -> int:
        return len(self.lipid_types)


@dataclass(frozen=True)
class ModelConfig:
    in_channels: int
    hidden_dim: int
    num_layers: int
    heads: int
    dropout: float
    comp_dim: int
    bonded_edge_attr_dim: int
    spatial_edge_attr_dim: int


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    batch_size: int
    num_workers: int
    learning_rate: float
    weight_decay: float
    seed: int
    patience: int
    lr_factor: float
    grad_clip_norm: float
    amp_dtype: str
    log_every_n_batches: int
    print_every_n_epochs: int


@dataclass(frozen=True)
class WandbConfig:
    project_prefix: str
    group: Optional[str]
    mode: str
    entity: Optional[str]


@dataclass(frozen=True)
class HpcConfig:
    group: str
    conda_env: str
    module_rocm: str
    partition_preprocess: str
    partition_train: str
    account: str
    work_subpath: str


@dataclass(frozen=True)
class MartiniPipelineBoxConfig:
    xy_nm: float
    z_nm: float
    salt_M: float
    water_type: str
    charge_mode: str
    center: bool
    pbc: str


@dataclass(frozen=True)
class MartiniPipelineRunConfig:
    nsteps_min: int
    nsteps_eq: int
    nsteps_prod: int
    nstenergy_eq: int
    save_forces: bool
    seed_strategy: str  # "deterministic" | "random" | "<int>"


@dataclass(frozen=True)
class MartiniPipelineGmxConfig:
    executable: str
    maxwarn: int
    mdrun_extra_args: List[str]


@dataclass(frozen=True)
class MartiniPipelineConfig:
    output_root: Path
    insane_cmd: str
    itp_dir: Path
    mdp_templates_dir: Path
    mdp_freeze: Path
    box: MartiniPipelineBoxConfig
    run: MartiniPipelineRunConfig
    gmx: MartiniPipelineGmxConfig


@dataclass(frozen=True)
class Config:
    paths: PathsConfig
    dataset: DatasetConfig
    vocab: VocabConfig
    model: ModelConfig
    training: TrainingConfig
    wandb: WandbConfig
    hpc: HpcConfig
    martini_pipeline: Optional[MartiniPipelineConfig] = None


def _build_paths(raw: dict) -> PathsConfig:
    return PathsConfig(
        data_dir=_resolve_path(raw["data_dir"]),
        props_dir=_resolve_path(raw["props_dir"]),
        resources_dir=_resolve_path(raw["resources_dir"]),
        ff_params_file=_resolve_path(raw["ff_params_file"]),
        ff_edge_params_file=_resolve_path(raw["ff_edge_params_file"]),
        ff_node_mapping_file=_resolve_path(raw["ff_node_mapping_file"]),
        chunks_dir=_resolve_path(raw["chunks_dir"]),
        subset_bundle_dir=_resolve_path(raw["subset_bundle_dir"]),
        results_dir=_resolve_path(raw["results_dir"]),
        training_results_dir=_resolve_path(raw["training_results_dir"]),
        logs_dir=_resolve_path(raw["logs_dir"]),
        wandb_dir=_resolve_path(raw["wandb_dir"]),
        topology_filename=raw["topology_filename"],
        trajectory_filename=raw["trajectory_filename"],
        trajectory_subdir=raw["trajectory_subdir"],
    )


def _build_martini_pipeline(raw: dict) -> MartiniPipelineConfig:
    box_raw = raw.get("box", {})
    run_raw = raw.get("run", {})
    gmx_raw = raw.get("gmx", {})
    return MartiniPipelineConfig(
        output_root=_resolve_path(raw.get("output_root", "data/martini_pipeline")),
        insane_cmd=raw.get("insane_cmd", "insane"),
        itp_dir=_resolve_path(raw.get("itp_dir", "resources/martini3/itp")),
        mdp_templates_dir=_resolve_path(raw.get("mdp_templates_dir", "lipid_gnn/martini_pipeline/templates")),
        mdp_freeze=_resolve_path(raw.get("mdp_freeze", "lipid_gnn/martini_pipeline/templates/_audit_freeze.json")),
        box=MartiniPipelineBoxConfig(
            xy_nm=float(box_raw.get("xy_nm", 11.0)),
            z_nm=float(box_raw.get("z_nm", 10.0)),
            salt_M=float(box_raw.get("salt_M", 0.15)),
            water_type=str(box_raw.get("water_type", "W")),
            charge_mode=str(box_raw.get("charge_mode", "auto")),
            center=bool(box_raw.get("center", True)),
            pbc=str(box_raw.get("pbc", "rectangular")),
        ),
        run=MartiniPipelineRunConfig(
            nsteps_min=int(run_raw.get("nsteps_min", 20000)),
            nsteps_eq=int(run_raw.get("nsteps_eq", 1000000)),
            nsteps_prod=int(run_raw.get("nsteps_prod", -1)),
            nstenergy_eq=int(run_raw.get("nstenergy_eq", 1000)),
            save_forces=bool(run_raw.get("save_forces", False)),
            seed_strategy=str(run_raw.get("seed_strategy", "deterministic")),
        ),
        gmx=MartiniPipelineGmxConfig(
            executable=str(gmx_raw.get("executable", "gmx")),
            maxwarn=int(gmx_raw.get("maxwarn", 2)),
            mdrun_extra_args=list(gmx_raw.get("mdrun_extra_args", [])),
        ),
    )


def load_config(path: Optional[Path] = None) -> Config:
    """Load and validate the YAML config.

    Environment overrides applied:
      - ``CHUNKS_DIR``    -> ``paths.chunks_dir``
      - ``WANDB_MODE``    -> ``wandb.mode``
      - ``WANDB_GROUP``   -> ``wandb.group``
    """
    path = path or DEFAULT_CONFIG_PATH
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Env overrides at the raw-dict layer (before dataclass construction).
    if env_chunks := os.environ.get("CHUNKS_DIR"):
        raw["paths"]["chunks_dir"] = env_chunks
    if env_mode := os.environ.get("WANDB_MODE"):
        raw["wandb"]["mode"] = env_mode
    if env_group := os.environ.get("WANDB_GROUP"):
        raw["wandb"]["group"] = env_group

    paths = _build_paths(raw["paths"])
    dataset = DatasetConfig(**raw["dataset"])
    vocab = VocabConfig(**raw["vocab"])
    model = ModelConfig(**raw["model"])
    training = TrainingConfig(**raw["training"])
    wandb_cfg = WandbConfig(**raw["wandb"])
    hpc = HpcConfig(**raw["hpc"])

    martini_pipeline: Optional[MartiniPipelineConfig] = None
    if "martini_pipeline" in raw and raw["martini_pipeline"] is not None:
        martini_pipeline = _build_martini_pipeline(raw["martini_pipeline"])

    # Validation
    if model.spatial_edge_attr_dim != dataset.rbf_num_gaussians:
        raise ValueError(
            f"model.spatial_edge_attr_dim ({model.spatial_edge_attr_dim}) must equal "
            f"dataset.rbf_num_gaussians ({dataset.rbf_num_gaussians})."
        )
    missing = set(vocab.active_properties) - set(vocab.all_properties)
    if missing:
        raise ValueError(
            f"vocab.active_properties contains unknown entries: {sorted(missing)}. "
            f"Valid: {vocab.all_properties}."
        )

    return Config(
        paths=paths,
        dataset=dataset,
        vocab=vocab,
        model=model,
        training=training,
        wandb=wandb_cfg,
        hpc=hpc,
        martini_pipeline=martini_pipeline,
    )


CONFIG = load_config()
