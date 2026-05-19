import os
from pathlib import Path

import pytest
import yaml

from lipid_gnn.config import CONFIG, load_config, DEFAULT_CONFIG_PATH


def test_default_load_succeeds():
    cfg = load_config()
    assert cfg is CONFIG or cfg == CONFIG  # dataclasses are frozen; value equality is fine.


def test_paths_resolved_absolute():
    assert CONFIG.paths.data_dir.is_absolute()
    assert CONFIG.paths.ff_params_file.is_absolute()
    assert CONFIG.paths.chunks_dir.is_absolute()


def test_rbf_stop_equals_spatial_cutoff():
    assert CONFIG.dataset.rbf_stop == CONFIG.dataset.spatial_cutoff


def test_active_properties_subset_of_all():
    assert set(CONFIG.vocab.active_properties).issubset(set(CONFIG.vocab.all_properties))


def test_chunks_dir_env_override(tmp_path, monkeypatch):
    override = tmp_path / "alt_chunks"
    monkeypatch.setenv("CHUNKS_DIR", str(override))
    cfg = load_config()
    assert cfg.paths.chunks_dir == override


def test_wandb_mode_env_override(monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "offline")
    cfg = load_config()
    assert cfg.wandb.mode == "offline"


def test_rejects_invalid_active_property(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["vocab"]["active_properties"] = ["not_a_real_property"]
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="active_properties"):
        load_config(bad_path)


def test_rejects_mismatched_rbf_dim(tmp_path):
    raw = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text())
    raw["model"]["spatial_edge_attr_dim"] = raw["dataset"]["rbf_num_gaussians"] + 1
    bad_path = tmp_path / "bad.yaml"
    bad_path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="rbf_num_gaussians"):
        load_config(bad_path)
