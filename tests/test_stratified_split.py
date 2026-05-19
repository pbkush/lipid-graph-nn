"""
Tests for the stratified system-level split in
scripts/training/preprocess_graphs._stratified_split_systems.

The original random-shuffle split (split_seed=0, N=70 systems, 15% test)
happened to produce a test set with std ≈ 4× narrower than train on
lipid_packing — making test MSE artificially low and meaningless as a
generalization signal. These tests assert that the stratified replacement
guarantees per-split y-range coverage on the stratification properties.
"""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.training.preprocess_graphs import (
    _composition_of,
    _split_from_json,
    _stratified_split_systems,
    _write_split_json,
)


def _make_sim_tuples(tmp_path, y_per_system, prop_names):
    """
    Write one mock .h5 (pickled (mean_dict, None) tuple) per synthetic
    system and return a sim_tuples list compatible with
    _stratified_split_systems. tpr/xtc paths are dummies — the function
    only opens the props path.
    """
    sim_tuples = []
    for i, y in enumerate(y_per_system):
        h5_path = tmp_path / f"sys_{i:03d}.h5"
        mean_dict = {p: float(v) for p, v in zip(prop_names, y)}
        with open(h5_path, "wb") as f:
            pickle.dump((mean_dict, None), f)
        sim_tuples.append((tmp_path / f"sys_{i:03d}.tpr",
                           tmp_path / f"sys_{i:03d}.xtc",
                           h5_path))
    return sim_tuples


def _per_split_std(sim_tuples, stratify_on):
    """Read y back from the mock h5s and return ndarray (N, P)."""
    Y = []
    for _, _, h5 in sim_tuples:
        with open(h5, "rb") as f:
            md, _ = pickle.load(f)
        Y.append([md[p] for p in stratify_on])
    return np.asarray(Y)


def test_stratified_split_covers_y_range_2d(tmp_path):
    """
    With N=70 systems and a 2-D y distribution containing one tight
    'mode' near the median, a random split can cluster the mode into one
    holdout. The stratified split must spread the modes across all three
    splits — assert each split's std is within 0.5x–2x of train's std on
    each stratified property.
    """
    rng = np.random.default_rng(42)
    n = 70
    p_outer = rng.uniform(2.5, 4.0, size=n // 2)
    t_outer = rng.uniform(35.0, 42.0, size=n // 2)
    p_inner = rng.normal(3.0, 0.05, size=n - n // 2)
    t_inner = rng.normal(38.5, 0.3, size=n - n // 2)
    y = np.column_stack([
        np.concatenate([p_outer, p_inner]),
        np.concatenate([t_outer, t_inner]),
    ])
    rng.shuffle(y)

    props = ["lipid_packing", "thickness"]
    sim_tuples = _make_sim_tuples(tmp_path, y, props)

    train, val, test = _stratified_split_systems(
        sim_tuples,
        stratify_on=props,
        val_frac=0.15,
        test_frac=0.15,
        split_seed=0,
    )

    Y_train = _per_split_std(train, props)
    Y_val   = _per_split_std(val,   props)
    Y_test  = _per_split_std(test,  props)

    for j, p in enumerate(props):
        s_train = Y_train[:, j].std()
        s_val   = Y_val[:, j].std()
        s_test  = Y_test[:, j].std()
        assert s_test >= 0.5 * s_train, (
            f"{p}: test std {s_test:.4f} < 0.5 * train std {s_train:.4f} — "
            f"stratification failed to cover the y-range on this property."
        )
        assert s_val >= 0.5 * s_train, (
            f"{p}: val std {s_val:.4f} < 0.5 * train std {s_train:.4f}"
        )
        assert s_test <= 2.0 * s_train, (
            f"{p}: test std {s_test:.4f} > 2x train std {s_train:.4f} — "
            f"holdout is unrealistically wide vs train."
        )


def test_stratified_split_disjoint(tmp_path):
    """All three splits must be pairwise disjoint at the system level."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=(70, 2))
    sim_tuples = _make_sim_tuples(tmp_path, y, ["a", "b"])

    train, val, test = _stratified_split_systems(
        sim_tuples, stratify_on=["a", "b"],
        val_frac=0.15, test_frac=0.15, split_seed=0,
    )

    train_paths = {t[2] for t in train}
    val_paths   = {t[2] for t in val}
    test_paths  = {t[2] for t in test}
    assert not (train_paths & val_paths)
    assert not (train_paths & test_paths)
    assert not (val_paths & test_paths)
    assert len(train_paths) + len(val_paths) + len(test_paths) == 70


def test_stratified_split_deterministic(tmp_path):
    """Same split_seed → same partition (reproducibility)."""
    rng = np.random.default_rng(0)
    y = rng.normal(size=(70, 2))
    sim_tuples = _make_sim_tuples(tmp_path, y, ["a", "b"])

    a1, b1, c1 = _stratified_split_systems(
        sim_tuples, stratify_on=["a", "b"],
        val_frac=0.15, test_frac=0.15, split_seed=0,
    )
    a2, b2, c2 = _stratified_split_systems(
        sim_tuples, stratify_on=["a", "b"],
        val_frac=0.15, test_frac=0.15, split_seed=0,
    )
    assert [t[2] for t in a1] == [t[2] for t in a2]
    assert [t[2] for t in b1] == [t[2] for t in b2]
    assert [t[2] for t in c1] == [t[2] for t in c2]


def test_stratified_split_4d_tier_a(tmp_path):
    """
    Tier A configuration (4 properties). Variation is intentionally
    near-constant (R²≈0.5 ceiling in real data) — the stratified split
    must still produce non-collapsed std on it.
    """
    rng = np.random.default_rng(1)
    n = 70
    y = np.column_stack([
        rng.uniform(2.5, 4.0, n),       # lipid_packing
        rng.uniform(35.0, 42.0, n),     # thickness
        rng.normal(0.26, 0.008, n),     # variation (narrow)
        rng.uniform(2.0, 2.4, n),       # thickness_std
    ])
    props = ["lipid_packing", "thickness", "variation", "thickness_std"]
    sim_tuples = _make_sim_tuples(tmp_path, y, props)

    train, val, test = _stratified_split_systems(
        sim_tuples, stratify_on=props,
        val_frac=0.15, test_frac=0.15, split_seed=0,
    )

    Y_train = _per_split_std(train, props)
    Y_test  = _per_split_std(test,  props)
    for j, p in enumerate(props):
        ratio = Y_test[:, j].std() / Y_train[:, j].std()
        assert 0.5 <= ratio <= 2.0, (
            f"{p}: test/train std ratio {ratio:.3f} outside [0.5, 2.0]"
        )


def _make_canonical_sim_tuples(tmp_path, comp_names):
    """Build sim_tuples whose tpr paths follow the production layout
    ``<root>/<comp>/run/prun.tpr`` so ``_composition_of`` returns ``<comp>``."""
    sim_tuples = []
    for comp in comp_names:
        run_dir = tmp_path / comp / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        sim_tuples.append((
            run_dir / "prun.tpr",
            run_dir / "prun.xtc",
            tmp_path / f"{comp}.h5",  # h5 sibling, unused in these tests
        ))
    return sim_tuples


def test_composition_of_matches_production_layout(tmp_path):
    sims = _make_canonical_sim_tuples(tmp_path, ["POPC100", "DPPC50_DOPC50"])
    assert _composition_of(sims[0]) == "POPC100"
    assert _composition_of(sims[1]) == "DPPC50_DOPC50"


def test_split_json_roundtrip(tmp_path):
    """Write a split via _write_split_json, read it back via _split_from_json,
    and confirm composition-membership is preserved exactly."""
    comps = [f"SYS{i:02d}" for i in range(20)]
    sims = _make_canonical_sim_tuples(tmp_path, comps)
    train_sims = sims[:14]
    val_sims = sims[14:17]
    test_sims = sims[17:]

    json_path = tmp_path / "split.json"
    _write_split_json(train_sims, val_sims, test_sims, json_path, source_run="unit_test")
    assert json_path.exists()

    train2, val2, test2 = _split_from_json(sims, json_path)
    assert [_composition_of(s) for s in train2] == [_composition_of(s) for s in train_sims]
    assert [_composition_of(s) for s in val2] == [_composition_of(s) for s in val_sims]
    assert [_composition_of(s) for s in test2] == [_composition_of(s) for s in test_sims]


def test_split_from_json_rejects_unassigned_composition(tmp_path):
    """A composition present in sim_tuples but absent from the JSON is a
    hard error — silent drops are how paired comparisons get corrupted."""
    comps = [f"SYS{i:02d}" for i in range(10)]
    sims = _make_canonical_sim_tuples(tmp_path, comps)

    spec = {
        "source_run": "partial",
        "train": comps[:6],
        "val": comps[6:8],
        "test": comps[8:9],  # SYS09 deliberately omitted
    }
    json_path = tmp_path / "partial_split.json"
    with open(json_path, "w") as f:
        import json as _json
        _json.dump(spec, f)

    with pytest.raises(ValueError, match="no split assignment"):
        _split_from_json(sims, json_path)


def test_split_from_json_warns_on_extra(tmp_path, capsys):
    """A composition in the JSON but not in sim_tuples is allowed — just
    warn; the JSON may have been written on a larger corpus."""
    comps = [f"SYS{i:02d}" for i in range(5)]
    sims = _make_canonical_sim_tuples(tmp_path, comps)

    spec = {
        "source_run": "superset",
        "train": comps[:3] + ["SYS99"],  # SYS99 not in sims
        "val": [comps[3]],
        "test": [comps[4]],
    }
    json_path = tmp_path / "superset_split.json"
    with open(json_path, "w") as f:
        import json as _json
        _json.dump(spec, f)

    train, val, test = _split_from_json(sims, json_path)
    assert {_composition_of(s) for s in train} == set(comps[:3])
    assert {_composition_of(s) for s in val} == {comps[3]}
    assert {_composition_of(s) for s in test} == {comps[4]}
    out = capsys.readouterr().out
    assert "not found" in out and "SYS99" in out


def test_split_from_json_rejects_duplicate_membership(tmp_path):
    """A composition that appears in two splits is malformed JSON; the
    loader must refuse rather than picking one silently."""
    comps = [f"SYS{i:02d}" for i in range(4)]
    sims = _make_canonical_sim_tuples(tmp_path, comps)

    spec = {
        "source_run": "bad",
        "train": [comps[0], comps[1]],
        "val": [comps[1]],  # SYS01 in both train and val
        "test": [comps[2], comps[3]],
    }
    json_path = tmp_path / "dup_split.json"
    with open(json_path, "w") as f:
        import json as _json
        _json.dump(spec, f)

    with pytest.raises(ValueError, match="multiple splits"):
        _split_from_json(sims, json_path)
