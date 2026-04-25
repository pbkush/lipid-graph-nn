"""
Tests for the stratified system-level split in
scripts/training/prepare_colab_subset._stratified_split_systems.

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

from scripts.training.prepare_colab_subset import _stratified_split_systems


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
