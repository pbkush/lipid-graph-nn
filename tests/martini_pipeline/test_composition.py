"""Tests for lipid_gnn.martini_pipeline.composition."""
from __future__ import annotations

import math
import os

import pytest

from lipid_gnn.martini_pipeline.composition import (
    Composition,
    counts_per_leaflet,
    parse_name,
    validate_fractions,
)

_LEGACY_DATA_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "data", "membrane_only")
)
_HAS_LEGACY_DATA = os.path.isdir(_LEGACY_DATA_ROOT)


def _legacy_names() -> list[str]:
    if not _HAS_LEGACY_DATA:
        return []
    return sorted(
        entry for entry in os.listdir(_LEGACY_DATA_ROOT)
        if os.path.isdir(os.path.join(_LEGACY_DATA_ROOT, entry))
    )


# ---------------------------------------------------------------------------
# 1. validate_fractions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fractions", [
    {"POPC": 1.0},
    {"POPC": 0.3, "DOPC": 0.7},
    {"POPC": 0.5, "DOPC": 0.5},
    {"A": 0.5, "B": 0.3, "C": 0.2},
    {"POPC": 0.7, "DOPC": 0.3},
])
def test_validate_fractions_valid(fractions):
    validate_fractions(fractions)


@pytest.mark.parametrize("fractions, match", [
    ({}, "at least one"),
    ({"POPC": 1.5}, "must be in"),
    ({"POPC": -0.1, "DOPC": 1.1}, "must be in"),
    ({"POPC": 0.6, "DOPC": 0.5}, "sum to 1.0"),
    ({"POPC": 0.3, "DOPC": 0.6}, "sum to 1.0"),
    ({"POPC": 0.0, "DOPC": 1.0}, "must be in"),
])
def test_validate_fractions_invalid(fractions, match):
    with pytest.raises(ValueError, match=match):
        validate_fractions(fractions)


# ---------------------------------------------------------------------------
# 2. Composition construction
# ---------------------------------------------------------------------------

def test_composition_valid():
    comp = Composition({"POPC": 0.3, "DOPC": 0.7})
    assert math.isclose(comp.fractions["POPC"], 0.3, rel_tol=1e-6)
    assert math.isclose(comp.fractions["DOPC"], 0.7, rel_tol=1e-6)


def test_composition_non_integer_percentage_raises():
    """Fractions that cannot be expressed as integer percentages must be rejected."""
    with pytest.raises(ValueError, match="integer"):
        Composition({"POPC": 1 / 3, "DOPC": 1 / 3, "DPPC": 1 / 3})


def test_composition_percentages_not_summing_to_100_raises():
    """Even if fractions sum to ~1.0 they must produce integer percentages summing to 100."""
    # 0.33 + 0.67 = 1.00 but 33 + 67 = 100 → valid
    Composition({"POPC": 0.33, "DOPC": 0.67})
    # 0.33 + 0.66 = 0.99 → validate_fractions catches sum error before integer check
    with pytest.raises(ValueError, match="sum to 1.0"):
        Composition({"POPC": 0.33, "DOPC": 0.66})


# ---------------------------------------------------------------------------
# 3. Canonical naming (Decision 7: descending fraction, alpha tiebreak)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fractions, expected_name", [
    ({"POPC": 1.0}, "POPC100"),
    ({"DOPC": 1.0}, "DOPC100"),
    ({"POPC": 0.3, "DOPC": 0.7}, "DOPC70_POPC30"),
    ({"DOPC": 0.7, "POPC": 0.3}, "DOPC70_POPC30"),   # insertion order irrelevant
    ({"POPC": 0.5, "DOPC": 0.5}, "DOPC50_POPC50"),   # tie: D < P alphabetically
    ({"POPC": 0.5, "DPPC": 0.5}, "DPPC50_POPC50"),   # tie: D < P
    ({"A": 0.5, "B": 0.3, "C": 0.2}, "A50_B30_C20"),
    ({"POPC": 0.7, "DOPC": 0.3}, "POPC70_DOPC30"),   # POPC dominant
    ({"POPC": 0.9, "CHOL": 0.1}, "POPC90_CHOL10"),   # legacy-style name preserved for POPC-dominant
])
def test_canonical_name(fractions, expected_name):
    assert Composition(fractions).name == expected_name


def test_lipid_types_order():
    comp = Composition({"POPC": 0.3, "DOPC": 0.7})
    assert comp.lipid_types == ("DOPC", "POPC")


# ---------------------------------------------------------------------------
# 4. parse_name — valid inputs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected_fracs", [
    ("POPC100", {"POPC": 1.0}),
    ("DOPC100", {"DOPC": 1.0}),
    ("DOPC70_POPC30", {"DOPC": 0.7, "POPC": 0.3}),
    ("POPC30_DOPC70", {"POPC": 0.3, "DOPC": 0.7}),   # non-canonical input parses fine
    ("A50_B30_C20", {"A": 0.5, "B": 0.3, "C": 0.2}),
    ("POPC90_CHOL10", {"POPC": 0.9, "CHOL": 0.1}),
])
def test_parse_name_valid(name, expected_fracs):
    comp = parse_name(name)
    for lipid, frac in expected_fracs.items():
        assert math.isclose(comp.fractions[lipid], frac, rel_tol=1e-6)


@pytest.mark.parametrize("name, match", [
    ("", "empty"),
    ("POPC30", "sum to 1.0"),           # single lipid at 30% → fracs sum to 0.3
    ("POPC30_DOPC60", "sum to 1.0"),    # 30 + 60 = 90 → fracs sum to 0.9
    ("POPC30_POPC70", "Duplicate"),
    ("popc30_dopc70", "Invalid token"),
    ("POPC_30", "Invalid token"),
    ("30POPC", "Invalid token"),
])
def test_parse_name_invalid(name, match):
    with pytest.raises(ValueError, match=match):
        parse_name(name)


# ---------------------------------------------------------------------------
# 5. Round-trip: parse_name(comp.name) reproduces comp.fractions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fractions", [
    {"POPC": 1.0},
    {"POPC": 0.3, "DOPC": 0.7},
    {"POPC": 0.5, "DOPC": 0.5},
    {"A": 0.5, "B": 0.3, "C": 0.2},
    {"POPC": 0.7, "DOPC": 0.3},
    {"POPC": 0.9, "CHOL": 0.1},
])
def test_round_trip(fractions):
    comp = Composition(fractions)
    recovered = parse_name(comp.name)
    assert set(comp.fractions) == set(recovered.fractions)
    for lipid in comp.fractions:
        assert math.isclose(comp.fractions[lipid], recovered.fractions[lipid], rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 6. counts_per_leaflet
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("fractions, n, expected", [
    ({"POPC": 1.0}, 100, {"POPC": 100}),
    ({"POPC": 0.3, "DOPC": 0.7}, 100, {"POPC": 30, "DOPC": 70}),
    ({"POPC": 0.5, "DOPC": 0.5}, 200, {"POPC": 100, "DOPC": 100}),
    ({"POPC": 0.7, "DOPC": 0.3}, 100, {"POPC": 70, "DOPC": 30}),
    ({"POPC": 0.3, "DOPC": 0.7}, 10, {"POPC": 3, "DOPC": 7}),
    ({"POPC": 0.5, "DOPC": 0.5}, 2, {"POPC": 1, "DOPC": 1}),
])
def test_counts_per_leaflet_valid(fractions, n, expected):
    comp = Composition(fractions)
    counts = counts_per_leaflet(comp, n)
    assert counts == expected
    assert sum(counts.values()) == n


def test_counts_per_leaflet_non_integer_raises():
    """30% * 33 = 9.9 — not an integer count."""
    comp = Composition({"POPC": 0.3, "DOPC": 0.7})
    with pytest.raises(ValueError, match="non-integer"):
        counts_per_leaflet(comp, 33)


def test_counts_per_leaflet_single_lipid():
    comp = Composition({"POPC": 1.0})
    assert counts_per_leaflet(comp, 128) == {"POPC": 128}


# ---------------------------------------------------------------------------
# 7. Immutability
# ---------------------------------------------------------------------------

def test_immutability_fractions_mapping():
    """fractions is a MappingProxyType: item assignment must raise TypeError."""
    comp = Composition({"POPC": 1.0})
    with pytest.raises(TypeError):
        comp.fractions["POPC"] = 0.5  # type: ignore[index]


def test_immutability_attribute():
    """Replacing the fractions attribute on a frozen dataclass must raise."""
    comp = Composition({"POPC": 1.0})
    with pytest.raises(Exception):   # dataclasses.FrozenInstanceError ⊂ AttributeError
        comp.fractions = {"POPC": 0.5}  # type: ignore[misc]


def test_hashable():
    """Compositions must be usable as dict keys and set members."""
    c1 = Composition({"POPC": 0.3, "DOPC": 0.7})
    c2 = Composition({"DOPC": 0.7, "POPC": 0.3})
    assert hash(c1) == hash(c2)
    assert c1 == c2
    seen = {c1}
    assert c2 in seen


# ---------------------------------------------------------------------------
# 8. Legacy name compatibility
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_LEGACY_DATA, reason="legacy data not available")
@pytest.mark.parametrize("legacy_name", _legacy_names())
def test_legacy_names_parse(legacy_name):
    """Every name under data/membrane_only/ must parse without error."""
    comp = parse_name(legacy_name)
    assert isinstance(comp, Composition)


@pytest.mark.skipif(not _HAS_LEGACY_DATA, reason="legacy data not available")
@pytest.mark.parametrize("legacy_name", _legacy_names())
def test_legacy_names_fractions_round_trip(legacy_name):
    """Parsing a legacy name and re-serialising preserves the fractions exactly."""
    comp = parse_name(legacy_name)
    recovered = parse_name(comp.name)
    assert set(comp.fractions) == set(recovered.fractions)
    for lipid in comp.fractions:
        assert math.isclose(comp.fractions[lipid], recovered.fractions[lipid], rel_tol=1e-6)


# ---------------------------------------------------------------------------
# 9. Invalid name characters (whitespace, lowercase, non-ASCII)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    " POPC100",
    "POPC100 ",
    "POPC 100",
    "POPC\t100",
    "POPC\n100",
    "popc100",
    "Popc100",
    "POPC100_",
    "_POPC100",
])
def test_invalid_name_characters(name):
    with pytest.raises(ValueError):
        parse_name(name)
