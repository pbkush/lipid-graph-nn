"""Tests for lipid_gnn.martini_pipeline.lipid_registry."""
from __future__ import annotations

import json
import os

import pytest

from lipid_gnn.martini_pipeline.lipid_registry import (
    LIPID_REGISTRY,
    LipidEntry,
    ResourceCheck,
    check_resources,
    get_lipid,
    lipid_names,
    register_lipid,
    validate_lipid,
)

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LEGACY_ITP_DIR = os.path.join(_REPO_ROOT, "data", "membrane_only", "POPC100", "toppar")
_NODE_MAPPING_PATH = os.path.join(_REPO_ROOT, "resources", "martini_ff_node_mapping.json")
_HAS_LEGACY_ITP = os.path.isdir(_LEGACY_ITP_DIR)
_HAS_NODE_MAPPING = os.path.isfile(_NODE_MAPPING_PATH)

_DEFAULT_LIPIDS = {"DIPC", "DLPC", "DOPC", "DPPC", "POPC", "DOPE", "DPPE", "POPE", "DOPS", "POPS", "CHOL"}
# DLPC is the modern M3 name for the same di-C18:2 PC lipid called DIPC in the
# legacy 70-system corpus.  Both registry entries point at the same itp_file /
# moleculetype / beads — only the user-facing token differs, so the canonical
# composition names can migrate via submit_simulations.sh --rename-lipid.


def _fake_entry(**overrides) -> LipidEntry:
    defaults = dict(
        name="FAKE", resname="FAKE",
        itp_file="fake.itp", moleculetype="FAKE",
        beads=("A", "B", "C"),
        family="phospholipid", insane_keyword="FAKE",
    )
    defaults.update(overrides)
    return LipidEntry(**defaults)


# ---------------------------------------------------------------------------
# 1. Default registry shape
# ---------------------------------------------------------------------------

def test_default_registry_names():
    assert set(lipid_names()) == _DEFAULT_LIPIDS


def test_default_registry_immutable():
    with pytest.raises(TypeError):
        LIPID_REGISTRY["FAKE"] = _fake_entry()  # type: ignore[index]


# ---------------------------------------------------------------------------
# 2. Default entries validate without error
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", sorted(_DEFAULT_LIPIDS))
def test_default_entries_validate(name):
    validate_lipid(get_lipid(name))


# ---------------------------------------------------------------------------
# 3. Bead cross-check against resources/martini_ff_node_mapping.json
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_NODE_MAPPING, reason="resources/martini_ff_node_mapping.json not present")
@pytest.mark.parametrize("name", sorted(_DEFAULT_LIPIDS))
def test_default_beads_match_node_mapping(name):
    with open(_NODE_MAPPING_PATH) as fh:
        mapping = json.load(fh)
    assert name in mapping, f"Lipid {name!r} missing from node mapping."
    expected = tuple(mapping[name].keys())
    entry = get_lipid(name)
    assert entry.beads == expected, (
        f"Bead mismatch for {name!r}. "
        f"Registry: {entry.beads}. Node mapping: {expected}."
    )


# ---------------------------------------------------------------------------
# 4. get_lipid — unknown name raises
# ---------------------------------------------------------------------------

def test_get_lipid_known():
    entry = get_lipid("POPC")
    assert entry.name == "POPC"
    assert entry.family == "phospholipid"


def test_get_lipid_unknown_raises():
    with pytest.raises(KeyError, match="NOPE"):
        get_lipid("NOPE")


def test_get_lipid_unknown_error_lists_known():
    with pytest.raises(KeyError) as exc_info:
        get_lipid("NOPE")
    msg = str(exc_info.value)
    assert any(k in msg for k in ["POPC", "DOPC"])


# ---------------------------------------------------------------------------
# 5. register_lipid
# ---------------------------------------------------------------------------

def test_register_lipid_happy():
    new_reg = register_lipid(LIPID_REGISTRY, _fake_entry())
    assert "FAKE" in new_reg
    assert new_reg["FAKE"].name == "FAKE"
    assert "FAKE" not in LIPID_REGISTRY  # original unchanged


def test_register_lipid_returns_dict():
    new_reg = register_lipid(LIPID_REGISTRY, _fake_entry())
    assert isinstance(new_reg, dict)


def test_register_lipid_duplicate_raises():
    with pytest.raises(ValueError, match="already registered"):
        register_lipid(LIPID_REGISTRY, _fake_entry(name="POPC", resname="POPC",
                                                    moleculetype="POPC", insane_keyword="POPC"))


def test_register_lipid_invalid_entry_raises():
    with pytest.raises(ValueError):
        register_lipid(LIPID_REGISTRY, _fake_entry(name=""))


def test_register_lipid_chained():
    r1 = register_lipid(LIPID_REGISTRY, _fake_entry(name="FAKE1", insane_keyword="FAKE1"))
    r2 = register_lipid(r1, _fake_entry(name="FAKE2", insane_keyword="FAKE2"))
    assert "FAKE1" in r2
    assert "FAKE2" in r2
    assert "FAKE1" not in LIPID_REGISTRY


# ---------------------------------------------------------------------------
# 6. validate_lipid — invalid entries
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("overrides, match", [
    ({"name": ""}, "non-empty"),
    ({"name": "popc"}, "uppercase"),
    ({"name": "123"}, "uppercase"),
    ({"resname": ""}, "non-empty"),
    ({"itp_file": ""}, "non-empty"),
    ({"moleculetype": ""}, "non-empty"),
    ({"beads": ()}, "non-empty"),
    ({"beads": ("A", "A", "B")}, "Duplicate"),
    ({"family": "glycolipid"}, "_KNOWN_FAMILIES"),
    ({"family": ""}, "_KNOWN_FAMILIES"),
    ({"insane_keyword": ""}, "non-empty"),
])
def test_validate_lipid_invalid(overrides, match):
    with pytest.raises(ValueError, match=match):
        validate_lipid(_fake_entry(**overrides))


# ---------------------------------------------------------------------------
# 7. check_resources — tmp_path fixtures
# ---------------------------------------------------------------------------

def test_check_resources_all_skipped():
    entry = get_lipid("POPC")
    result = check_resources(entry)
    assert result.ok
    assert result.itp_present is None
    assert result.moleculetype_declared is None
    assert result.beads_match_node_mapping is None
    assert result.errors == ()


def test_check_resources_itp_present_and_moleculetype_declared(tmp_path):
    itp_content = (
        "[ moleculetype ]\n"
        "; molname  nrexcl\n"
        "FAKE   1\n"
        "\n"
        "[ atoms ]\n"
        "; dummy\n"
    )
    itp_file = tmp_path / "fake.itp"
    itp_file.write_text(itp_content)
    entry = _fake_entry()
    result = check_resources(entry, itp_dir=tmp_path)
    assert result.itp_present is True
    assert result.moleculetype_declared is True
    assert result.ok


def test_check_resources_itp_missing(tmp_path):
    entry = _fake_entry()
    result = check_resources(entry, itp_dir=tmp_path)
    assert result.itp_present is False
    assert result.moleculetype_declared is None
    assert not result.ok
    assert any("not found" in e for e in result.errors)


def test_check_resources_moleculetype_missing(tmp_path):
    itp_content = (
        "[ moleculetype ]\n"
        "; molname  nrexcl\n"
        "OTHER   1\n"
    )
    (tmp_path / "fake.itp").write_text(itp_content)
    entry = _fake_entry()
    result = check_resources(entry, itp_dir=tmp_path)
    assert result.itp_present is True
    assert result.moleculetype_declared is False
    assert not result.ok
    assert any("FAKE" in e for e in result.errors)


def test_check_resources_node_mapping_match(tmp_path):
    nm = tmp_path / "mapping.json"
    nm.write_text(json.dumps({"FAKE": {"A": "type1", "B": "type2", "C": "type3"}}))
    entry = _fake_entry()
    result = check_resources(entry, node_mapping_path=nm)
    assert result.beads_match_node_mapping is True
    assert result.ok


def test_check_resources_node_mapping_bead_mismatch(tmp_path):
    nm = tmp_path / "mapping.json"
    nm.write_text(json.dumps({"FAKE": {"X": "type1", "Y": "type2"}}))
    entry = _fake_entry()
    result = check_resources(entry, node_mapping_path=nm)
    assert result.beads_match_node_mapping is False
    assert not result.ok
    assert any("mismatch" in e.lower() for e in result.errors)


def test_check_resources_node_mapping_lipid_absent(tmp_path):
    nm = tmp_path / "mapping.json"
    nm.write_text(json.dumps({"OTHER": {"A": "t"}}))
    entry = _fake_entry()
    result = check_resources(entry, node_mapping_path=nm)
    assert result.beads_match_node_mapping is False
    assert not result.ok
    assert any("FAKE" in e for e in result.errors)


def test_check_resources_node_mapping_file_missing(tmp_path):
    entry = _fake_entry()
    result = check_resources(entry, node_mapping_path=tmp_path / "missing.json")
    assert result.beads_match_node_mapping is False
    assert not result.ok
    assert any("not found" in e for e in result.errors)


def test_check_resources_combined_all_pass(tmp_path):
    itp_content = (
        "[ moleculetype ]\n"
        "; comment\n"
        "FAKE   1\n"
    )
    (tmp_path / "fake.itp").write_text(itp_content)
    nm = tmp_path / "mapping.json"
    nm.write_text(json.dumps({"FAKE": {"A": "t1", "B": "t2", "C": "t3"}}))
    entry = _fake_entry()
    result = check_resources(entry, itp_dir=tmp_path, node_mapping_path=nm)
    assert result.ok
    assert result.itp_present is True
    assert result.moleculetype_declared is True
    assert result.beads_match_node_mapping is True


# ---------------------------------------------------------------------------
# 8. Integration: check_resources against legacy data
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not (_HAS_LEGACY_ITP and _HAS_NODE_MAPPING),
    reason="legacy data or node mapping not present",
)
@pytest.mark.parametrize(
    "name",
    # DIPC and DLPC excluded: both registry entries point to the v2 ITP
    # (DLPC moleculetype, 12 beads), while the legacy ITP set uses v1
    # (DIPC moleculetype).  Same root cause; see lipid_registry.py comment
    # on the DIPC/DLPC entries and Decision 49 in martini_pipeline_plan.md.
    sorted(_DEFAULT_LIPIDS - {"DIPC", "DLPC"}),
)
def test_check_resources_legacy_integration(name):
    entry = get_lipid(name)
    result = check_resources(
        entry,
        itp_dir=_LEGACY_ITP_DIR,
        node_mapping_path=_NODE_MAPPING_PATH,
    )
    assert result.ok, f"check_resources failed for {name!r}: {result.errors}"
