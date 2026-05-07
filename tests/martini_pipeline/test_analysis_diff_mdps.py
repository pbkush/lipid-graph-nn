"""Tests for lipid_gnn.martini_pipeline.analysis.diff_mdps (step 3)."""
from __future__ import annotations

import json
import os
import shutil
import stat
import textwrap

import pytest

from lipid_gnn.martini_pipeline.analysis import (
    MDPAuditReport,
    MDPDeviation,
    MDPStageAudit,
    _audit_stage,
    _canonicalise_key,
    _dump_tpr_inputrec,
    _mode,
    _normalise_value,
    _parse_mdp_file,
    diff_mdps,
)

_FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures", "mdp")
_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_LEGACY_ROOT = os.path.join(_REPO_ROOT, "data", "membrane_only")
_HAS_LEGACY_DATA = os.path.isdir(_LEGACY_ROOT)
_HAS_GMX = shutil.which("gmx") is not None


def _write_gmx_mock(tmp_path, inputrec_block: str) -> str:
    """Write a fake gmx script that prints a hand-crafted inputrec block to stdout."""
    script = tmp_path / "gmx"
    script.write_text(
        "#!/bin/sh\n"
        'echo "inputrec:"\n'
        + "".join(f'echo "   {line}"\n' for line in inputrec_block.splitlines())
        + 'echo "topology:"\n'
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def _make_system_tree(tmp_path, systems: dict[str, dict[str, str]]) -> str:
    """Build a mini systems_root with run.mdp per system."""
    root = tmp_path / "systems"
    root.mkdir()
    for name, kv in systems.items():
        sys_dir = root / name
        sys_dir.mkdir()
        lines = "\n".join(f"{k} = {v}" for k, v in kv.items())
        (sys_dir / "run.mdp").write_text(lines + "\n")
    return str(root)


# ---------------------------------------------------------------------------
# 1. _parse_mdp_file
# ---------------------------------------------------------------------------

def test_parse_mdp_simple():
    parsed = _parse_mdp_file(os.path.join(_FIXTURES, "simple.mdp"))
    assert parsed["integrator"] == "md"
    assert parsed["dt"] == "0.02"
    assert parsed["nsteps"] == "1000"
    assert parsed["ref-t"] == "310"
    assert "inline comment" not in parsed.get("ref-t", "")


def test_parse_mdp_inline_comment_stripped():
    parsed = _parse_mdp_file(os.path.join(_FIXTURES, "simple.mdp"))
    assert parsed["ref-t"] == "310"


def test_parse_mdp_duplicate_key_last_wins():
    parsed = _parse_mdp_file(os.path.join(_FIXTURES, "duplicate_key.mdp"))
    assert parsed["integrator"] == "md"


def test_parse_mdp_underscore_to_hyphen():
    parsed = _parse_mdp_file(os.path.join(_FIXTURES, "underscore_keys.mdp"))
    assert "ref-t" in parsed
    assert "tau-t" in parsed
    assert "tc-grps" in parsed
    assert "ref_t" not in parsed


def test_parse_mdp_blank_and_comment_lines_ignored():
    parsed = _parse_mdp_file(os.path.join(_FIXTURES, "blank_and_comment.mdp"))
    assert "integrator" in parsed
    assert "dt" in parsed
    assert len(parsed) == 2


# ---------------------------------------------------------------------------
# 2. _dump_tpr_inputrec with mock gmx
# ---------------------------------------------------------------------------

def test_dump_tpr_inputrec_mock(tmp_path):
    inputrec = textwrap.dedent("""\
        integrator                     = md
        dt                             = 0.01
        nsteps                         = 250000
        tcoupl                         = V-rescale
        ref-t                          = 310
    """)
    gmx = _write_gmx_mock(tmp_path, inputrec)
    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = str(tmp_path) + os.pathsep + old_path
    try:
        result = _dump_tpr_inputrec(tmp_path / "fake.tpr", gmx_binary=gmx)
    finally:
        os.environ["PATH"] = old_path
    assert result["integrator"] == "md"
    assert result["dt"] == "0.01"
    assert result["nsteps"] == "250000"
    assert result["ref-t"] == "310"


def test_dump_tpr_inputrec_skips_matrix_lines(tmp_path):
    inputrec = textwrap.dedent("""\
        integrator                     = md
        compressibility (3x3):
            compressibility[    0]={ 3.00000e-05,  0.00000e+00,  0.00000e+00}
        ref-t                          = 310
    """)
    gmx = _write_gmx_mock(tmp_path, inputrec)
    result = _dump_tpr_inputrec(tmp_path / "fake.tpr", gmx_binary=gmx)
    assert "ref-t" in result
    assert not any("compressibility" in k and "[" in k for k in result)


# ---------------------------------------------------------------------------
# 3. _dump_tpr_inputrec gmx failure modes
# ---------------------------------------------------------------------------

def test_dump_tpr_inputrec_gmx_nonzero(tmp_path):
    script = tmp_path / "gmx"
    script.write_text("#!/bin/sh\necho 'error output' >&2\nexit 1\n")
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(RuntimeError, match="gmx dump failed"):
        _dump_tpr_inputrec(tmp_path / "fake.tpr", gmx_binary=str(script))


def test_dump_tpr_inputrec_gmx_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        _dump_tpr_inputrec(tmp_path / "fake.tpr", gmx_binary="/nonexistent/gmx")


# ---------------------------------------------------------------------------
# 4. _mode helper
# ---------------------------------------------------------------------------

def test_mode_majority():
    assert _mode(["a", "b", "a", "a"]) == "a"


def test_mode_tie_first_seen():
    assert _mode(["b", "a", "b", "a"]) == "b"


# ---------------------------------------------------------------------------
# 5. diff_mdps over synthetic tree — deviations detected
# ---------------------------------------------------------------------------

def test_diff_mdps_identical_systems(tmp_path):
    root = _make_system_tree(tmp_path, {
        "SYS1": {"dt": "0.02", "nsteps": "1000"},
        "SYS2": {"dt": "0.02", "nsteps": "1000"},
        "SYS3": {"dt": "0.02", "nsteps": "1000"},
    })
    report = diff_mdps(root, stages=("run",))
    run = report.stages["run"]
    assert run.n_systems == 3
    assert run.deviations == ()
    assert run.canonical["dt"] == "0.02"


def test_diff_mdps_one_dissenter(tmp_path):
    root = _make_system_tree(tmp_path, {
        "SYS1": {"dt": "0.02", "ref-t": "310"},
        "SYS2": {"dt": "0.02", "ref-t": "310"},
        "SYS3": {"dt": "0.02", "ref-t": "320"},
    })
    report = diff_mdps(root, stages=("run",))
    run = report.stages["run"]
    assert run.canonical["ref-t"] == "310"
    assert len(run.deviations) == 1
    assert run.deviations[0].system == "SYS3"
    assert run.deviations[0].value == "320"
    assert run.deviations[0].canonical == "310"


# ---------------------------------------------------------------------------
# 6. Skip keys — gen-seed differences not reported
# ---------------------------------------------------------------------------

def test_diff_mdps_skips_seed_keys(tmp_path):
    root = _make_system_tree(tmp_path, {
        "SYS1": {"dt": "0.02", "gen-seed": "111"},
        "SYS2": {"dt": "0.02", "gen-seed": "222"},
        "SYS3": {"dt": "0.02", "gen-seed": "333"},
    })
    report = diff_mdps(root, stages=("run",))
    assert report.stages["run"].deviations == ()
    assert "gen-seed" not in report.stages["run"].canonical


def test_diff_mdps_custom_skip_keys(tmp_path):
    root = _make_system_tree(tmp_path, {
        "SYS1": {"dt": "0.02", "mykey": "x"},
        "SYS2": {"dt": "0.02", "mykey": "y"},
    })
    report = diff_mdps(root, stages=("run",), skip_keys=frozenset({"mykey"}))
    assert report.stages["run"].deviations == ()


# ---------------------------------------------------------------------------
# 7. Missing files handling
# ---------------------------------------------------------------------------

def test_diff_mdps_missing_run_mdp(tmp_path):
    root = tmp_path / "systems"
    root.mkdir()
    (root / "SYS1").mkdir()
    (root / "SYS1" / "run.mdp").write_text("dt = 0.02\n")
    (root / "SYS2").mkdir()
    # SYS2 has no run.mdp

    report = diff_mdps(str(root), stages=("run",))
    run = report.stages["run"]
    assert run.n_systems == 1
    assert "SYS2" in run.missing_systems


def test_diff_mdps_empty_root(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    report = diff_mdps(str(root), stages=("run",))
    assert report.stages["run"].n_systems == 0
    assert report.total_deviations == 0


# ---------------------------------------------------------------------------
# 8. to_markdown and to_freeze_json
# ---------------------------------------------------------------------------

def test_to_markdown_structure(tmp_path):
    root = _make_system_tree(tmp_path, {
        "SYS1": {"dt": "0.02", "nsteps": "1000"},
        "SYS2": {"dt": "0.02", "nsteps": "2000"},
    })
    report = diff_mdps(root, stages=("run",))
    md = report.to_markdown()
    assert "## Stage: `run`" in md
    assert "Canonical parameter values" in md
    assert "Deviations from canonical" in md
    assert "dt" in md
    assert "nsteps" in md


def test_to_markdown_cross_stage_section(tmp_path):
    root = _make_system_tree(tmp_path, {
        "SYS1": {"dt": "0.02", "nsteps": "1000"},
    })
    # Patch to include a second fake stage by using two separate single-stage reports
    report = diff_mdps(root, stages=("run",))
    md = report.to_markdown()
    # Single-stage report has no cross-stage section
    assert "Cross-stage" not in md


def test_to_freeze_json(tmp_path):
    root = _make_system_tree(tmp_path, {
        "SYS1": {"dt": "0.02", "nsteps": "1000"},
        "SYS2": {"dt": "0.02", "nsteps": "1000"},
    })
    report = diff_mdps(root, stages=("run",))
    freeze = json.loads(report.to_freeze_json())
    assert "run" in freeze
    assert freeze["run"]["dt"] == "0.02"
    assert freeze["run"]["nsteps"] == "1000"


def test_total_deviations_property(tmp_path):
    root = _make_system_tree(tmp_path, {
        "SYS1": {"dt": "0.02"},
        "SYS2": {"dt": "0.03"},
    })
    report = diff_mdps(root, stages=("run",))
    assert report.total_deviations == 1


# ---------------------------------------------------------------------------
# 9. Legacy integration tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _HAS_LEGACY_DATA, reason="legacy data not present")
def test_diff_mdps_legacy_run_stage_no_deviations():
    """All 70 run.mdp files are byte-identical — verified during planning via md5."""
    report = diff_mdps(_LEGACY_ROOT, stages=("run",))
    run = report.stages["run"]
    assert run.n_systems == 70
    assert run.deviations == (), f"Unexpected run deviations: {run.deviations}"
    assert run.missing_systems == ()


@pytest.mark.skipif(not _HAS_LEGACY_DATA, reason="legacy data not present")
def test_diff_mdps_legacy_run_canonical_values():
    report = diff_mdps(_LEGACY_ROOT, stages=("run",))
    canon = report.stages["run"].canonical
    assert canon["dt"] == "0.02"
    assert canon["ref-t"] == "310"
    assert canon["cutoff-scheme"].lower() == "verlet"
    assert canon["coulombtype"].lower() == "cutoff"
    for key in ("nsteps", "tau-t", "tau-p", "rcoulomb", "rvdw"):
        assert key in canon, f"Expected key {key!r} in canonical"


@pytest.mark.skipif(
    not (_HAS_LEGACY_DATA and _HAS_GMX),
    reason="legacy data or gmx not present",
)
def test_diff_mdps_legacy_all_stages():
    report = diff_mdps(_LEGACY_ROOT, stages=("run", "equilibration", "minimization"))
    for stage in ("run", "equilibration", "minimization"):
        assert stage in report.stages
        audit = report.stages[stage]
        assert audit.n_systems == 70, f"{stage}: expected 70 systems, got {audit.n_systems}"
    # Cross-stage report should render without error
    md = report.to_markdown()
    assert "Cross-stage parameter comparison" in md


@pytest.mark.skipif(
    not (_HAS_LEGACY_DATA and _HAS_GMX),
    reason="legacy data or gmx not present",
)
def test_diff_mdps_legacy_cross_stage_integrator_differs():
    """Integrator should differ across stages: md (run/eq) vs steep (minimization)."""
    report = diff_mdps(_LEGACY_ROOT, stages=("run", "equilibration", "minimization"))
    run_int = report.stages["run"].canonical.get("integrator", "")
    em_int = report.stages["minimization"].canonical.get("integrator", "")
    assert run_int.lower() == "md"
    assert em_int.lower() == "steep"
