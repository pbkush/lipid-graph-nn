"""Tests for scripts/python/scan_completed_systems.py.

Runs the script as a subprocess against synthetic output roots.  The script's
core function (scan_root) is also unit-tested directly.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_SCRIPT = _REPO_ROOT / "scripts" / "python" / "scan_completed_systems.py"

# Allow direct-import for unit-testing the helper without the CLI wrapping.
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "python"))
from scan_completed_systems import scan_root  # noqa: E402


def _make_legacy_dir(root: Path, dirname: str) -> None:
    """Legacy system: directory with run/prun.xtc but no manifest."""
    d = root / dirname / "run"
    d.mkdir(parents=True)
    (d / "prun.xtc").write_bytes(b"\x00")


def _make_pipeline_dir(root: Path, dirname: str, status: str = "ok") -> None:
    """Pipeline system: directory with manifest.json overall_status=<status>."""
    d = root / dirname
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({
        "schema_version": "1.0",
        "overall_status": status,
        "stages": [],
    }))


class TestScanRoot(unittest.TestCase):
    def test_pipeline_dir_with_ok_manifest(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_pipeline_dir(root, "POPC100", "ok")
            rows = scan_root(root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["canonical_name"], "POPC100")
        self.assertEqual(rows[0]["status"], "ok")
        self.assertEqual(rows[0]["source_root"], str(root))

    def test_legacy_dir_canonicalises_non_canonical_name(self):
        """Non-canonical legacy dirname (POPC10_DIPC90) → canonical (DIPC90_POPC10)."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_legacy_dir(root, "POPC10_DIPC90")
            rows = scan_root(root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["canonical_name"], "DIPC90_POPC10")
        self.assertEqual(rows[0]["source_dir"], "POPC10_DIPC90")
        self.assertEqual(rows[0]["status"], "legacy_no_manifest")

    def test_unparseable_dirname_excluded(self):
        """A non-composition directory (e.g. 'fixtures') is skipped, not crashed-on."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_legacy_dir(root, "fixtures")
            _make_pipeline_dir(root, "POPC100", "ok")
            rows = scan_root(root)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["canonical_name"], "POPC100")

    def test_nonexistent_root_returns_empty(self):
        rows = scan_root(Path("/tmp/_definitely_not_a_real_root_xyz"))
        self.assertEqual(rows, [])

    def test_directory_without_manifest_or_xtc_excluded(self):
        """A dir with neither manifest nor prun.xtc is NOT marked as done."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "POPC100").mkdir()
            rows = scan_root(root)
        self.assertEqual(rows, [])


class TestScanCli(unittest.TestCase):
    def test_cli_writes_csv_with_header(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_pipeline_dir(root, "POPC100", "ok")
            _make_legacy_dir(root, "POPC10_DIPC90")  # non-canonical
            out_csv = root / "done.csv"
            result = subprocess.run(
                [sys.executable, str(_SCRIPT),
                 "--output-roots", str(root),
                 "--out", str(out_csv)],
                capture_output=True, text=True, cwd=str(_REPO_ROOT),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(out_csv.exists())
            with open(out_csv) as fh:
                rows = list(csv.DictReader(fh))
            names = {r["canonical_name"] for r in rows}
            # Both should appear, with the legacy one canonicalised
            self.assertEqual(names, {"POPC100", "DIPC90_POPC10"})

    def test_cli_dedupes_across_roots_first_wins(self):
        """Same canonical name in two roots → only one row."""
        with tempfile.TemporaryDirectory() as d:
            root1 = Path(d) / "r1"; root1.mkdir()
            root2 = Path(d) / "r2"; root2.mkdir()
            _make_legacy_dir(root1, "POPC100")
            _make_pipeline_dir(root2, "POPC100", "ok")
            out_csv = Path(d) / "done.csv"
            result = subprocess.run(
                [sys.executable, str(_SCRIPT),
                 "--output-roots", str(root1), str(root2),
                 "--out", str(out_csv)],
                capture_output=True, text=True, cwd=str(_REPO_ROOT),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out_csv) as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            # First root wins (root1 is legacy)
            self.assertEqual(rows[0]["status"], "legacy_no_manifest")

    def test_cli_status_filter_drops_failed(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            _make_pipeline_dir(root, "POPC100", "ok")
            _make_pipeline_dir(root, "DPPC100", "failed_at_equilibration")
            out_csv = root / "done.csv"
            result = subprocess.run(
                [sys.executable, str(_SCRIPT),
                 "--output-roots", str(root),
                 "--status-filter", "ok",
                 "--out", str(out_csv)],
                capture_output=True, text=True, cwd=str(_REPO_ROOT),
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with open(out_csv) as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["canonical_name"], "POPC100")


if __name__ == "__main__":
    unittest.main()
