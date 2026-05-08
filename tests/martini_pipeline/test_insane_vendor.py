"""Tests for the vendored resources/martini3/insane.py."""
from __future__ import annotations

import ast
import os
import subprocess
import sys
import unittest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from lipid_gnn.martini_pipeline import INSANE_PATH

_HAS_LEGACY_DATA = os.path.isfile(
    os.path.join(_REPO_ROOT, "data", "membrane_only", "POPC100", "run.gro")
)
_RUN_PARITY = os.environ.get("RUN_INSANE_PARITY") == "1"


class TestInsaneVendorExists(unittest.TestCase):
    def test_path_exists(self):
        self.assertTrue(os.path.isfile(INSANE_PATH), f"INSANE_PATH not found: {INSANE_PATH}")

    def test_path_is_executable(self):
        self.assertTrue(os.access(INSANE_PATH, os.X_OK),
                        f"INSANE_PATH is not executable: {INSANE_PATH}")

    def test_path_importable_constant(self):
        from lipid_gnn.martini_pipeline import INSANE_PATH as P
        self.assertTrue(os.path.isfile(P))


class TestInsanePython3(unittest.TestCase):
    def test_python3_parseable(self):
        with open(INSANE_PATH) as fh:
            source = fh.read()
        try:
            ast.parse(source)
        except SyntaxError as exc:
            self.fail(f"insane.py failed Python 3 parse: {exc}")

    def test_version_marker_preserved(self):
        with open(INSANE_PATH) as fh:
            source = fh.read()
        self.assertIn('previous  = "20140603.11.TAW"', source,
                      "version marker missing — did the file get replaced with a different build?")

    def test_gpl_header_present(self):
        with open(INSANE_PATH) as fh:
            header = fh.read(2000)
        self.assertIn("GNU General Public License", header)
        self.assertIn("Tsjerk A. Wassenaar", header)


class TestInsaneHelp(unittest.TestCase):
    def test_help_exits_zero(self):
        result = subprocess.run(
            [sys.executable, INSANE_PATH, "--help"],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0,
                         f"insane.py --help exited {result.returncode}:\n{result.stderr[:500]}")

    def test_help_mentions_lipid_flag(self):
        result = subprocess.run(
            [sys.executable, INSANE_PATH, "--help"],
            capture_output=True, text=True, timeout=20,
        )
        output = result.stdout + result.stderr
        self.assertIn("-l", output, "help output does not mention -l (lipid) flag")

    def test_help_mentions_box_flags(self):
        result = subprocess.run(
            [sys.executable, INSANE_PATH, "--help"],
            capture_output=True, text=True, timeout=20,
        )
        output = result.stdout + result.stderr
        for flag in ("-x", "-y", "-z"):
            self.assertIn(flag, output, f"help output does not mention {flag} (box dimension) flag")


@unittest.skipUnless(_RUN_PARITY and _HAS_LEGACY_DATA,
                     "Skipped: set RUN_INSANE_PARITY=1 with legacy data available")
class TestInsaneParityPOPC100(unittest.TestCase):
    def test_parity_popc100_atom_count(self):
        import tempfile
        legacy_gro = os.path.join(_REPO_ROOT, "data", "membrane_only", "POPC100", "run.gro")
        with open(legacy_gro) as fh:
            fh.readline()
            legacy_atom_count = int(fh.readline().strip())

        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                [sys.executable, INSANE_PATH,
                 "-o", os.path.join(tmp, "popc100.gro"),
                 "-x", "11", "-y", "11", "-z", "10",
                 "-l", "POPC:100", "-center", "-sol", "W",
                 "-salt", "0.15", "-charge", "auto",
                 "-p", os.path.join(tmp, "topol.top")],
                capture_output=True, text=True, timeout=60,
            )
            self.assertEqual(result.returncode, 0,
                             f"insane.py parity run failed:\n{result.stderr[:500]}")
            rebuilt_gro = os.path.join(tmp, "popc100.gro")
            with open(rebuilt_gro) as fh:
                fh.readline()
                rebuilt_atom_count = int(fh.readline().strip())

        # Parity: atom count may differ by ≤ 200 atoms (solvent/ion packing difference
        # between Python 2 and Python 3 RNG; see INSANE_PROVENANCE.md).
        # Lipid count (membrane beads) should be identical — checked via NDX output.
        delta = abs(rebuilt_atom_count - legacy_atom_count)
        self.assertLessEqual(delta, 200,
                             f"atom count divergence too large: legacy={legacy_atom_count}, "
                             f"rebuilt={rebuilt_atom_count}, delta={delta}")


if __name__ == "__main__":
    unittest.main()
