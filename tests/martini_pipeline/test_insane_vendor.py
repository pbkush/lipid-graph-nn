"""Tests for the insane membrane builder (installed package, not vendored file)."""
from __future__ import annotations

import shutil
import subprocess
import unittest

_HAS_INSANE = shutil.which("insane") is not None


@unittest.skipUnless(_HAS_INSANE, "insane command not found on PATH")
class TestInsaneCommand(unittest.TestCase):
    def test_insane_on_path(self):
        self.assertIsNotNone(shutil.which("insane"))

    def test_help_exits_zero(self):
        result = subprocess.run(
            ["insane", "--help"],
            capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(result.returncode, 0,
                         f"insane --help exited {result.returncode}:\n{result.stderr[:500]}")

    def test_help_mentions_lipid_flag(self):
        result = subprocess.run(
            ["insane", "--help"],
            capture_output=True, text=True, timeout=20,
        )
        output = result.stdout + result.stderr
        self.assertIn("-l", output, "help output does not mention -l (lipid) flag")

    def test_help_mentions_box_flags(self):
        result = subprocess.run(
            ["insane", "--help"],
            capture_output=True, text=True, timeout=20,
        )
        output = result.stdout + result.stderr
        for flag in ("-x", "-y", "-z"):
            self.assertIn(flag, output, f"help output does not mention {flag}")

    def test_help_mentions_pbc_flag(self):
        result = subprocess.run(
            ["insane", "--help"],
            capture_output=True, text=True, timeout=20,
        )
        self.assertIn("-pbc", result.stdout + result.stderr)

    def test_insane_cmd_constant(self):
        from lipid_gnn.martini_pipeline import INSANE_CMD
        self.assertEqual(INSANE_CMD, "insane")
        self.assertIsNotNone(shutil.which(INSANE_CMD),
                             "INSANE_CMD not on PATH")

    def test_insane_version_detectable(self):
        import importlib.metadata
        try:
            version = importlib.metadata.version("insane")
            self.assertRegex(version, r"^\d+\.\d+")
        except importlib.metadata.PackageNotFoundError:
            self.skipTest("insane package metadata not available")


if __name__ == "__main__":
    unittest.main()
