"""Tests for lipid_gnn.martini_pipeline.system_builder."""
from __future__ import annotations

import os
import stat
import sys
import textwrap
import unittest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from lipid_gnn.martini_pipeline.system_builder import (
    BoxParams,
    BuildResult,
    _MARTINI3_ITPS,
    build_command,
    build_system,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_GRO = textwrap.dedent("""\
    INSANE! Membrane test
       5
        1POPC   BB    1   0.000   0.000   0.000
        1POPC   PO4   2   0.001   0.001   0.001
        1POPC   GL1   3   0.002   0.002   0.002
        1POPC   GL2   4   0.003   0.003   0.003
        2W      W     5   5.000   5.000   5.000
       11.00000  11.00000  10.00000
""")

_FAKE_TOP = textwrap.dedent("""\
    #include "martini.itp"

    [ system ]
    ; INSANE! test
    INSANE! Membrane

    [ molecules ]
    ; name  number
    POPC          196
    POPC          196
    W            5305
    NA              58
    CL              58
""")


def _write_fake_insane(path: str) -> None:
    """Write a fake insane script that emits minimal gro + top and exits 0."""
    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import argparse, os, sys
        p = argparse.ArgumentParser()
        p.add_argument("-o"); p.add_argument("-p")
        p.add_argument("-x"); p.add_argument("-y"); p.add_argument("-z")
        p.add_argument("-pbc")
        p.add_argument("-l", action="append", dest="lipids")
        p.add_argument("-sol"); p.add_argument("-salt"); p.add_argument("-charge")
        p.add_argument("-center", action="store_true")
        args = p.parse_args()
        with open(args.o, "w") as fh: fh.write({_FAKE_GRO!r})
        with open(args.p, "w") as fh: fh.write({_FAKE_TOP!r})
        print("fake insane done")
    """)
    with open(path, "w") as fh:
        fh.write(script)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def _write_fake_itp_dir(itp_dir: str) -> None:
    os.makedirs(itp_dir, exist_ok=True)
    for name in _MARTINI3_ITPS:
        with open(os.path.join(itp_dir, name), "w") as fh:
            fh.write(f"; fake {name}\n")


# ---------------------------------------------------------------------------
# build_command tests (pure function, no filesystem)
# ---------------------------------------------------------------------------

class TestBuildCommand(unittest.TestCase):
    def test_single_lipid(self):
        argv = build_command({"POPC": 1.0}, BoxParams(), "/tmp/a.gro", "/tmp/a.top")
        self.assertIn("-l", argv)
        li = argv.index("-l")
        self.assertEqual(argv[li + 1], "POPC:100")

    def test_binary_mixture_ratios(self):
        argv = build_command({"POPC": 0.7, "DOPC": 0.3}, BoxParams(), "/tmp/a.gro", "/tmp/a.top")
        l_args = [argv[i + 1] for i, a in enumerate(argv) if a == "-l"]
        self.assertEqual(len(l_args), 2)
        total = sum(int(a.split(":")[1]) for a in l_args)
        self.assertEqual(total, 100)

    def test_pbc_always_present(self):
        # pbc must always be explicit — insane defaults to hexagonal, not rectangular
        argv = build_command({"POPC": 1.0}, BoxParams(pbc="rectangular"), "/tmp/a.gro", "/tmp/a.top")
        self.assertIn("-pbc", argv)
        self.assertEqual(argv[argv.index("-pbc") + 1], "rectangular")

    def test_pbc_nondefault(self):
        argv = build_command({"POPC": 1.0}, BoxParams(pbc="hexagonal"), "/tmp/a.gro", "/tmp/a.top")
        self.assertEqual(argv[argv.index("-pbc") + 1], "hexagonal")

    def test_center_flag_true(self):
        argv = build_command({"POPC": 1.0}, BoxParams(center=True), "/tmp/a.gro", "/tmp/a.top")
        self.assertIn("-center", argv)

    def test_center_flag_false(self):
        argv = build_command({"POPC": 1.0}, BoxParams(center=False), "/tmp/a.gro", "/tmp/a.top")
        self.assertNotIn("-center", argv)

    def test_box_dims(self):
        argv = build_command({"POPC": 1.0}, BoxParams(xy_nm=9.0, z_nm=8.0), "/tmp/a.gro", "/tmp/a.top")
        self.assertIn("9.0", argv)
        self.assertIn("8.0", argv)

    def test_insane_cmd_is_first_token(self):
        argv = build_command({"POPC": 1.0}, BoxParams(), "/tmp/a.gro", "/tmp/a.top",
                             insane_cmd="myinsane")
        self.assertEqual(argv[0], "myinsane")


# ---------------------------------------------------------------------------
# build_system tests (use fake insane executable + fake itp dir)
# ---------------------------------------------------------------------------

class TestBuildSystem(unittest.TestCase):
    def setUp(self):
        import tempfile
        self._tmpdir = tempfile.mkdtemp()
        self._insane = os.path.join(self._tmpdir, "fake_insane.py")
        self._itp_dir = os.path.join(self._tmpdir, "itp")
        self._out_dir = os.path.join(self._tmpdir, "build")
        _write_fake_insane(self._insane)
        _write_fake_itp_dir(self._itp_dir)

    def tearDown(self):
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _build(self, **kwargs):
        return build_system(
            {"POPC": 1.0},
            self._out_dir,
            insane_cmd=self._insane,
            itp_dir=self._itp_dir,
            gmx_executable="gmx_does_not_exist_XYZ",
            **kwargs,
        )

    def test_creates_gro(self):
        result = self._build()
        self.assertTrue(os.path.isfile(result.gro_path))

    def test_topology_finalised(self):
        result = self._build()
        with open(result.top_path) as fh:
            content = fh.read()
        self.assertIn('#include "toppar/martini_v3.0.0.itp"', content)
        self.assertNotIn('#include "martini.itp"', content)

    def test_all_32_includes_present(self):
        result = self._build()
        with open(result.top_path) as fh:
            content = fh.read()
        for itp in _MARTINI3_ITPS:
            self.assertIn(f'toppar/{itp}', content, f"Missing include for {itp}")

    def test_ffbonded_before_lipids(self):
        result = self._build()
        with open(result.top_path) as fh:
            content = fh.read()
        ffbonded_pos = content.index("martini_v3.0.0_ffbonded_v2.itp")
        pc_pos = content.index("martini_v3.0.0_phospholipids_PC_v2.itp")
        self.assertLess(ffbonded_pos, pc_pos)

    def test_itps_staged(self):
        result = self._build()
        for itp in _MARTINI3_ITPS:
            staged = os.path.join(result.out_dir, "toppar", itp)
            self.assertTrue(os.path.isfile(staged), f"Not staged: {itp}")

    def test_log_written(self):
        result = self._build()
        self.assertTrue(os.path.isfile(result.log_path))
        with open(result.log_path) as fh:
            self.assertIn("fake insane done", fh.read())

    def test_molecule_counts(self):
        result = self._build()
        self.assertEqual(result.molecule_counts.get("POPC"), 392)  # 196+196
        self.assertEqual(result.molecule_counts.get("W"), 5305)

    def test_total_atoms(self):
        result = self._build()
        self.assertEqual(result.total_atoms, 5)

    def test_insane_failure_raises(self):
        bad_insane = os.path.join(self._tmpdir, "bad_insane.py")
        with open(bad_insane, "w") as fh:
            fh.write("#!/usr/bin/env python3\nimport sys; sys.exit(1)\n")
        os.chmod(bad_insane, os.stat(bad_insane).st_mode | stat.S_IEXEC)
        with self.assertRaises(RuntimeError):
            build_system(
                {"POPC": 1.0}, self._out_dir,
                insane_cmd=bad_insane, itp_dir=self._itp_dir,
                gmx_executable="gmx_does_not_exist_XYZ",
            )

    def test_no_martini_include_raises(self):
        no_include_top = _FAKE_TOP.replace('#include "martini.itp"', '; removed include')
        bad_insane = os.path.join(self._tmpdir, "bad_top_insane.py")
        script = textwrap.dedent(f"""\
            #!/usr/bin/env python3
            import argparse
            p = argparse.ArgumentParser()
            p.add_argument("-o"); p.add_argument("-p")
            p.add_argument("-x"); p.add_argument("-y"); p.add_argument("-z")
            p.add_argument("-pbc")
            p.add_argument("-l", action="append"); p.add_argument("-sol")
            p.add_argument("-salt"); p.add_argument("-charge")
            p.add_argument("-center", action="store_true")
            args = p.parse_args()
            with open(args.o, "w") as fh: fh.write({_FAKE_GRO!r})
            with open(args.p, "w") as fh: fh.write({no_include_top!r})
        """)
        with open(bad_insane, "w") as fh:
            fh.write(script)
        import stat as _stat
        os.chmod(bad_insane, os.stat(bad_insane).st_mode | _stat.S_IEXEC)
        with self.assertRaises(ValueError):
            build_system(
                {"POPC": 1.0}, self._out_dir,
                insane_cmd=bad_insane, itp_dir=self._itp_dir,
                gmx_executable="gmx_does_not_exist_XYZ",
            )

    def test_no_gmx_skips_ndx(self):
        result = self._build()
        self.assertIsNone(result.ndx_path)

    def test_missing_itp_preflight_raises(self):
        import shutil, tempfile
        empty_itp_dir = tempfile.mkdtemp()
        try:
            with self.assertRaises(FileNotFoundError):
                build_system(
                    {"POPC": 1.0}, self._out_dir,
                    insane_cmd=self._insane, itp_dir=empty_itp_dir,
                    gmx_executable="gmx_does_not_exist_XYZ",
                )
        finally:
            shutil.rmtree(empty_itp_dir)


if __name__ == "__main__":
    unittest.main()
