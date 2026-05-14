"""Tests for lipid_gnn.martini_pipeline.pipeline.

gmx is mocked via a fake Python script on PATH (tmp_bin/fake_gmx.py).
insane is mocked similarly, reusing the helper from test_system_builder.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
import textwrap
import unittest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from lipid_gnn.martini_pipeline.system_builder import _MARTINI3_ITPS
from lipid_gnn.martini_pipeline.pipeline import (
    PipelineResult,
    StageResult,
    _derive_seed,
    run as pipeline_run,
)

# ---------------------------------------------------------------------------
# Fake file content constants (must match test_system_builder)
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


# ---------------------------------------------------------------------------
# Helpers to write fake binaries
# ---------------------------------------------------------------------------

def _write_fake_insane(path: str) -> None:
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


def _write_fake_gmx(path: str) -> None:
    """Fake gmx that handles make_ndx, grompp, mdrun, --version."""
    script = textwrap.dedent("""\
        #!/usr/bin/env python3
        import argparse, os, sys

        if len(sys.argv) < 2:
            sys.exit(1)
        subcmd = sys.argv[1]

        if subcmd == "--version":
            print("GROMACS - Machine Made For Scientists")
            print("GROMACS version:    2024.1-fake")
            sys.exit(0)

        elif subcmd == "make_ndx":
            p = argparse.ArgumentParser()
            p.add_argument("-f"); p.add_argument("-o")
            args, _ = p.parse_known_args(sys.argv[2:])
            with open(args.o, "w") as fh:
                fh.write("[ System ]\\n 0 1 2 3 4\\n")
            print("fake make_ndx done")
            sys.exit(0)

        elif subcmd == "grompp":
            p = argparse.ArgumentParser()
            p.add_argument("-f"); p.add_argument("-c"); p.add_argument("-p")
            p.add_argument("-n"); p.add_argument("-o"); p.add_argument("-maxwarn")
            args, _ = p.parse_known_args(sys.argv[2:])
            with open(args.o, "w") as fh:
                fh.write("fake tpr\\n")
            print("fake grompp done")
            sys.exit(0)

        elif subcmd == "mdrun":
            p = argparse.ArgumentParser()
            p.add_argument("-deffnm")
            args, _ = p.parse_known_args(sys.argv[2:])
            deffnm = args.deffnm
            for ext in [".gro", ".log", ".edr", ".xtc", ".cpt"]:
                with open(deffnm + ext, "w") as fh:
                    fh.write(f"fake {ext}\\n")
            print("fake mdrun done")
            sys.exit(0)

        else:
            print(f"fake gmx: unknown subcmd {subcmd!r}", file=sys.stderr)
            sys.exit(1)
    """)
    with open(path, "w") as fh:
        fh.write(script)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)


def _write_fake_gmx_fail_stage(path: str, fail_stage: str, fail_step: str = "grompp") -> None:
    """Fake gmx that exits 1 on *fail_step* when the stage matches *fail_stage*."""
    script = textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import argparse, os, sys

        subcmd = sys.argv[1] if len(sys.argv) > 1 else ""

        if subcmd == "--version":
            print("GROMACS version:    2024.1-fake")
            sys.exit(0)

        elif subcmd == "make_ndx":
            p = argparse.ArgumentParser()
            p.add_argument("-f"); p.add_argument("-o")
            args, _ = p.parse_known_args(sys.argv[2:])
            with open(args.o, "w") as fh:
                fh.write("[ System ]\\n 0 1 2 3 4\\n")
            sys.exit(0)

        elif subcmd == "grompp":
            p = argparse.ArgumentParser()
            p.add_argument("-f"); p.add_argument("-c"); p.add_argument("-p")
            p.add_argument("-n"); p.add_argument("-o"); p.add_argument("-maxwarn")
            args, _ = p.parse_known_args(sys.argv[2:])
            # Detect stage from mdp filename
            mdp = args.f or ""
            is_eq = "eq.mdp" in mdp
            should_fail = is_eq and "{fail_step}" == "grompp" and "{fail_stage}" == "equilibration"
            if should_fail:
                print("fake grompp failure", file=sys.stderr)
                sys.exit(1)
            with open(args.o, "w") as fh:
                fh.write("fake tpr\\n")
            sys.exit(0)

        elif subcmd == "mdrun":
            p = argparse.ArgumentParser()
            p.add_argument("-deffnm")
            args, _ = p.parse_known_args(sys.argv[2:])
            deffnm = args.deffnm
            for ext in [".gro", ".log", ".edr", ".xtc", ".cpt"]:
                with open(deffnm + ext, "w") as fh:
                    fh.write(f"fake {{ext}}\\n")
            sys.exit(0)

        else:
            sys.exit(1)
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
# Base test setup
# ---------------------------------------------------------------------------

class _PipelineTestBase(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._insane = os.path.join(self._tmpdir, "fake_insane.py")
        self._gmx = os.path.join(self._tmpdir, "fake_gmx.py")
        self._itp_dir = os.path.join(self._tmpdir, "itp")
        self._out_dir = os.path.join(self._tmpdir, "out")
        _write_fake_insane(self._insane)
        _write_fake_gmx(self._gmx)
        _write_fake_itp_dir(self._itp_dir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _run(self, composition=None, **kwargs):
        return pipeline_run(
            composition or {"POPC": 1.0},
            self._out_dir,
            gmx_executable=self._gmx,
            insane_cmd=self._insane,
            itp_dir=self._itp_dir,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPipelineDirectoryLayout(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._insane = os.path.join(self._tmpdir, "fake_insane.py")
        self._gmx = os.path.join(self._tmpdir, "fake_gmx.py")
        self._itp_dir = os.path.join(self._tmpdir, "itp")
        self._out_dir = os.path.join(self._tmpdir, "out")
        _write_fake_insane(self._insane)
        _write_fake_gmx(self._gmx)
        _write_fake_itp_dir(self._itp_dir)

    def tearDown(self):
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def _run(self, **kwargs):
        return pipeline_run(
            {"POPC": 1.0}, self._out_dir,
            gmx_executable=self._gmx,
            insane_cmd=self._insane,
            itp_dir=self._itp_dir,
            **kwargs,
        )

    def test_run_creates_per_stage_dirs(self):
        self._run()
        for stage in ("minimization", "equilibration", "run"):
            self.assertTrue(
                os.path.isdir(os.path.join(self._out_dir, stage)),
                f"Missing stage dir: {stage}",
            )

    def test_run_writes_tpr_per_stage(self):
        self._run()
        expected = [
            ("minimization", "martini_em.tpr"),
            ("equilibration", "martini_eq.tpr"),
            ("run", "prun.tpr"),
        ]
        for stage_dir, fname in expected:
            path = os.path.join(self._out_dir, stage_dir, fname)
            self.assertTrue(os.path.isfile(path), f"Missing: {stage_dir}/{fname}")

    def test_run_writes_handoff_files(self):
        self._run()
        expected = [
            ("minimization", "minimized.gro"),
            ("equilibration", "equilibrated.gro"),
            ("run", "prun.gro"),
        ]
        for stage_dir, fname in expected:
            path = os.path.join(self._out_dir, stage_dir, fname)
            self.assertTrue(os.path.isfile(path), f"Missing handoff: {stage_dir}/{fname}")

    def test_mdp_written_per_stage_subdir(self):
        self._run()
        self.assertTrue(os.path.isfile(os.path.join(self._out_dir, "minimization", "em.mdp")))
        self.assertTrue(os.path.isfile(os.path.join(self._out_dir, "equilibration", "eq.mdp")))
        self.assertTrue(os.path.isfile(os.path.join(self._out_dir, "run", "run.mdp")))

    def test_manifest_written(self):
        result = self._run()
        self.assertTrue(os.path.isfile(result.manifest_path))

    def test_manifest_is_valid_json(self):
        result = self._run()
        with open(result.manifest_path) as fh:
            data = json.load(fh)
        self.assertEqual(data["overall_status"], "ok")
        self.assertEqual(data["schema_version"], "1.0")


class TestGromppInvocations(_PipelineTestBase):
    def _run(self, **kwargs):
        return pipeline_run(
            {"POPC": 1.0}, self._out_dir,
            gmx_executable=self._gmx,
            insane_cmd=self._insane,
            itp_dir=self._itp_dir,
            **kwargs,
        )

    def test_grompp_cmd_in_manifest(self):
        result = self._run()
        with open(result.manifest_path) as fh:
            data = json.load(fh)
        for stage_data in data["stages"]:
            if stage_data["status"] != "skipped":
                self.assertIn("grompp", stage_data["grompp_cmd"][1])

    def test_maxwarn_in_grompp_cmd(self):
        result = self._run(maxwarn=5)
        with open(result.manifest_path) as fh:
            data = json.load(fh)
        for stage_data in data["stages"]:
            if stage_data["status"] != "skipped":
                cmd = stage_data["grompp_cmd"]
                self.assertIn("-maxwarn", cmd)
                idx = cmd.index("-maxwarn")
                self.assertEqual(cmd[idx + 1], "5")

    def test_mdrun_deffnm_in_manifest(self):
        result = self._run()
        with open(result.manifest_path) as fh:
            data = json.load(fh)
        deffnms = {
            "minimization": "martini_em",
            "equilibration": "martini_eq",
            "production": "prun",
        }
        for stage_data in data["stages"]:
            name = stage_data["name"]
            if stage_data["status"] != "skipped":
                cmd = stage_data["mdrun_cmd"]
                self.assertIn("-deffnm", cmd)
                idx = cmd.index("-deffnm")
                self.assertEqual(cmd[idx + 1], deffnms[name])

    def test_mdrun_extra_args_propagate(self):
        result = self._run(mdrun_extra_args=("-ntomp", "4"))
        with open(result.manifest_path) as fh:
            data = json.load(fh)
        for stage_data in data["stages"]:
            if stage_data["status"] != "skipped":
                cmd = stage_data["mdrun_cmd"]
                self.assertIn("-ntomp", cmd)
                self.assertIn("4", cmd)

    def test_mdrun_extra_args_string_form_is_split(self):
        """Single-string extra_args (e.g. '-ntomp 4 -nb cpu') must be tokenised
        before being passed to mdrun — bash workers pass --mdrun-args as one
        quoted token and argparse.REMAINDER stores it as a 1-element list."""
        result = self._run(mdrun_extra_args=("-ntomp 4 -nb cpu",))
        with open(result.manifest_path) as fh:
            data = json.load(fh)
        for stage_data in data["stages"]:
            if stage_data["status"] != "skipped":
                cmd = stage_data["mdrun_cmd"]
                # Each token must appear as its own argv element, not a fused string
                self.assertIn("-ntomp", cmd)
                self.assertIn("4", cmd)
                self.assertIn("-nb", cmd)
                self.assertIn("cpu", cmd)
                self.assertNotIn("-ntomp 4 -nb cpu", cmd)


class TestIdempotency(_PipelineTestBase):
    def _run(self, **kwargs):
        return pipeline_run(
            {"POPC": 1.0}, self._out_dir,
            gmx_executable=self._gmx,
            insane_cmd=self._insane,
            itp_dir=self._itp_dir,
            **kwargs,
        )

    def test_idempotency_skips_completed_stage(self):
        # Pre-create the minimization handoff file
        os.makedirs(os.path.join(self._out_dir, "minimization"), exist_ok=True)
        handoff = os.path.join(self._out_dir, "minimization", "minimized.gro")
        # Handoff must have valid gro content for equilibration to read it
        with open(handoff, "w") as fh:
            fh.write(_FAKE_GRO)
        result = self._run()
        with open(result.manifest_path) as fh:
            data = json.load(fh)
        min_stage = next(s for s in data["stages"] if s["name"] == "minimization")
        self.assertEqual(min_stage["status"], "skipped")
        eq_stage = next(s for s in data["stages"] if s["name"] == "equilibration")
        self.assertEqual(eq_stage["status"], "ok")

    def test_force_rerun_overrides_idempotency(self):
        # First run
        self._run()
        # Touch minimized.gro with a sentinel value
        handoff = os.path.join(self._out_dir, "minimization", "minimized.gro")
        mtime_before = os.path.getmtime(handoff)
        # Second run with force_rerun
        import time
        time.sleep(0.01)
        self._run(force_rerun=True)
        mtime_after = os.path.getmtime(handoff)
        self.assertGreater(mtime_after, mtime_before, "force_rerun should overwrite handoff")


class TestSeedBehaviour(_PipelineTestBase):
    def _run(self, **kwargs):
        return pipeline_run(
            {"POPC": 1.0}, self._out_dir,
            gmx_executable=self._gmx,
            insane_cmd=self._insane,
            itp_dir=self._itp_dir,
            **kwargs,
        )

    def test_seed_deterministic_same_composition(self):
        s1 = _derive_seed("POPC100")
        s2 = _derive_seed("POPC100")
        self.assertEqual(s1, s2)

    def test_different_compositions_different_seeds(self):
        self.assertNotEqual(_derive_seed("POPC100"), _derive_seed("DOPC100"))

    def test_seed_in_manifest(self):
        result = self._run(seed=99999)
        with open(result.manifest_path) as fh:
            data = json.load(fh)
        self.assertEqual(data["seed"], 99999)

    def test_deterministic_seed_in_manifest(self):
        """Rerunning without explicit seed produces the same seed in manifest."""
        r1 = self._run()
        with open(r1.manifest_path) as fh:
            d1 = json.load(fh)
        shutil.rmtree(self._out_dir)
        r2 = self._run()
        with open(r2.manifest_path) as fh:
            d2 = json.load(fh)
        self.assertEqual(d1["seed"], d2["seed"])


class TestFailureHandling(_PipelineTestBase):
    def test_failure_at_equilibration_writes_manifest(self):
        bad_gmx = os.path.join(self._tmpdir, "bad_gmx.py")
        _write_fake_gmx_fail_stage(bad_gmx, fail_stage="equilibration", fail_step="grompp")

        with self.assertRaises(RuntimeError):
            pipeline_run(
                {"POPC": 1.0}, self._out_dir,
                gmx_executable=bad_gmx,
                insane_cmd=self._insane,
                itp_dir=self._itp_dir,
            )

        manifest_path = os.path.join(self._out_dir, "manifest.json")
        self.assertTrue(os.path.isfile(manifest_path), "manifest not written on failure")
        with open(manifest_path) as fh:
            data = json.load(fh)
        self.assertIn("failed", data["overall_status"])

    def test_gmx_not_found_raises(self):
        with self.assertRaises(FileNotFoundError):
            pipeline_run(
                {"POPC": 1.0}, self._out_dir,
                gmx_executable="gmx_absolutely_does_not_exist_XYZ",
                insane_cmd=self._insane,
                itp_dir=self._itp_dir,
            )


class TestOverallResult(_PipelineTestBase):
    def _run(self, **kwargs):
        return pipeline_run(
            {"POPC": 1.0}, self._out_dir,
            gmx_executable=self._gmx,
            insane_cmd=self._insane,
            itp_dir=self._itp_dir,
            **kwargs,
        )

    def test_overall_status_ok(self):
        result = self._run()
        self.assertEqual(result.overall_status, "ok")

    def test_three_stages_returned(self):
        result = self._run()
        self.assertEqual(len(result.stages), 3)

    def test_all_stages_ok(self):
        result = self._run()
        for s in result.stages:
            self.assertEqual(s.status, "ok")

    def test_build_result_in_pipeline_result(self):
        result = self._run()
        self.assertIsNotNone(result.build)
        self.assertTrue(os.path.isfile(result.build.gro_path))


if __name__ == "__main__":
    unittest.main()
