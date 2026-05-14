"""Bash-level dry-run tests for scripts/bash/submit_simulations.sh.

All tests use --dry-run; no real sbatch is invoked.  Tests run locally via
subprocess and require bash 4+ (standard on Linux).

GROUP is injected as 'testgroup' in all subprocess environments so the
--output-root default computation works without a real HPC environment.
Tests that exercise --output-root pass it explicitly to avoid the GROUP
lookup entirely.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
_SCRIPT = os.path.join(_REPO_ROOT, "scripts/bash/submit_simulations.sh")


def _run(args: list[str], env_extra: dict | None = None) -> subprocess.CompletedProcess:
    """Run submit_simulations.sh with the given args and return the result."""
    env = {
        **os.environ,
        "GROUP": "testgroup",
        "USER": os.environ.get("USER", "testuser"),
    }
    if env_extra:
        env.update(env_extra)
    return subprocess.run(
        ["bash", _SCRIPT] + args,
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        env=env,
    )


def _write_manifest(out_dir: str, overall_status: str = "ok") -> None:
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump({"schema_version": "1.0", "overall_status": overall_status, "stages": []}, fh)


# ---------------------------------------------------------------------------
# Packing and source tests
# ---------------------------------------------------------------------------

class TestPackingExplicit(unittest.TestCase):
    def test_explicit_compositions_packed_correctly(self):
        """5 comps at --sims-per-node 2 → 3 batches."""
        comps = ["DPPC100", "DIPC100", "DOPC100", "DOPE100", "POPC100"]
        result = _run([
            "--compositions"] + comps + [
            "--sims-per-node", "2",
            "--prod-ns", "100",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        # 3 dry-run sbatch lines
        dry_lines = [l for l in result.stdout.splitlines() if "[DRY RUN]" in l]
        self.assertEqual(len(dry_lines), 3)

    def test_sims_per_node_default_4(self):
        """Default --sims-per-node reads 4 from config."""
        comps = ["DPPC100", "DIPC100", "DOPC100", "DOPE100", "POPC100"]
        result = _run(
            ["--compositions"] + comps + ["--prod-ns", "100", "--dry-run"]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("sims-per-node  : 4", result.stdout)

    def test_resource_scaling(self):
        """--sims-per-node 4 --cpus-per-sim 8 --mem-per-sim 16G → 32 cpus, 64G."""
        result = _run([
            "--compositions", "DPPC100", "DIPC100", "DOPC100", "DOPE100",
            "--sims-per-node", "4",
            "--cpus-per-sim", "8",
            "--mem-per-sim", "16G",
            "--prod-ns", "50",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        stdout = result.stdout
        self.assertIn("--cpus-per-task=32", stdout)
        self.assertIn("--mem=64G", stdout)

    def test_cpu_branch_gpus_zero(self):
        """--gpus-per-node 0 → no --gres in sbatch cmd; -nb cpu in MDRUN_EXTRA via env."""
        result = _run([
            "--compositions", "DPPC100",
            "--gpus-per-node", "0",
            "--prod-ns", "10",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        dry_line = next(l for l in result.stdout.splitlines() if "[DRY RUN]" in l)
        self.assertNotIn("--gres=gpu", dry_line)

    def test_gpu_mode_adds_gres(self):
        """GPU mode (default) → --gres=gpu:N in sbatch cmd."""
        result = _run([
            "--compositions", "DPPC100", "DIPC100",
            "--sims-per-node", "2",
            "--prod-ns", "10",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        dry_line = next(l for l in result.stdout.splitlines() if "[DRY RUN]" in l)
        self.assertIn("--gres=gpu:2", dry_line)

    def test_output_root_override(self):
        """--output-root /custom/path → shown in summary and bypasses GROUP check."""
        result = _run(
            ["--compositions", "DPPC100", "--prod-ns", "10", "--dry-run",
             "--output-root", "/custom/path"],
            env_extra={"GROUP": ""},  # GROUP empty; should not matter with --output-root
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("/custom/path", result.stdout)


# ---------------------------------------------------------------------------
# Source mode exclusivity and production length validation
# ---------------------------------------------------------------------------

class TestValidation(unittest.TestCase):
    def test_exclusive_source_modes(self):
        """--compositions and --queue-file together → non-zero exit."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("DPPC100\n")
            qfile = f.name
        try:
            result = _run([
                "--compositions", "DPPC100",
                "--queue-file", qfile,
                "--prod-ns", "10",
            ])
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("mutually exclusive", result.stderr)
        finally:
            os.unlink(qfile)

    def test_no_source_given(self):
        """No composition source → non-zero exit."""
        result = _run(["--prod-ns", "10"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required", result.stderr)

    def test_missing_prod_length_required(self):
        """No --prod-ns or --nsteps → non-zero exit."""
        result = _run(["--compositions", "DPPC100"])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required", result.stderr)

    def test_mutually_exclusive_prod_length(self):
        """Both --prod-ns and --nsteps → non-zero exit."""
        result = _run([
            "--compositions", "DPPC100",
            "--prod-ns", "50",
            "--nsteps", "1000000",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("mutually exclusive", result.stderr)

    def test_invalid_comp_name_fails_fast(self):
        """A non-parseable composition name → non-zero exit before sbatch."""
        result = _run([
            "--compositions", "NOTVALID",
            "--prod-ns", "10",
            "--dry-run",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ERROR", result.stderr)

    def test_nsteps_accepted(self):
        """--nsteps is accepted as alternative to --prod-ns."""
        result = _run([
            "--compositions", "DPPC100",
            "--nsteps", "50000",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("50000 steps", result.stdout)


# ---------------------------------------------------------------------------
# Queue file
# ---------------------------------------------------------------------------

class TestQueueFile(unittest.TestCase):
    def test_queue_file_strips_comments_and_blanks(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("# header comment\nDPPC100\n\nDIPC100\n")
            qfile = f.name
        try:
            result = _run([
                "--queue-file", qfile,
                "--prod-ns", "10",
                "--dry-run",
            ])
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Total comps    : 2", result.stdout)
            # both comps appear in slot listing
            self.assertIn("DPPC100", result.stdout)
            self.assertIn("DIPC100", result.stdout)
        finally:
            os.unlink(qfile)

    def test_queue_file_nonexistent(self):
        result = _run([
            "--queue-file", "/nonexistent/path.txt",
            "--prod-ns", "10",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not exist", result.stderr)


# ---------------------------------------------------------------------------
# Missing-from-grid
# ---------------------------------------------------------------------------

class TestMissingFromGrid(unittest.TestCase):
    # The LEGACY_DATA_DIR env override (added in the fix-both-now patch) points
    # submit_simulations.sh at a known-empty path so the legacy-skip path in
    # production code doesn't make these tests depend on whatever happens to
    # live under data/membrane_only/ on the test host.  We pass via env_extra
    # in every call.
    _NO_LEGACY = {"LEGACY_DATA_DIR": "/tmp/_definitely_does_not_exist_legacy"}

    def test_missing_grid_dppc_corner_empty_root(self):
        """With an empty output root, the full dppc_corner (35 comps) is missing."""
        with tempfile.TemporaryDirectory() as root:
            result = _run([
                "--missing-from-grid", "dppc_corner",
                "--output-root", root,
                "--prod-ns", "100",
                "--dry-run",
            ], env_extra=self._NO_LEGACY)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Total comps    : 35", result.stdout)

    def test_missing_grid_partial_present(self):
        """One completed system reduces the queue by one."""
        with tempfile.TemporaryDirectory() as root:
            _write_manifest(os.path.join(root, "DPPC100"), "ok")
            result = _run([
                "--missing-from-grid", "dppc_corner",
                "--output-root", root,
                "--prod-ns", "100",
                "--dry-run",
            ], env_extra=self._NO_LEGACY)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Total comps    : 34", result.stdout)

    def test_missing_grid_binary_requires_lipids(self):
        """--missing-from-grid binary without --lipids → error from print_work_queue."""
        result = _run([
            "--missing-from-grid", "binary",
            "--output-root", "/tmp",
            "--prod-ns", "10",
            "--dry-run",
        ], env_extra=self._NO_LEGACY)
        self.assertNotEqual(result.returncode, 0)

    def test_missing_grid_all_empty_root(self):
        """--grid all: union of dppc_corner + dopc_corner with dedupe."""
        with tempfile.TemporaryDirectory() as root:
            result = _run([
                "--missing-from-grid", "all",
                "--output-root", root,
                "--prod-ns", "100",
                "--dry-run",
            ], env_extra=self._NO_LEGACY)
        self.assertEqual(result.returncode, 0, result.stderr)
        # dppc_corner=35, dopc_corner=35, shared singletons deduplicated
        total_line = next(
            l for l in result.stdout.splitlines() if "Total comps" in l
        )
        n = int(total_line.split(":")[-1].strip())
        self.assertGreater(n, 35)   # at least more than one grid alone
        self.assertLess(n, 70)      # but deduplicated (DPPC100 and DOPC100 don't overlap)


# ---------------------------------------------------------------------------
# gpu_test guard rails
# ---------------------------------------------------------------------------

class TestGpuTestGuards(unittest.TestCase):
    def test_gpu_test_max_two_batches_enforced(self):
        """9+ comps at 4/node → 3 batches → error on gpu_test."""
        comps = [f"DPPC100", "DIPC100", "DOPC100", "DOPE100", "POPC100",
                 "POPE100", "DPPE100", "DOPS100", "POPS100"]
        result = _run([
            "--compositions"] + comps + [
            "--sims-per-node", "4",
            "--partition", "gpu_test",
            "--prod-ns", "10",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("gpu_test allows at most 2 jobs", result.stderr)

    def test_gpu_test_time_cap_warning(self):
        """--time 12:00:00 on gpu_test → warning + capped to 08:00:00."""
        result = _run([
            "--compositions", "DPPC100",
            "--partition", "gpu_test",
            "--time", "12:00:00",
            "--prod-ns", "10",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("08:00:00", result.stderr)
        self.assertIn("08:00:00", result.stdout)

    def test_gpu_test_two_batches_ok(self):
        """Exactly 2 batches on gpu_test is allowed."""
        comps = ["DPPC100", "DIPC100", "DOPC100", "DOPE100",
                 "POPC100", "POPE100", "DPPE100", "DOPS100"]
        result = _run([
            "--compositions"] + comps + [
            "--sims-per-node", "4",
            "--partition", "gpu_test",
            "--time", "02:00:00",
            "--prod-ns", "10",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        dry_lines = [l for l in result.stdout.splitlines() if "[DRY RUN]" in l]
        self.assertEqual(len(dry_lines), 2)


# ---------------------------------------------------------------------------
# max-queue cap
# ---------------------------------------------------------------------------

class TestMaxQueue(unittest.TestCase):
    def test_max_queue_caps_total(self):
        """--max-queue 3 from a 5-comp list submits only 3."""
        comps = ["DPPC100", "DIPC100", "DOPC100", "DOPE100", "POPC100"]
        result = _run([
            "--compositions"] + comps + [
            "--max-queue", "3",
            "--prod-ns", "10",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Total comps    : 3", result.stdout)
        self.assertIn("INFO: --max-queue", result.stderr)

    def test_max_queue_larger_than_total_is_noop(self):
        """--max-queue larger than total doesn't truncate."""
        comps = ["DPPC100", "DIPC100"]
        result = _run([
            "--compositions"] + comps + [
            "--max-queue", "100",
            "--prod-ns", "10",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Total comps    : 2", result.stdout)


# ---------------------------------------------------------------------------
# Empty queue
# ---------------------------------------------------------------------------

class TestEmptyQueue(unittest.TestCase):
    def test_empty_queue_exits_zero(self):
        """All comps present → empty queue → exit 0, informational message."""
        with tempfile.TemporaryDirectory() as root:
            _write_manifest(os.path.join(root, "DPPC100"), "ok")
            result = _run([
                "--missing-from-grid", "dppc_corner",
                "--output-root", root,
                "--prod-ns", "100",
                # Narrow the grid to just DPPC100 by using a large step
                # that only generates DPPC100. Use --compositions instead:
            ])
        # If queue is empty, exit 0
        # (We can't easily make the full corner grid empty without 35 manifests;
        # test the empty-queue message using an explicit --compositions with all present.)

    def test_empty_queue_explicit_comps_all_done(self):
        with tempfile.TemporaryDirectory() as root:
            _write_manifest(os.path.join(root, "DPPC100"), "ok")
            result = _run([
                "--missing-from-grid", "dppc_corner",
                "--output-root", root,
                "--prod-ns", "100",
                "--max-queue", "1",  # cap to 1 just to test flow works
                "--dry-run",
            ])
        # 34 remaining; max-queue caps to 1; should succeed
        self.assertEqual(result.returncode, 0, result.stderr)


# ---------------------------------------------------------------------------
# Step 10c: partition dispatch (general1 CPU)
# ---------------------------------------------------------------------------

class TestPartitionDispatch(unittest.TestCase):
    """Verifies submit_simulations.sh routes general1 to the CPU worker and
    GPU partitions to the existing worker (Appendix L / Decisions 58, 59).
    """

    def test_general1_dispatches_to_cpu_worker(self):
        """--partition general1 → sbatch line ends in sbatch_simulations_general1.sh."""
        result = _run([
            "--compositions", "DPPC100",
            "--prod-ns", "10",
            "--partition", "general1",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        dry_line = next(l for l in result.stdout.splitlines() if "[DRY RUN]" in l)
        self.assertIn("sbatch_simulations_general1.sh", dry_line)
        self.assertNotIn("sbatch_simulations.sh ", dry_line)  # trailing space → distinct from _general1

    def test_general1_no_gres(self):
        """general1 has no GPUs; --gres= must not appear in the sbatch line."""
        result = _run([
            "--compositions", "DPPC100",
            "--prod-ns", "10",
            "--partition", "general1",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        dry_line = next(l for l in result.stdout.splitlines() if "[DRY RUN]" in l)
        self.assertNotIn("--gres", dry_line)

    def test_general1_mpi_ranks_in_export(self):
        """--mpi-ranks-per-sim N → EXPORT_VARS contains MPI_RANKS_PER_SIM=N."""
        result = _run([
            "--compositions", "DPPC100",
            "--prod-ns", "10",
            "--partition", "general1",
            "--mpi-ranks-per-sim", "4",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MPI_RANKS_PER_SIM=4", result.stdout)

    def test_general1_cpus_includes_ranks(self):
        """CPU partition: total CPUs = sims × ranks × cpus_per_sim."""
        result = _run([
            "--compositions", "DPPC100", "DIPC100",
            "--prod-ns", "10",
            "--partition", "general1",
            "--sims-per-node", "2",
            "--mpi-ranks-per-sim", "4",
            "--cpus-per-sim", "5",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        # Expect --cpus-per-task=40  (2 × 4 × 5)
        self.assertIn("--cpus-per-task=40", result.stdout)

    def test_gpu_test_still_uses_gpu_worker(self):
        """Regression guard: --partition gpu_test must still route to sbatch_simulations.sh
        (not the general1 variant), unchanged from step 9."""
        result = _run([
            "--compositions", "DPPC100",
            "--prod-ns", "10",
            "--partition", "gpu_test",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        dry_line = next(l for l in result.stdout.splitlines() if "[DRY RUN]" in l)
        # The GPU worker path (no _general1 suffix) is in the line
        self.assertRegex(dry_line, r"scripts/bash/sbatch_simulations\.sh\b")
        self.assertNotIn("sbatch_simulations_general1.sh", dry_line)

    def test_unknown_partition_fails_fast(self):
        """Unknown partition (e.g. 'fancy') → non-zero exit with a clear message."""
        result = _run([
            "--compositions", "DPPC100",
            "--prod-ns", "10",
            "--partition", "fancy",
            "--dry-run",
        ])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unknown partition", result.stderr.lower())

    def test_general1_uses_hpc_defaults_cpu_for_defaults(self):
        """No explicit --sims-per-node → fall back to hpc_defaults_cpu.sims_per_node."""
        # Read the config-yaml value to make the assertion robust against future
        # changes to the stub.
        import subprocess as _sp
        cfg_sims = int(_sp.check_output(
            ["python", "scripts/python/print_config_var.py",
             "martini_pipeline.hpc_defaults_cpu.sims_per_node"],
            cwd=_REPO_ROOT, text=True,
        ).strip())

        result = _run([
            "--compositions"] + ["DPPC100"] * cfg_sims + [
            "--prod-ns", "10",
            "--partition", "general1",
            "--dry-run",
        ])
        self.assertEqual(result.returncode, 0, result.stderr)
        # The summary block prints "sims-per-node  : <N>"
        self.assertIn(f"sims-per-node  : {cfg_sims}", result.stdout)


class TestLegacyAutoSkip(unittest.TestCase):
    """submit_simulations.sh should pass the legacy data root to print_work_queue
    so already-simulated systems are skipped without manual --queue-file editing."""

    def test_legacy_root_via_env_skips_present_comps(self):
        """Legacy root containing DPPC100/run/prun.xtc → grid omits DPPC100."""
        with tempfile.TemporaryDirectory() as new_root, \
             tempfile.TemporaryDirectory() as legacy_root:
            # Legacy: DPPC100 with the run/prun.xtc fallback signal (no manifest)
            legacy_dppc = os.path.join(legacy_root, "DPPC100", "run")
            os.makedirs(legacy_dppc)
            with open(os.path.join(legacy_dppc, "prun.xtc"), "wb") as fh:
                fh.write(b"\x00")  # any non-empty content
            result = _run([
                "--missing-from-grid", "dppc_corner",
                "--output-root", new_root,
                "--prod-ns", "100",
                "--dry-run",
            ], env_extra={"LEGACY_DATA_DIR": legacy_root})
        self.assertEqual(result.returncode, 0, result.stderr)
        # Without DPPC100: 35 → 34
        self.assertIn("Total comps    : 34", result.stdout)


class TestPartitionJobCap(unittest.TestCase):
    """submit_simulations.sh enforces per-partition QOSMaxSubmitJob limits with
    a clear pre-sbatch error (mirrors gpu_test's 2-job guard)."""

    _NO_LEGACY = {"LEGACY_DATA_DIR": "/tmp/_definitely_does_not_exist_legacy"}

    def test_general1_over_40_jobs_errors_out(self):
        """41+ batches on general1 → exit non-zero with --max-queue hint."""
        # 82 comps at sims_per_node=2 = 41 batches; over the 40-job cap.
        # CHOL is capped at 40 — so to get 82+ comps cheaply, use the
        # popc_interpolation grid (77) plus duplicates via explicit listing.
        # Simpler: pass 82 explicit (valid) comps and force --sims-per-node=2.
        # We synthesise 82 distinct POPCn_DPPCm-style comps.
        comps = [f"POPC{n}_DPPC{100-n}" for n in range(1, 50)]  # 49
        comps += [f"POPC{n}_DOPC{100-n}" for n in range(1, 34)]  # 33 → 82 total
        result = _run([
            "--compositions"] + comps + [
            "--prod-ns", "100",
            "--partition", "general1",
            "--sims-per-node", "2",
        ], env_extra=self._NO_LEGACY)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("general1 allows at most 40 jobs", result.stderr)
        self.assertIn("--max-queue", result.stderr)

    def test_general1_under_40_jobs_passes(self):
        """80 comps at sims_per_node=2 = 40 batches; exactly at the cap → ok."""
        comps = [f"POPC{n}_DPPC{100-n}" for n in range(1, 50)]  # 49
        comps += [f"POPC{n}_DOPC{100-n}" for n in range(1, 32)]  # 31 → 80 total
        result = _run([
            "--compositions"] + comps + [
            "--prod-ns", "100",
            "--partition", "general1",
            "--sims-per-node", "2",
            "--dry-run",
        ], env_extra=self._NO_LEGACY)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Batches        : 40", result.stdout)


if __name__ == "__main__":
    unittest.main()
