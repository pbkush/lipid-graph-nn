"""Unit tests for scripts/python/analyze_benchmark.py.

All tests run locally without SLURM.  Fixture files live under
tests/martini_pipeline/fixtures/benchmark/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_FIXTURES = Path(__file__).parent / "fixtures" / "benchmark"
_SCRIPT = _REPO_ROOT / "scripts" / "python" / "analyze_benchmark.py"
_BENCH_SH = _REPO_ROOT / "scripts" / "simulation" / "benchmark_hpc.sh"

sys.path.insert(0, str(_REPO_ROOT / "scripts" / "python"))

from analyze_benchmark import (
    PointResult,
    parse_perf,
    parse_rocm_smi,
    load_point,
    recommend,
    to_dataframe,
    _isnan,
)


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

class TestParsePerfLine(unittest.TestCase):
    def test_parse_ok_log(self):
        """Well-formed log → (57.6, 1500.0)."""
        ns, wall = parse_perf(_FIXTURES / "slot_ok" / "bench.log")
        self.assertAlmostEqual(ns, 57.6, places=5)
        self.assertAlmostEqual(wall, 1500.0, places=1)

    def test_parse_missing_perf_returns_none(self):
        """Log without Performance: line → (None, None)."""
        ns, wall = parse_perf(_FIXTURES / "slot_no_perf" / "bench.log")
        self.assertIsNone(ns)
        self.assertIsNone(wall)

    def test_parse_nonexistent_file_returns_none(self):
        ns, wall = parse_perf(_FIXTURES / "does_not_exist.log")
        self.assertIsNone(ns)
        self.assertIsNone(wall)

    def test_parse_indented_time_line(self):
        """Real gmx 2022 emits 'Time:' indented with leading whitespace
        (e.g. '       Time:  46.328  11.582  400.0').  The regex must
        tolerate the leading whitespace or we silently lose wall-time
        and the recommend() filter drops every point as NaN-scored."""
        ns, wall = parse_perf(_FIXTURES / "slot_indented" / "bench.log")
        self.assertAlmostEqual(ns, 9108.65, places=2)
        self.assertAlmostEqual(wall, 11.582, places=3)


# ---------------------------------------------------------------------------
# rocm-smi parsing
# ---------------------------------------------------------------------------

class TestParseRocmSmi(unittest.TestCase):
    def test_parse_fixture_tsv(self):
        """Fixture TSV → expected mean util, mean power, max VRAM."""
        util, power, vram = parse_rocm_smi(_FIXTURES / "rocm-smi.tsv")
        self.assertAlmostEqual(util, (85.5 + 88.0 + 90.0) / 3, places=4)
        self.assertAlmostEqual(power, (280.0 + 282.5 + 285.0) / 3, places=4)
        self.assertAlmostEqual(vram, 4100.0, places=0)

    def test_missing_tsv_returns_nan(self):
        util, power, vram = parse_rocm_smi(_FIXTURES / "no_rocm.tsv")
        self.assertTrue(_isnan(util))
        self.assertTrue(_isnan(power))
        self.assertTrue(_isnan(vram))


# ---------------------------------------------------------------------------
# Aggregate and score calculation
# ---------------------------------------------------------------------------

class TestAggregateAndScore(unittest.TestCase):
    def _make_point(self, ns_list, wall_list, sims=2) -> PointResult:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "point_meta.json").write_text(json.dumps({
                "label": "test", "sims_per_node": sims,
                "gpus_per_node": 2, "cpus_per_sim": 8,
                "mem_per_sim": "16G", "partition": "gpu_test",
            }))
            for i, (ns, wall) in enumerate(zip(ns_list, wall_list)):
                slot = root / f"slot_{i}"
                slot.mkdir()
                log_content = (
                    f"Time:          {wall * 8:.3f}    {wall:.3f}        800.0\n\n"
                    f"Performance:       {ns:.1f}       {24 / ns:.3f}\n"
                    f"             ns/day   hours/ns\n"
                )
                (slot / "bench.log").write_text(log_content)
            return load_point(root)

    def test_two_slot_aggregate(self):
        p = self._make_point([300.0, 320.0], [1800.0, 1820.0])
        self.assertAlmostEqual(p.aggregate_ns_per_day, 620.0, places=4)

    def test_score_calculation(self):
        """620 ns/day in 0.5 node-hours → score = 1240."""
        p = self._make_point([300.0, 320.0], [1800.0, 1800.0])
        expected_score = 620.0 / (1800.0 / 3600.0)  # 1240.0
        self.assertAlmostEqual(p.score, expected_score, places=3)

    def test_score_falls_back_to_aggregate_when_walltime_missing(self):
        """If wall-time parse fails (no 'Time:' line), score falls back to
        aggregate_ns_per_day so recommend() can still rank — see the user's
        first general1 sweep where logs were truncated."""
        # Write logs WITHOUT the Time: line — only Performance:
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "point_meta.json").write_text(json.dumps({
                "label": "x", "sims_per_node": 2, "gpus_per_node": 0,
                "cpus_per_sim": 20, "mem_per_sim": "16G", "partition": "general1",
                "mpi_ranks_per_sim": 1, "device": "cpu",
            }))
            for i, ns in enumerate([6589.0, 6589.0]):
                slot = root / f"slot_{i}"
                slot.mkdir()
                (slot / "bench.log").write_text(
                    f"Performance:    {ns:.2f}    {24/ns:.4f}\n"
                    "             ns/day   hours/ns\n"
                )
            p = load_point(root)
        # aggregate = 13178; node_hours = NaN; score should fall back to aggregate
        self.assertAlmostEqual(p.aggregate_ns_per_day, 13178.0, places=1)
        self.assertAlmostEqual(p.score, 13178.0, places=1)


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------

def _make_pr(label, sims, gpus, cpus, mem_g, score) -> PointResult:
    """Construct a synthetic PointResult with a known score for ranking tests."""
    p = PointResult(
        label=label, sims_per_node=sims, gpus_per_node=gpus,
        cpus_per_sim=cpus, mem_per_sim=f"{mem_g}G", partition="gpu_test",
        status="ok", n_slots_ok=sims, n_slots_total=sims,
        aggregate_ns_per_day=score * 0.5,  # node_hours=0.5
        max_wall_t_s=1800.0,
        node_hours=0.5,
        score=score,
    )
    return p


class TestRecommendation(unittest.TestCase):
    def test_picks_top_score_under_mem_cap(self):
        """Point disqualified by memory cap is excluded; next-best wins."""
        p_fast = _make_pr("fast_over_mem", 8, 8, 8, 16, 1500.0)  # 128G total > 0.7*256=179G — OK
        p_huge = _make_pr("huge_mem", 8, 4, 8, 24, 1600.0)        # 192G total > 179G — disqualified
        p_mid  = _make_pr("mid",  4, 4, 8, 16, 1200.0)             # 64G total — OK

        rec = recommend([p_fast, p_huge, p_mid], node_mem_GB=256.0, headroom=0.70)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.label, "fast_over_mem")

    def test_mem_cap_disqualifies_correctly(self):
        """Only the over-budget point is excluded."""
        p_ok  = _make_pr("ok",   4, 4, 8, 16, 900.0)   # 64G ≤ 179G
        p_bad = _make_pr("bad",  8, 8, 8, 24, 1800.0)  # 192G > 179G

        rec = recommend([p_ok, p_bad], node_mem_GB=256.0, headroom=0.70)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.label, "ok")

    def test_tie_break_by_gpu_count(self):
        """Equal score → prefer fewer GPUs."""
        p_4gpu = _make_pr("four_gpu", 4, 4, 8, 16, 1000.0)
        p_8gpu = _make_pr("eight_gpu", 8, 8, 8, 16, 1000.0)

        rec = recommend([p_8gpu, p_4gpu], node_mem_GB=256.0, headroom=0.70)
        self.assertIsNotNone(rec)
        self.assertEqual(rec.label, "four_gpu")

    def test_recommend_refuses_partial_sweep(self):
        """Incomplete points raise an error via the CLI without --allow-partial."""
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            pts = root / "points"
            (pts / "done" ).mkdir(parents=True)
            (pts / "broken").mkdir(parents=True)
            (pts / "done" / "point_meta.json").write_text(json.dumps({
                "label": "done", "sims_per_node": 2, "gpus_per_node": 2,
                "cpus_per_sim": 8, "mem_per_sim": "16G", "partition": "gpu_test",
            }))
            (pts / "broken" / "point_meta.json").write_text(json.dumps({
                "label": "broken", "sims_per_node": 2, "gpus_per_node": 2,
                "cpus_per_sim": 8, "mem_per_sim": "16G", "partition": "gpu_test",
            }))
            # 'done' has no slot logs either — both are "failed"
            result = subprocess.run(
                [sys.executable, str(_SCRIPT), "--root", str(root), "--recommend"],
                capture_output=True, text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("incomplete", result.stderr.lower())

    def test_no_qualifying_point_returns_none(self):
        """All points over memory budget → recommend returns None."""
        p = _make_pr("fat", 4, 4, 8, 64, 1000.0)  # 256G = exactly node_mem → > 0.7*256
        rec = recommend([p], node_mem_GB=256.0, headroom=0.70)
        self.assertIsNone(rec)


def _make_cpu_pr(label, sims, mpi, cpus, score) -> PointResult:
    p = PointResult(
        label=label, sims_per_node=sims, gpus_per_node=0,
        cpus_per_sim=cpus, mem_per_sim="16G", partition="general1",
        mpi_ranks_per_sim=mpi, device="cpu",
        status="ok", n_slots_ok=sims, n_slots_total=sims,
        aggregate_ns_per_day=score * 0.5,
        max_wall_t_s=1800.0, node_hours=0.5,
        score=score,
    )
    return p


class TestRecommendCpu(unittest.TestCase):
    def test_cpu_branch_ignores_mem_headroom(self):
        """CPU branch should pick top score regardless of memory (per K.12 Q10)."""
        # A "huge mem" CPU point would be disqualified in the GPU branch but
        # must NOT be in the CPU branch.
        p_huge = _make_cpu_pr("huge", sims=8, mpi=1, cpus=5, score=2000.0)
        p_mid  = _make_cpu_pr("mid",  sims=4, mpi=1, cpus=10, score=1500.0)
        rec = recommend([p_huge, p_mid], node_mem_GB=256.0, headroom=0.10, device="cpu")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.label, "huge")

    def test_cpu_tie_break_prefers_fewer_mpi_ranks(self):
        """Equal score → prefer fewer mpi ranks (less comm overhead)."""
        p_4rank = _make_cpu_pr("four_rank", sims=1, mpi=4, cpus=10, score=1000.0)
        p_1rank = _make_cpu_pr("one_rank",  sims=1, mpi=1, cpus=40, score=1000.0)
        rec = recommend([p_4rank, p_1rank], node_mem_GB=256.0, headroom=0.70, device="cpu")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.label, "one_rank")

    def test_cpu_ignores_gpu_points(self):
        """CPU recommend must not return a GPU point even if it scores higher."""
        p_gpu_fast = _make_pr("gpu_fast", 4, 4, 8, 16, 5000.0)
        p_cpu_slow = _make_cpu_pr("cpu_slow", sims=1, mpi=1, cpus=40, score=200.0)
        rec = recommend([p_gpu_fast, p_cpu_slow], node_mem_GB=256.0, headroom=0.70, device="cpu")
        self.assertIsNotNone(rec)
        self.assertEqual(rec.label, "cpu_slow")

    def test_cpu_yaml_emits_hpc_defaults_cpu(self):
        """_rec_yaml for a CPU point should emit hpc_defaults_cpu with the
        partition + module fields."""
        from analyze_benchmark import _rec_yaml
        p = _make_cpu_pr("good", sims=4, mpi=1, cpus=10, score=1500.0)
        yaml = _rec_yaml(p)
        self.assertIn("hpc_defaults_cpu:", yaml)
        self.assertIn("mpi_ranks_per_sim: 1", yaml)
        self.assertIn('partition: "general1"', yaml)
        self.assertIn("gromacs/2022.4-gcc-11.3.1-zx2wwcx", yaml)
        self.assertIn("mpi/openmpi/5.0.0", yaml)
        # Must NOT contain GPU-only fields
        self.assertNotIn("gpus_per_node", yaml)


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

class TestOutputHelpers(unittest.TestCase):
    def test_summary_df_has_expected_columns(self):
        p = _make_pr("pt", 4, 4, 8, 16, 800.0)
        df = to_dataframe([p])
        for col in ("label", "aggregate_ns_per_day", "score_ns_day_per_node_hour", "status"):
            self.assertIn(col, df.columns)

    def test_yaml_block_round_trips_config(self):
        """Recommended YAML block parses back into MartiniPipelineHpcDefaultsConfig."""
        from lipid_gnn.config import _build_martini_pipeline_hpc_defaults
        p = _make_pr("winner", 4, 4, 8, 16, 1000.0)
        raw = {
            "sims_per_node": p.sims_per_node,
            "cpus_per_sim": p.cpus_per_sim,
            "mem_per_sim": p.mem_per_sim,
            "gpus_per_node": p.gpus_per_node,
        }
        cfg = _build_martini_pipeline_hpc_defaults(raw)
        self.assertEqual(cfg.sims_per_node, 4)
        self.assertEqual(cfg.cpus_per_sim, 8)
        self.assertEqual(cfg.mem_per_sim, "16G")
        self.assertEqual(cfg.gpus_per_node, 4)


# ---------------------------------------------------------------------------
# benchmark_hpc.sh dry-run
# ---------------------------------------------------------------------------

def _run_bench_sh(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "GROUP": "testgroup", "USER": os.environ.get("USER", "testuser")}
    return subprocess.run(
        ["bash", str(_BENCH_SH)] + args,
        capture_output=True, text=True,
        cwd=str(_REPO_ROOT), env=env,
    )


class TestBenchmarkHshDryRun(unittest.TestCase):
    def test_dry_run_emits_one_sbatch_per_point(self):
        """--dry-run with default 7-point GPU-only TSV → 7 [DRY RUN] sbatch lines."""
        with tempfile.TemporaryDirectory() as d:
            result = _run_bench_sh([
                "--dry-run",
                "--bench-root", d,
                "--partition", "gpu_test",
            ])
        self.assertEqual(result.returncode, 0, result.stderr)
        dry_lines = [l for l in result.stdout.splitlines() if "[DRY RUN]" in l]
        self.assertEqual(len(dry_lines), 7)

    def test_dry_run_with_multi_comp(self):
        """Two reference comps → two REFERENCE_TPRS entries in export string."""
        with tempfile.TemporaryDirectory() as d:
            result = _run_bench_sh([
                "--dry-run",
                "--reference-comp", "POPC100", "DPPC100",
                "--bench-root", d,
                "--partition", "gpu_test",
            ])
        self.assertEqual(result.returncode, 0, result.stderr)
        # Both comps appear in the REFERENCE_TPRS export value
        self.assertIn("POPC100", result.stdout)
        self.assertIn("DPPC100", result.stdout)


_BENCH_CPU_SH = _REPO_ROOT / "scripts/simulation/benchmark_hpc_general1.sh"


def _run_bench_cpu_sh(args: list[str]) -> subprocess.CompletedProcess:
    env = {**os.environ, "GROUP": "testgroup", "USER": os.environ.get("USER", "testuser")}
    return subprocess.run(
        ["bash", str(_BENCH_CPU_SH)] + args,
        capture_output=True, text=True,
        cwd=str(_REPO_ROOT), env=env,
    )


class TestBenchmarkCpuHshDryRun(unittest.TestCase):
    def test_dry_run_emits_one_sbatch_per_point(self):
        """--dry-run default mode → 1 Phase-1 setup line + 7 Phase-2 bench lines."""
        with tempfile.TemporaryDirectory() as d:
            result = _run_bench_cpu_sh([
                "--dry-run",
                "--bench-root", d,
            ])
        self.assertEqual(result.returncode, 0, result.stderr)
        dry_lines = [l for l in result.stdout.splitlines() if "[DRY RUN]" in l]
        self.assertEqual(len(dry_lines), 8)
        # First line is the Phase 1 setup
        self.assertIn("sbatch_setup_general1.sh", dry_lines[0])
        # Remaining 7 are bench points
        for l in dry_lines[1:]:
            self.assertIn("sbatch_benchmark_hpc_general1.sh", l)

    def test_dry_run_reference_tpr_skips_phase_1(self):
        """--reference-tpr should skip Phase 1 → only 7 bench dry-run lines."""
        with tempfile.TemporaryDirectory() as d:
            result = _run_bench_cpu_sh([
                "--dry-run",
                "--bench-root", d,
                "--reference-tpr", "/tmp/some.tpr",
            ])
        self.assertEqual(result.returncode, 0, result.stderr)
        dry_lines = [l for l in result.stdout.splitlines() if "[DRY RUN]" in l]
        self.assertEqual(len(dry_lines), 7)
        self.assertIn("SKIPPED (--reference-tpr provided)", result.stdout)
        # All bench lines reference the user-supplied tpr
        for l in dry_lines:
            self.assertIn("/tmp/some.tpr", l)

    def test_dry_run_setup_overrides_propagate(self):
        """--setup-comp / --setup-nsteps-eq should appear in the Phase 1 export."""
        with tempfile.TemporaryDirectory() as d:
            result = _run_bench_cpu_sh([
                "--dry-run",
                "--bench-root", d,
                "--setup-comp", "DPPC100",
                "--setup-nsteps-eq", "5000",
            ])
        self.assertEqual(result.returncode, 0, result.stderr)
        setup_line = next(l for l in result.stdout.splitlines()
                          if "[DRY RUN]" in l and "sbatch_setup_general1.sh" in l)
        self.assertIn("COMP=DPPC100", setup_line)
        self.assertIn("NSTEPS_EQ=5000", setup_line)

    def test_dry_run_all_points_use_general1_partition(self):
        """Every sbatch line should target --partition=general1 by default."""
        with tempfile.TemporaryDirectory() as d:
            result = _run_bench_cpu_sh(["--dry-run", "--bench-root", d])
        self.assertEqual(result.returncode, 0, result.stderr)
        sbatch_lines = [l for l in result.stdout.splitlines() if "sbatch " in l]
        self.assertTrue(all("--partition=general1" in l for l in sbatch_lines))

    def test_dry_run_exports_mpi_ranks(self):
        """Worker env must include MPI_RANKS_PER_SIM (absent in the GPU benchmark)."""
        with tempfile.TemporaryDirectory() as d:
            result = _run_bench_cpu_sh(["--dry-run", "--bench-root", d])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("MPI_RANKS_PER_SIM=", result.stdout)

    def test_dry_run_total_cpus_capped_at_40(self):
        """Every default point should request --cpus-per-task=40 (full general1 node)
        — except the lone half-node 1sim_1rank_20omp probe at 20.
        """
        with tempfile.TemporaryDirectory() as d:
            result = _run_bench_cpu_sh(["--dry-run", "--bench-root", d])
        self.assertEqual(result.returncode, 0, result.stderr)
        cpus_seen = set()
        for line in result.stdout.splitlines():
            for tok in line.split():
                if tok.startswith("--cpus-per-task="):
                    cpus_seen.add(int(tok.split("=")[1]))
        # We expect {20, 40} from the default 7-point sweep
        self.assertTrue(cpus_seen.issubset({20, 40}),
                        f"unexpected --cpus-per-task values: {cpus_seen}")
        self.assertIn(40, cpus_seen)


if __name__ == "__main__":
    unittest.main()
