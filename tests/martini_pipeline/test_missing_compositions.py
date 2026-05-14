"""Tests for analysis.py grid generators, summarise_systems, and missing_compositions."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from lipid_gnn.martini_pipeline.analysis import (
    SystemStatus,
    binary_grid,
    dopc_corner_grid,
    dppc_corner_grid,
    missing_compositions,
    popc_interpolation_grid,
    summarise_systems,
    ternary_grid,
)
from lipid_gnn.martini_pipeline.composition import Composition, parse_name


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_manifest(out_dir: str, overall_status: str = "ok") -> None:
    """Write a minimal manifest.json to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "schema_version": "1.0",
        "overall_status": overall_status,
        "stages": [
            {"name": "minimization", "status": "ok", "walltime_s": 1.0},
            {"name": "equilibration", "status": "ok", "walltime_s": 2.0},
            {"name": "production", "status": "ok", "walltime_s": 3.0},
        ],
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as fh:
        json.dump(manifest, fh)


def _write_prun_xtc(out_dir: str) -> None:
    """Write a stub run/prun.xtc to out_dir to simulate a legacy system."""
    run_dir = os.path.join(out_dir, "run")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "prun.xtc"), "wb") as fh:
        fh.write(b"stub")


# ---------------------------------------------------------------------------
# Grid generator tests
# ---------------------------------------------------------------------------

class TestBinaryGrid(unittest.TestCase):
    def test_binary_grid_count(self):
        result = binary_grid("POPC", "DOPC", step=10)
        # 9 mixtures (10%–90%) + 2 pures = 11
        self.assertEqual(len(result), 11)

    def test_binary_grid_step_5(self):
        result = binary_grid("POPC", "DOPC", step=5)
        # 19 mixtures (5%–95%) + 2 pures = 21
        self.assertEqual(len(result), 21)

    def test_binary_grid_pure_included(self):
        result = binary_grid("POPC", "DOPC", step=10)
        names = {c.name for c in result}
        self.assertIn("POPC100", names)
        self.assertIn("DOPC100", names)

    def test_binary_grid_no_pure(self):
        result = binary_grid("POPC", "DOPC", step=10, include_pure=False)
        names = {c.name for c in result}
        self.assertNotIn("POPC100", names)
        self.assertNotIn("DOPC100", names)
        self.assertEqual(len(result), 9)

    def test_binary_grid_fractions_sum_to_one(self):
        for comp in binary_grid("DPPC", "DIPC", step=10):
            self.assertAlmostEqual(sum(comp.fractions.values()), 1.0, places=9)


class TestTernaryGrid(unittest.TestCase):
    def test_ternary_grid_simplex_count(self):
        result = ternary_grid(["POPC", "DOPC", "DPPC"], step=20)
        # full 3-component simplex: C(100/step + 2, 2) = C(7,2) = 21 points
        self.assertEqual(len(result), 21)

    def test_ternary_grid_interior_only(self):
        result = ternary_grid(["POPC", "DOPC", "DPPC"], step=20, include_edges=False)
        # only strictly interior points (all fracs > 0): 6 for step=20
        self.assertEqual(len(result), 6)
        for comp in result:
            for frac in comp.fractions.values():
                self.assertGreater(frac, 0.0)

    def test_ternary_grid_step_10_count(self):
        result = ternary_grid(["POPC", "DOPC", "DPPC"], step=10)
        # C(12, 2) = 66
        self.assertEqual(len(result), 66)

    def test_ternary_grid_wrong_lipid_count(self):
        with self.assertRaises(ValueError):
            ternary_grid(["POPC", "DOPC"], step=10)

    def test_ternary_grid_fractions_sum_to_one(self):
        for comp in ternary_grid(["POPC", "DOPC", "DPPC"], step=20):
            self.assertAlmostEqual(sum(comp.fractions.values()), 1.0, places=9)


class TestCornerGrids(unittest.TestCase):
    def test_dppc_corner_grid_includes_pure_dppc(self):
        names = {c.name for c in dppc_corner_grid()}
        self.assertIn("DPPC100", names)

    def test_dopc_corner_grid_includes_pure_dopc(self):
        names = {c.name for c in dopc_corner_grid()}
        self.assertIn("DOPC100", names)

    def test_corner_grids_use_canonical_names(self):
        for comp in dppc_corner_grid() + dopc_corner_grid():
            parsed = parse_name(comp.name)
            self.assertEqual(comp.name, parsed.name)

    def test_dppc_corner_chol_capped_at_40(self):
        for comp in dppc_corner_grid():
            if "CHOL" in comp.fractions:
                self.assertLessEqual(comp.fractions["CHOL"], 0.40 + 1e-9)

    def test_dopc_corner_chol_capped_at_40(self):
        for comp in dopc_corner_grid():
            if "CHOL" in comp.fractions:
                self.assertLessEqual(comp.fractions["CHOL"], 0.40 + 1e-9)

    def test_dppc_corner_anchor_gte_50(self):
        for comp in dppc_corner_grid():
            if "DPPC" in comp.fractions:
                self.assertGreaterEqual(comp.fractions["DPPC"], 0.50 - 1e-9)

    def test_dopc_corner_anchor_gte_50(self):
        for comp in dopc_corner_grid():
            if "DOPC" in comp.fractions:
                self.assertGreaterEqual(comp.fractions["DOPC"], 0.50 - 1e-9)

    def test_dppc_corner_no_duplicates(self):
        names = [c.name for c in dppc_corner_grid()]
        self.assertEqual(len(names), len(set(names)))

    def test_dopc_corner_no_duplicates(self):
        names = [c.name for c in dopc_corner_grid()]
        self.assertEqual(len(names), len(set(names)))


class TestPopcInterpolationGrid(unittest.TestCase):
    """Subgoal 3a — POPC-anchored binaries across the full POPC fraction range.
    Distinct from the corner grids (which cap at anchor>=50%): used for
    densifying the GNN training-domain regime, not for extrapolation.
    """

    def test_includes_pure_popc(self):
        names = {c.name for c in popc_interpolation_grid()}
        self.assertIn("POPC100", names)

    def test_excludes_pure_partners(self):
        """Pure-partner endpoints belong in extrapolation grids, not here."""
        for comp in popc_interpolation_grid():
            self.assertIn("POPC", comp.fractions,
                          f"non-POPC composition in popc_interpolation: {comp.name}")

    def test_step_10_default_total_count(self):
        """At step=10: POPC100 + 8 non-CHOL partners × 9 fractions + CHOL × 4 = 77."""
        grid = popc_interpolation_grid(step=10)
        self.assertEqual(len(grid), 1 + 8 * 9 + 4)

    def test_chol_capped_at_40(self):
        for comp in popc_interpolation_grid():
            if "CHOL" in comp.fractions:
                self.assertLessEqual(comp.fractions["CHOL"], 0.40 + 1e-9)

    def test_full_popc_range_covered(self):
        """Both POPC10_X90 (low-POPC) and POPC90_X10 (high-POPC) must appear
        for non-CHOL partners — that's the point of interpolation vs corner."""
        names = {c.name for c in popc_interpolation_grid()}
        # Low-POPC corner of the binary: DPPC90_POPC10 (canonical order)
        self.assertIn("DPPC90_POPC10", names)
        # High-POPC end: POPC90_DPPC10
        self.assertIn("POPC90_DPPC10", names)

    def test_uses_canonical_names(self):
        for comp in popc_interpolation_grid():
            self.assertEqual(comp.name, parse_name(comp.name).name)

    def test_no_duplicates(self):
        names = [c.name for c in popc_interpolation_grid()]
        self.assertEqual(len(names), len(set(names)))

    def test_step_validation(self):
        with self.assertRaises(ValueError):
            popc_interpolation_grid(step=0)
        with self.assertRaises(ValueError):
            popc_interpolation_grid(step=101)

    def test_step_5_doubles_density(self):
        """At step=5, each non-CHOL partner gets 19 fractions (5..95) instead of 9."""
        grid5 = popc_interpolation_grid(step=5)
        # POPC100 + 8 non-CHOL partners × 19 fractions + CHOL × 8 (5..40) = 161
        self.assertEqual(len(grid5), 1 + 8 * 19 + 8)


# ---------------------------------------------------------------------------
# missing_compositions tests
# ---------------------------------------------------------------------------

class TestMissingCompositions(unittest.TestCase):
    def test_missing_compositions_empty_grid(self):
        result = missing_compositions([], ["/nonexistent"])
        self.assertEqual(result, [])

    def test_missing_compositions_empty_root(self):
        grid = [Composition({"POPC": 0.5, "DOPC": 0.5})]
        result = missing_compositions(grid, ["/nonexistent_dir_XYZ"])
        self.assertEqual(len(result), 1)

    def test_missing_compositions_with_manifest_ok(self):
        grid = [
            Composition({"POPC": 0.5, "DOPC": 0.5}),
            Composition({"DPPC": 1.0}),
        ]
        with tempfile.TemporaryDirectory() as root:
            comp_dir = os.path.join(root, grid[0].name)
            _write_manifest(comp_dir, "ok")
            result = missing_compositions(grid, [root])
        # first comp is present, second is missing
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].name, "DPPC100")

    def test_missing_compositions_with_manifest_failed(self):
        comp = Composition({"POPC": 0.5, "DOPC": 0.5})
        with tempfile.TemporaryDirectory() as root:
            comp_dir = os.path.join(root, comp.name)
            _write_manifest(comp_dir, "failed_at_equilibration")
            result = missing_compositions([comp], [root])
        # failed status → still missing
        self.assertEqual(len(result), 1)

    def test_missing_compositions_legacy_fallback_on(self):
        comp = Composition({"POPC": 0.5, "DOPC": 0.5})
        with tempfile.TemporaryDirectory() as root:
            comp_dir = os.path.join(root, comp.name)
            _write_prun_xtc(comp_dir)  # no manifest
            result = missing_compositions([comp], [root], legacy_fallback=True)
        self.assertEqual(result, [])

    def test_missing_compositions_legacy_fallback_off(self):
        comp = Composition({"POPC": 0.5, "DOPC": 0.5})
        with tempfile.TemporaryDirectory() as root:
            comp_dir = os.path.join(root, comp.name)
            _write_prun_xtc(comp_dir)  # no manifest
            result = missing_compositions([comp], [root], legacy_fallback=False)
        # without legacy fallback, no manifest → missing
        self.assertEqual(len(result), 1)

    def test_missing_compositions_union_across_roots(self):
        comp = Composition({"POPC": 0.5, "DOPC": 0.5})
        with tempfile.TemporaryDirectory() as root_a, \
             tempfile.TemporaryDirectory() as root_b:
            _write_manifest(os.path.join(root_a, comp.name), "ok")
            # root_b has nothing
            result = missing_compositions([comp], [root_a, root_b])
        # comp is present in root_a → not missing
        self.assertEqual(result, [])

    def test_missing_compositions_canonical_aliasing(self):
        # POPC10_DOPC90 and DOPC90_POPC10 should canonicalise to the same name
        comp_target = parse_name("DOPC90_POPC10")
        with tempfile.TemporaryDirectory() as root:
            # directory named with the canonical form
            comp_dir = os.path.join(root, comp_target.name)  # DOPC90_POPC10
            _write_manifest(comp_dir, "ok")
            result = missing_compositions([comp_target], [root])
        self.assertEqual(result, [])

    def test_missing_compositions_deduplicates_grid(self):
        comp = Composition({"POPC": 0.5, "DOPC": 0.5})
        grid = [comp, comp, comp]  # duplicates
        with tempfile.TemporaryDirectory() as root:
            result = missing_compositions(grid, [root])
        # deduped to 1, all missing
        self.assertEqual(len(result), 1)

    def test_invalid_manifest_counted_missing(self):
        comp = Composition({"POPC": 0.5, "DOPC": 0.5})
        with tempfile.TemporaryDirectory() as root:
            comp_dir = os.path.join(root, comp.name)
            os.makedirs(comp_dir, exist_ok=True)
            with open(os.path.join(comp_dir, "manifest.json"), "w") as fh:
                fh.write("{corrupt json")
            result = missing_compositions([comp], [root])
        # corrupt manifest → treated as missing
        self.assertEqual(len(result), 1)


# ---------------------------------------------------------------------------
# summarise_systems tests
# ---------------------------------------------------------------------------

class TestSummariseSystems(unittest.TestCase):
    def test_summarise_systems_returns_status_per_dir(self):
        comps = [
            Composition({"POPC": 1.0}),
            Composition({"DOPC": 1.0}),
            Composition({"DPPC": 1.0}),
        ]
        with tempfile.TemporaryDirectory() as root:
            _write_manifest(os.path.join(root, comps[0].name), "ok")
            _write_manifest(os.path.join(root, comps[1].name), "failed_at_production")
            # third has no manifest but has prun.xtc
            _write_prun_xtc(os.path.join(root, comps[2].name))

            statuses = summarise_systems(root, legacy_fallback=True)

        self.assertEqual(len(statuses), 3)
        by_name = {s.canonical_name: s for s in statuses}

        self.assertTrue(by_name["POPC100"].has_manifest)
        self.assertEqual(by_name["POPC100"].overall_status, "ok")
        self.assertAlmostEqual(by_name["POPC100"].walltime_s, 6.0)

        self.assertTrue(by_name["DOPC100"].has_manifest)
        self.assertEqual(by_name["DOPC100"].overall_status, "failed_at_production")

        self.assertFalse(by_name["DPPC100"].has_manifest)
        self.assertIsNone(by_name["DPPC100"].overall_status)
        self.assertTrue(by_name["DPPC100"].has_prun_xtc)

    def test_summarise_systems_empty_root(self):
        result = summarise_systems("/nonexistent_dir_XYZ")
        self.assertEqual(result, [])

    def test_summarise_systems_skips_no_manifest_without_fallback(self):
        comp = Composition({"POPC": 1.0})
        with tempfile.TemporaryDirectory() as root:
            _write_prun_xtc(os.path.join(root, comp.name))
            result = summarise_systems(root, legacy_fallback=False)
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
