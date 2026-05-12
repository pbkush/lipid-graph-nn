"""End-to-end smoke test for the Martini pipeline with real gmx.

Opt-in: set RUN_MARTINI_E2E=1 in the environment before running.
Requires: insane on PATH, gmx on PATH, resources/martini3/itp/ populated.

Run:
    RUN_MARTINI_E2E=1 pytest tests/martini_pipeline/test_e2e_smoke.py -v
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

_RUN_E2E = os.environ.get("RUN_MARTINI_E2E", "0") == "1"

_NSTEPS_SMOKE = 5000       # 50 ps at dt=0.01 ps — fast but non-trivial
_APL_MIN_NM2 = 0.55        # wide tolerance for a 50 ps run
_APL_MAX_NM2 = 0.80


@unittest.skipUnless(_RUN_E2E, "Set RUN_MARTINI_E2E=1 to run end-to-end tests")
class TestE2ESmokeTest(unittest.TestCase):
    """Runs DIPC100 for 50 ps and checks physical plausibility."""

    @classmethod
    def setUpClass(cls):
        cls._tmpdir = tempfile.mkdtemp(prefix="martini_e2e_")
        from lipid_gnn.martini_pipeline.mdp_writer import MDPParams
        from lipid_gnn.martini_pipeline.pipeline import run as pipeline_run

        cls._result = pipeline_run(
            {"DIPC": 1.0},
            os.path.join(cls._tmpdir, "DIPC100"),
            mdp_params=MDPParams(nsteps_prod=_NSTEPS_SMOKE),
        )

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_overall_status_ok(self):
        self.assertEqual(self._result.overall_status, "ok")

    def test_prun_xtc_exists(self):
        xtc = os.path.join(self._result.out_dir, "run", "prun.xtc")
        self.assertTrue(os.path.isfile(xtc), f"prun.xtc not found at {xtc}")

    def test_prun_xtc_has_frames(self):
        import MDAnalysis as mda
        xtc = os.path.join(self._result.out_dir, "run", "prun.xtc")
        tpr = os.path.join(self._result.out_dir, "run", "prun.tpr")
        u = mda.Universe(tpr, xtc)
        self.assertGreaterEqual(len(u.trajectory), 2, "prun.xtc should have ≥ 2 frames")

    def test_manifest_parses_and_status_ok(self):
        with open(self._result.manifest_path) as fh:
            data = json.load(fh)
        self.assertEqual(data["overall_status"], "ok")
        self.assertEqual(data["schema_version"], "1.0")

    def test_apl_in_physical_range(self):
        """Mean APL over last half of trajectory should be in [0.55, 0.80] nm²."""
        import MDAnalysis as mda
        import numpy as np

        xtc = os.path.join(self._result.out_dir, "run", "prun.xtc")
        tpr = os.path.join(self._result.out_dir, "run", "prun.tpr")
        u = mda.Universe(tpr, xtc)

        with open(self._result.manifest_path) as fh:
            manifest = json.load(fh)
        n_lipids = manifest["build_stats"]["molecule_counts"].get("DIPC", 0)
        if n_lipids == 0:
            self.skipTest("DIPC count not in manifest")

        n_frames = len(u.trajectory)
        start = max(0, n_frames // 2)
        apls = []
        for ts in u.trajectory[start:]:
            lx = ts.dimensions[0] / 10.0  # Å → nm
            ly = ts.dimensions[1] / 10.0
            apls.append(lx * ly / (n_lipids / 2))

        mean_apl = float(np.mean(apls))
        self.assertGreaterEqual(mean_apl, _APL_MIN_NM2,
                                f"APL {mean_apl:.4f} nm² below minimum {_APL_MIN_NM2}")
        self.assertLessEqual(mean_apl, _APL_MAX_NM2,
                             f"APL {mean_apl:.4f} nm² above maximum {_APL_MAX_NM2}")

    def test_no_energy_blowup(self):
        """Potential energy should not contain NaN or exceed 1e8 kJ/mol."""
        try:
            import pyedr
            import numpy as np
            edr = os.path.join(self._result.out_dir, "run", "prun.edr")
            data = pyedr.get_edr(edr)
            epot = data.get("Potential", data.get("potential", None))
            if epot is None:
                self.skipTest("Potential not in edr")
            self.assertFalse(np.any(np.isnan(epot)), "NaN found in potential energy")
            self.assertLess(float(np.max(np.abs(epot))), 1e8, "Potential energy blowup detected")
        except ImportError:
            self.skipTest("pyedr not installed")


if __name__ == "__main__":
    unittest.main()
