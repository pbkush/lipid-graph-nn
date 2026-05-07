"""Tests for lipid_gnn.martini_pipeline.mdp_writer."""
from __future__ import annotations

import json
import os
import shutil
import string
import sys
import unittest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from lipid_gnn.martini_pipeline.analysis import _parse_mdp_file
from lipid_gnn.martini_pipeline.mdp_writer import (
    MDPParams,
    STAGES,
    _DEFAULT_FREEZE,
    _DEFAULT_TEMPLATES,
    _nstfout,
    render_mdp,
    write_mdps,
)

_HAS_LEGACY_DATA = os.path.isfile(
    os.path.join(_REPO_ROOT, "data", "membrane_only", "POPC100", "run.mdp")
)
_HAS_GMX = bool(shutil.which("gmx"))


# ---------------------------------------------------------------------------
# 1. freeze_missing_raises
# ---------------------------------------------------------------------------

class TestFreezeMissingRaises(unittest.TestCase):
    def test_missing_freeze_raises_file_not_found(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "no_such_freeze.json")
            with self.assertRaises(FileNotFoundError) as ctx:
                write_mdps(tmp, freeze_path=missing)
            self.assertIn("audit_mdps.py", str(ctx.exception))


# ---------------------------------------------------------------------------
# 2. render_em_minimal
# ---------------------------------------------------------------------------

class TestRenderEM(unittest.TestCase):
    def setUp(self):
        with open(_DEFAULT_FREEZE) as fh:
            self.freeze = json.load(fh)
        tmpl_path = os.path.join(_DEFAULT_TEMPLATES, "em.mdp.tmpl")
        with open(tmpl_path) as fh:
            self.tmpl = fh.read()

    def test_em_keys_present(self):
        content = render_mdp("minimization", MDPParams(), self.freeze["minimization"], self.tmpl)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mdp", delete=False) as fh:
            fh.write(content)
            path = fh.name
        try:
            parsed = _parse_mdp_file(path)
        finally:
            os.unlink(path)
        for key in ("integrator", "emtol", "emstep", "cutoff-scheme", "coulombtype",
                    "coulomb-modifier", "rcoulomb", "vdw-type", "rvdw", "tcoupl", "pcoupl"):
            self.assertIn(key, parsed, f"key '{key}' missing from rendered em.mdp")

    def test_em_nsteps_from_params(self):
        content = render_mdp("minimization", MDPParams(nsteps_min=5000),
                             self.freeze["minimization"], self.tmpl)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mdp", delete=False) as fh:
            fh.write(content)
            path = fh.name
        try:
            parsed = _parse_mdp_file(path)
        finally:
            os.unlink(path)
        self.assertEqual(parsed["nsteps"], "5000")

    def test_em_integrator_steep(self):
        content = render_mdp("minimization", MDPParams(), self.freeze["minimization"], self.tmpl)
        self.assertIn("steep", content)


# ---------------------------------------------------------------------------
# 3. render_eq_overrides_legacy (Decision 14 commitments)
# ---------------------------------------------------------------------------

class TestRenderEQ(unittest.TestCase):
    def setUp(self):
        with open(_DEFAULT_FREEZE) as fh:
            self.freeze = json.load(fh)
        tmpl_path = os.path.join(_DEFAULT_TEMPLATES, "eq.mdp.tmpl")
        with open(tmpl_path) as fh:
            self.tmpl = fh.read()

    def _render_and_parse(self, params=None):
        if params is None:
            params = MDPParams()
        content = render_mdp("equilibration", params, self.freeze["equilibration"], self.tmpl)
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mdp", delete=False) as fh:
            fh.write(content)
            path = fh.name
        try:
            return _parse_mdp_file(path)
        finally:
            os.unlink(path)

    def test_compressibility_3e4(self):
        parsed = self._render_and_parse()
        self.assertEqual(parsed["compressibility"], "3e-4 3e-4",
                         "compressibility must be 3e-4 (Decision 14)")

    def test_nsteps_1e6(self):
        parsed = self._render_and_parse()
        self.assertEqual(parsed["nsteps"], "1000000",
                         "eq nsteps must be 1 000 000 (Decision 14)")

    def test_nstenergy_1000(self):
        parsed = self._render_and_parse()
        self.assertEqual(parsed["nstenergy"], "1000",
                         "nstenergy must be 1000 (Decision 14)")

    def test_gen_vel_yes(self):
        parsed = self._render_and_parse()
        self.assertEqual(parsed["gen-vel"], "yes",
                         "gen-vel must be yes (Decision 14)")

    def test_gen_temp_310(self):
        parsed = self._render_and_parse()
        self.assertEqual(parsed["gen-temp"], "310",
                         "gen-temp must be 310 (Decision 14)")

    def test_pcoupl_berendsen(self):
        parsed = self._render_and_parse()
        self.assertIn(parsed["pcoupl"].lower(), ("berendsen",),
                      "eq must use Berendsen barostat")

    def test_pcoupltype_semiisotropic(self):
        parsed = self._render_and_parse()
        self.assertIn("semiisotropic", parsed["pcoupltype"].lower())

    def test_nsteps_overrideable(self):
        parsed = self._render_and_parse(MDPParams(nsteps_eq=500_000))
        self.assertEqual(parsed["nsteps"], "500000")

    def test_nstenergy_overrideable(self):
        parsed = self._render_and_parse(MDPParams(nstenergy_eq=500))
        self.assertEqual(parsed["nstenergy"], "500")


# ---------------------------------------------------------------------------
# 4. render_run_clones_legacy
# ---------------------------------------------------------------------------

class TestRenderRun(unittest.TestCase):
    def setUp(self):
        with open(_DEFAULT_FREEZE) as fh:
            self.freeze = json.load(fh)
        tmpl_path = os.path.join(_DEFAULT_TEMPLATES, "run.mdp.tmpl")
        with open(tmpl_path) as fh:
            self.tmpl = fh.read()

    def _render_and_parse(self, params):
        import tempfile
        content = render_mdp("run", params, self.freeze["run"], self.tmpl)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mdp", delete=False) as fh:
            fh.write(content)
            path = fh.name
        try:
            return _parse_mdp_file(path)
        finally:
            os.unlink(path)

    def test_run_clones_legacy_keys(self):
        """Keys shared between run template and freeze.run must match canonical."""
        params = MDPParams(gen_seed=42)
        parsed = self._render_and_parse(params)
        run_freeze = self.freeze["run"]
        # Keys that are placeholders — exempt from equality check
        placeholder_keys = {"nsteps", "nstfout", "gen-seed", "gen-temp",
                            "gen-vel", "gen-seed"}
        for key, val in run_freeze.items():
            if key in placeholder_keys:
                continue
            if key in parsed:
                self.assertEqual(parsed[key], val,
                                 f"run.mdp key '{key}' differs from freeze canonical")

    def test_run_nsteps_default_minus1(self):
        parsed = self._render_and_parse(MDPParams(gen_seed=1))
        self.assertEqual(parsed["nsteps"], "-1")

    def test_run_nsteps_override(self):
        parsed = self._render_and_parse(MDPParams(nsteps_prod=50_000, gen_seed=1))
        self.assertEqual(parsed["nsteps"], "50000")


# ---------------------------------------------------------------------------
# 5. save_forces toggle
# ---------------------------------------------------------------------------

class TestSaveForces(unittest.TestCase):
    def setUp(self):
        with open(_DEFAULT_FREEZE) as fh:
            self.freeze = json.load(fh)
        tmpl_path = os.path.join(_DEFAULT_TEMPLATES, "run.mdp.tmpl")
        with open(tmpl_path) as fh:
            self.tmpl = fh.read()

    def _render_nstfout(self, save_forces: bool) -> str:
        import tempfile
        params = MDPParams(save_forces=save_forces, gen_seed=1)
        content = render_mdp("run", params, self.freeze["run"], self.tmpl)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mdp", delete=False) as fh:
            fh.write(content)
            path = fh.name
        try:
            return _parse_mdp_file(path)["nstfout"]
        finally:
            os.unlink(path)

    def test_save_forces_false_gives_zero(self):
        self.assertEqual(self._render_nstfout(False), "0")

    def test_save_forces_true_matches_nstxout_compressed(self):
        nstxout_compressed = self.freeze["run"]["nstxout-compressed"]
        self.assertEqual(self._render_nstfout(True), nstxout_compressed)

    def test_nstfout_helper_false(self):
        self.assertEqual(_nstfout(False, {"nstxout-compressed": "75000"}), 0)

    def test_nstfout_helper_true(self):
        self.assertEqual(_nstfout(True, {"nstxout-compressed": "75000"}), 75000)

    def test_nstfout_helper_fallback(self):
        self.assertEqual(_nstfout(True, {}), 75000)


# ---------------------------------------------------------------------------
# 6. seed handling
# ---------------------------------------------------------------------------

class TestSeed(unittest.TestCase):
    def setUp(self):
        with open(_DEFAULT_FREEZE) as fh:
            self.freeze = json.load(fh)
        tmpl_path = os.path.join(_DEFAULT_TEMPLATES, "run.mdp.tmpl")
        with open(tmpl_path) as fh:
            self.tmpl = fh.read()

    def _get_seed_from_content(self, content: str) -> str:
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".mdp", delete=False) as fh:
            fh.write(content)
            path = fh.name
        try:
            return _parse_mdp_file(path)["gen-seed"]
        finally:
            os.unlink(path)

    def test_explicit_seed_round_trips(self):
        params = MDPParams(gen_seed=12345)
        content = render_mdp("run", params, self.freeze["run"], self.tmpl)
        self.assertEqual(self._get_seed_from_content(content), "12345")

    def test_random_seed_is_positive_int(self):
        params = MDPParams(gen_seed=None)
        content = render_mdp("run", params, self.freeze["run"], self.tmpl)
        seed_val = int(self._get_seed_from_content(content))
        self.assertGreater(seed_val, 0)
        self.assertLessEqual(seed_val, 2**31 - 1)

    def test_two_random_calls_differ(self):
        params = MDPParams(gen_seed=None)
        c1 = render_mdp("run", params, self.freeze["run"], self.tmpl)
        c2 = render_mdp("run", params, self.freeze["run"], self.tmpl)
        s1 = self._get_seed_from_content(c1)
        s2 = self._get_seed_from_content(c2)
        self.assertNotEqual(s1, s2, "two calls with gen_seed=None must produce different seeds")

    def test_write_mdps_seed_consistent_across_stages(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            written = write_mdps(tmp, params=MDPParams(gen_seed=None))
            seed_eq = _parse_mdp_file(written["equilibration"])["gen-seed"]
            seed_run = _parse_mdp_file(written["run"])["gen-seed"]
            self.assertEqual(seed_eq, seed_run,
                             "eq and run must share the same seed within one write_mdps call")


# ---------------------------------------------------------------------------
# 7. write_mdps round-trip
# ---------------------------------------------------------------------------

class TestWriteMdpsRoundtrip(unittest.TestCase):
    def test_all_three_files_written(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            written = write_mdps(tmp, params=MDPParams(gen_seed=1))
            self.assertEqual(set(written.keys()), set(STAGES))
            for path in written.values():
                self.assertTrue(os.path.isfile(path), f"expected file: {path}")

    def test_files_parse_cleanly(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            written = write_mdps(tmp, params=MDPParams(gen_seed=99))
            for stage, path in written.items():
                parsed = _parse_mdp_file(path)
                self.assertGreater(len(parsed), 5,
                                   f"stage {stage}: parsed too few keys")

    def test_out_dir_created(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            new_dir = os.path.join(tmp, "sub", "dir")
            write_mdps(new_dir, params=MDPParams(gen_seed=1))
            self.assertTrue(os.path.isdir(new_dir))

    def test_deterministic_with_explicit_seed(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp1, \
             tempfile.TemporaryDirectory() as tmp2:
            w1 = write_mdps(tmp1, params=MDPParams(gen_seed=7))
            w2 = write_mdps(tmp2, params=MDPParams(gen_seed=7))
            for stage in STAGES:
                with open(w1[stage]) as f1, open(w2[stage]) as f2:
                    self.assertEqual(f1.read(), f2.read(),
                                     f"stage {stage} not deterministic with same seed")


# ---------------------------------------------------------------------------
# 8. strict template substitution
# ---------------------------------------------------------------------------

class TestTemplateStrict(unittest.TestCase):
    def setUp(self):
        with open(_DEFAULT_FREEZE) as fh:
            self.freeze = json.load(fh)

    def test_unknown_placeholder_raises(self):
        broken_tmpl = "integrator = md\nnsteps = ${unknown_knob}\n"
        with self.assertRaises((KeyError, ValueError)):
            render_mdp("run", MDPParams(), self.freeze["run"], broken_tmpl)


# ---------------------------------------------------------------------------
# 9. [CONFIG:] markers
# ---------------------------------------------------------------------------

class TestConfigMarkers(unittest.TestCase):
    def test_rendered_files_contain_config_markers(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            written = write_mdps(tmp, params=MDPParams(gen_seed=1))
            for stage, path in written.items():
                with open(path) as fh:
                    content = fh.read()
                self.assertIn("[CONFIG:", content,
                              f"stage {stage}: no [CONFIG:] marker found in rendered file")

    def test_all_param_fields_covered_in_templates(self):
        """Every MDPParams field must appear as [CONFIG: <field>] somewhere across the templates."""
        import dataclasses
        fields = {f.name for f in dataclasses.fields(MDPParams)}
        combined = ""
        for tmpl_name in ("em.mdp.tmpl", "eq.mdp.tmpl", "run.mdp.tmpl"):
            with open(os.path.join(_DEFAULT_TEMPLATES, tmpl_name)) as fh:
                combined += fh.read()
        for field_name in fields:
            self.assertIn(f"[CONFIG: {field_name}]", combined,
                          f"MDPParams field '{field_name}' has no [CONFIG: {field_name}] marker "
                          "in any template")


# ---------------------------------------------------------------------------
# 10. grompp smoke (opt-in)
# ---------------------------------------------------------------------------

_RUN_GROMPP = os.environ.get("RUN_MDP_GROMPP") == "1"
_LEGACY_POPC100 = os.path.join(_REPO_ROOT, "data", "membrane_only", "POPC100")


@unittest.skipUnless(_RUN_GROMPP and _HAS_GMX and _HAS_LEGACY_DATA,
                     "Skipped: set RUN_MDP_GROMPP=1 with gmx and legacy data available")
class TestGromppSmoke(unittest.TestCase):
    def test_grompp_accepts_em_mdp(self):
        import subprocess
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            written = write_mdps(tmp, params=MDPParams(gen_seed=1))
            gro = os.path.join(_LEGACY_POPC100, "run.gro")
            top = os.path.join(_LEGACY_POPC100, "topol.top")
            em_mdp = written["minimization"]
            result = subprocess.run(
                ["gmx", "grompp", "-f", em_mdp, "-c", gro, "-p", top,
                 "-o", os.path.join(tmp, "em.tpr"), "-maxwarn", "1"],
                capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 0,
                             f"gmx grompp failed for em.mdp:\n{result.stderr}")


if __name__ == "__main__":
    unittest.main()
