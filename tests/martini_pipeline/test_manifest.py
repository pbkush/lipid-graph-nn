"""Tests for lipid_gnn.martini_pipeline.manifest."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)

from lipid_gnn.martini_pipeline.manifest import (
    Manifest,
    detect_gmx_version,
    detect_insane_version,
    hash_file,
    read_manifest,
    write_manifest,
)

_SAMPLE_MANIFEST = Manifest(
    schema_version="1.0",
    composition={"POPC": 1.0},
    canonical_name="POPC100",
    out_dir="/tmp/POPC100",
    created_utc="2026-05-12T00:00:00+00:00",
    gmx_version="2024.1",
    insane_version="1.2.0",
    insane_cmd=["insane", "-o", "run.gro", "-p", "topol.top"],
    seed=12345,
    box={"xy_nm": 11.0, "z_nm": 10.0},
    mdp_params={"nsteps_prod": 5000},
    mdp_hashes={"em.mdp": "sha256:abc", "eq.mdp": "sha256:def", "run.mdp": "sha256:123"},
    stages=[{"name": "minimization", "status": "ok"}],
    build_stats={"total_atoms": 1000, "molecule_counts": {"POPC": 392}},
    overall_status="ok",
    host={"hostname": "testhost", "platform": "Linux", "python": "3.11.0"},
)


class TestManifestRoundTrip(unittest.TestCase):
    def test_manifest_round_trip(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_manifest(path, _SAMPLE_MANIFEST)
            m2 = read_manifest(path)
            self.assertEqual(_SAMPLE_MANIFEST, m2)
        finally:
            os.unlink(path)

    def test_write_is_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_manifest(path, _SAMPLE_MANIFEST)
            with open(path) as fh:
                data = json.load(fh)
            self.assertIn("schema_version", data)
        finally:
            os.unlink(path)

    def test_manifest_schema_fields(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            write_manifest(path, _SAMPLE_MANIFEST)
            with open(path) as fh:
                data = json.load(fh)
            required = [
                "schema_version", "composition", "canonical_name", "out_dir",
                "created_utc", "gmx_version", "insane_version", "insane_cmd",
                "seed", "box", "mdp_params", "mdp_hashes", "stages",
                "build_stats", "overall_status", "host",
            ]
            for field in required:
                self.assertIn(field, data, f"Missing field: {field}")
        finally:
            os.unlink(path)


class TestHashFile(unittest.TestCase):
    def test_hash_starts_with_sha256(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"test content")
            path = f.name
        try:
            h = hash_file(path)
            self.assertTrue(h.startswith("sha256:"))
        finally:
            os.unlink(path)

    def test_mdp_hash_changes_with_content(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"original content")
            path = f.name
        try:
            h1 = hash_file(path)
            with open(path, "wb") as fh:
                fh.write(b"different content")
            h2 = hash_file(path)
            self.assertNotEqual(h1, h2)
        finally:
            os.unlink(path)

    def test_hash_is_deterministic(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            f.write(b"stable content")
            path = f.name
        try:
            h1 = hash_file(path)
            h2 = hash_file(path)
            self.assertEqual(h1, h2)
        finally:
            os.unlink(path)


class TestDetectVersions(unittest.TestCase):
    def test_detect_insane_version(self):
        version = detect_insane_version()
        # insane is installed in this env; version should be parseable
        if version != "unknown":
            self.assertRegex(version, r"^\d+\.\d+")

    def test_detect_gmx_version_absent(self):
        result = detect_gmx_version("gmx_does_not_exist_XYZ")
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
