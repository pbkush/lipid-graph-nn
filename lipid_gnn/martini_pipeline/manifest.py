"""JSON manifest writer/reader for a Martini pipeline run."""
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Manifest:
    schema_version: str
    composition: dict
    canonical_name: str
    out_dir: str
    created_utc: str
    gmx_version: Optional[str]
    insane_version: str
    insane_cmd: list
    seed: int
    box: dict
    mdp_params: dict
    mdp_hashes: dict
    stages: list
    build_stats: dict
    overall_status: str
    host: dict


def write_manifest(path: str, m: Manifest) -> None:
    """Atomically write manifest JSON (tempfile + rename)."""
    data = {
        "schema_version": m.schema_version,
        "composition": m.composition,
        "canonical_name": m.canonical_name,
        "out_dir": m.out_dir,
        "created_utc": m.created_utc,
        "gmx_version": m.gmx_version,
        "insane_version": m.insane_version,
        "insane_cmd": m.insane_cmd,
        "seed": m.seed,
        "box": m.box,
        "mdp_params": m.mdp_params,
        "mdp_hashes": m.mdp_hashes,
        "stages": m.stages,
        "build_stats": m.build_stats,
        "overall_status": m.overall_status,
        "host": m.host,
    }
    tmp_dir = os.path.dirname(os.path.abspath(path))
    fd, tmp_path = tempfile.mkstemp(dir=tmp_dir, suffix=".json.tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=2)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def read_manifest(path: str) -> Manifest:
    with open(path) as fh:
        data = json.load(fh)
    return Manifest(**data)


def hash_file(path: str) -> str:
    """Return 'sha256:<hex>' for the file at *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return f"sha256:{h.hexdigest()}"


def detect_gmx_version(gmx_exe: str) -> Optional[str]:
    """Return GROMACS version string or None if the binary is absent."""
    import shutil
    if not shutil.which(gmx_exe):
        return None
    try:
        result = subprocess.run(
            [gmx_exe, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("GROMACS version"):
                parts = stripped.split()
                return parts[-1] if len(parts) >= 3 else stripped
        # Fallback: return first non-empty line capped at 80 chars
        for line in result.stdout.splitlines():
            if line.strip():
                return line.strip()[:80]
    except Exception:
        pass
    return None


def detect_insane_version() -> str:
    """Return insane package version or 'unknown'."""
    try:
        import importlib.metadata
        return importlib.metadata.version("insane")
    except Exception:
        pass
    return "unknown"


def host_info() -> dict:
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
    }
