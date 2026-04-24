#!/usr/bin/env python3
"""Print a single resolved value from config.yaml. Used by bash scripts.

Usage:
    python scripts/python/print_config_var.py dataset.spatial_cutoff
    python scripts/python/print_config_var.py vocab.active_properties
    python scripts/python/print_config_var.py paths.chunks_dir

Dotted key walks the Config dataclass tree. Lists are printed space-separated
so they can be consumed directly as bash arguments.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repo root is on sys.path when invoked by bash from arbitrary cwd.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lipid_gnn.config import CONFIG  # noqa: E402


def resolve(dotted: str):
    obj = CONFIG
    for part in dotted.split("."):
        obj = getattr(obj, part)
    return obj


def format_value(v) -> str:
    if isinstance(v, (list, tuple)):
        return " ".join(str(x) for x in v)
    return str(v)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: print_config_var.py <dotted.key>", file=sys.stderr)
        return 2
    try:
        value = resolve(sys.argv[1])
    except AttributeError as e:
        print(f"unknown config key {sys.argv[1]!r}: {e}", file=sys.stderr)
        return 1
    print(format_value(value))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
