"""Pickle-based load/save helpers.

Replaces the relevant surface of ``lipid_gnn.functions_emil.functions`` for
the in-project call sites (``dataset.py``, ``prepare_colab_subset.py``,
``smoke_test_sweep.py``). The historical helper supported glob expansion and
imported ``cv2`` / ``nglview`` at module top; neither is used here.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def pkl_load(path: str | Path, verbose: bool = True) -> Any:
    """Load a single pickled object from disk.

    ``verbose`` is accepted for backwards-compat with the legacy helper; it
    only controls a "file not found" message.
    """
    p = Path(path)
    if not p.exists():
        if verbose:
            print(f"Cannot find file: {p}")
        return None
    with p.open("rb") as fh:
        return pickle.load(fh)


def pkl_save(path: str | Path, obj: Any) -> None:
    """Write ``obj`` to ``path`` using ``pickle.HIGHEST_PROTOCOL``."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
