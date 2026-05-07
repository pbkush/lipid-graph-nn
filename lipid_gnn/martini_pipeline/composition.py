"""Martini 3 membrane composition: validation, canonical naming, and leaflet counts."""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

_TOL_FRACTION = 1e-6
_TOL_PERCENT = 0.01  # tolerance on the 0-100 scale for integer-percentage check
_TOKEN_RE = re.compile(r"^([A-Z]+)(\d{1,3})$")


def validate_fractions(fractions: Mapping[str, float], tol: float = _TOL_FRACTION) -> None:
    """Raise ValueError if *fractions* is not a valid composition spec.

    Checks: non-empty, all keys non-empty strings, all fractions in (0, 1],
    fractions sum to 1.0 within *tol*.
    """
    if not fractions:
        raise ValueError("Composition must contain at least one lipid.")
    for name, frac in fractions.items():
        if not isinstance(name, str) or not name:
            raise ValueError(f"Lipid name must be a non-empty string, got {name!r}.")
        if frac <= 0.0 or frac > 1.0 + tol:
            raise ValueError(
                f"Mole fraction for {name!r} must be in (0, 1], got {frac}."
            )
    total = sum(fractions.values())
    if abs(total - 1.0) > tol:
        raise ValueError(
            f"Mole fractions must sum to 1.0 (got {total:.8g})."
        )


def _canonical_order(fractions: Mapping[str, float]) -> tuple[str, ...]:
    """Return lipid names sorted by descending fraction, alphabetical tiebreak."""
    return tuple(sorted(fractions, key=lambda k: (-fractions[k], k)))


def _to_integer_percentages(fractions: Mapping[str, float]) -> dict[str, int]:
    """Convert fractions to integer percentages, raising if any fraction is non-representable."""
    result: dict[str, int] = {}
    for name, frac in fractions.items():
        raw_pct = frac * 100.0
        rounded = round(raw_pct)
        if abs(raw_pct - rounded) > _TOL_PERCENT:
            raise ValueError(
                f"Mole fraction {frac!r} for {name!r} does not correspond to an integer "
                f"percentage (got {raw_pct:.6g}%). "
                "Use a fraction representable as N/100 (e.g. 0.3, 0.25, 0.1)."
            )
        result[name] = rounded
    total = sum(result.values())
    if total != 100:
        raise ValueError(
            f"Integer percentages sum to {total}, not 100. "
            "Choose fractions representable as N/100 that sum exactly to 100."
        )
    return result


@dataclass(frozen=True)
class Composition:
    """Immutable membrane composition: lipid name -> mol fraction.

    Fractions must sum to 1.0 and each must be representable as an integer
    percentage (e.g. 0.30, 0.25, 0.10 — not 0.333).
    """

    fractions: Mapping[str, float]

    def __post_init__(self) -> None:
        validate_fractions(self.fractions)
        _to_integer_percentages(self.fractions)
        # Store as immutable mapping so callers cannot mutate the internal state.
        object.__setattr__(self, "fractions", MappingProxyType(dict(self.fractions)))

    @property
    def name(self) -> str:
        """Canonical name: descending mol fraction, alphabetical tiebreak.

        Examples: 'POPC100', 'DOPC70_POPC30', 'DIPC50_POPC50'.
        """
        order = _canonical_order(self.fractions)
        pcts = _to_integer_percentages(self.fractions)
        return "_".join(f"{lipid}{pcts[lipid]}" for lipid in order)

    @property
    def lipid_types(self) -> tuple[str, ...]:
        """Lipid names in canonical order (descending fraction, alpha tiebreak)."""
        return _canonical_order(self.fractions)

    def __hash__(self) -> int:
        # MappingProxyType is not hashable, so we hash on sorted items.
        return hash(tuple(sorted(self.fractions.items())))


def parse_name(name: str) -> Composition:
    """Parse a composition name like 'DOPC70_POPC30' into a Composition.

    The input does not need to be in canonical order; the returned Composition's
    .name will always be canonical. Raises ValueError on any malformed input.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"Cannot parse empty name {name!r}.")
    tokens = name.split("_")
    fractions: dict[str, float] = {}
    for token in tokens:
        m = _TOKEN_RE.match(token)
        if m is None:
            raise ValueError(
                f"Invalid token {token!r} in composition name {name!r}. "
                "Each segment must be an uppercase lipid name followed by an integer "
                "percentage, e.g. 'POPC30'."
            )
        lipid, pct_str = m.group(1), m.group(2)
        if lipid in fractions:
            raise ValueError(f"Duplicate lipid {lipid!r} in name {name!r}.")
        fractions[lipid] = int(pct_str) / 100.0
    return Composition(fractions)


def counts_per_leaflet(comp: Composition, n_lipids_per_leaflet: int) -> dict[str, int]:
    """Compute integer lipid counts per leaflet for a given total leaflet size.

    Raises ValueError if any lipid's integer percentage does not divide
    *n_lipids_per_leaflet* evenly (i.e. pct * n % 100 != 0).
    """
    pcts = _to_integer_percentages(comp.fractions)
    result: dict[str, int] = {}
    for lipid, pct in pcts.items():
        numerator = pct * n_lipids_per_leaflet
        if numerator % 100 != 0:
            raise ValueError(
                f"Percentage {pct}% for {lipid!r} with n={n_lipids_per_leaflet} "
                f"gives non-integer count {numerator / 100:.6g}. "
                "Choose n_lipids_per_leaflet such that every percentage times n is "
                "divisible by 100 (n=100 always works)."
            )
        result[lipid] = numerator // 100
    return result
