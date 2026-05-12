"""Martini 3 lipid registry: data, validation, and on-disk resource checks."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

_KNOWN_FAMILIES: frozenset[str] = frozenset({"phospholipid", "sterol"})

_ITP_SECTION_RE = re.compile(r"^\s*\[\s*moleculetype\s*\]\s*$", re.IGNORECASE)
_NAME_RE = re.compile(r"^[A-Z][A-Z0-9]*$")


@dataclass(frozen=True)
class LipidEntry:
    """Metadata for a single Martini 3 lipid."""

    name: str
    resname: str
    itp_file: str
    moleculetype: str
    beads: tuple[str, ...]
    family: str
    insane_keyword: str


@dataclass(frozen=True)
class ResourceCheck:
    """Result of an on-disk resource verification."""

    lipid: str
    itp_present: bool | None
    moleculetype_declared: bool | None
    beads_match_node_mapping: bool | None
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        if self.errors:
            return False
        for field in (self.itp_present, self.moleculetype_declared, self.beads_match_node_mapping):
            if field is False:
                return False
        return True


def validate_lipid(entry: LipidEntry) -> None:
    """Raise ValueError if *entry* is malformed."""
    if not entry.name or not _NAME_RE.match(entry.name):
        raise ValueError(
            f"Lipid name must be non-empty uppercase alphanumeric starting with a letter, "
            f"got {entry.name!r}."
        )
    if not entry.resname:
        raise ValueError(f"resname must be non-empty for {entry.name!r}.")
    if not entry.itp_file:
        raise ValueError(f"itp_file must be non-empty for {entry.name!r}.")
    if not entry.moleculetype:
        raise ValueError(f"moleculetype must be non-empty for {entry.name!r}.")
    if not entry.beads:
        raise ValueError(f"beads must be non-empty for {entry.name!r}.")
    if len(entry.beads) != len(set(entry.beads)):
        dupes = [b for b in entry.beads if entry.beads.count(b) > 1]
        raise ValueError(f"Duplicate bead names {dupes!r} for {entry.name!r}.")
    if entry.family not in _KNOWN_FAMILIES:
        raise ValueError(
            f"Unknown family {entry.family!r} for {entry.name!r}. "
            f"Known families: {sorted(_KNOWN_FAMILIES)}. "
            "Add the new family to _KNOWN_FAMILIES to register it."
        )
    if not entry.insane_keyword:
        raise ValueError(f"insane_keyword must be non-empty for {entry.name!r}.")


def _parse_moleculetypes(itp_text: str) -> set[str]:
    """Return the set of moleculetype names declared in *itp_text*."""
    names: list[str] = []
    in_section = False
    for line in itp_text.splitlines():
        stripped = line.strip()
        if _ITP_SECTION_RE.match(stripped):
            in_section = True
            continue
        if in_section and stripped and not stripped.startswith(";"):
            names.append(stripped.split()[0])
            in_section = False
    return set(names)


def check_resources(
    entry: LipidEntry,
    *,
    itp_dir: str | os.PathLike | None = None,
    node_mapping_path: str | os.PathLike | None = None,
) -> ResourceCheck:
    """Verify on-disk resources for *entry*. Skips checks for which the path is None."""
    errors: list[str] = []
    itp_present: bool | None = None
    moleculetype_declared: bool | None = None
    beads_match: bool | None = None

    if itp_dir is not None:
        itp_path = os.path.join(itp_dir, entry.itp_file)
        if not os.path.isfile(itp_path):
            itp_present = False
            errors.append(f"ITP file not found: {itp_path}")
        else:
            itp_present = True
            itp_text = open(itp_path).read()
            declared = _parse_moleculetypes(itp_text)
            if entry.moleculetype in declared:
                moleculetype_declared = True
            else:
                moleculetype_declared = False
                errors.append(
                    f"Moleculetype {entry.moleculetype!r} not declared in {itp_path}. "
                    f"Found: {sorted(declared)}"
                )

    if node_mapping_path is not None:
        nm_path = str(node_mapping_path)
        if not os.path.isfile(nm_path):
            beads_match = False
            errors.append(f"Node mapping file not found: {nm_path}")
        else:
            with open(nm_path) as fh:
                mapping: dict = json.load(fh)
            if entry.name not in mapping:
                beads_match = False
                errors.append(
                    f"Lipid {entry.name!r} not found in node mapping {nm_path}."
                )
            else:
                expected = tuple(mapping[entry.name].keys())
                if entry.beads == expected:
                    beads_match = True
                else:
                    beads_match = False
                    errors.append(
                        f"Bead mismatch for {entry.name!r}. "
                        f"Registry: {entry.beads}. Node mapping: {expected}."
                    )

    return ResourceCheck(
        lipid=entry.name,
        itp_present=itp_present,
        moleculetype_declared=moleculetype_declared,
        beads_match_node_mapping=beads_match,
        errors=tuple(errors),
    )


def register_lipid(
    registry: Mapping[str, LipidEntry], entry: LipidEntry
) -> dict[str, LipidEntry]:
    """Return a new registry with *entry* added. Raises on duplicate name or invalid entry."""
    validate_lipid(entry)
    if entry.name in registry:
        raise ValueError(
            f"Lipid {entry.name!r} is already registered. "
            "Use a different name or update the existing entry."
        )
    return dict(registry) | {entry.name: entry}


def get_lipid(name: str) -> LipidEntry:
    """Return the entry for *name* from the default registry. Raises KeyError if absent."""
    if name not in LIPID_REGISTRY:
        raise KeyError(
            f"Lipid {name!r} not in registry. "
            f"Known lipids: {sorted(LIPID_REGISTRY)}."
        )
    return LIPID_REGISTRY[name]


def lipid_names() -> tuple[str, ...]:
    """Return all registered lipid names in alphabetical order."""
    return tuple(sorted(LIPID_REGISTRY))


_PHOSPHOLIPID_ITP = "martini_v3.0.0_phospholipids_v1.itp"
_STEROL_ITP = "martini_v3.0_sterols_v1.0.itp"

_REGISTRY_DATA: dict[str, LipidEntry] = {
    e.name: e for e in [
        # Composition token "DIPC" maps to v2 moleculetype "DLPC" — the M3-Lipid-Parameters
        # v2 set renamed di-C18:2 PC from legacy "DIPC" to "DLPC".  We keep the user-facing
        # token as "DIPC" (matches legacy 70-system directory naming and composition.py
        # parsing) but pass "DLPC" to insane and the v2 PC ITP.
        LipidEntry(
            name="DIPC", resname="DLPC", itp_file="martini_v3.0.0_phospholipids_PC_v2.itp",
            moleculetype="DLPC",
            beads=("NC3", "PO4", "GL1", "GL2", "C1A", "D2A", "D3A", "C4A", "C1B", "D2B", "D3B", "C4B"),
            family="phospholipid", insane_keyword="DLPC",
        ),
        LipidEntry(
            name="DOPC", resname="DOPC", itp_file=_PHOSPHOLIPID_ITP, moleculetype="DOPC",
            beads=("NC3", "PO4", "GL1", "GL2", "C1A", "D2A", "C3A", "C4A", "C1B", "D2B", "C3B", "C4B"),
            family="phospholipid", insane_keyword="DOPC",
        ),
        LipidEntry(
            name="DPPC", resname="DPPC", itp_file=_PHOSPHOLIPID_ITP, moleculetype="DPPC",
            beads=("NC3", "PO4", "GL1", "GL2", "C1A", "C2A", "C3A", "C4A", "C1B", "C2B", "C3B", "C4B"),
            family="phospholipid", insane_keyword="DPPC",
        ),
        LipidEntry(
            name="POPC", resname="POPC", itp_file=_PHOSPHOLIPID_ITP, moleculetype="POPC",
            beads=("NC3", "PO4", "GL1", "GL2", "C1A", "D2A", "C3A", "C4A", "C1B", "C2B", "C3B", "C4B"),
            family="phospholipid", insane_keyword="POPC",
        ),
        LipidEntry(
            name="DOPE", resname="DOPE", itp_file=_PHOSPHOLIPID_ITP, moleculetype="DOPE",
            beads=("NH3", "PO4", "GL1", "GL2", "C1A", "D2A", "C3A", "C4A", "C1B", "D2B", "C3B", "C4B"),
            family="phospholipid", insane_keyword="DOPE",
        ),
        LipidEntry(
            name="DPPE", resname="DPPE", itp_file=_PHOSPHOLIPID_ITP, moleculetype="DPPE",
            beads=("NH3", "PO4", "GL1", "GL2", "C1A", "C2A", "C3A", "C4A", "C1B", "C2B", "C3B", "C4B"),
            family="phospholipid", insane_keyword="DPPE",
        ),
        LipidEntry(
            name="POPE", resname="POPE", itp_file=_PHOSPHOLIPID_ITP, moleculetype="POPE",
            beads=("NH3", "PO4", "GL1", "GL2", "C1A", "D2A", "C3A", "C4A", "C1B", "C2B", "C3B", "C4B"),
            family="phospholipid", insane_keyword="POPE",
        ),
        LipidEntry(
            name="DOPS", resname="DOPS", itp_file=_PHOSPHOLIPID_ITP, moleculetype="DOPS",
            beads=("CNO", "PO4", "GL1", "GL2", "C1A", "D2A", "C3A", "C4A", "C1B", "D2B", "C3B", "C4B"),
            family="phospholipid", insane_keyword="DOPS",
        ),
        LipidEntry(
            name="POPS", resname="POPS", itp_file=_PHOSPHOLIPID_ITP, moleculetype="POPS",
            beads=("CNO", "PO4", "GL1", "GL2", "C1A", "D2A", "C3A", "C4A", "C1B", "C2B", "C3B", "C4B"),
            family="phospholipid", insane_keyword="POPS",
        ),
        LipidEntry(
            name="CHOL", resname="CHOL", itp_file=_STEROL_ITP, moleculetype="CHOL",
            beads=("ROH", "R1", "R2", "R3", "R4", "R5", "R6", "C1", "C2"),
            family="sterol", insane_keyword="CHOL",
        ),
    ]
}

LIPID_REGISTRY: Mapping[str, LipidEntry] = MappingProxyType(_REGISTRY_DATA)
