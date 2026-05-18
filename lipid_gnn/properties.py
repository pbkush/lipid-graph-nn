"""Bilayer property calculations for Martini 3 membrane systems.

Replaces :mod:`lipid_gnn.functions_emil.calculate_properties`. Two modes:

- ``legacy=False`` (default) — bug-fixed implementations of every property.
- ``legacy=True`` — reproduction of the historical pipeline, including the
  bugs documented in ``docs/functions_emil_cleanup_plan.md`` §2. Used for
  re-deriving the original training labels under controlled (seeded) RNG.

Output schema: :func:`compute_all` returns ``(mean_dict, raw_dict)`` whose
keys match the historical 8-property schema, with one rename:

- ``thickness_inhomogeneity`` is the corrected name for what was previously
  called ``compressibility`` (it is *not* an area-compressibility modulus).
  For backwards compat the old key ``compressibility`` is emitted as an
  alias of the same value.

Per-property functions exposed individually so they can be tested in
isolation. The orchestrator :func:`compute_all` reuses cached intermediate
fields (height-interpolation grid, leaflet split) across properties.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import mdtraj as md
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
from scipy.spatial import Voronoi
import shapely.geometry


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

ALL_PROPERTIES: tuple[str, ...] = (
    "lipid_packing",
    "thickness",
    "thickness_std",
    "thickness_inhomogeneity",
    "bending_modulus",
    "persistence",
    "diffusivity",
    "variation",
)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

@dataclass
class GridSpec:
    """Lateral grid used for the height-field interpolation."""

    box_xy: tuple[float, float]
    step: float = 0.1
    margin: float = 1.5

    def build(self) -> tuple[np.ndarray, np.ndarray]:
        x = np.arange(self.margin, self.box_xy[0] - self.margin + 1e-9, self.step)
        y = np.arange(self.margin, self.box_xy[1] - self.margin + 1e-9, self.step)
        # indexing="ij" → X[i,j] = x[i], Y[i,j] = y[j], shape (nx, ny).
        # Keeps physical axis identity aligned with array axis for downstream
        # FFT code (see compute_bending_modulus_from_field).
        return np.meshgrid(x, y, indexing="ij")


def _split_leaflets(z: np.ndarray) -> float:
    """Return the z-coordinate that splits a 1D sorted distribution into two
    populations (largest gap in sorted z)."""
    z_sorted = np.sort(z)
    if len(z_sorted) < 2:
        return float(z_sorted[0]) if len(z_sorted) else 0.0
    gap_idx = int(np.argmax(np.diff(z_sorted)))
    return float((z_sorted[gap_idx] + z_sorted[gap_idx + 1]) / 2)


def _leaflet_assignment(head_z: np.ndarray, cutoff: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (lower_mask, upper_mask) given head-bead z coordinates and a
    z-cutoff."""
    lower = head_z < cutoff
    upper = head_z > cutoff
    return lower, upper


# ---------------------------------------------------------------------------
# 1) lipid_packing
# ---------------------------------------------------------------------------

def compute_lipid_packing(traj: md.Trajectory, n_lipids: int | None = None
                          ) -> tuple[float, np.ndarray]:
    """Lipid number density (lipids / nm²), averaged over frames.

    Counts unique head-bead residues if ``n_lipids`` is not supplied.
    Box-area is taken per-frame, so the function is correct under NPT.
    """
    if n_lipids is None:
        head_idx = traj.topology.select("name PO4 ROH")
        n_lipids = int(len(head_idx))
    area = np.prod(traj.unitcell_lengths[:, :2], axis=1)
    series = n_lipids / area
    return float(np.mean(series)), series


# ---------------------------------------------------------------------------
# 2) thickness / thickness_std / thickness_inhomogeneity / midplane field
# ---------------------------------------------------------------------------

def _height_fields(traj: md.Trajectory, leaflet_head_indices: np.ndarray,
                   grid: GridSpec
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                              np.ndarray]:
    """Interpolate per-frame upper/lower leaflet height onto a fixed lateral
    grid. Frames whose interpolation yields any NaN are skipped.

    Returns ``(xy_thickness, xy_midplane, X, Y, frame_mask)`` where
    ``frame_mask[i]`` is True for retained frames.
    """
    X, Y = grid.build()
    xy_thickness = []
    xy_midplane = []
    frame_mask = np.zeros(len(traj), dtype=bool)
    for f in range(len(traj)):
        z = traj.xyz[f, leaflet_head_indices, 2]
        cutoff = _split_leaflets(z)
        xyz = traj.xyz[f, leaflet_head_indices]
        lower = xyz[xyz[:, 2] < cutoff]
        upper = xyz[xyz[:, 2] > cutoff]
        if len(lower) < 3 or len(upper) < 3:
            continue
        lo = LinearNDInterpolator(lower[:, :2], lower[:, 2])(X, Y)
        hi = LinearNDInterpolator(upper[:, :2], upper[:, 2])(X, Y)
        if np.isnan(lo).any() or np.isnan(hi).any():
            continue
        xy_thickness.append(hi - lo)
        xy_midplane.append((hi + lo) / 2.0)
        frame_mask[f] = True
    return (np.asarray(xy_thickness), np.asarray(xy_midplane), X, Y, frame_mask)


def _legacy_height_fields(traj: md.Trajectory, po4_indices: np.ndarray,
                          grid: GridSpec
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                     np.ndarray]:
    """Legacy variant — leaflet split uses PO4 only (bug #6), and the
    ``xy_membrane`` half-thickness is returned where the rewrite returns the
    midplane (bug #4)."""
    X, Y = grid.build()
    xy_thickness = []
    xy_membrane_half = []
    frame_mask = np.zeros(len(traj), dtype=bool)
    for f in range(len(traj)):
        po4_xyz = traj.xyz[f, po4_indices]
        z = po4_xyz[:, 2]
        cutoff = _split_leaflets(z)
        lower = po4_xyz[po4_xyz[:, 2] < cutoff]
        upper = po4_xyz[po4_xyz[:, 2] > cutoff]
        if len(lower) < 3 or len(upper) < 3:
            continue
        lo = LinearNDInterpolator(lower[:, :2], lower[:, 2])(X, Y)
        hi = LinearNDInterpolator(upper[:, :2], upper[:, 2])(X, Y)
        if np.isnan(lo).any() or np.isnan(hi).any():
            continue
        xy_thickness.append(hi - lo)
        xy_membrane_half.append((hi - lo) / 2.0)
        frame_mask[f] = True
    return (np.asarray(xy_thickness), np.asarray(xy_membrane_half), X, Y, frame_mask)


def thickness_summary(xy_thickness: np.ndarray, frame_mask: np.ndarray | None = None
                      ) -> tuple[float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    """Reduce an ``(n_kept, nx, ny)`` thickness field to scalar + per-frame
    series.

    Parameters
    ----------
    xy_thickness
        Filtered thickness field (only frames that passed interpolation).
    frame_mask
        Optional boolean array of length ``n_frames_total`` marking which
        frames were retained. When supplied, returned series are length
        ``n_frames_total`` with ``NaN`` at dropped-frame positions, so that
        per-property series from different properties can be co-indexed.

    Returns
    ``(thickness_A, thickness_std_A, thickness_inhomogeneity,
    thickness_series, thickness_std_series, inhomogeneity_series)``, all in
    Ångström / Å² where appropriate. Mean values are taken over kept frames
    only (NaN-free).
    """
    if xy_thickness.size == 0:
        nan = float("nan")
        if frame_mask is not None:
            n = len(frame_mask)
            return (nan, nan, nan,
                    np.full(n, nan), np.full(n, nan), np.full(n, nan))
        return nan, nan, nan, np.array([]), np.array([]), np.array([])
    flat = xy_thickness.reshape(xy_thickness.shape[0], -1)
    kept_thickness = flat.mean(axis=1) * 10.0     # nm → Å
    kept_std = flat.std(axis=1) * 10.0
    kept_inhomog = kept_std ** 2                  # Å² (already squared Å std)

    if frame_mask is not None:
        n = len(frame_mask)
        thickness_series = np.full(n, np.nan)
        thickness_std_series = np.full(n, np.nan)
        inhomogeneity_series = np.full(n, np.nan)
        thickness_series[frame_mask] = kept_thickness
        thickness_std_series[frame_mask] = kept_std
        inhomogeneity_series[frame_mask] = kept_inhomog
    else:
        thickness_series = kept_thickness
        thickness_std_series = kept_std
        inhomogeneity_series = kept_inhomog

    return (
        float(np.nanmean(thickness_series)),
        float(np.nanmean(thickness_std_series)),
        float(np.nanmean(inhomogeneity_series)),
        thickness_series,
        thickness_std_series,
        inhomogeneity_series,
    )


# ---------------------------------------------------------------------------
# 3) bending modulus from a height field via the Helfrich spectrum
# ---------------------------------------------------------------------------

def _undulation_model(q: np.ndarray, kappa: float, kBT: float = 1.0) -> np.ndarray:
    return kBT / (kappa * q**4)


def compute_bending_modulus_from_field(Z: np.ndarray, X: np.ndarray, Y: np.ndarray,
                                       kBT: float = 1.0,
                                       q_min: float = 0.1,
                                       n_bins: int = 50
                                       ) -> tuple[float, dict]:
    """Fit κ from the Helfrich undulation spectrum on a midplane height field.

    Continuous Helfrich: ``⟨|h(q)|²⟩ = kBT / (κ q⁴ A)`` where
    ``h(q) = (1/A) ∫ h(r) e^(-i q·r) d²r``. For a discrete grid of N samples
    on area A with spacings (Δx, Δy), using ``np.fft.fft2(..., norm="ortho")``
    one has ``⟨|H_k|²⟩ = kBT / (κ q⁴ Δx Δy)``. The raw fit therefore returns
    ``κ_fit = κ · Δx Δy``; this function divides by ``Δx Δy`` so the returned
    value is the **physical bending modulus in units of kBT**.

    ``Z`` is expected with shape ``(n_frames, nx, ny)`` and ``X, Y`` built
    with ``indexing="ij"`` so that ``X[i,j] = x[i], Y[i,j] = y[j]``.

    Returns ``(kappa, diag)``. The caller must pass the midplane field
    ``(upper + lower) / 2`` — *not* the half-thickness (cleanup-plan §2
    bug #4).
    """
    if Z.ndim != 3 or Z.shape[0] < 2:
        return float("nan"), {"reason": "need >= 2 frames"}
    n_frames, nx, ny = Z.shape
    # Actual grid spacing (not Lx/nx — that's an off-by-one for half-open
    # arange grids and silently swaps axes for indexing="xy" meshgrids).
    step_x = float(X[1, 0] - X[0, 0]) if nx > 1 else 1.0
    step_y = float(Y[0, 1] - Y[0, 0]) if ny > 1 else 1.0
    Zf = Z - Z.mean(axis=0)
    fft_frames = np.fft.fft2(Zf, axes=(1, 2), norm="ortho")
    ps_avg = np.mean(np.abs(fft_frames) ** 2, axis=0)
    qx = np.fft.fftfreq(nx, d=step_x) * 2 * np.pi
    qy = np.fft.fftfreq(ny, d=step_y) * 2 * np.pi
    QX, QY = np.meshgrid(qx, qy, indexing="ij")
    q_flat = np.sqrt(QX**2 + QY**2).flatten()
    ps_flat = ps_avg.flatten()

    q_edges = np.linspace(0.0, q_flat.max(), n_bins + 1)
    q_centers = 0.5 * (q_edges[:-1] + q_edges[1:])
    ps_binned = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    bin_idx = np.searchsorted(q_edges, q_flat) - 1
    for b, p in zip(bin_idx, ps_flat):
        if 0 <= b < n_bins:
            ps_binned[b] += p
            counts[b] += 1
    ps_binned /= np.maximum(counts, 1)

    valid = (q_centers > q_min) & (counts > 5) & np.isfinite(ps_binned) & (ps_binned > 0)
    if valid.sum() < 3:
        return float("nan"), {"reason": "not enough q bins", "q": q_centers, "ps": ps_binned}
    try:
        popt, _ = curve_fit(
            lambda q, k: _undulation_model(q, k, kBT=kBT),
            q_centers[valid], ps_binned[valid], p0=[1.0],
        )
    except Exception as exc:  # pragma: no cover — handled by NaN return
        return float("nan"), {"reason": f"curve_fit failed: {exc}"}
    kappa_fit = float(popt[0])
    kappa_phys = kappa_fit / (step_x * step_y)
    return kappa_phys, {
        "q": q_centers[valid],
        "ps": ps_binned[valid],
        "kappa_raw_fit": kappa_fit,
        "step_x": step_x,
        "step_y": step_y,
    }


# ---------------------------------------------------------------------------
# 4) Voronoi cell-area coefficient of variation
# ---------------------------------------------------------------------------

def _voronoi_cv(points: np.ndarray, box: tuple[float, float, float, float],
                periodic: bool) -> float:
    xmin, xmax, ymin, ymax = box
    if periodic:
        Lx, Ly = xmax - xmin, ymax - ymin
        # Replicate across 8 neighbours, build Voronoi on 9× set, keep central
        shifts = [(dx * Lx, dy * Ly) for dx in (-1, 0, 1) for dy in (-1, 0, 1)]
        repl = np.vstack([points + np.array([sx, sy]) for sx, sy in shifts])
        # Central image lives at shift index (0, 0). With the shift order
        # (-1,-1), (-1,0), (-1,1), (0,-1), (0,0), … the central block is
        # index 4 of the 9.
        central_slice = slice(4 * len(points), 5 * len(points))
        vor = Voronoi(repl)
        areas = []
        for pidx in range(central_slice.start, central_slice.stop):
            region_idx = vor.point_region[pidx]
            verts = vor.regions[region_idx]
            if -1 in verts or len(verts) == 0:
                continue
            # No bbox clipping: replicated neighbours already bound the cell.
            poly = shapely.geometry.Polygon(vor.vertices[verts])
            areas.append(poly.area)
    else:
        vor = Voronoi(points)
        bbox = shapely.geometry.box(xmin, ymin, xmax, ymax)
        areas = []
        for pidx, region_idx in enumerate(vor.point_region):
            verts = vor.regions[region_idx]
            if -1 in verts or len(verts) == 0:
                continue
            poly = shapely.geometry.Polygon(vor.vertices[verts])
            clipped = poly.intersection(bbox)
            if not clipped.is_empty:
                areas.append(clipped.area)
    if not areas:
        return float("nan")
    areas_np = np.asarray(areas)
    m = areas_np.mean()
    return float(areas_np.std() / m) if m > 0 else float("nan")


def compute_variation(traj: md.Trajectory, *, legacy: bool = False
                      ) -> tuple[float, np.ndarray]:
    """Mean coefficient of variation of Voronoi cell areas, averaged over
    the two leaflets and over frames.

    ``legacy=False`` uses a periodic Voronoi (replicate points across 8
    neighbouring boxes, take cells in the central image). ``legacy=True``
    reproduces the bug #10 behaviour (raw points clipped to the box).
    """
    po4_indices = traj.topology.select("name PO4")
    per_frame = []
    for f in range(len(traj)):
        po4_xyz = traj.xyz[f, po4_indices]
        z = po4_xyz[:, 2]
        cutoff = _split_leaflets(z)
        lower = po4_xyz[po4_xyz[:, 2] < cutoff][:, :2]
        upper = po4_xyz[po4_xyz[:, 2] > cutoff][:, :2]
        Lx, Ly = traj.unitcell_lengths[f, 0], traj.unitcell_lengths[f, 1]
        bbox = (0.0, float(Lx), 0.0, float(Ly))
        per_frame.append([
            _voronoi_cv(lower, bbox, periodic=not legacy),
            _voronoi_cv(upper, bbox, periodic=not legacy),
        ])
    with warnings.catch_warnings():
        # All-NaN slice → NaN propagation is the intended behaviour here
        warnings.filterwarnings("ignore", message="Mean of empty slice")
        series = np.nanmean(per_frame, axis=1)
        return float(np.nanmean(series)) if series.size else float("nan"), series


# ---------------------------------------------------------------------------
# 5) persistence (probability a contact survives `lag` frames)
# ---------------------------------------------------------------------------

def _select_leaflet_residues(frame_xyz: np.ndarray, head_indices: np.ndarray,
                             cutoff: float, upper: bool) -> np.ndarray:
    if upper:
        return np.where(frame_xyz[head_indices, 2] > cutoff)[0]
    return np.where(frame_xyz[head_indices, 2] < cutoff)[0]


def compute_persistence(traj: md.Trajectory, *, lag: int = 50,
                        contact_cutoff: float = 0.7,
                        probe_size: int = 10,
                        rng: np.random.Generator,
                        legacy: bool = False
                        ) -> tuple[float, np.ndarray]:
    """Fraction of randomly-sampled lipid–lipid contacts that survive ``lag``
    frames.

    Bug fixes vs ``legacy=True``:

    - **#1** "upper" branch correctly uses ``z > cutoff``; the legacy path
      always sampled from the lower leaflet.
    - **#2** contact residues are looked up via ``other_indices[j]`` (global
      atom index), not the positional index that the legacy code passed.
    - **#3** the survival check at frame ``+lag`` recomputes contacts
      directly against the chosen target residue, instead of intersecting
      bead-index sets with positional indices.
    """
    head_indices = traj.topology.select("name PO4 ROH")
    n_lipids = int(len(head_indices))
    residues = [
        np.array([atom.index for atom in r.atoms])
        for r in list(traj.topology.residues)[:n_lipids]
    ]
    beads = list(traj.topology.atoms)
    per_frame = []
    nf = len(traj)
    for f in range(nf - lag):
        frames = traj[[f, f + lag]]
        head_z0 = frames.xyz[0, head_indices, 2]
        cutoff = _split_leaflets(head_z0)
        samples = []
        for _ in range(probe_size):
            if legacy:
                # bug #1: both arms identical (always lower)
                resids = np.where(frames.xyz[0, head_indices, 2] < cutoff)[0]
            else:
                upper = bool(rng.random() > 0.5)
                resids = _select_leaflet_residues(frames.xyz[0], head_indices, cutoff, upper)
            if len(resids) < 2:
                continue
            i = int(rng.choice(resids))
            lipid_atoms = residues[i]
            other_resids = [r for r in resids if r != i]
            if not other_resids:
                continue
            other_atoms = np.concatenate([residues[j] for j in other_resids])
            pairs = [[h, k] for k in other_atoms for h in lipid_atoms]
            d = np.min(
                md.compute_distances(frames, pairs).reshape(2, -1, len(lipid_atoms)),
                axis=2,
            )  # shape (2, n_other_atoms)
            in_contact_t0 = d[0] < contact_cutoff
            if not in_contact_t0.any():
                continue
            if legacy:
                # bug #2: position-into-other_indices treated as global bead index
                contact_positions = np.where(in_contact_t0)[0]
                contacts_resids = np.unique([beads[j].residue.index for j in contact_positions])
                if len(contacts_resids) == 0:
                    continue
                target_res = int(rng.choice(contacts_resids))
                # bug #3: set-of-bead-indices vs positions-into-other_indices
                set_target_atoms = set(int(a) for a in residues[target_res])
                positions_t1 = set(int(p) for p in np.where(d[1] < contact_cutoff)[0])
                samples.append(1.0 if set_target_atoms.intersection(positions_t1) else 0.0)
            else:
                contact_positions = np.where(in_contact_t0)[0]
                contact_atoms = other_atoms[contact_positions]
                contacts_resids = np.unique([beads[int(a)].residue.index for a in contact_atoms])
                target_res = int(rng.choice(contacts_resids))
                target_atoms = residues[target_res]
                target_atoms_set = set(int(a) for a in target_atoms)
                # boolean vector: is each other_atom in the target residue?
                in_target = np.fromiter(
                    (int(a) in target_atoms_set for a in other_atoms), dtype=bool,
                    count=len(other_atoms),
                )
                samples.append(float(((d[1] < contact_cutoff) & in_target).any()))
        per_frame.append(float(np.mean(samples)) if samples else float("nan"))
    series = np.asarray(per_frame, dtype=float)
    return float(np.nanmean(series)) if series.size else float("nan"), series


# ---------------------------------------------------------------------------
# 6) diffusivity
# ---------------------------------------------------------------------------

def _pbc_displacement_2d(r0: np.ndarray, r1: np.ndarray,
                         box: np.ndarray) -> np.ndarray:
    """Minimum-image XY displacement r1 − r0 under orthorhombic box.

    ``box`` is the *current* box (the displacement is measured against the
    nearest periodic image, so passing the frame-1 box is the standard
    convention)."""
    dr = r1 - r0
    dr -= np.round(dr / box) * box
    return dr


def compute_diffusivity(traj: md.Trajectory, *, lag: int = 10,
                        probe_size: int = 10,
                        rng: np.random.Generator,
                        legacy: bool = False
                        ) -> tuple[float, np.ndarray]:
    """Lateral mean squared displacement per lag (Å² / lag).

    ``legacy=False`` measures single-lipid lab-frame MSD with explicit PBC
    minimum-image unwrap (bug #9 fix) and samples both leaflets equally
    (bug #1 fix). ``legacy=True`` reproduces the historical pair-relative
    pivot-and-rewrap formulation AND its leaflet bias (always samples the
    lower leaflet — same bug #1 fingerprint as legacy persistence).
    """
    head_indices = traj.topology.select("name PO4 ROH")
    per_frame = []
    nf = len(traj)
    for f in range(nf - lag):
        frames = traj[[f, f + lag]]
        head_z0 = frames.xyz[0, head_indices, 2]
        cutoff = _split_leaflets(head_z0)
        samples = []
        for _ in range(probe_size):
            if legacy:
                resids = np.where(frames.xyz[0, head_indices, 2] < cutoff)[0]
            else:
                upper = bool(rng.random() > 0.5)
                resids = _select_leaflet_residues(frames.xyz[0], head_indices, cutoff, upper)
            if len(resids) < 2:
                continue
            i = int(rng.choice(resids))
            if legacy:
                j = int(rng.choice(resids))
                while j == i:
                    j = int(rng.choice(resids))
                box = frames.unitcell_lengths[:, :2].copy()
                xyz = frames.xyz.copy()
                xyz[:, :, :2] -= xyz[:, head_indices[j], :2][:, None, :]
                xyz[:, :, :2] += box[:, None, :] / 2.0
                xyz[:, :, :2] %= box[:, None, :]
                new_pos = xyz[1, head_indices[i], :2]
                old_pos = xyz[0, head_indices[i], :2]
                delta = (new_pos / box[1] - old_pos / box[0])
                new_pos = new_pos - np.round(delta) * box[1]
                disp = new_pos - old_pos
            else:
                box = frames.unitcell_lengths[1, :2]
                old_pos = frames.xyz[0, head_indices[i], :2]
                new_pos = frames.xyz[1, head_indices[i], :2]
                disp = _pbc_displacement_2d(old_pos, new_pos, box)
            samples.append(float(np.sum(disp ** 2)))
        per_frame.append(float(np.mean(samples)) if samples else float("nan"))
    series = np.asarray(per_frame, dtype=float) * 100.0  # nm² → Å²
    return float(np.nanmean(series)) if series.size else float("nan"), series


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def compute_all(traj: md.Trajectory, *,
                box_xy: tuple[float, float] | None = None,
                grid_step: float = 0.1,
                grid_margin: float = 1.5,
                contact_cutoff: float = 0.7,
                lag_persistence: int = 50,
                lag_diffusivity: int = 10,
                probe_size: int = 10,
                properties: Iterable[str] | None = None,
                seed: int | None = None,
                legacy: bool = False,
                verbose: bool = False,
                ) -> tuple[dict, dict]:
    """Compute the selected bilayer properties.

    Parameters
    ----------
    traj
        ``mdtraj.Trajectory`` (head beads selected via ``"name PO4 ROH"``).
    box_xy
        Lateral box ``(Lx, Ly)`` in nm for the interpolation grid. If
        ``None`` the per-frame box mean is used.
    properties
        Iterable of property names to compute. ``None`` → all of
        :data:`ALL_PROPERTIES`.
    seed
        RNG seed (NumPy ``default_rng``). Stochastic properties become
        deterministic w.r.t. this seed.
    legacy
        If ``True``, reproduce the bugs from
        ``functions_emil.calculate_properties`` documented in cleanup-plan
        §2. Use only for re-deriving historical labels.

    Returns
    -------
    (mean_dict, raw_dict)
        ``mean_dict`` has scalar entries; ``raw_dict`` has per-frame
        time series. The renamed ``thickness_inhomogeneity`` is emitted
        under both its new name and the legacy alias ``compressibility``.
    """
    requested = set(properties) if properties is not None else set(ALL_PROPERTIES)
    unknown = requested - set(ALL_PROPERTIES) - {"compressibility"}
    if unknown:
        raise ValueError(f"Unknown properties: {sorted(unknown)}. "
                         f"Available: {ALL_PROPERTIES}")
    if "compressibility" in requested:
        requested.add("thickness_inhomogeneity")
        requested.discard("compressibility")

    rng = np.random.default_rng(seed)

    if box_xy is None:
        # use frame-mean box; the grid is static across frames so this is the
        # natural choice for NPT data with small box fluctuations.
        box_xy = (
            float(traj.unitcell_lengths[:, 0].mean()),
            float(traj.unitcell_lengths[:, 1].mean()),
        )
    grid = GridSpec(box_xy=box_xy, step=grid_step, margin=grid_margin)

    mean: dict = {}
    raw: dict = {}

    # ---- lipid_packing ----
    if "lipid_packing" in requested:
        m, s = compute_lipid_packing(traj)
        mean["lipid_packing"], raw["lipid_packing"] = m, s

    # ---- thickness / thickness_std / thickness_inhomogeneity / bending ----
    needs_height = bool(requested & {
        "thickness", "thickness_std", "thickness_inhomogeneity", "bending_modulus",
    })
    if needs_height:
        if legacy:
            po4 = traj.topology.select("name PO4")
            xy_thickness, xy_membrane_half, X, Y, frame_mask = _legacy_height_fields(
                traj, po4, grid,
            )
        else:
            heads = traj.topology.select("name PO4 ROH")
            xy_thickness, xy_midplane, X, Y, frame_mask = _height_fields(
                traj, heads, grid,
            )
        t_mean, t_std_mean, inh_mean, t_series, t_std_series, inh_series = thickness_summary(
            xy_thickness, frame_mask=frame_mask,
        )
        if "thickness" in requested:
            mean["thickness"] = t_mean
            raw["thickness"] = t_series
        if "thickness_std" in requested:
            mean["thickness_std"] = t_std_mean
            raw["thickness_std"] = t_std_series
        if "thickness_inhomogeneity" in requested:
            mean["thickness_inhomogeneity"] = inh_mean
            raw["thickness_inhomogeneity"] = inh_series
            # backwards-compat alias under the historical (mislabelled) key
            mean["compressibility"] = inh_mean
            raw["compressibility"] = inh_series
        if "bending_modulus" in requested:
            mid = xy_membrane_half if legacy else xy_midplane
            kappa_phys, diag = compute_bending_modulus_from_field(mid, X, Y, kBT=1.0)
            if not np.isfinite(kappa_phys):
                mean["bending_modulus"] = float("nan")
            elif legacy:
                # Reproduce the historical (dimensionally muddled) label:
                # κ_fit · 1000, where κ_fit = κ_phys · Δx · Δy. The factor
                # of 1000 was tagged "kT/Å³" in the legacy code but is not
                # a real bending modulus — preserved here only to make
                # legacy=True bit-for-bit reproducible.
                kappa_raw = diag.get("kappa_raw_fit", kappa_phys * diag["step_x"] * diag["step_y"])
                mean["bending_modulus"] = kappa_raw * 1000.0
            else:
                # Physical κ in kBT (already normalised by Δx Δy in the fit).
                mean["bending_modulus"] = kappa_phys
            raw["bending_modulus"] = {
                "q_centers": diag.get("q"),
                "ps_binned": diag.get("ps"),
            }

    # ---- variation ----
    if "variation" in requested:
        m, s = compute_variation(traj, legacy=legacy)
        mean["variation"], raw["variation"] = m, s

    # ---- persistence ----
    if "persistence" in requested:
        m, s = compute_persistence(
            traj, lag=lag_persistence, contact_cutoff=contact_cutoff,
            probe_size=probe_size, rng=rng, legacy=legacy,
        )
        mean["persistence"], raw["persistence"] = m, s

    # ---- diffusivity ----
    if "diffusivity" in requested:
        m, s = compute_diffusivity(
            traj, lag=lag_diffusivity, probe_size=probe_size,
            rng=rng, legacy=legacy,
        )
        mean["diffusivity"], raw["diffusivity"] = m, s

    if verbose:
        for k, v in mean.items():
            print(f"  {k:<26s} = {v}")
    return mean, raw
