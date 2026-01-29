import numpy as np
import mdtraj as md
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm
from scipy.fft import fft2, fftshift, fftfreq
from scipy.spatial import Voronoi
import shapely.geometry
from scipy.optimize import curve_fit

def undulation_model(q, kappa, kBT=2.5):
    return kBT / (kappa * q**4)

def voronoi_area_cv(points, bounding_box=None):
    """
    CREDITS: chatgpt
    
    Compute the coefficient of variation (CV) of Voronoi cell areas
    for a set of 2D points within a bounding box.
    
    Parameters:
    - points: (N, 2) array of xy points
    - bounding_box: tuple (xmin, xmax, ymin, ymax). If None, will auto-compute.

    Returns:
    - CV: float (std / mean of areas)
    """
    if bounding_box is None:
        xmin, ymin = points.min(axis=0) - 1.0
        xmax, ymax = points.max(axis=0) + 1.0
    else:
        xmin, xmax, ymin, ymax = bounding_box

    bbox = shapely.geometry.box(xmin, ymin, xmax, ymax)

    vor = Voronoi(points)
    areas = []

    for point_idx, region_idx in enumerate(vor.point_region):
        vertex_indices = vor.regions[region_idx]
        if -1 in vertex_indices or len(vertex_indices) == 0:
            # Open region (extends to infinity) → skip or clip
            continue

        polygon = shapely.geometry.Polygon(vor.vertices[vertex_indices])
        polygon_clipped = polygon.intersection(bbox)
        
        if not polygon_clipped.is_empty:
            areas.append(polygon_clipped.area)

    areas = np.array(areas)
    mean_area = np.mean(areas)
    std_area = np.std(areas)

    return std_area / mean_area if mean_area > 0 else 0.0

def compute_bending_modulus(Z, X, Y, kBT=2.5, plot=False):
    """
    Compute membrane bending modulus from time series of membrane height fields Z(x, y, t)
    
    Parameters:
    - Z: array of shape (n_frames, n_x, n_y) – height values over time
    - X, Y: meshgrid arrays of shape (n_x, n_y) – x/y positions (uniform grid)
    - kBT: thermal energy (default: 2.5 in kBT units)
    - plot: whether to plot undulation spectrum and fit

    Returns:
    - kappa: estimated bending modulus in kBT
    """

    n_frames, nx, ny = Z.shape
    Lx = X.max() - X.min()
    Ly = Y.max() - Y.min()

    # Subtract time average height → fluctuations
    Z_fluct = Z - np.mean(Z, axis=0)

    # FFT of each frame → average power spectrum
    fft_frames = fft2(Z_fluct, axes=(1, 2), norm='ortho')  # shape: (n_frames, nx, ny)
    fft_power = np.abs(fft_frames) ** 2
    ps_avg = np.mean(fft_power, axis=0)

    # Get q values
    qx = fftfreq(nx, d=Lx / nx) * 2 * np.pi
    qy = fftfreq(ny, d=Ly / ny) * 2 * np.pi
    QX, QY = np.meshgrid(qx, qy, indexing='ij')
    q_vals = np.sqrt(QX**2 + QY**2).flatten()
    ps_vals = ps_avg.flatten()
        
    def undulation_model(q, kappa):
        return kBT / (kappa * q**4)

    # Bin power spectrum radially
    n_bins = 50
    q_bins = np.linspace(0, q_vals.max(), n_bins + 1)
    q_centers = 0.5 * (q_bins[:-1] + q_bins[1:])
    ps_binned = np.zeros(n_bins)
    counts = np.zeros(n_bins)

    for i in range(len(q_vals)):
        bin_idx = np.searchsorted(q_bins, q_vals[i]) - 1
        if 0 <= bin_idx < n_bins:
            ps_binned[bin_idx] += ps_vals[i]
            counts[bin_idx] += 1

    ps_binned /= np.maximum(counts, 1)

    # Only fit where q > 0
    valid = (q_centers > 0.1) & (counts > 5) & np.isfinite(ps_binned)
    popt, _ = curve_fit(undulation_model, q_centers[valid], ps_binned[valid])
    kappa = popt[0]

    if plot:
        plt.figure(figsize=(6, 4))
        plt.loglog(q_centers[valid], ps_binned[valid], 'o', label="Data")
        plt.loglog(q_centers[valid], undulation_model(q_centers[valid], *popt), '--',
                   label=f"Fit: κ = {kappa:.2f} kBT")
        plt.xlabel("q (1/nm)")
        plt.ylabel(r"$\langle |h(q)|^2 \rangle$")
        plt.title("Membrane Undulation Spectrum")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return kappa, [q_centers[valid], ps_binned[valid]]

def compute_properties(trajectory,
                       box_xy:list = [11,11],
                       contact_cutoff:float = .7,
                       lag_persistence:int = 50,
                       lag_diffusivity:int = 10,
                       probe_size:int = 10,  # for each frame, how many random lipids?
                       verbose=True):
    
    """
    Compute various physical and structural properties of a lipid bilayer from a molecular dynamics trajectory.

    Parameters
    ----------
    trajectory : mdtraj.Trajectory
        Molecular dynamics trajectory containing lipid molecules.
    box_xy : list of float, optional
        The XY dimensions of the simulation box to define the grid for surface interpolation. Default is [11, 11].
    contact_cutoff : float, optional
        Distance cutoff (in nm) to define lipid-lipid contacts. Default is 0.7.
    lag_persistence : int, optional
        Number of frames to compute contact persistence. Default is 50.
    lag_diffusivity : int, optional
        Number of frames to compute lipid diffusivity. Default is 10.
    probe_size : int, optional
        Number of randomly selected lipids per frame for persistence and diffusivity calculations. Default is 10.
    verbose : bool, optional
        If True, displays a progress bar for trajectory processing. Default is True.

    Returns
    -------
    mean_dict : dict
        Dictionary of mean values for each computed property:
            - 'lipid_packing': average lipid packing density
            - 'thickness': average bilayer thickness (Å)
            - 'thickness_std': frame-wise standard deviation of thickness (Å)
            - 'compressibility': bilayer compressibility (Å^3 / kT)
            - 'bending_modulus': bending modulus (kT / Å^3)
            - 'persistence': mean contact persistence
            - 'diffusivity': mean lipid diffusivity (Å^2 / lag)
            - 'variation': mean coefficient of variation of Voronoi cell areas

    raw_dict : dict
        Dictionary of time-series arrays for each property across frames.

    Notes
    -----
    - Uses LinearNDInterpolator to estimate upper and lower leaflet surfaces for thickness calculation.
    - Contact persistence and diffusivity are sampled over a subset of lipids determined by `probe_size`.
    - Voronoi tessellation is used to calculate the coefficient of variation of lipid packing.
    - All distance-based calculations assume a periodic boundary condition and the minimum image convention.
    - Thickness and diffusivity values are converted to Å and Å^2 units, respectively.
    """

    
    # grid definition
    X, Y = np.meshgrid(np.arange(1.5, box_xy[0]-1.49, .1),
                   np.arange(1.5, box_xy[1]-1.49, .1))
    XY = np.array(list(zip(X.ravel(), Y.ravel())))
    
    # general data about lipids
    po4_indices = trajectory.topology.select('name PO4')
    heads_indices = trajectory.topology.select('name PO4 ROH')
    n_lipids = len(heads_indices)
    residues = [np.array([atom.index for atom in residues.atoms]) 
                for residues in trajectory.topology.residues][:n_lipids]
    beads = list(trajectory.topology.atoms)
    
    # initializing lists
    xy_thickness = []
    xy_membrane = []

    persistence = []  # probability that after lag time still in contact
    diffusivity = []  # how far diffused in lag time
    variation = []  # voronoi coefficient of variation

    # Main loop
    for frame_index in tqdm(range(len(trajectory)), position=0, disable=not verbose):

        # upper and lower leaflet indices
        po4_xyz = trajectory.xyz[frame_index, po4_indices]
        z = np.sort(po4_xyz[:, 2])
        cutoff = np.argmax(np.diff(z))
        cutoff = (z[cutoff] + z[cutoff + 1]) / 2

        # upper and lower leaflet positions
        po4_lower_xyz = po4_xyz[po4_xyz[:, 2] < cutoff]
        po4_upper_xyz = po4_xyz[po4_xyz[:, 2] > cutoff]

        # upper and lower leaflet interpolation
        lower_surface = LinearNDInterpolator(po4_lower_xyz[:, :2],
                                             po4_lower_xyz[:, 2])(X, Y)
        upper_surface = LinearNDInterpolator(po4_upper_xyz[:, :2],
                                             po4_upper_xyz[:, 2])(X, Y)

        # thickness update (only if no nans)
        if not np.sum(np.isnan(upper_surface)) and not np.sum(np.isnan(lower_surface)):  
            xy_thickness.append(upper_surface - lower_surface)
            xy_membrane.append((upper_surface - lower_surface) / 2.)

        """
        Persistence
        """

        if frame_index < len(trajectory) - lag_persistence:

            frames = trajectory[[frame_index, frame_index + lag_persistence]]
            persistence.append([])

            for _ in range(probe_size):

                # choose one leaflet
                if np.random.random() > .5:
                    resids = np.where(frames.xyz[0, heads_indices, 2] < cutoff)[0]  # lower leaflet
                else:
                    resids = np.where(frames.xyz[0, heads_indices, 2] < cutoff)[0]  # upper leaflet

                # the chosen one and the others
                i = np.random.choice(resids)
                lipid_indices = residues[i]
                other_indices = np.concatenate([residues[j] for j in resids if j != i])

                # which lipids are in contact with the chosen one?
                d = np.min(md.compute_distances(frames, [[h, k] for k in other_indices for h in lipid_indices]
                                    ).reshape(2, -1, len(lipid_indices)), axis=2)
                # index 0: frame
                # index 1: other_indices
                # value: minimum distance between lipid and other_indices

                # lipid-other_indices contacts?
                contacts_indices = np.where(d[0] < contact_cutoff)[0]

                # trace back to resid
                contacts_resids = np.unique([beads[j].residue.index for j in contacts_indices])

                # choose one residue in contact
                other_indices = set(residues[np.random.choice(contacts_resids)])

                # is the contact still there after the lag?
                if len(other_indices.intersection(np.where(d[1] < contact_cutoff)[0])):
                    persistence[-1].append(1.)
                else:
                    persistence[-1].append(0.)

        """
        Diffusivity.
        """

        if frame_index < len(trajectory) - lag_diffusivity:

            frames = trajectory[[frame_index, frame_index + lag_diffusivity]]
            diffusivity.append([])

            for _ in range(probe_size):

                # choose one leaflet
                if np.random.random() > .5:
                    resids = np.where(frames.xyz[0, heads_indices, 2] < cutoff)[0]  # lower leaflet
                else:
                    resids = np.where(frames.xyz[0, heads_indices, 2] < cutoff)[0]  # upper leaflet

                # choose two lipids (relative movement)
                i = np.random.choice(resids)
                while (j := np.random.choice(resids)) == i:
                    continue

                # center pivot
                box = frames.unitcell_lengths[:, :2]
                frames.xyz[:, :, :2] -= frames.xyz[:, heads_indices[j], :2][:, None, :2]
                frames.xyz[:, :, :2] += box[:, None, :] / 2.
                frames.xyz[:, :, :2] %= box[:, None, :]

                new_pos = frames.xyz[1, heads_indices[i], :2]
                old_pos = frames.xyz[0, heads_indices[i], :2]

                # apply minimum image convention
                delta = (new_pos / box[1] - old_pos / box[0])
                new_pos -= np.round(delta) * box[1]

                # actual displacement
                delta = new_pos - old_pos

                # square for diffusivity
                diffusivity[-1].append(np.sum(delta ** 2))

        """
        coefficient of variation (Voronoi cell area).
        """

        # for both leaflets
        variation.append([
        voronoi_area_cv(po4_lower_xyz[:, :2], [0, trajectory.unitcell_lengths[frame_index, 0],
                                           0, trajectory.unitcell_lengths[frame_index, 1]]),
        voronoi_area_cv(po4_upper_xyz[:, :2], [0, trajectory.unitcell_lengths[frame_index, 0],
                                           0, trajectory.unitcell_lengths[frame_index, 1]])])
        
    xy_thickness = np.array(xy_thickness)
    xy_membrane = np.array(xy_membrane)

    # thickness [A]
    thickness_series = np.mean(xy_thickness.reshape(-1, np.prod(X.shape)), axis=1) * 10.  # A
    thickness = np.mean(xy_thickness) * 10.  # A

    # thickness std (frame-wise) [A]
    # still a measure of compressibility-curvature
    thickness_std_series = np.std(xy_thickness.reshape(-1, np.prod(X.shape)), axis=1) * 10.  # A
    thickness_std = np.mean(thickness_std_series) # A

    # compressibility [A^3 / [kT]]
    compressibility_series = np.std(xy_thickness.reshape(-1, np.prod(X.shape)) -
                                    thickness_series[:, None], axis=1) ** 2 * 100.
    compressibility = np.std(xy_thickness - thickness) ** 2 * 100.

    # bending modulus (conver to [[kT] / A^3] units)
    bending_modulus, bending_modulus_series = compute_bending_modulus(xy_membrane, X, Y, kBT=1)
    bending_modulus = bending_modulus * 1000.

    # total persistence [1/lag]
    persistence_series = np.mean(persistence, axis=1)
    mean_persistence = np.mean(persistence)

    # diffusivity [A^2/lag]
    diffusivity_series = np.mean(diffusivity, axis=1) * 100.
    mean_diffusivity = np.mean(diffusivity) * 100.

    # packing [lipids / A^2]
    packing_series = n_lipids / np.prod(trajectory.unitcell_lengths[:, :2], axis=1)
    packing = np.mean(packing_series)

    # coefficient of variation [A^2] (of Voronoi cell areas)
    variation_series = np.mean(variation, axis=1)
    mean_variation = np.mean(variation_series)

    mean_dict = {'lipid_packing'   : packing,
                 'thickness'       : thickness,
                 'thickness_std'   : thickness_std,
                 'compressibility' : compressibility,
                 'bending_modulus' : bending_modulus,
                 'persistence'     : mean_persistence,
                 'diffusivity'     : mean_diffusivity,
                 'variation'       : mean_variation,}

    raw_dict = {'lipid_packing'   : packing_series,
                'thickness'       : thickness_series,
                'thickness_std'   : thickness_std_series,
                'compressibility' : compressibility_series,
                'bending_modulus' : bending_modulus_series,
                'persistence'     : persistence_series,
                'diffusivity'     : diffusivity_series,
                'variation'       : variation_series,}
        
    return mean_dict, raw_dict