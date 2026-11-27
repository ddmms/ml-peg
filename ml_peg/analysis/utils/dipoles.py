"""Shared dipole-extraction helpers for the water benchmarks."""

from __future__ import annotations

from ase import Atoms
from matscipy.neighbours import neighbour_list
import numpy as np

# Oxygen point-charge magnitude (e) used to estimate the water dipole. Chosen to
# match revPBE-D3 (see https://arxiv.org/abs/2603.04228v1). Hydrogen carries +q/2.
DEFAULT_WATER_CHARGE = 0.5562
OH_NEIGHBOUR_LIST_CUTOFF = 1.2


def get_z_dipoles(
    frames: list[Atoms], *, q: float = DEFAULT_WATER_CHARGE
) -> np.ndarray:
    """
    Total z-dipole per unit xy-area for each frame, via a point-charge model.

    Places charge ``-q`` on each oxygen and ``+q/2`` on each hydrogen. For water
    this is charge neutral, so the total dipole is independent of the coordinate
    origin. The z-component is summed and normalised by the xy cross-sectional
    area of the (first frame's) cell.

    Parameters
    ----------
    frames
        Trajectory frames, each an ASE Atoms object of water (O and H atoms).
    q
        Oxygen point-charge magnitude in e; hydrogen carries ``q/2``. Default is
        `DEFAULT_WATER_CHARGE`.

    Returns
    -------
    numpy.ndarray
        Total z-dipole per unit area for each frame, in e/Å.
    """
    dipoles = np.zeros(len(frames))
    for i, struc in enumerate(frames):
        o_index = [atom.index for atom in struc if atom.number == 8]
        h_index = [atom.index for atom in struc if atom.number == 1]
        dipoles[i] = (
            np.sum(struc.positions[o_index, 2]) * (-q)
            + np.sum(struc.positions[h_index, 2]) * q / 2
        )
    area = frames[0].cell[0, 0] * frames[0].cell[1, 1]
    return dipoles / area


def get_z_dipoles_frames(
    frames: list[Atoms],
    *,
    q: float = DEFAULT_WATER_CHARGE,
    surface_idx_1: int = 90,
    surface_idx_2: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-frame z-dipole and the z-resolved per-molecule dipole profile.

    Builds each water molecule's dipole from a point-charge model: the vector sum
    of its O->H bond vectors scaled by ``q/2`` (equivalent to placing ``-q`` on the
    oxygen and ``+q/2`` on each hydrogen). H atoms are assigned to an oxygen via a
    neighbour list with an O-H cutoff of ``OH_NEIGHBOUR_LIST_CUTOFF``. The per-frame
    total z-dipole and every molecule's z-dipole (against its oxygen's height above
    the surface) are both normalised by the xy cross-sectional area of the first
    frame's cell.

    Parameters
    ----------
    frames
        Trajectory frames, each an ASE Atoms object of water (O and H atoms).
    q
        Oxygen point-charge magnitude in e; hydrogen carries ``q/2``. Default is
        `DEFAULT_WATER_CHARGE`.
    surface_idx_1
        Start index of the atom slice whose mean z defines the surface height that
        molecular oxygen heights are measured from. Default is 90.
    surface_idx_2
        Stop index (exclusive) of that surface-defining atom slice. Default is 120.

    Returns
    -------
    numpy.ndarray
        Per-frame total z-dipole per unit area, in e/Å.
    numpy.ndarray
        Pooled per-molecule points across all frames, shape ``(N, 2)``: column 0 is
        the oxygen z relative to the surface (Å), column 1 the molecular z-dipole per
        unit area (e/Å).
    """
    o_indexes = [atom.index for atom in frames[0] if atom.number == 8]
    dipoles = np.zeros(len(frames))
    all_dipoles = []

    for i, struc in enumerate(frames):
        # Get neighbour list for water atoms
        idx_i, dist = neighbour_list(
            quantities="iD",
            atoms=struc,
            cutoff={("O", "H"): OH_NEIGHBOUR_LIST_CUTOFF},
        )

        top_pos = np.average(struc.positions[surface_idx_1:surface_idx_2], axis=0)[2]
        net_dipole = np.array([0.0, 0.0, 0.0])
        all_dipoles_sub = []
        for oi in o_indexes:
            h_indexes = np.where(idx_i == oi)
            if len(h_indexes) > 0:
                dipole = np.sum(dist[h_indexes], axis=0)
                opos = struc[oi].position
                norm = np.linalg.norm(dipole)
                if norm == 0:
                    dipole = np.array([0.0, 0.0, 0.0])
                else:
                    dipole = dipole * 0.5 * q
                net_dipole += dipole
                all_dipoles_sub.append(
                    (
                        opos[2] - top_pos,
                        dipole[2],
                    )
                )

        dipoles[i] = net_dipole[2]
        all_dipoles_sub = sorted(all_dipoles_sub, key=lambda x: x[0])
        all_dipoles.append(all_dipoles_sub)

    area = frames[0].cell[0, 0] * frames[0].cell[1, 1]
    dipoles /= area

    # Pool every molecule's (z, z-dipole) point across all frames into a single
    # (N, 2) array, matching what get_z_dipoles_integrated_profile expects. Only the
    # dipole column is normalised by area; column 0 stays a z position in Å.
    all_dipoles = np.array([point for frame in all_dipoles for point in frame])
    all_dipoles[:, 1] /= area
    return dipoles, all_dipoles


def get_z_dipoles_average_integrated_profile(
    all_dipoles: np.ndarray,
    *,
    n_frames: int,
    bin_width: float = 0.1,
    x_lower=0.0,
    x_upper: float = 35.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Per-frame-average cumulative (integrated) z-dipole profile.

    For each z, sums the z-dipole of every molecule below that height (pooled over
    all frames) and divides by the number of frames. This is the ``integrated``
    output of ``plot_dipole_distribution`` (mode ``'average'``): the mean over frames
    of each frame's cumulative dipole below z. Dividing by ``n_frames`` makes the
    profile independent of trajectory length, so reference and model curves (which
    have different frame counts) are on the same scale.

    Parameters
    ----------
    all_dipoles
        Pooled per-molecule points, shape ``(N, 2)``: column 0 the oxygen z relative
        to the surface (Å), column 1 the molecular z-dipole per unit area (e/Å), as
        returned by :func:`get_z_dipoles_frames`.
    n_frames
        Number of trajectory frames pooled into ``all_dipoles``; the cumulative sum is
        divided by this to give a per-frame average.
    bin_width
        Width of the bins in Å.
    x_lower
        Lower bound of the integration in Å.
    x_upper
        Upper bound of the integration in Å.

    Returns
    -------
    numpy.ndarray
        The z bin edges, in Å.
    numpy.ndarray
        Per-frame-average cumulative z-dipole per unit area below each z, in e/Å.
    """
    z_bins = np.arange(x_lower, x_upper, bin_width)
    integrated_profile = np.zeros(len(z_bins))
    for i in range(len(z_bins)):
        integrated_profile[i] = (
            np.sum(all_dipoles[all_dipoles[:, 0] < z_bins[i], 1]) / n_frames
        )

    return z_bins, integrated_profile
