"""Shared dipole-extraction helpers for the water benchmarks."""

from __future__ import annotations

from ase import Atoms
import numpy as np

# Oxygen point-charge magnitude (e) used to estimate the water dipole. Chosen to
# match revPBE-D3 (see https://arxiv.org/abs/2603.04228v1). Hydrogen carries +q/2.
DEFAULT_WATER_CHARGE = 0.5562


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
