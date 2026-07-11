"""
Shared physicality diagnostics for carbon binding-curve benchmarks.

These mirror the ``physicality/diatomics`` metrics (force-direction flips, number
of energy minima, energy inflections, and Spearman correlations on the repulsive
and attractive branches) but derive the restoring force from the energy gradient
rather than atomic forces, so they apply equally to symmetric bulk cells where
per-atom forces vanish. The location and depth of the energy minimum are also
returned so the analysis stage can score them against a reference curve.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from scipy.signal import find_peaks

# Metric column names shared by the binding-curve benchmarks. ``r_min``/``e_min``
# are returned by ``curve_shape_metrics`` but are consumed to build the min-error
# columns rather than reported directly.
SHAPE_METRICS = (
    "Force flips",
    "Energy minima",
    "Energy inflections",
    "ρ(E, repulsion)",
    "ρ(E, attraction)",
)


def count_sign_changes(array: np.ndarray, tol: float) -> int:
    """
    Count sign changes in a sequence while ignoring small magnitudes.

    Parameters
    ----------
    array
        Input values.
    tol
        Absolute tolerance below which values are treated as zero.

    Returns
    -------
    int
        Number of sign changes exceeding the specified tolerance.
    """
    if array.size < 3:
        return 0
    clipped = array[np.abs(array) > tol]
    if clipped.size < 2:
        return 0
    signs = np.sign(clipped)
    return int(np.sum(signs[:-1] != signs[1:]))


def curve_shape_metrics(
    distances: np.ndarray, energies: np.ndarray
) -> dict[str, float] | None:
    """
    Compute diatomics-style shape diagnostics for one binding curve.

    Parameters
    ----------
    distances
        Scan coordinate (Angstrom).
    energies
        Energies per atom (eV), referenced so the large-separation limit is ~0.

    Returns
    -------
    dict[str, float] | None
        Shape metrics plus ``r_min`` (Angstrom) and ``e_min`` (eV) for the energy
        minimum, or ``None`` if there are too few finite points.
    """
    d = np.asarray(distances, dtype=float)
    e = np.asarray(energies, dtype=float)
    finite = np.isfinite(d) & np.isfinite(e)
    d, e = d[finite], e[finite]
    if d.size < 3:
        return None

    # Sort by distance then reverse to descending order to match the diatomics
    # convention (reference zero is the largest-separation energy).
    ascending = np.argsort(d)
    d, e = d[ascending][::-1], e[ascending][::-1]
    e = e - e[0]

    energy_gradient = np.gradient(e, d)
    energy_curvature = np.gradient(energy_gradient, d)
    # Restoring force along the scan coordinate.
    force = -energy_gradient

    force_flips = count_sign_changes(force, tol=1e-2)

    minima_indices, _ = find_peaks(-e, prominence=0.1, width=1)
    minima = len(minima_indices)

    inflections = count_sign_changes(energy_curvature, tol=0.5)

    well_index = int(np.argmin(e))
    spearman_repulsion = np.nan
    spearman_attraction = np.nan
    if d[well_index:].size > 1:
        spearman_repulsion = float(
            stats.spearmanr(d[well_index:], e[well_index:]).statistic
        )
    if d[:well_index].size > 1:
        spearman_attraction = float(
            stats.spearmanr(d[:well_index], e[:well_index]).statistic
        )

    return {
        "Force flips": float(force_flips),
        "Energy minima": float(minima),
        "Energy inflections": float(inflections),
        "ρ(E, repulsion)": spearman_repulsion,
        "ρ(E, attraction)": spearman_attraction,
        "r_min": float(d[well_index]),
        "e_min": float(e[well_index]),
    }


def reference_minimum(ref_x: list[float], ref_y: list[float]) -> tuple[float, float]:
    """
    Return the location and value of the minimum of a reference curve.

    Parameters
    ----------
    ref_x
        Reference scan coordinate.
    ref_y
        Reference energies (in the reference's native unit).

    Returns
    -------
    tuple[float, float]
        Position and value of the reference minimum.
    """
    x = np.asarray(ref_x, dtype=float)
    y = np.asarray(ref_y, dtype=float)
    i = int(np.argmin(y))
    return float(x[i]), float(y[i])
