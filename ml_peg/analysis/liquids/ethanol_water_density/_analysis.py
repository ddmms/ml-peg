"""Analyse ethanol-water density curves."""

from __future__ import annotations

import numpy as np

M_WATER = 18.01528  # g/mol
M_ETOH = 46.06844  # g/mol

# Pick densities consistent with your reference conditions (T,P!)
# If your ref curve is at ~20°C, these are around:
RHO_WATER_PURE = 0.9982  # g/cm^3
RHO_ETH_PURE = 0.7893  # g/cm^3


def weight_to_mole_fraction(w):
    r"""
    Convert ethanol weight fraction to mole fraction.

    Parameters
    ----------
    w : array-like
        Ethanol weight fraction :math:`m_\mathrm{ethanol} / m_\mathrm{total}`.

    Returns
    -------
    numpy.ndarray
        Ethanol mole fraction.
    """
    n_e = w / M_ETOH
    n_w = (1 - w) / M_WATER
    return n_e / (n_e + n_w)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute root-mean-square error between two arrays.

    Parameters
    ----------
    a : numpy.ndarray
        First array.
    b : numpy.ndarray
        Second array.

    Returns
    -------
    float
        Root-mean-square error.
    """
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


def _interp_1d(x_src: np.ndarray, y_src: np.ndarray, x_tgt: np.ndarray) -> np.ndarray:
    """
    Linearly interpolate onto target x values.

    Parameters
    ----------
    x_src : numpy.ndarray
        Source x grid.
    y_src : numpy.ndarray
        Source y values.
    x_tgt : numpy.ndarray
        Target x positions.

    Returns
    -------
    numpy.ndarray
        Interpolated y values at ``x_tgt``.
    """
    if np.any(x_tgt < x_src.min() - 1e-12) or np.any(x_tgt > x_src.max() + 1e-12):
        raise ValueError("Target x values fall outside reference interpolation range.")
    return np.interp(x_tgt, x_src, y_src)


def _excess_volume(x: np.ndarray, rhos: np.ndarray) -> np.ndarray:
    """
    Compute excess volume given molar fraction and density respectively.

    Parameters
    ----------
    x : numpy.ndarray
        Composition grid (mol fraction).
    rhos : numpy.ndarray
        Density.

    Returns
    -------
    numpy.ndarray
        Excess values ``y - y_linear``.
    """
    return (x * M_ETOH + (1 - x) * M_WATER) / rhos - (
        x * M_ETOH / rhos[-1] + (1 - x) * M_WATER / rhos[0]
    )


def _peak_x_quadratic(x: np.ndarray, y: np.ndarray) -> float:
    """
    Estimate x position of the minimum by local quadratic fitting.

    Parameters
    ----------
    x : numpy.ndarray
        Composition grid.
    y : numpy.ndarray
        Property values.

    Returns
    -------
    float
        Estimated composition of the minimum.
    """
    if len(x) < 3:
        return float(x[int(np.argmin(y))])

    i = int(np.argmin(y))
    if i == 0 or i == len(x) - 1:
        return float(x[i])

    # Fit a parabola to (i-1, i, i+1)
    xs = x[i - 1 : i + 2]
    ys = y[i - 1 : i + 2]

    # y = ax^2 + bx + c
    a, b, _c = np.polyfit(xs, ys, deg=2)
    if abs(a) < 1e-16:
        return float(x[i])

    xv = -b / (2.0 * a)

    # Clamp to local bracket so we don't get silly extrapolation
    return float(np.clip(xv, xs.min(), xs.max()))
