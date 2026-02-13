"""Analyse ethanol-water density curves."""

from __future__ import annotations

import numpy as np

M_WATER = 18.01528  # g/mol
M_ETOH = 46.06844  # g/mol

# Pick densities consistent with your reference conditions (T,P!)
# If your ref curve is at ~20°C, these are around:
RHO_WATER_PURE = 0.9982  # g/cm^3
RHO_ETH_PURE = 0.7893  # g/cm^3


def x_to_phi_ethanol(
    x,
    rho_mix,
    *,
    m_eth=M_ETOH,
    m_water=M_WATER,
    rho_eth=RHO_ETH_PURE,
    rho_water=RHO_WATER_PURE,
):  # TODO: double check formula
    """
    Convert ethanol mole fraction to ethanol volume fraction.

    Parameters
    ----------
    x : array-like
        Ethanol mole fraction.
    rho_mix : array-like
        Mixture density in g/cm^3 at each composition.
    m_eth : float, optional
        Ethanol molar mass in g/mol.
    m_water : float, optional
        Water molar mass in g/mol.
    rho_eth : float, optional
        Pure ethanol density in g/cm^3.
    rho_water : float, optional
        Pure water density in g/cm^3.

    Returns
    -------
    numpy.ndarray
        Ethanol volume fraction for each input composition.
    """
    x = np.asarray(x, dtype=float)
    rho_mix = np.asarray(rho_mix, dtype=float)

    m_eth = x * m_eth
    m_wat = (1.0 - x) * m_water

    v_mix = (m_eth + m_wat) / rho_mix  # cm^3 per "1 mol mixture basis"
    v_eth = m_eth / rho_eth  # cm^3 (proxy)
    return v_eth / v_mix


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


def _endpoints_at_0_1(
    x: np.ndarray, y: np.ndarray, tol: float = 1e-8
) -> tuple[float, float]:
    """
    Return y values at x=0 and x=1.

    Parameters
    ----------
    x : numpy.ndarray
        Composition grid.
    y : numpy.ndarray
        Property values.
    tol : float, optional
        Absolute tolerance used to identify endpoint compositions.

    Returns
    -------
    tuple[float, float]
        Pair ``(y0, y1)`` for x=0 and x=1.
    """
    i0 = np.where(np.isclose(x, 0.0, atol=tol))[0]
    i1 = np.where(np.isclose(x, 1.0, atol=tol))[0]
    if len(i0) != 1 or len(i1) != 1:
        raise ValueError("Curve must include x=0 and x=1 to define linear baseline.")
    return float(y[i0[0]]), float(y[i1[0]])


def _linear_baseline(x: np.ndarray, y0: float, y1: float) -> np.ndarray:
    """
    Build the straight line connecting values at x=0 and x=1.

    Parameters
    ----------
    x : numpy.ndarray
        Composition grid.
    y0 : float
        Value at x=0.
    y1 : float
        Value at x=1.

    Returns
    -------
    numpy.ndarray
        Linear baseline evaluated at ``x``.
    """
    return y0 + x * (y1 - y0)


def _excess_curve(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute excess curve relative to endpoint linear interpolation.

    Parameters
    ----------
    x : numpy.ndarray
        Composition grid.
    y : numpy.ndarray
        Property values.

    Returns
    -------
    numpy.ndarray
        Excess values ``y - y_linear``.
    """
    y0, y1 = _endpoints_at_0_1(x, y)
    return y - _linear_baseline(x, y0, y1)


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
