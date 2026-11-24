"""
Birch-Murnaghan fit and EOS comparison measures of the ACWF verification study.

The fit is adapted from ``eosfit.py`` v3.1 (K. Lejaeghere, Ghent University, LGPL-2.1+),
as distributed in ``3-analyze/eos_utils/eosfit_31_adapted.py`` of
https://github.com/aiidateam/acwf-verification-scripts.

The comparison measures (Δ, ε and ν) are adapted from
``acwf_paper_plots/quantities_for_comparison.py`` of the same repository (MIT licence),
at commit 8eb1e9a35b512780c4462ceb8f3db815096c25cf. They are kept identical to the
published implementation so that values can be compared with:

E. Bosoni et al., "How to verify the precision of density-functional-theory
implementations via reproducible and universal workflows", Nat. Rev. Phys. 6, 45
(2024), https://doi.org/10.1038/s42254-023-00655-3

Throughout, volumes are in Å³/atom, energies in eV/atom and bulk moduli in eV/Å³.
In the comparison functions, ``(v0w, b0w, b1w)`` are the parameters of the curve being
tested (the MLIP) and ``(v0f, b0f, b1f)`` those of the reference.
"""

from __future__ import annotations

import numpy as np

# Weights of the relative errors on B0 and B1 in ν, and the prefactor of ν (Bosoni
# et al., 2024).
NU_WEIGHT_B0 = 1.0 / 20.0
NU_WEIGHT_B1 = 1.0 / 400.0
NU_PREFACTOR = 100.0

# Fractional half-width of the volume range used by ACWF: the EOS is sampled at
# 0.94, 0.96, ..., 1.06 times the central volume, and Δ and ε integrate over
# 0.94-1.06 times the average V0 of the two curves.
VOLUME_RANGE = 0.06


def birch_murnaghan(
    volume: float | np.ndarray, e0: float, v0: float, b0: float, b1: float
) -> float | np.ndarray:
    """
    Evaluate the third-order Birch-Murnaghan equation of state.

    Parameters
    ----------
    volume
        Volume(s) at which to evaluate the energy.
    e0
        Energy at the minimum.
    v0
        Volume at the minimum.
    b0
        Bulk modulus at the minimum.
    b1
        Pressure derivative of the bulk modulus at the minimum.

    Returns
    -------
    float | np.ndarray
        Energy at each volume.
    """
    ratio = (v0 / volume) ** (2.0 / 3.0)
    return e0 + 9.0 / 16.0 * b0 * v0 * (
        (ratio - 1.0) ** 3 * b1 + (ratio - 1.0) ** 2 * (6.0 - 4.0 * ratio)
    )


def fit_birch_murnaghan(
    volumes: np.ndarray, energies: np.ndarray
) -> dict[str, float] | None:
    """
    Fit a Birch-Murnaghan equation of state as in the ACWF reference data.

    The energy is fitted as a cubic polynomial in V^(-2/3), which is a linear least
    squares problem, and the parameters are obtained from its derivatives at the
    minimum.

    Parameters
    ----------
    volumes
        Volumes.
    energies
        Energies at each volume.

    Returns
    -------
    dict[str, float] | None
        ``E0``, ``V0``, ``B0``, ``B1`` and ``residuals`` (1 - R²) of the fit, or
        ``None`` if there are fewer than four points or the fitted polynomial has no
        minimum.
    """
    volumes = np.asarray(volumes, dtype=float)
    energies = np.asarray(energies, dtype=float)
    if volumes.size < 4:
        return None

    coefficients, ssr, *_ = np.polyfit(volumes ** (-2.0 / 3.0), energies, 3, full=True)
    sst = np.sum((energies - np.mean(energies)) ** 2)
    deriv0 = np.poly1d(coefficients)
    deriv1 = np.polyder(deriv0, 1)
    deriv2 = np.polyder(deriv1, 1)
    deriv3 = np.polyder(deriv2, 1)

    for root in np.roots(deriv1):
        if abs(root.imag) < 1e-8 and root.real > 0 and deriv2(root.real) > 0:
            x = root.real
            break
    else:
        return None

    deriv_v2 = 4.0 / 9.0 * x**5 * deriv2(x)
    deriv_v3 = -20.0 / 9.0 * x ** (13.0 / 2.0) * deriv2(x) - 8.0 / 27.0 * x ** (
        15.0 / 2.0
    ) * deriv3(x)

    return {
        "E0": float(deriv0(x)),
        "V0": float(x ** (-3.0 / 2.0)),
        "B0": float(deriv_v2 / x ** (3.0 / 2.0)),
        "B1": float(-1.0 - x ** (-3.0 / 2.0) * deriv_v3 / deriv_v2),
        "residuals": float(ssr[0] / sst) if ssr.size and sst > 0 else 0.0,
    }


def _antider_e(v0: float, b0: float, b1: float, volume: float) -> float:
    """
    Antiderivative of the Birch-Murnaghan E(V), with E0 = 0.

    Parameters
    ----------
    v0
        Volume at the minimum.
    b0
        Bulk modulus.
    b1
        Pressure derivative of the bulk modulus.
    volume
        Volume at which to evaluate the antiderivative.

    Returns
    -------
    float
        Value of the antiderivative.
    """
    return (
        9
        * b0
        * v0
        * (
            -((-6 + b1) * volume)
            - ((-4 + b1) * v0**2) / volume
            + 3 * (-14 + 3 * b1) * v0 * (v0 / volume) ** (1 / 3)
            + 3 * (-16 + 3 * b1) * volume * (v0 / volume) ** (2 / 3)
        )
    ) / 16


def _antider_e2(v0: float, b0: float, b1: float, volume: float) -> float:
    """
    Antiderivative of the squared Birch-Murnaghan E(V), with E0 = 0.

    Parameters
    ----------
    v0
        Volume at the minimum.
    b0
        Bulk modulus.
    b1
        Pressure derivative of the bulk modulus.
    volume
        Volume at which to evaluate the antiderivative.

    Returns
    -------
    float
        Value of the antiderivative.
    """
    return (
        81
        * b0**2
        * v0**2
        * (
            (-6 + b1) ** 2 * volume
            + (4 * (124 + 5 * (-10 + b1) * b1) * v0**2) / volume
            - ((-4 + b1) ** 2 * v0**4) / (3.0 * volume**3)
            - (
                3
                * (v0 / volume) ** (2 / 3)
                * (
                    10 * (-6 + b1) * (-16 + 3 * b1) * volume**2
                    + (324 + 5 * b1 * (-28 + 3 * b1)) * v0**2
                )
            )
            / (5.0 * volume)
            + (v0 / volume) ** (1 / 3)
            * (
                -3 * (424 + 5 * b1 * (-32 + 3 * b1)) * v0
                + (6 * (-4 + b1) * (-14 + 3 * b1) * v0**3) / (7.0 * volume**2)
            )
        )
    ) / 256.0


def _antider_e12_sq(
    v0w: float,
    b0w: float,
    b1w: float,
    v0f: float,
    b0f: float,
    b1f: float,
    volume: float,
) -> float:
    """
    Antiderivative of (E1(V) - E2(V))², for two Birch-Murnaghan curves with E0 = 0.

    Parameters
    ----------
    v0w, b0w, b1w
        Parameters of the first curve.
    v0f, b0f, b1f
        Parameters of the second curve.
    volume
        Volume at which to evaluate the antiderivative.

    Returns
    -------
    float
        Value of the antiderivative.
    """
    vol = volume
    return (
        81
        * (
            6
            * b0w
            * b0f
            * (-16 + 3 * b1w)
            * (-16 + 3 * b1f)
            * vol
            * v0w
            * (v0w / vol) ** (2 / 3)
            * v0f
            * (v0f / vol) ** (2 / 3)
            - 2
            * b0w
            * b0f
            * (-14 + 3 * b1w)
            * (-16 + 3 * b1f)
            * vol
            * v0w
            * (v0w / vol) ** (4 / 3)
            * v0f
            * (v0f / vol) ** (2 / 3)
            - 2
            * b0w
            * b0f
            * (-16 + 3 * b1w)
            * (-14 + 3 * b1f)
            * vol
            * v0w
            * (v0w / vol) ** (2 / 3)
            * v0f
            * (v0f / vol) ** (4 / 3)
            + (
                6
                * b0w
                * b0f
                * (-14 + 3 * b1w)
                * (-14 + 3 * b1f)
                * vol
                * v0w
                * (v0w / vol) ** (4 / 3)
                * v0f
                * (v0f / vol) ** (4 / 3)
            )
            / 5.0
            + vol * (b0w * (-6 + b1w) * v0w - b0f * (-6 + b1f) * v0f) ** 2
            - (b0w * (-4 + b1w) * v0w**3 - b0f * (-4 + b1f) * v0f**3) ** 2
            / (3.0 * vol**3)
            + (
                3
                * b0f
                * (v0f / vol) ** (7 / 3)
                * (
                    -2
                    * b0w
                    * (-14 + 3 * b1f)
                    * v0w
                    * (-7 * (-6 + b1w) * vol**2 + (-4 + b1w) * v0w**2)
                    - 7 * b0f * (424 + 5 * b1f * (-32 + 3 * b1f)) * vol**2 * v0f
                    + 2 * b0f * (-4 + b1f) * (-14 + 3 * b1f) * v0f**3
                )
            )
            / 7.0
            - (
                3
                * b0f
                * (v0f / vol) ** (5 / 3)
                * (
                    -2
                    * b0w
                    * (-16 + 3 * b1f)
                    * v0w
                    * (5 * (-6 + b1w) * vol**2 + (-4 + b1w) * v0w**2)
                    + 10 * b0f * (-6 + b1f) * (-16 + 3 * b1f) * vol**2 * v0f
                    + b0f * (324 + 5 * b1f * (-28 + 3 * b1f)) * v0f**3
                )
            )
            / 5.0
            + (
                4 * b0w**2 * (124 + 5 * (-10 + b1w) * b1w) * v0w**4
                - 2 * b0w * b0f * (-4 + b1w) * (-6 + b1f) * v0w**3 * v0f
                - 2 * b0w * b0f * (-6 + b1w) * (-4 + b1f) * v0w * v0f**3
                + 4 * b0f**2 * (124 + 5 * (-10 + b1f) * b1f) * v0f**4
            )
            / vol
            + (
                3
                * b0w
                * (v0w / vol) ** (7 / 3)
                * (
                    -7 * b0w * (424 + 5 * b1w * (-32 + 3 * b1w)) * vol**2 * v0w
                    + 2 * b0w * (-4 + b1w) * (-14 + 3 * b1w) * v0w**3
                    + 2
                    * b0f
                    * (-14 + 3 * b1w)
                    * v0f
                    * (7 * (-6 + b1f) * vol**2 - (-4 + b1f) * v0f**2)
                )
            )
            / 7.0
            - (
                3
                * b0w
                * (v0w / vol) ** (5 / 3)
                * (
                    10 * b0w * (-6 + b1w) * (-16 + 3 * b1w) * vol**2 * v0w
                    + b0w * (324 + 5 * b1w * (-28 + 3 * b1w)) * v0w**3
                    - 2
                    * b0f
                    * (-16 + 3 * b1w)
                    * v0f
                    * (5 * (-6 + b1f) * vol**2 + (-4 + b1f) * v0f**2)
                )
            )
            / 5.0
        )
    ) / 256.0


def delta(
    v0w: float, b0w: float, b1w: float, v0f: float, b0f: float, b1f: float
) -> float:
    """
    Get the Δ gauge between two Birch-Murnaghan curves.

    Δ is the root-mean-square energy difference between the two curves, aligned at
    their minima, over 0.94-1.06 times their average V0 (Lejaeghere et al., Science
    351, aad3000, 2016).

    Parameters
    ----------
    v0w, b0w, b1w
        Parameters of the tested curve.
    v0f, b0f, b1f
        Parameters of the reference curve.

    Returns
    -------
    float
        Δ in meV/atom.
    """
    v_i = (1 - VOLUME_RANGE) * (v0w + v0f) / 2.0
    v_f = (1 + VOLUME_RANGE) * (v0w + v0f) / 2.0

    a3f = 9.0 * v0f**3.0 * b0f / 16.0 * (b1f - 4.0)
    a2f = 9.0 * v0f ** (7.0 / 3.0) * b0f / 16.0 * (14.0 - 3.0 * b1f)
    a1f = 9.0 * v0f ** (5.0 / 3.0) * b0f / 16.0 * (3.0 * b1f - 16.0)
    a0f = 9.0 * v0f * b0f / 16.0 * (6.0 - b1f)

    a3w = 9.0 * v0w**3.0 * b0w / 16.0 * (b1w - 4.0)
    a2w = 9.0 * v0w ** (7.0 / 3.0) * b0w / 16.0 * (14.0 - 3.0 * b1w)
    a1w = 9.0 * v0w ** (5.0 / 3.0) * b0w / 16.0 * (3.0 * b1w - 16.0)
    a0w = 9.0 * v0w * b0w / 16.0 * (6.0 - b1w)

    x = [
        (a0f - a0w) ** 2,
        6.0 * (a1f - a1w) * (a0f - a0w),
        -3.0 * (2.0 * (a2f - a2w) * (a0f - a0w) + (a1f - a1w) ** 2.0),
        -2.0 * (a3f - a3w) * (a0f - a0w) - 2.0 * (a2f - a2w) * (a1f - a1w),
        -3.0 / 5.0 * (2.0 * (a3f - a3w) * (a1f - a1w) + (a2f - a2w) ** 2.0),
        -6.0 / 7.0 * (a3f - a3w) * (a2f - a2w),
        -1.0 / 3.0 * (a3f - a3w) ** 2.0,
    ]

    f_i = sum(x[n] * v_i ** (-(2.0 * n - 3.0) / 3.0) for n in range(7))
    f_f = sum(x[n] * v_f ** (-(2.0 * n - 3.0) / 3.0) for n in range(7))

    return float(1000.0 * np.sqrt((f_f - f_i) / (v_f - v_i)))


def epsilon(
    v0w: float, b0w: float, b1w: float, v0f: float, b0f: float, b1f: float
) -> float:
    """
    Get the ε measure between two Birch-Murnaghan curves.

    ε is the root-mean-square energy difference between the two curves, aligned at
    their minima, divided by the geometric mean of their standard deviations, over
    0.94-1.06 times their average V0 (Bosoni et al., 2024).

    As in the published implementation, the mean energy of the tested curve is used
    when computing the variance of both curves. Compared with using the mean of each
    curve, this lowers ε by a few percent.

    Parameters
    ----------
    v0w, b0w, b1w
        Parameters of the tested curve.
    v0f, b0f, b1f
        Parameters of the reference curve.

    Returns
    -------
    float
        Value of ε (dimensionless).
    """
    v_i = (1 - VOLUME_RANGE) * (v0w + v0f) / 2.0
    v_f = (1 + VOLUME_RANGE) * (v0w + v0f) / 2.0
    delta_v = v_f - v_i

    int_diff2 = _antider_e12_sq(v0w, b0w, b1w, v0f, b0f, b1f, v_f) - _antider_e12_sq(
        v0w, b0w, b1w, v0f, b0f, b1f, v_i
    )
    int_ew = _antider_e(v0w, b0w, b1w, v_f) - _antider_e(v0w, b0w, b1w, v_i)
    int_ef = _antider_e(v0f, b0f, b1f, v_f) - _antider_e(v0f, b0f, b1f, v_i)
    int_ew2 = _antider_e2(v0w, b0w, b1w, v_f) - _antider_e2(v0w, b0w, b1w, v_i)
    int_ef2 = _antider_e2(v0f, b0f, b1f, v_f) - _antider_e2(v0f, b0f, b1f, v_i)

    mean_ew = int_ew / delta_v
    var_w = int_ew2 - 2 * mean_ew * int_ew + delta_v * mean_ew**2
    var_f = int_ef2 - 2 * mean_ew * int_ef + delta_v * mean_ew**2

    # The antiderivatives cancel to high order for near-identical curves, so rounding
    # can make the ratio slightly negative.
    return float(np.sqrt(abs(int_diff2 / np.sqrt(var_w * var_f))))


def relative_difference(tested: float, reference: float) -> float:
    """
    Get the relative difference used by ACWF, 2 (w - f) / (w + f).

    Parameters
    ----------
    tested
        Tested value.
    reference
        Reference value.

    Returns
    -------
    float
        Relative difference (dimensionless).
    """
    return 2.0 * (tested - reference) / (tested + reference)


def nu(v0w: float, b0w: float, b1w: float, v0f: float, b0f: float, b1f: float) -> float:
    """
    Get the ν measure between two sets of Birch-Murnaghan parameters.

    ν is the weighted length of the vector of relative differences in V0, B0 and B1,
    with weights 1, 1/20 and 1/400, multiplied by 100 (Bosoni et al., 2024).

    Parameters
    ----------
    v0w, b0w, b1w
        Parameters of the tested curve.
    v0f, b0f, b1f
        Parameters of the reference curve.

    Returns
    -------
    float
        Value of ν (dimensionless).
    """
    return float(
        NU_PREFACTOR
        * np.sqrt(
            relative_difference(v0w, v0f) ** 2
            + (NU_WEIGHT_B0 * relative_difference(b0w, b0f)) ** 2
            + (NU_WEIGHT_B1 * relative_difference(b1w, b1f)) ** 2
        )
    )
