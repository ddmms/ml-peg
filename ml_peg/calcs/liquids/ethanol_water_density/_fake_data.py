"""Generate fake data for debugging density-curve metrics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ml_peg.analysis.liquids.ethanol_water_density.io_tools import read_ref_curve


@dataclass(frozen=True)
class FakeCurveParams:
    """
    Parameters controlling synthetic curve perturbations.

    Attributes
    ----------
    severity : float
        Global scaling of all enabled perturbation components.
    bias : float
        Additive offset magnitude.
    scale : float
        Multiplicative distortion magnitude.
    tilt : float
        Linear-in-composition additive distortion magnitude.
    warp : float
        Smooth nonlinear distortion magnitude.
    noise_sigma : float
        Standard deviation for additive white noise.
    corr_len : float
        Correlation length for correlated noise component.
    bump_amp : float
        Localized bump amplitude.
    bump_center : float
        Composition location for localized bump.
    bump_width : float
        Width of localized bump in composition units.
    """

    # Master knob: 0 -> perfect match, 1 -> very poor
    severity: float = 0.0

    # Individual error components (interpreted as "max at severity=1")
    bias: float = 0.0  # additive offset in y-units
    scale: float = 0.0  # multiplicative: y *= (1 + scale*...)
    tilt: float = 0.0  # linear-in-x additive distortion
    warp: float = 0.0  # smooth nonlinear additive distortion

    noise_sigma: float = 0.0  # iid gaussian noise in y-units
    corr_len: float = 0.0  # if >0, adds correlated noise along x

    bump_amp: float = 0.0  # amplitude of local bump(s)
    bump_center: float = 0.5  # x location of bump
    bump_width: float = 0.08  # bump width (in x units)


def _smooth_random_field(
    xs: np.ndarray, corr_len: float, rng: np.random.Generator
) -> np.ndarray:
    """
    Create a smooth random field along the composition grid.

    Parameters
    ----------
    xs : numpy.ndarray
        Composition grid.
    corr_len : float
        Correlation length in composition units.
    rng : numpy.random.Generator
        Random number generator.

    Returns
    -------
    numpy.ndarray
        Approximately zero-mean, unit-variance correlated field.
    """
    if corr_len <= 0:
        return np.zeros_like(xs)

    dx = xs[:, None] - xs[None, :]
    k = np.exp(-0.5 * (dx / corr_len) ** 2)
    # sample correlated normal: K^(1/2) z via cholesky (add jitter for stability)
    z = rng.standard_normal(len(xs))
    field = np.linalg.cholesky(k + 1e-12 * np.eye(len(xs))) @ z
    field = field - field.mean()
    return field / (field.std() + 1e-12)


def make_fake_curve_from_ref(
    xs_ref: list[float],
    ys_ref: list[float],
    *,
    params: FakeCurveParams,
    seed: int | None = 0,
    clip: tuple[float | None, float | None] = (None, None),
) -> tuple[list[float], list[float]]:
    """
    Build a synthetic curve from reference data and perturbation settings.

    Parameters
    ----------
    xs_ref : list[float]
        Reference x-grid.
    ys_ref : list[float]
        Reference y-values.
    params : FakeCurveParams
        Perturbation parameters controlling distortion type and magnitude.
    seed : int | None, optional
        Seed for deterministic random perturbations.
    clip : tuple[float | None, float | None], optional
        Optional ``(min, max)`` clipping bounds for generated y values.

    Returns
    -------
    tuple[list[float], list[float]]
        Synthetic x and y arrays.
    """
    sev = float(np.clip(params.severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)

    xs = np.asarray(xs_ref, dtype=float)
    y = np.asarray(ys_ref, dtype=float).copy()

    # Normalize x into [-1, 1] for stable “tilt/warp” magnitudes
    x01 = (xs - xs.min()) / (xs.max() - xs.min() + 1e-12)
    xpm = 2.0 * x01 - 1.0

    # Small randomization so multiple models with same severity aren’t identical
    # (but still deterministic for a given seed).
    def jitter():
        """
        Return a small random multiplier around unity.

        Returns
        -------
        float
            Random scale factor in a narrow range near 1.
        """
        return 0.85 + 0.30 * rng.random()

    # 1) multiplicative scale error
    if params.scale != 0.0 and sev > 0:
        y *= 1.0 + (params.scale * sev * jitter())

    # 2) additive bias
    if params.bias != 0.0 and sev > 0:
        y += params.bias * sev * jitter()

    # 3) linear tilt (additive)
    if params.tilt != 0.0 and sev > 0:
        y += (params.tilt * sev * jitter()) * xpm

    # 4) smooth nonlinear warp (additive): use a low-order smooth basis
    if params.warp != 0.0 and sev > 0:
        # cubic-ish shape distortion with zero mean
        w = xpm**3 - xpm * np.mean(xpm**2)
        w = w - w.mean()
        w = w / (np.std(w) + 1e-12)
        y += (params.warp * sev * jitter()) * w

    # 5) local bump to simulate specific composition failure
    if params.bump_amp != 0.0 and sev > 0:
        bump = np.exp(
            -0.5 * ((xs - params.bump_center) / (params.bump_width + 1e-12)) ** 2
        )
        bump = bump / (bump.max() + 1e-12)
        y += (params.bump_amp * sev * jitter()) * bump

    # 6) noise: iid + optional correlated component
    if params.noise_sigma != 0.0 and sev > 0:
        y += rng.normal(0.0, params.noise_sigma * sev, size=len(xs))

    if params.corr_len > 0.0 and params.noise_sigma != 0.0 and sev > 0:
        field = _smooth_random_field(xs, params.corr_len, rng)
        y += field * (params.noise_sigma * sev * 0.8)

    lo, hi = clip
    if lo is not None:
        y = np.maximum(y, lo)
    if hi is not None:
        y = np.minimum(y, hi)

    return xs, y


def make_fake_curve(
    kind: str | int,
    seed: int | None = 0,
) -> tuple[list[float], list[float]]:
    """
    Generate a synthetic density curve with predefined quality level.

    Parameters
    ----------
    kind : str | int
        Quality label or index (perfect/good/medium/bad or 0-3).
    seed : int | None, optional
        Seed for deterministic randomness.

    Returns
    -------
    tuple[list[float], list[float]]
        Synthetic x and y arrays.
    """
    xs_ref, ys_ref = read_ref_curve()

    kind = kind.lower().strip() if isinstance(kind, str) else kind
    if kind == "perfect" or kind == 0:
        params = FakeCurveParams(severity=0.0)
    elif kind == "good" or kind == 1:
        params = FakeCurveParams(
            severity=0.15,
            bias=0.0,
            scale=0.01,
            tilt=0.003,
            warp=0.002,
            noise_sigma=0.001,
            corr_len=0.12,
            bump_amp=0.0,
        )
    elif kind in {"medium", "ok"} or kind == 2:
        params = FakeCurveParams(
            severity=0.45,
            bias=0.0,
            scale=0.03,
            tilt=0.01,
            warp=0.01,
            noise_sigma=0.004,
            corr_len=0.15,
            bump_amp=0.01,
            bump_center=0.6,
            bump_width=0.10,
        )
    elif kind == "bad" or kind == 3:
        params = FakeCurveParams(
            severity=0.85,
            bias=0.02,
            scale=0.06,
            tilt=0.03,
            warp=0.04,
            noise_sigma=0.01,
            corr_len=0.18,
            bump_amp=0.05,
            bump_center=0.4,
            bump_width=0.08,
        )
    else:
        raise ValueError(f"Unknown kind={kind!r} (use perfect/good/medium/bad)")

    return make_fake_curve_from_ref(xs_ref, ys_ref, params=params, seed=seed)


def make_fake_density_timeseries(
    rho_eq: float,
    n_steps: int,
    *,
    seed: int,
    start_offset: float = 0.02,  # initial deviation from eq
    tau: float = 0.15,  # relaxation rate (bigger -> faster)
    noise_sigma: float = 0.001,  # per-step noise
) -> list[float]:
    """
    Generate a fake density time series with relaxation dynamics.

    Parameters
    ----------
    rho_eq : float
        Equilibrium density target.
    n_steps : int
        Number of time steps to generate.
    seed : int
        Seed for deterministic random noise.
    start_offset : float, optional
        Magnitude of initial offset from equilibrium.
    tau : float, optional
        Relaxation factor per step.
    noise_sigma : float, optional
        Standard deviation of additive Gaussian noise per step.

    Returns
    -------
    list[float]
        Generated density values.
    """
    rng = np.random.default_rng(seed)
    rho0 = rho_eq + start_offset * (2 * rng.random() - 1)

    series = []
    rho = rho0
    for _ in range(n_steps):
        # exponential-ish relaxation to rho_eq
        rho += tau * (rho_eq - rho)
        # add noise
        rho += rng.normal(0.0, noise_sigma)
        series.append(float(rho))
    return series
