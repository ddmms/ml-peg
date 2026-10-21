"""
Uncertainty estimates for time-averaged MD observables.

The statistical inefficiency is estimated from the autocorrelation function, truncated
at its first non-positive value after ``minimum_samples`` lags, as suggested by Swope
and used in Chodera et al. [1]_. The autocovariance is calculated by FFT, adapted from
the ``"positive"`` sequence estimator in ``red.variance.get_variance_initial_sequence``
(https://github.com/fjclark/red, MIT licence). Similar functionality is used in OpenFF
Evaluator (``openff/evaluator/utils/timeseries.py``, MIT licence), itself based on
pymbar. The standard error, effective sample size and t-based confidence interval
follow ``red.confidence_intervals.get_conf_int_init_seq``. Truncating at the first
negative value is a fairly naive approach and can underestimate the uncertainty more
than some more sophisticated methods, but is simple and sufficient when care has been
taken to simulate systems for many times the correlation time of the observable of
interest. When this isn't the case, all methods will underestimate the uncertainty
anyway (though using independent replicas can help).

These estimates are only valid for the time average of a single scalar observable
sampled from a stationary trajectory (e.g. the density from an NPT run, after
discarding equilibration).

References
----------
.. [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the
   weighted histogram analysis method for the analysis of simulated and parallel
   tempering simulations. JCTC 3(1):26-41, 2007.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
import warnings

import numpy as np
from scipy.fft import next_fast_len
from scipy.stats import t as t_dist

ACF_METHOD = "autocorrelation (Swope, Chodera et al.)"


@dataclass(frozen=True)
class TimeSeriesEstimate:
    """
    Mean of a time series and its uncertainty.

    Attributes
    ----------
    mean
        Mean over all samples.
    sem
        Standard error of the mean, accounting for autocorrelation. NaN if it could not
        be estimated.
    n_eff
        Effective number of uncorrelated samples.
    ci_95
        Half-width of the 95 % confidence interval of the mean. NaN if it could not be
        estimated.
    method
        Name of the method used to estimate the uncertainty.
    """

    mean: float
    sem: float
    n_eff: float
    ci_95: float
    method: str = ACF_METHOD


@dataclass(frozen=True)
class MetricEstimate:
    """
    Value of a metric and its confidence interval.

    Attributes
    ----------
    value
        Metric calculated from the mean of each prediction.
    ci_low, ci_high
        Bounds of the confidence interval from simulation sampling noise alone. `None`
        if not available.
    ci_low_ref, ci_high_ref
        Bounds of the confidence interval including reference uncertainties. `None` if
        no reference uncertainties were given.
    n_eff_min
        Smallest effective sample size of the predictions. `None` if not available.
    method
        Name of the method used to estimate the uncertainty of each prediction.
    note
        Optional note to display alongside the confidence interval.
    """

    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    ci_low_ref: float | None = None
    ci_high_ref: float | None = None
    n_eff_min: float | None = None
    method: str | None = None
    note: str | None = None


def get_autocovariance(time_series: Sequence[float]) -> np.ndarray:
    """
    Calculate the autocovariance function of a time series by FFT.

    Uses the biased estimator, normalised by the number of samples at every lag, as
    in ``red`` and ``statsmodels.tsa.stattools.acovf(adjusted=False)``. The series is
    zero-padded to avoid circular correlation.

    Parameters
    ----------
    time_series
        Time series to analyse, with shape (n_samples,).

    Returns
    -------
    np.ndarray
        Autocovariance at lags 0 to n_samples - 1.
    """
    shifted = np.asarray(time_series, dtype=float)
    shifted = shifted - shifted.mean()
    n_samples = len(shifted)

    # Find the padding to avoid wrap-around correlation, and to make the FFT fast.
    n_fft = next_fast_len(2 * n_samples - 1, real=True)

    # Wiener-Khinchin relation: the autocovariance is the inverse FFT of the power
    # spectrum |X|^2.
    spectrum = np.fft.rfft(shifted, n_fft)
    autocov = np.fft.irfft(spectrum.real**2 + spectrum.imag**2, n_fft)[:n_samples]
    return autocov / n_samples


def get_statistical_inefficiency(
    time_series: Sequence[float], minimum_samples: int = 3
) -> float:
    """
    Calculate the statistical inefficiency of a time series.

    The autocorrelation function is summed up to, but not including, its first
    non-positive value after ``minimum_samples`` lags, as suggested by Swope
    and used in Chodera et al. [1]_. This gives the same result as OpenFF
    Evaluator's ``_statistical_inefficiency``, but calculates the autocorrelation
    function by FFT, so it's faster for long time series with long correlation
    times (for short time series with short correlation times, both methods
    will be fast).

    Parameters
    ----------
    time_series
        Time series to analyse, with shape (n_samples,).
    minimum_samples
        Minimum number of lags to include before truncating the autocorrelation
        function. Default is 3.

    Returns
    -------
    float
        Statistical inefficiency, g >= 1. Returns 1 if the time series has fewer
        than two samples or zero variance (with a warning), as there are no
        fluctuations to be correlated.

    Raises
    ------
    ValueError
        If the time series is not one-dimensional or contains non-finite values.

    References
    ----------
    .. [1] J. D. Chodera, W. C. Swope, J. W. Pitera, C. Seok, and K. A. Dill. Use of the
       weighted histogram analysis method for the analysis of simulated and parallel
       tempering simulations. JCTC 3(1):26-41, 2007.
    """
    data = np.asarray(time_series, dtype=float)
    if data.ndim != 1:
        raise ValueError("Time series must be one-dimensional.")
    if not np.all(np.isfinite(data)):
        raise ValueError("Time series contains non-finite values.")
    if len(data) < 2:
        return 1.0

    # Check the range rather than the variance, which may be non-zero from rounding
    if np.ptp(data) == 0:
        warnings.warn(
            "Time series has zero variance, returning g = 1.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 1.0
    autocov = get_autocovariance(data)

    # Lags 1 to n_samples - 2, matching the OpenFF loop
    acf = autocov[1:-1] / autocov[0]
    lags = np.arange(1, len(acf) + 1)

    # Keep the ACF up to (excluding) the first non-positive value after minimum_samples
    is_cutoff = (acf <= 0.0) & (lags > minimum_samples)
    if is_cutoff.any():
        first_cutoff = np.argmax(is_cutoff)  # Index of the first True
        acf = acf[:first_cutoff]

    return max(1.0 + 2.0 * float(acf.sum()), 1.0)


def estimate_mean(
    time_series: Sequence[float],
    alpha: float = 0.05,
    min_n_eff: float = 50,
    label: str | None = None,
) -> TimeSeriesEstimate:
    """
    Estimate the mean of a time series and its confidence interval.

    All samples are used to calculate the mean (no subsampling). The standard error
    accounts for autocorrelation through the statistical inefficiency.

    Parameters
    ----------
    time_series
        Time series to analyse, with shape (n_samples,). Equilibration should already
        have been discarded.
    alpha
        Two-tailed significance level of the confidence interval. Default is 0.05.
    min_n_eff
        Warn if the effective sample size is below this. Default is 50.
    label
        Label to identify the time series in warnings.

    Returns
    -------
    TimeSeriesEstimate
        Mean, standard error, effective sample size and confidence interval
        half-width. The standard error and confidence interval are NaN if there are
        fewer than two effective samples. All values are NaN, with zero effective
        samples, if the time series is empty or contains non-finite values.
    """
    data = np.asarray(time_series, dtype=float)
    if data.ndim != 1:
        raise ValueError("Time series must be one-dimensional.")
    warning_name = f" for {label}" if label else ""
    n_samples = len(data)

    if n_samples == 0 or not np.all(np.isfinite(data)):
        warnings.warn(
            f"No samples or non-finite values{warning_name}. "
            "Mean and uncertainty cannot be estimated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return TimeSeriesEstimate(mean=np.nan, sem=np.nan, n_eff=0.0, ci_95=np.nan)

    mean = float(data.mean())
    n_eff = n_samples / get_statistical_inefficiency(data)

    if n_eff < 2:
        warnings.warn(
            f"Insufficient sampling{warning_name}: {n_eff:.1f} effective samples. "
            "Uncertainty cannot be estimated.",
            RuntimeWarning,
            stacklevel=2,
        )
        return TimeSeriesEstimate(mean=mean, sem=np.nan, n_eff=n_eff, ci_95=np.nan)

    if n_eff < min_n_eff:
        warnings.warn(
            f"Low effective sample size{warning_name}: {n_eff:.1f}. "
            "Confidence intervals may be unreliable.",
            RuntimeWarning,
            stacklevel=2,
        )

    # Correct for bias due to unknown mean with n_eff - 1,
    # though bias is small for reasonable n_eff.
    sem = float(np.sqrt(data.var() / (n_eff - 1)))
    t_val = float(t_dist.ppf(1 - alpha / 2, n_eff - 1))
    ci_95 = t_val * sem

    return TimeSeriesEstimate(mean=mean, sem=sem, n_eff=n_eff, ci_95=ci_95)


def _get_percentile_interval(
    samples: np.ndarray, alpha: float
) -> tuple[float | None, float | None]:
    """
    Get percentile interval of bootstrap samples.

    Parameters
    ----------
    samples
        Bootstrapped values of the metric.
    alpha
        Two-tailed significance level.

    Returns
    -------
    tuple[float | None, float | None]
        Lower and upper bounds, or `None` if any sample is not finite.
    """
    if not np.all(np.isfinite(samples)):
        return None, None
    low, high = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(low), float(high)


def bootstrap_metric(
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    ref: Sequence[float],
    pred: Sequence[TimeSeriesEstimate | float | None],
    ref_sigma_1sd: Sequence[float] | None = None,
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
    note: str | None = None,
) -> MetricEstimate | None:
    """
    Estimate a confidence interval for a metric with a parametric bootstrap.

    Each prediction is resampled from a normal distribution (we rely on the central
    limit theorem here, assuming reasonable n_eff) with its mean and standard error,
    and the metric is recalculated for each draw. The interval is the 95 % range of
    the metric under sampling noise.

    The seed is fixed so that every model uses the same random draws, keeping the
    results reproducible and making comparisons between models fairer.

    Parameters
    ----------
    statistic_fn
        Function of (reference, prediction) arrays, each with shape (n_systems,),
        returning the metric, e.g. ``lambda ref, pred: np.mean(np.abs(pred - ref))``.
    ref
        Reference values, with shape (n_systems,).
    pred
        Predictions for each system. Plain floats are treated as having no known
        uncertainty, in which case no confidence interval is returned. Likewise if
        the uncertainty of any prediction could not be estimated.
    ref_sigma_1sd
        Standard uncertainty (one standard deviation) of each reference value. If
        given, a second interval that also includes these is returned.
    n_boot
        Number of bootstrap draws. Default is 10,000.
    alpha
        Two-tailed significance level. Default is 0.05.
    seed
        Random seed. Default is 0.
    note
        Optional note to display alongside the confidence interval.

    Returns
    -------
    MetricEstimate | None
        Metric value and confidence intervals, or `None` if any prediction is `None`.
    """
    # Missing predictions: skip the metric, as for an incomplete subset
    if any(p is None for p in pred):
        return None

    # Reported value: the metric of the time-averaged predictions
    ref = np.asarray(ref, dtype=float)
    means = np.array([p.mean if isinstance(p, TimeSeriesEstimate) else p for p in pred])
    value = float(statistic_fn(ref, means))

    # Standard errors, NaN where a prediction has no time series
    estimates = [p for p in pred if isinstance(p, TimeSeriesEstimate)]
    method = estimates[0].method if estimates else None
    sems = np.array(
        [p.sem if isinstance(p, TimeSeriesEstimate) else np.nan for p in pred]
    )

    # Some predictions are plain floats (e.g. precomputed values) with no standard
    # error. Treating them as exact would understate the uncertainty, so give no CI
    if len(estimates) < len(pred):
        return MetricEstimate(value=value, method=method, note="no time series")
    n_eff_min = float(min(p.n_eff for p in estimates))

    # The bootstrap cannot run if the metric is undefined (e.g. missing or
    # non-finite data), or if any standard error is NaN (fewer than two
    # effective samples)
    n_no_sem = int(np.sum(~np.isfinite(sems)))
    if not np.isfinite(value) or n_no_sem:
        if not np.isfinite(value):
            reason = "metric is not finite (missing or invalid data)"
        else:
            reason = (
                f"fewer than 2 effective samples in {n_no_sem} of {len(sems)} "
                "systems; run longer"
            )
        return MetricEstimate(
            value=value,
            n_eff_min=n_eff_min,
            method=method,
            note=f"no CI, {reason}",
        )

    # Each row of pred_draws, with shape (n_boot, n_systems), is a plausible rerun of
    # every simulation: each mean is perturbed independently by its standard error,
    # as time averages are normal by the central limit theorem (assuming reasonable
    # n_eff). The fixed seed gives every model the same draws.
    rng = np.random.default_rng(seed)
    pred_draws = means + sems * rng.standard_normal((n_boot, len(means)))

    # Recompute the metric for each rerun, and take the percentile interval
    samples = np.array([statistic_fn(ref, draw) for draw in pred_draws])
    ci_low, ci_high = _get_percentile_interval(samples, alpha)

    # Optionally also perturb the reference values by their uncertainties, reusing
    # the same prediction draws. Kept separate as the reference uncertainty is
    # shared by all models, so could hide differences between them
    ci_low_ref = ci_high_ref = None
    if ref_sigma_1sd is not None:
        ref_sigma = np.asarray(ref_sigma_1sd, dtype=float)
        ref_draws = ref + ref_sigma * rng.standard_normal((n_boot, len(ref)))
        samples_ref = np.array(
            [statistic_fn(r, p) for r, p in zip(ref_draws, pred_draws, strict=True)]
        )
        ci_low_ref, ci_high_ref = _get_percentile_interval(samples_ref, alpha)

    return MetricEstimate(
        value=value,
        ci_low=ci_low,
        ci_high=ci_high,
        ci_low_ref=ci_low_ref,
        ci_high_ref=ci_high_ref,
        n_eff_min=n_eff_min,
        method=method,
        note=note,
    )
