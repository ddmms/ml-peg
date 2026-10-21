"""Test uncertainty estimates for time series."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from ml_peg.analysis.utils.timeseries import (
    TimeSeriesEstimate,
    bootstrap_metric,
    estimate_mean,
    statistical_inefficiency,
)
from ml_peg.analysis.utils.utils import mae


def ar_series(n_samples: int, phis: list[float], seed: int = 0) -> np.ndarray:
    """
    Generate an autoregressive time series with unit noise.

    Parameters
    ----------
    n_samples
        Number of samples.
    phis
        Autoregressive coefficients.
    seed
        Random seed.

    Returns
    -------
    np.ndarray
        Time series, after discarding a burn-in period.
    """
    rng = np.random.default_rng(seed)
    burn_in = 1000
    noise = rng.standard_normal(n_samples + burn_in)
    series = np.zeros(n_samples + burn_in)
    for i in range(len(phis), n_samples + burn_in):
        series[i] = noise[i] + sum(
            phi * series[i - lag - 1] for lag, phi in enumerate(phis)
        )
    return series[burn_in:]


def openff_statistical_inefficiency(
    time_series: np.ndarray, minimum_samples: int = 3
) -> float:
    """
    Calculate the statistical inefficiency with the OpenFF Evaluator loop.

    Reference implementation, adapted from OpenFF Evaluator's
    ``_statistical_inefficiency`` for one-dimensional data.

    Parameters
    ----------
    time_series
        Time series to analyse.
    minimum_samples
        Minimum number of lags to include before truncating.

    Returns
    -------
    float
        Statistical inefficiency.
    """
    shifted = time_series - time_series.mean()
    n_samples = len(shifted)
    sigma_squared = np.mean(shifted * shifted)

    statistical_ineff = 1.0
    for lag in range(1, n_samples - 1):
        acf = np.dot(shifted[: n_samples - lag], shifted[lag:]) / (
            (n_samples - lag) * sigma_squared
        )
        if acf <= 0.0 and lag > minimum_samples:
            break
        statistical_ineff += 2.0 * acf * (1.0 - lag / n_samples)
    return max(statistical_ineff, 1.0)


@pytest.mark.parametrize(
    "n_samples, phis",
    [
        (5, [0.5]),
        (97, [0.9]),
        (1000, [-0.5]),
        (4999, [0.95]),
        (10_000, [1.6, -0.8]),
    ],
)
def test_statistical_inefficiency_matches_openff(n_samples, phis):
    """Test the FFT estimate matches the OpenFF Evaluator loop."""
    series = ar_series(n_samples, phis)
    assert statistical_inefficiency(series) == pytest.approx(
        openff_statistical_inefficiency(series), rel=1e-8
    )


def test_statistical_inefficiency_invalid():
    """Test errors are raised for invalid time series."""
    with pytest.raises(ValueError, match="non-finite"):
        statistical_inefficiency([1.0, np.nan, 2.0])
    with pytest.raises(ValueError, match="one-dimensional"):
        statistical_inefficiency(np.ones((2, 3)))


def test_statistical_inefficiency_ar1():
    """Test the statistical inefficiency of an AR(1) process is recovered."""
    phi = 0.9
    expected = (1 + phi) / (1 - phi)
    g = statistical_inefficiency(ar_series(100_000, [phi]))
    assert g == pytest.approx(expected, rel=0.1)


@pytest.mark.parametrize("value", [1.0, 0.1, 1003.7])
def test_statistical_inefficiency_constant(value):
    """Test a constant time series warns and gives g = 1."""
    with pytest.warns(RuntimeWarning, match="zero variance"):
        assert statistical_inefficiency(np.full(100, value)) == 1.0


def test_statistical_inefficiency_single_sample():
    """Test a single sample gives g = 1."""
    assert statistical_inefficiency([1.0]) == 1.0


def test_estimate_mean_white_noise():
    """Test white noise estimates match the uncorrelated standard error."""
    series = np.random.default_rng(0).normal(size=10_000)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        estimate = estimate_mean(series)

    # No subsampling
    assert estimate.mean == pytest.approx(np.mean(series))
    assert estimate.n_eff == pytest.approx(len(series), rel=0.1)
    expected_sem = np.std(series, ddof=1) / np.sqrt(len(series))
    assert estimate.sem == pytest.approx(expected_sem, rel=0.1)
    assert estimate.ci_95 == pytest.approx(1.96 * estimate.sem, rel=0.01)


def test_estimate_mean_low_samples_warns():
    """Test a warning is raised for fewer than 50 effective samples."""
    series = ar_series(1000, [0.95])
    with pytest.warns(RuntimeWarning, match="Low effective sample size for test"):
        estimate = estimate_mean(series, label="test")
    assert estimate.n_eff < 50
    assert np.isfinite(estimate.ci_95)


def test_estimate_mean_insufficient_samples():
    """Test uncertainties are NaN with fewer than two effective samples."""
    with pytest.warns(RuntimeWarning, match="Insufficient sampling"):
        estimate = estimate_mean([1.0])
    assert estimate.mean == pytest.approx(1.0)
    assert np.isnan(estimate.sem)
    assert np.isnan(estimate.ci_95)


def test_estimate_mean_constant():
    """Test a constant time series has zero uncertainty."""
    with pytest.warns(RuntimeWarning, match="zero variance"):
        estimate = estimate_mean(np.ones(100))
    assert estimate.mean == pytest.approx(1.0)
    assert estimate.n_eff == 100
    assert estimate.sem == 0.0
    assert estimate.ci_95 == 0.0


@pytest.mark.parametrize("series", [[], [1.0, np.nan, 2.0], [1.0, np.inf]])
def test_estimate_mean_invalid(series):
    """Test empty or non-finite time series give NaN estimates."""
    with pytest.warns(RuntimeWarning, match="non-finite"):
        estimate = estimate_mean(series)
    assert np.isnan(estimate.mean)
    assert np.isnan(estimate.sem)
    assert estimate.n_eff == 0


def test_estimate_mean_coverage():
    """Test 95 % confidence intervals contain the true mean ~95 % of the time."""
    n_repeats = 200
    covered = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for seed in range(n_repeats):
            estimate = estimate_mean(ar_series(2000, [0.7], seed=seed))
            # True mean is zero so no need to subtract it
            covered += abs(estimate.mean) <= estimate.ci_95
    assert 0.88 <= covered / n_repeats <= 0.98


def estimate(mean: float, sem: float) -> TimeSeriesEstimate:
    """
    Create a time series estimate.

    Parameters
    ----------
    mean
        Mean of the time series.
    sem
        Standard error of the mean.

    Returns
    -------
    TimeSeriesEstimate
        Estimate with 100 effective samples.
    """
    return TimeSeriesEstimate(mean=mean, sem=sem, n_eff=100, ci_95=1.98 * sem)


def test_bootstrap_metric_no_noise():
    """Test the interval collapses to the value without uncertainties."""
    ref = [1.0, 2.0, 3.0]
    pred = [estimate(1.5, 0.0), estimate(2.0, 0.0), estimate(2.0, 0.0)]
    result = bootstrap_metric(mae, ref, pred)

    assert result.value == pytest.approx(0.5)
    assert result.ci_low == pytest.approx(0.5)
    assert result.ci_high == pytest.approx(0.5)
    assert result.ci_low_ref is None
    assert result.n_eff_min == 100


def test_bootstrap_metric_reference_uncertainty():
    """Test reference uncertainties widen a separate interval."""
    ref = [1.0, 2.0, 3.0]
    pred = [estimate(1.5, 0.05), estimate(2.5, 0.05), estimate(2.0, 0.05)]
    result = bootstrap_metric(mae, ref, pred, ref_sigma_1sd=[0.2, 0.2, 0.2])

    assert result.ci_low < result.value < result.ci_high
    width = result.ci_high - result.ci_low
    assert result.ci_high_ref - result.ci_low_ref > width


def test_bootstrap_metric_errors_near_zero():
    """Test metrics with errors comparable to the uncertainty can be bootstrapped."""
    ref = [1.0, 2.0]
    pred = [estimate(1.0, 0.1), estimate(2.0, 0.1)]
    result = bootstrap_metric(mae, ref, pred)

    # Noise biases the MAE upwards, so the value is below the interval
    assert result.value == pytest.approx(0.0)
    assert 0.0 < result.ci_low < result.ci_high


def test_bootstrap_metric_no_time_series():
    """Test no interval is given if any prediction has no uncertainty."""
    result = bootstrap_metric(mae, [1.0, 2.0], [estimate(1.5, 0.1), 2.0])

    assert result.value == pytest.approx(0.25)
    assert result.ci_low is None
    assert result.ci_high is None
    assert result.note == "no time series"


def test_bootstrap_metric_insufficient_sampling():
    """Test no interval is given if any uncertainty could not be estimated."""
    result = bootstrap_metric(
        mae, [1.0, 2.0], [estimate(1.5, 0.1), estimate(2.0, np.nan)]
    )

    assert result.ci_low is None
    assert result.note == (
        "no CI, fewer than 2 effective samples in 1 of 2 systems; run longer"
    )


def test_bootstrap_metric_non_finite():
    """Test no interval is given if the metric is not finite."""
    result = bootstrap_metric(
        mae, [1.0, 2.0], [estimate(1.5, 0.1), estimate(np.nan, 0.1)]
    )

    assert result.ci_low is None
    assert result.note == "no CI, metric is not finite (missing or invalid data)"


def test_bootstrap_metric_missing_prediction():
    """Test None is returned if any prediction is missing."""
    assert bootstrap_metric(mae, [1.0, 2.0], [estimate(1.5, 0.1), None]) is None
