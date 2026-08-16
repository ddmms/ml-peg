"""Regression checks for analysis utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from ml_peg.analysis.utils.utils import block_estimate, correlator, maze


def test_correlator():
    """Test correlator helper function."""
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([2.0, 4.0, 6.0])
    expected = np.mean(x * y) - np.mean(x) * np.mean(y)
    assert correlator(x, y) == pytest.approx(expected)


def test_block_estimate_mean():
    """Test block error and mean estimate."""
    values = np.arange(8.0)

    mean, stderr = block_estimate(values, block_size=2)
    block_means = np.array([0.5, 2.5, 4.5, 6.5])

    assert mean == pytest.approx(np.mean(block_means))
    assert stderr == pytest.approx(
        np.std(block_means, ddof=1) / np.sqrt(len(block_means))
    )


def test_block_estimate_custom_estimator():
    """Test block error and mean estimate with custom estimator."""
    values = np.arange(8.0)

    mean, stderr = block_estimate(values, block_size=2, estimator=np.var)

    block_values = np.array(
        [
            np.var([0.0, 1.0]),
            np.var([2.0, 3.0]),
            np.var([4.0, 5.0]),
            np.var([6.0, 7.0]),
        ]
    )

    assert mean == pytest.approx(np.mean(block_values))
    assert stderr == pytest.approx(
        np.std(block_values, ddof=1) / np.sqrt(len(block_values))
    )


def test_block_estimate_multiple_series():
    """Test block error and mean estimate for multiple series."""
    x = np.arange(8.0)
    y = 2 * x

    mean, stderr = block_estimate(x, y, block_size=2, estimator=correlator)

    block_values = np.array(
        [
            correlator(x[0:2], y[0:2]),
            correlator(x[2:4], y[2:4]),
            correlator(x[4:6], y[4:6]),
            correlator(x[6:8], y[6:8]),
        ]
    )

    assert mean == pytest.approx(np.mean(block_values))
    assert stderr == pytest.approx(
        np.std(block_values, ddof=1) / np.sqrt(len(block_values))
    )


def test_block_estimate_requires_two_blocks():
    """Test block error and mean estimate minumum requirement of two blocks."""
    mean, stderr = block_estimate(np.arange(4.0), block_size=4)
    assert np.isnan(mean)
    assert np.isnan(stderr)


def test_block_estimate_requires_equal_lengths():
    """Test block error and mean estimate requirement of homogeneous data."""
    with pytest.raises(ValueError):
        block_estimate(np.arange(8.0), np.arange(7.0), block_size=2)


def test_maze():
    """Test Mean Absolute Zeta Error."""
    ref = [1.0, 2.0, 3.0]
    prediction = [1.1, 1.8, 3.3]
    stderr = [0.1, 0.2, 0.3]

    assert maze(ref, prediction, stderr) == pytest.approx(1.0)


def test_maze_nan_prediction():
    """Test Mean Absolute Zeta Error handling NaNs."""
    assert np.isnan(maze([1.0, 2.0], [1.0, np.nan], [0.1, 0.1]))


def test_maze_zero_stderr():
    """Test Mean Absolute Zeta Error handling zero error."""
    assert np.isnan(maze([1.0, 2.0], [1.1, 2.1], [0.1, 0.0]))
