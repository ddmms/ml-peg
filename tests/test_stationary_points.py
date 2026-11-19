"""Unit tests for the Landscape17 stationary points Hessian metrics."""

from __future__ import annotations

import numpy as np
import pytest

from ml_peg.analysis.landscapes.stationary_points.hessian_metrics import (
    ratio_agreement,
)


@pytest.mark.parametrize("ratio", [2.0, 0.5])
def test_too_sharp_scores_like_too_flat_by_the_same_factor(ratio: float) -> None:
    """
    A barrier top twice too sharp is as wrong as one twice too flat.

    Parameters
    ----------
    ratio
        Model curvature over DFT's: twice and half.
    """
    assert ratio_agreement(ratio) == pytest.approx(0.5)


def test_exact_agreement_scores_one() -> None:
    """A ratio of 1 is full agreement."""
    assert ratio_agreement(1.0) == pytest.approx(1.0)


def test_missing_ratio_stays_missing() -> None:
    """A saddle the model has no barrier at stays out of the summary."""
    assert np.isnan(ratio_agreement(np.nan))
