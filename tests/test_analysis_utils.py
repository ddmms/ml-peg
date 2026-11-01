"""Test analysis utility functions."""

from __future__ import annotations

import numpy as np
import pytest

from ml_peg.analysis.utils.utils import build_dispersion_name_map, mape
from ml_peg.models.get_models import get_model_names


def test_dispersion_name_map_all_models():
    """Test display names are built for all models, not just those being analysed."""
    all_models = get_model_names()
    name_map = build_dispersion_name_map()

    # Display names must be available for models outside the current analysis run
    assert name_map == build_dispersion_name_map(all_models)
    assert set(name_map) <= set(all_models)
    assert name_map

    subset = build_dispersion_name_map(all_models[:1])
    assert set(subset) <= set(name_map)
    assert all(name_map[model].startswith(model) for model in name_map)


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_mape_non_finite_is_nan(bad):
    """Test a non-finite reference or prediction gives NaN rather than raising."""
    assert np.isnan(mape([1.0, 2.0], [1.0, bad]))
    assert np.isnan(mape([1.0, bad], [1.0, 2.0]))
