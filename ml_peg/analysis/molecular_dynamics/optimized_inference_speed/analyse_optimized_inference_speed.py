"""Analyse optimised inference speed benchmark."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ml_peg.analysis.molecular_dynamics.optimized_inference_speed.aggregate import (
    analyse_campaign,
)
from ml_peg.calcs import CALCS_ROOT

DEFAULT_CAMPAIGN = (
    CALCS_ROOT / "molecular_dynamics" / "optimized_inference_speed" / "outputs"
)


@pytest.fixture
def campaign() -> Path:
    """
    Get the campaign directory to analyse.

    Returns
    -------
    Path
        ``$MLPEG_INFERENCE_CAMPAIGN``, or the calculation's outputs directory.
    """
    path = Path(os.environ.get("MLPEG_INFERENCE_CAMPAIGN", DEFAULT_CAMPAIGN))
    if not (path / "in" / "spec.json").is_file():
        pytest.skip(f"No campaign at {path}; set MLPEG_INFERENCE_CAMPAIGN.")
    return path


def test_optimized_inference_speed(campaign: Path) -> None:
    """
    Validate, aggregate and plot a campaign.

    Parameters
    ----------
    campaign
        Campaign directory.
    """
    analyse_campaign(campaign)
