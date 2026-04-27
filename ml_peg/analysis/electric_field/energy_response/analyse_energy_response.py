"""Analyse scaling_pol benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
import numpy as np
from ase.io import read, write
import pytest

from ml_peg.analysis.electric_field.energy_response.energy_response import get_energy_response
#from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    #build_dispersion_name_map,
    load_metrics_config,
    #mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
#DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "electric_field" / "energy_response" / "outputs"
OUT_PATH = APP_ROOT / "data" / "electric_field" / "energy_response"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


# Linear organic molecules data sets for electric field response.
DATASETS = [
    "ALKANES",
    "CUMULENES",
]


def test_energy_response(metrics: dict[str, dict]) -> None:
    """
    Run energy_response test.

    Parameters
    ----------
    metrics
        All energy_response metrics.
    """
    get_energy_response(
        datasets=DATASETS,
        calc_path=CALC_PATH,
        out_path=OUT_PATH,
        thresholds=DEFAULT_THRESHOLDS,
        metric_tooltips=DEFAULT_TOOLTIPS,
        weights=DEFAULT_WEIGHTS,
    )

    return
