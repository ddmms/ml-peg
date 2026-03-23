"""Analyse the electric field property benchmarks within the GSCDB138 collection."""

from __future__ import annotations

from pathlib import Path

from ml_peg.analysis.utils.analyse_gscdb138 import get_gscdb138_metrics
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT

CALC_PATH = CALCS_ROOT / "electric_field" / "GSCDB138_field" / "outputs"
OUT_PATH = APP_ROOT / "data" / "electric_field" / "GSCDB138_field"
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")

DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

DATASETS = ["Dip146", "HR46", "OEEF", "Pol130", "T144", "V30"]


def test_gscdb138() -> None:
    """Run electric field property GSCDB138 test."""
    get_gscdb138_metrics(
        datasets=DATASETS,
        calc_path=CALC_PATH,
        out_path=OUT_PATH,
        thresholds=DEFAULT_THRESHOLDS,
        metric_tooltips=DEFAULT_TOOLTIPS,
        weights=DEFAULT_WEIGHTS,
    )
