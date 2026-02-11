"""Analyse the transition metal complex energy datasets in GSCDB138 database."""

from __future__ import annotations

from pathlib import Path

from ml_peg.analysis.utils.analyse_gscdb138 import get_gscdb138_metrics
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT

CALC_PATH = CALCS_ROOT / "tm_complexes" / "GSCDB138_tm_complexes" / "outputs"
OUT_PATH = APP_ROOT / "data" / "tm_complexes" / "GSCDB138_tm_complexes"
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")

DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

DATASETS = [
    "3d4dIPSS",
    "CUAGAU83",
    "DAPD",
    "MME52",
    "MOBH28",
    "MOR13",
    "ROST61",
    "TMB11",
    "TMD10",
]


def test_gscdb138() -> None:
    """Run transition metal complexes GSCDB138 test."""
    get_gscdb138_metrics(
        datasets=DATASETS,
        calc_path=CALC_PATH,
        out_path=OUT_PATH,
        thresholds=DEFAULT_THRESHOLDS,
        metric_tooltips=DEFAULT_TOOLTIPS,
        weights=DEFAULT_WEIGHTS,
    )
