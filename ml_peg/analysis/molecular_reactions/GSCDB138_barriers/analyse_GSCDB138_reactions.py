"""Analyse the reaction barriers datasets within the GSCDB138 collection."""

from __future__ import annotations

from pathlib import Path

from ml_peg.analysis.utils.analyse_gscdb138 import get_gscdb138_metrics
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT

CALC_PATH = CALCS_ROOT / "molecular_reactions" / "GSCDB138_barriers" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "GSCDB138_barriers"
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")

DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

DATASETS = [
    "BH28",
    "BH46",
    "BH876",
    "BHDIV7",
    "BHPERI11",
    "BHROT27",
    "CRBH14",
    "DBH22",
    "INV23",
    "ORBH35",
    "PX9",
    "WCPT26",
]


def test_gscdb138() -> None:
    """Run analysis for reaction barriers datasets within GSCDB138."""
    get_gscdb138_metrics(
        datasets=DATASETS,
        calc_path=CALC_PATH,
        out_path=OUT_PATH,
        thresholds=DEFAULT_THRESHOLDS,
        metric_tooltips=DEFAULT_TOOLTIPS,
        weights=DEFAULT_WEIGHTS,
    )
