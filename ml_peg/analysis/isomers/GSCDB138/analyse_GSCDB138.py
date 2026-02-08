"""Analyse the isomer energy benchmarks within the GSCDB138 collection."""

from __future__ import annotations

from pathlib import Path

from ml_peg.analysis.utils.analyse_gscdb138 import get_gscdb138_metrics
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT

CALC_PATH = CALCS_ROOT / "isomers" / "GSCDB138" / "outputs"
OUT_PATH = APP_ROOT / "data" / "isomers" / "GSCDB138"
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")

DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

DATASETS = [
    "A19Rel6",
    "ACONF",
    "AlkIsomer11",
    "Amino20x4",
    "BUT14DIOL",
    # "C20C246",
    # "C60ISO7",
    "DIE60",
    "EIE22",
    # "H2O16Rel4",
    # "H2O20Rel9",
    "ICONF",
    "IDISP",
    "ISO34",
    # "ISOL23",
    "ISOMERIZATION20",
    "MCONF",
    "PArel",
    "PCONF21",
    # "Pentane13",
    "S66Rel7",
    "SCONF",
    # "Styrene42",
    # "SW49Rel28",
    "TAUT15",
    "UPU23",
]


def test_gscdb138() -> None:
    """Run GSCDB138 test."""
    get_gscdb138_metrics(
        datasets=DATASETS,
        calc_path=CALC_PATH,
        out_path=OUT_PATH,
        thresholds=DEFAULT_THRESHOLDS,
        metric_tooltips=DEFAULT_TOOLTIPS,
        weights=DEFAULT_WEIGHTS,
    )
