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

# Thermochemistry datasets.
DATASETS = [
    "AE11",
    "AE18",
    "AL2X6",
    "ALK8",
    "AlkAtom19ALKBDE10",
    "AlkIsod14",
    "BDE99MR",
    "BDE99nonMR",
    "BH76RC",
    "BSR36",
    "CR20",
    "DARC",
    "DC13",
    "DIPCS9",
    "EA50",
    "FH51",
    "G21EA",
    "G21IP",
    "G2RC24",
    "HAT707MR",
    "HAT707nonMR",
    "HEAVYSB11",
    "HNBrBDE18",
    "IP23",
    "IP30",
    "MB08-165",
    "MB16-43",
    "MX34",
    "NBPRC",
    "P34",
    "PA26",
    "PlatonicID6",
    "PlatonicIG6",
    "PlatonicTAE6",
    "RC21",
    "RSE43",
    "SIE4x4",
    "SN13",
    "TAE_W4-17MR",
    "TAE_W4-17nonMR",
    "WCPT6",
    "YBDE18",
]


def test_gscdb138() -> None:
    """Run isomer GSCDB138 test."""
    get_gscdb138_metrics(
        datasets=DATASETS,
        calc_path=CALC_PATH,
        out_path=OUT_PATH,
        thresholds=DEFAULT_THRESHOLDS,
        metric_tooltips=DEFAULT_TOOLTIPS,
        weights=DEFAULT_WEIGHTS,
    )
