"""Analyse the non covalent interactions benchmarks within the GSCDB138 collection."""

from __future__ import annotations

from pathlib import Path

from ml_peg.analysis.utils.analyse_gscdb138 import get_gscdb138_metrics
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT

CALC_PATH = CALCS_ROOT / "non_covalent_interactions" / "GSCDB138_NCIs" / "outputs"
OUT_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "GSCDB138_NCIs"
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")

DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

DATASETS = [
    "3B-69",
    "3BHET",
    "A19Rel6",
    "A24",
    "ADIM6",
    "AHB21",
    "Bauza30",
    "BzDC215",
    "CARBHB8",
    "CHB6",
    "CT20",
    "DS14",
    "FmH2O10",
    "H2O16Rel4",
    "H2O20Rel9",
    "HB262",
    "HB49",
    "HCP32",
    "He3",
    "HEAVY28",
    "HSG",
    "HW30",
    "HW6Cl5",
    "HW6F",
    "IHB100",
    "IHB100x2",
    "IL16",
    "NBC10",
    "NC11",
    "O24",
    "O24x4",
    "PNICO23",
    "RG10N",
    "RG18",
    "S22",
    "S66",
    "S66Rel7",
    "Shields38",
    "SW49Bind22",
    "SW49Rel28",
    "TA13",
    "WATER27",
    "X40",
    "X40x5",
    "XB25",
]


def test_gscdb138() -> None:
    """Run non covalent interactions GSCDB138 test."""
    get_gscdb138_metrics(
        datasets=DATASETS,
        calc_path=CALC_PATH,
        out_path=OUT_PATH,
        thresholds=DEFAULT_THRESHOLDS,
        metric_tooltips=DEFAULT_TOOLTIPS,
        weights=DEFAULT_WEIGHTS,
    )
