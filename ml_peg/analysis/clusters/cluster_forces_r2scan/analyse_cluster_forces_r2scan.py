"""Analyse small-cluster forces against r2SCAN references."""

from __future__ import annotations

from pathlib import Path

from ml_peg.analysis.utils.analyse_cluster_forces import analyse_cluster_forces
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT

CALC_PATH = CALCS_ROOT / "clusters" / "cluster_forces_r2scan" / "outputs"
OUT_PATH = APP_ROOT / "data" / "clusters" / "cluster_forces_r2scan"
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")

DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def test_cluster_forces_r2scan() -> None:
    """Write metrics, plots and structures for the r2SCAN cluster forces."""
    analyse_cluster_forces(
        title="r2SCAN",
        calc_path=CALC_PATH,
        out_path=OUT_PATH,
        thresholds=DEFAULT_THRESHOLDS,
        metric_tooltips=DEFAULT_TOOLTIPS,
        weights=DEFAULT_WEIGHTS,
    )
