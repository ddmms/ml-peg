"""Analyse small-cluster forces against ωB97M-V references."""

from __future__ import annotations

from pathlib import Path

from ml_peg.analysis.utils.analyse_cluster_forces import analyse_cluster_forces
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT

CALC_PATH = CALCS_ROOT / "clusters" / "cluster_forces_wb97mv" / "outputs"
OUT_PATH = APP_ROOT / "data" / "clusters" / "cluster_forces_wb97mv"
METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")

DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Models not trained on dispersion-corrected data are run with D3 added
DISPERSION_NAME_MAP = build_dispersion_name_map()


def test_cluster_forces_wb97mv() -> None:
    """Write metrics, plots and structures for the ωB97M-V cluster forces."""
    analyse_cluster_forces(
        title="ωB97M-V",
        calc_path=CALC_PATH,
        out_path=OUT_PATH,
        thresholds=DEFAULT_THRESHOLDS,
        metric_tooltips=DEFAULT_TOOLTIPS,
        weights=DEFAULT_WEIGHTS,
        mlip_name_map=DISPERSION_NAME_MAP,
    )
