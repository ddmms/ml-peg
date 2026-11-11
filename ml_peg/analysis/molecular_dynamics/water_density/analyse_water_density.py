"""Analyse the water density benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import Trajectory, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.timeseries import (
    MetricEstimate,
    TimeSeriesEstimate,
    bootstrap_metric,
    estimate_mean,
)
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_dispersion_name_map()

INTERVAL_PS = 0.1
EQUILIB_TIME_PS = 500

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "water_density" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "water_density"


METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EXPERIMENTAL_DATA = {
    "water_270.0_K": 0.9998,
    "water_290.0_K": 0.9991,
    "water_300.0_K": 0.9970,
    "water_330.0_K": 0.9802,
}
# Standard uncertainties (one standard deviation) of the experimental densities, in
# the same order as INFO["filenames"]. None if not known.
# TODO: Check if these are known and add them if so.
REF_SIGMA_1SD = None


INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.traj",
    index=0,
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
    include_filenames=True,
)

# Reference values are experimental, so are known for every system
REF_DENSITIES = [EXPERIMENTAL_DATA[label] for label in INFO["filenames"]]


def compute_density(fname, density_col=13, label=None) -> TimeSeriesEstimate:
    """
    Compute average density, and its uncertainty, from NPT log file.

    Parameters
    ----------
    fname
        Path to the log file.
    density_col
        Which column the density numbers are in.
    label
        Label to identify the simulation in warnings.

    Returns
    -------
    TimeSeriesEstimate
        Average density in g/cm3, with its uncertainty.
    """
    density_series = []
    with open(fname) as lines:
        for line in lines:
            items = line.strip().split()
            if len(items) != 15:
                continue
            density_series.append(float(items[density_col]))
    skip_frames = int(EQUILIB_TIME_PS / INTERVAL_PS)
    return estimate_mean(density_series[skip_frames:], label=label)


@pytest.fixture
def density_estimates() -> dict[str, list[TimeSeriesEstimate | None]]:
    """
    Get water density estimates for all models and temperatures.

    Returns
    -------
    dict[str, list[TimeSeriesEstimate | None]]
        Density estimates for each model, in the order of ``INFO["filenames"]``.
        None where the run is missing.
    """
    estimates = {mlip: [] for mlip in MODELS}
    for model_name in MODELS:
        for label in INFO["filenames"]:
            traj_path = CALC_PATH / model_name / f"{label}.traj"
            log_path = CALC_PATH / model_name / f"{label}.log"
            if not traj_path.exists() or not log_path.exists():
                estimates[model_name].append(None)
                continue
            estimates[model_name].append(
                compute_density(log_path, label=f"{model_name}/{label}")
            )
    return estimates


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_water_density.json",
    title="Densities",
    x_label="Predicted density / g/cm³",
    y_label="Reference density / g/cm³",
    hoverdata={
        "Labels": INFO["filenames"],
    },
)
def water_density(
    density_estimates: dict[str, list[TimeSeriesEstimate | None]],
) -> dict[str, list]:
    """
    Get water densities for all temperatures.

    Parameters
    ----------
    density_estimates
        Density estimates for each model.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted densities, and their 95 %
        confidence interval half-widths.
    """
    results = {"ref": list(REF_DENSITIES)} | {mlip: [] for mlip in MODELS}
    results["pred_err"] = {}
    if REF_SIGMA_1SD is not None:
        results["ref_err"] = [1.96 * sigma for sigma in REF_SIGMA_1SD]

    for model_name in MODELS:
        estimates = density_estimates[model_name]
        results[model_name] = [
            float("nan") if estimate is None else estimate.mean
            for estimate in estimates
        ]
        results["pred_err"][model_name] = [
            float("nan") if estimate is None else estimate.ci_95
            for estimate in estimates
        ]

        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)

        for label, estimate in zip(INFO["filenames"], estimates, strict=True):
            if estimate is None:
                continue
            traj_path = CALC_PATH / model_name / f"{label}.traj"

            # Write structures for app
            write(structs_dir / f"{label}.xyz", Trajectory(traj_path)[-1])

    return results


@pytest.fixture
def get_mae(
    water_density: dict[str, list],
    density_estimates: dict[str, list[TimeSeriesEstimate | None]],
) -> dict[str, MetricEstimate]:
    """
    Get mean absolute error for water densities, with confidence intervals.

    Parameters
    ----------
    water_density
        Dictionary of reference and predicted water densities.
    density_estimates
        Density estimates for each model.

    Returns
    -------
    dict[str, MetricEstimate]
        Dictionary of predicted water density errors for all models.
    """
    return {
        model_name: bootstrap_metric(
            mae,
            water_density["ref"],
            density_estimates[model_name],
            ref_sigma_1sd=REF_SIGMA_1SD,
        )
        for model_name in MODELS
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "water_density_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(get_mae: dict[str, MetricEstimate]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_mae,
    }


@pytest.mark.framework("mace-polar-1")
def test_water_density(metrics: dict[str, dict]) -> None:
    """
    Run water density test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
