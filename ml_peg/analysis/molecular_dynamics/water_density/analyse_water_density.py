"""Analyse the water density benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import Trajectory, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_dispersion_name_map(MODELS)

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


def labels() -> list:
    """
    Get list of system names.

    Returns
    -------
    list
        List of all system names.
    """
    for model_name in MODELS:
        return [path.stem for path in (CALC_PATH / model_name).glob("*.log")]
    return []


def compute_density(fname, density_col=13):
    """
    Compute average density from NPT log file.

    Parameters
    ----------
    fname
        Path to the log file.
    density_col
        Which column the density numbers are in.

    Returns
    -------
    float
        Average density in g/cm3.
    """
    density_series = []
    with open(fname) as lines:
        for line in lines:
            items = line.strip().split()
            if len(items) != 15:
                continue
            density_series.append(float(items[density_col]))
    skip_frames = int(EQUILIB_TIME_PS / INTERVAL_PS)
    return np.mean(density_series[skip_frames:])


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_water_density.json",
    title="Densities",
    x_label="Predicted density / kcal/mol",
    y_label="Reference density / kcal/mol",
    hoverdata={
        "Labels": labels(),
    },
)
def water_density() -> dict[str, list]:
    """
    Get water densities for all temperatures.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted densities.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        for label in labels():
            atoms = Trajectory(CALC_PATH / model_name / f"{label}.traj")[-1]

            results[model_name].append(
                compute_density(CALC_PATH / model_name / f"{label}.log")
            )
            if not ref_stored:
                results["ref"].append(EXPERIMENTAL_DATA[label])

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms)
        ref_stored = True
    return results


@pytest.fixture
def get_mae(water_density) -> dict[str, float]:
    """
    Get mean absolute error for water densities.

    Parameters
    ----------
    water_density
        Dictionary of reference and predicted water densities.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted water density errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(water_density["ref"], water_density[model_name])
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "water_density_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(get_mae: dict[str, float]) -> dict[str, dict]:
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


def test_water_density(metrics: dict[str, dict]) -> None:
    """
    Run water density test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
