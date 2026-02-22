"""Analyse water densities."""

from __future__ import annotations

from pathlib import Path

from ase import units
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "molecular_dynamics" / "WaterDensity" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "WaterDensity"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KCAL = units.mol / units.kcal
INFO = [270, 290, 300, 330]

experimental_data = {
    "temperature": np.array(
        [
            270,
            273.15,
            277,
            280,
            283,
            286,
            290,
            293,
            296,
            300,
            303,
            306,
            310,
            313,
            316,
            320,
            323,
            326,
            330,
            333,
            336,
            340,
        ]
    ),
    "density": np.array(
        [
            0.9998,
            0.9999,
            1.0000,
            0.9999,
            0.9998,
            0.9997,
            0.9991,
            0.9985,
            0.9978,
            0.9970,
            0.9957,
            0.9944,
            0.9927,
            0.9910,
            0.9893,
            0.9871,
            0.9849,
            0.9827,
            0.9802,
            0.9777,
            0.9752,
            0.9723,
        ]
    ),  # g/cm³
}

DATA_PATH = (
    download_s3_data(
        filename="WaterDensities.zip",
        key="inputs/molecular_dynamics/WaterDensities/WaterDensities.zip",
    )
    / "WaterDensities"
)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_WaterDensity.json",
    title="Water density",
    x_label="Predicted density / g/cm^3",
    y_label="Reference density / g/cm^3",
    hoverdata={"Labels": INFO},
)
def densities() -> dict[str, list]:
    """
    Get water density for all the teperatures.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted densities.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}

    results["ref"] = [
        density
        for density, temp in zip(
            experimental_data["density"], experimental_data["temperature"], strict=False
        )
        if temp in INFO
    ]

    for model_name in MODELS:
        for temperature in INFO:
            instantaneous_densities = []
            with open(
                DATA_PATH / model_name / f"water_T_{temperature:.1f}" / "water.log"
            ) as lines:
                skip_time_ps = 500
                for line in lines:
                    items = line.strip().split()
                    time = float(items[1])
                    if time < skip_time_ps:
                        continue
                    instantaneous_densities.append(float(items[13]))

            results[model_name].append(np.mean(instantaneous_densities))
    return results


@pytest.fixture
def get_mae(densities) -> dict[str, float]:
    """
    Get mean absolute error for densities for all temperatures.

    Parameters
    ----------
    densities
        Dictionary of reference and predicted densities.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted densities errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(densities["ref"], densities[model_name])
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "WaterDensity_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
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
    return {"MAE": get_mae}


def test_water_density(metrics: dict[str, dict]) -> None:
    """
    Run water density test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
