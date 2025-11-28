"""Analyse battery electrolyte benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config, rmse
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
REF_PATH = CALCS_ROOT / "bulk_liquids" / "battery_electrolyte" / "data"
CALC_PATH = CALCS_ROOT / "bulk_liquids" / "battery_electrolyte" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_liquids" / "battery_electrolyte"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)

property_metadata = {
    "Intra-Forces": ["arrays", "forces_intram"],
    "Inter-Forces": ["arrays", "forces_interm"],
    "Inter-Energy": ["info", "energy_interm"],
    "Intra-Virial": ["info", "virial_intram"],
    "Inter-Virial": ["info", "virial_interm"],
}


def get_property_results(prop_key: str) -> dict[str, float]:
    """
    Get inter-intra results for a specific property.

    Parameters
    ----------
    prop_key
        String of property name.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted inter-intra properties.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}

    stored, property = property_metadata[prop_key]

    for model in results.keys():
        if model == "ref":
            configs = read(REF_PATH / "Intra_Inter_data" / "intrainter_PBED3.xyz", ":")

        else:
            configs = read(
                CALC_PATH / "Intra_Inter_output" / f"intrainter_{model}_D3.xyz", ":"
            )

        for frame in configs:
            frame_data = getattr(frame, stored)
            property_data = frame_data[property]
            results[model].append(property_data.tolist())

        if "Forces" in prop_key:
            results[model] = np.concatenate(results[model]).flatten()
            results[model] = results[model].tolist()

        if "Virial" in prop_key:
            results[model] = np.array(results[model]).flatten()
            results[model] = results[model].tolist()

    return results


# @pytest.fixture
# @plot_parity(
#     filename=OUT_PATH / "test.json"
#     # title=f"{prop_key}",
#     # x_label="Predicted lattice energy / METRICS_CONFIG_PATH",
#     # y_label="Reference lattice energy / kJ/mol",
#     # hoverdata={
#     #     "System": get_system_names(),
#     # },
# )
# def intra_forces_plot() -> dict[str, dict]:
#     results = get_property_results("Intra-Forces")

#     return results


# def test_parity_plot(intra_forces_plot: None) -> None:
#     return


@pytest.fixture
def get_property_rmses() -> dict[str, dict]:
    """
    Get model prediction RMSEs for all inter-intra properties.

    Returns
    -------
    dict[str, dict]
        Dictionary of inter-intra properties and the respective RMSE per model.
    """
    property_rmse = {prop_key: {} for prop_key in property_metadata.keys()}

    for prop_key in property_metadata.keys():
        results = get_property_results(prop_key)

        for model in MODELS:
            model_rmse = rmse(results["ref"], results[model])
            property_rmse[prop_key][model] = model_rmse

    return property_rmse


@pytest.fixture
@build_table(
    filename=OUT_PATH / "Inter_intra_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def rmse_metrics(get_property_rmses: dict[str, dict]) -> dict[str, dict]:
    """
    Get all inter intra RMSE metrics.

    Parameters
    ----------
    get_property_rmses
        Dictionary for every property containing each model's RMSE.

    Returns
    -------
    dict[str, dict]
        Dictionary for every property containing each model's RMSE.
    """
    return get_property_rmses


def test_rmse_metrics(rmse_metrics: dict[str, dict]) -> None:
    """
    Run inter-intra property test.

    Parameters
    ----------
    rmse_metrics
        All inter-intra metrics.
    """
    return


# @pytest.fixture
# def solvent_volscan_results() -> list[float, float]:
#     """
#     Get results for the neat solvent Volume Scan.

#     Returns
#     -------
#     dict[str, dict]
#         Dictionary for every property containing each model's RMSE.
#     """
