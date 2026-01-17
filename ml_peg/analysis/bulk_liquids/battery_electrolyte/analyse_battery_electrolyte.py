"""Analyse battery electrolyte benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, rmse
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)

MODELS = MODELS[:-1]
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
conf_type = ["solvent", "electrolyte"]


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
        Dictionary of reference and predicted inter-intra property.
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


def plot_results(prop_key: str, results: dict[str, float]) -> None:
    """
    Plot inter-intra property parity plots.

    Parameters
    ----------
    prop_key
        Name of inter-intra property to be plotted.
    results
        Results from all models for a single property.
    """

    @plot_parity(
        filename=OUT_PATH / f"{prop_key}_parity.json",
        title=prop_key,
        x_label=f"Predicted {prop_key} / {DEFAULT_THRESHOLDS[prop_key]['unit']}",
        y_label=f"DFT {prop_key} / {DEFAULT_THRESHOLDS[prop_key]['unit']}",
        plot_combined=False,
    )
    def plot_result() -> dict[str, list[float]]:
        """
        Plot the inter-intra propery parity plots.

        Returns
        -------
        dict[str, tuple[list[float]]]
            Dictionary of reference and predicted inter-intra property.
        """
        return results

    plot_result()


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
        plot_results(prop_key, results)
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


def test_rmse_metrics(
    rmse_metrics: dict[str, dict],
) -> None:
    """
    Run inter-intra property test.

    Parameters
    ----------
    rmse_metrics
        All inter-intra metrics.
    """
    return


# @pytest.fixture
# def volscan_results(conf_type: str) -> tuple[dict[str, float], dict[str, float]]:
#     """
#     Get relative energies per atom for a type of Volume Scan.

#     Returns
#     -------
#         .
#     """
#     results = {"ref": []} | {mlip: [] for mlip in MODELS}
#     densities = {"ref": []} | {mlip: [] for mlip in MODELS}

#     for model in results.keys():
#         if model == "ref":
#             configs = read(REF_PATH / "Volume_Scan_data" /
#                               f"{type}_VS_PBED3.xyz", ":")

#         else:
#             configs = read(
#                 CALC_PATH / "Volume_Scan_output" / f"{type}_VS_{model}_D3.xyz", ":"
#             )

#         energies = [
#             frame.calc.__dict__["results"]["energy"] / len(frame) for frame in configs
#         ]
#         relative_energies = energies - min(energies)

#         results[model].append(relative_energies.tolist())
#         densities = [ea.get_density_gcm3(frame) for frame in configs]

#     return results, densities


# @pytest.fixture
# @plot_parity(
#     filename=OUT_PATH / "solvent_volscan_scatter.json",
#     title="Solvent Volume Scans",
#     x_label="Predicted intra-forces / eV/atom",
#     y_label="DFT intra-forces / eV/atom",
# )
# def solvent_volscan_scatter() -> dict[str, float]:
#     """
#     Solvent volume scan results of each model for parity plots.

#     Returns
#     -------
#     tuple[dict[str, float], dict[str, float]]
#         Volume scan energies and densities for each model.
#     """
#     solvent_vs_energies, solvent_vs_densities = volscan_results("solvent")
#     return solvent_vs_energies, solvent_vs_densities


# def plot_volscans(model: str) -> None:
#     """
#     Plot volume scans and save all structure files.

#     Parameters
#     ----------
#     model
#         Name of MLIP.
#     """

#     @plot_scatter(
#         filename=OUT_PATH / f"{conf_type}_VS_{model}.json",
#         title=f"Volume Scan of {conf_type} configs",
#         x_label="Density / g/cm3",
#         y_label="Energy / eV/atom",
#         show_line=False,
#     )
#     def plot_volscan() -> dict[str, tuple[list[float], list[float]]]:
#         """
#         Plot a Volume Scan and save the structure file.

#         Returns
#         -------
#         dict[str, tuple[list[float], list[float]]]
#             Dictionary of tuples of image/energy for each model.
#         """
#         results = {}
#         structs = read(
#             CALC_PATH / f"li_diffusion_{path.lower()}-{model}-neb-band.extxyz",
#             index=":",
#         )
#         results[model] = [
#             list(range(len(structs))),
#             [struct.get_potential_energy() for struct in structs],
#         ]
#         structs_dir = OUT_PATH / model
#         structs_dir.mkdir(parents=True, exist_ok=True)
#         write(structs_dir / f"{model}-{path.lower()}-neb-band.extxyz", structs)

#         return results

#     plot_neb()
