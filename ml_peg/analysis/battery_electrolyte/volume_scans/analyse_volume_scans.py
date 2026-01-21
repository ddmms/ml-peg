"""Analyse Volume Scans benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
from aseMolec import extAtoms
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, rmse
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)

REF_PATH = CALCS_ROOT / "battery_electrolyte" / "volume_scans" / "data"
CALC_PATH = CALCS_ROOT / "battery_electrolyte" / "volume_scans" / "outputs"
OUT_PATH = APP_ROOT / "data" / "battery_electrolyte" / "volume_scans"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)

conf_types = ["Solvent", "Electrolyte"]


def get_volscan_results(
    conf_type: str,
) -> tuple[dict[str, list[float]], dict[str, list[float]]]:
    """
    Get relative energies per atom for a type of Volume Scan.

    Parameters
    ----------
    conf_type
        Name of Volume Scan type to be plotted.

    Returns
    -------
    results
        Relative energy of each model per Volume Scan config density.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    densities = {"ref": []} | {mlip: [] for mlip in MODELS}

    for model in results.keys():
        if model == "ref":
            configs = read(REF_PATH / f"{conf_type.lower()}_VS_PBED3.xyz", ":")

        else:
            configs = read(CALC_PATH / f"{conf_type.lower()}_VS_{model}_D3.xyz", ":")

        energies = [
            frame.calc.__dict__["results"]["energy"] * 1000 / len(frame)
            for frame in configs
        ]
        relative_energies = energies - min(energies)
        densities = [extAtoms.get_density_gcm3(frame) for frame in configs]

        results[model].append(densities)
        results[model].append(relative_energies.tolist())

    return results


def plot_volscans(conf_type: str, model: str, results: dict[str, float]) -> None:
    """
    Plot Volume Scan scatter plots.

    Parameters
    ----------
    conf_type
        Name of Volume Scan type to be plotted.
    model
        Name of MLIP.
    results
        Results from all models for a single Volume Scan.
    """

    @plot_scatter(
        filename=OUT_PATH / f"{conf_type.lower()}_{model}_volscan_scatter.json",
        title=f"{conf_type} Volume Scan",
        x_label="Density / g cm<sup>-3</sup>",
        y_label="Energy wrt Minimum Energy / meV/atom",
        show_line=True,
    )
    def plot_result() -> dict[str, list[float]]:
        """
        Plot the Volume Scan plots.

        Returns
        -------
        model_results
            Dictionary of reference and a specific MLIP's Volume Scan energies.
        """
        return {"ref": results["ref"], model: results[model]}

    plot_result()


@pytest.fixture
def get_volscan_rmses() -> dict[str, dict]:
    """
    Get model prediction RMSEs for all volume scan energies.

    Returns
    -------
    volscan_rmse
        Dictionary of energy RMSE per model for each volume scan.
    """
    volscan_rmse = {conf_type: {} for conf_type in conf_types}

    for conf_type in conf_types:
        results = get_volscan_results(conf_type)
        for model in MODELS:
            model_rmse = rmse(results["ref"][1], results[model][1])
            volscan_rmse[conf_type][model] = model_rmse
            plot_volscans(conf_type, model, results)

    return volscan_rmse


@pytest.fixture
@build_table(
    filename=OUT_PATH / "vol_scan_rmses_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def vs_rmse_metrics(get_volscan_rmses: dict[str, dict]) -> dict[str, dict]:
    """
    Get all Volume Scan RMSE metrics.

    Parameters
    ----------
    get_volscan_rmses
        Dictionary of each model's RMSE for all Volume Scans.

    Returns
    -------
    dict[str, dict]
        Dictionary of each model's RMSE for all Volume Scans.
    """
    return get_volscan_rmses


def test_vs_rmse_metrics(
    vs_rmse_metrics: dict[str, dict],
) -> None:
    """
    Run Volume Scans test.

    Parameters
    ----------
    vs_rmse_metrics
        All Volume Scan metrics.
    """
    return
