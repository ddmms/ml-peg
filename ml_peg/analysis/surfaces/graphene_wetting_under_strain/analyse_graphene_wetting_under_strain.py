"""Analyse graphene wetting under strain benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
import ase.io
import numpy as np
import pytest
from scipy.optimize import curve_fit
import yaml

from ml_peg.analysis.utils.decorators import build_table, plot_parity, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "surfaces" / "graphene_wetting_under_strain" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "graphene_wetting_under_strain"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

with open(CALC_PATH / "database_info.yml") as fp:
    DATABASE_INFO = yaml.safe_load(fp)
ORIENTATIONS = DATABASE_INFO["orientations"]
STRAINS = DATABASE_INFO["strains"]


def get_molecule_distance(sys: Atoms) -> float:
    """
    Compute distance of water molecule from graphene sheet for one configuration.

    Parameters
    ----------
    sys
        Single frame of water molecule + graphene system.

    Returns
    -------
    float
        Water molecule distance.
    """
    assert np.sum(sys.symbols == "O") == 1
    assert np.sum(sys.symbols == "C") > 1
    oxygens = sys.positions[sys.symbols == "O"]
    carbons = sys.positions[sys.symbols == "C"]
    return oxygens[0, 2] - np.mean(carbons[:, 2])


def morse_potential(r: np.ndarray, de: float, a: float, re: float) -> np.ndarray:
    """
    Compute Morse potential.

    Parameters
    ----------
    r
        Radial coordinates.
    de
        Potential well depth.
    a
        Decay coefficient (related to spring constant).
    re
        Equilibrium length.

    Returns
    -------
    NDArray
        Potentials at corresponding radii.
    """
    return de * (((1.0 - np.exp(a * (re - r))) ** 2) - 1.0)


def get_binding_parameters(
    distances: list[float], adsorption_energies: list[float]
) -> tuple[float, float, float]:
    """
    Compute best-fit parameters for adsorption energy curve.

    Parameters
    ----------
    distances
        Water molecule distances.
    adsorption_energies
        Corresponding adsorption energies.

    Returns
    -------
    float
        Potential well depth (de) of Morse potential.
    float
        Decay coefficient (a) of Morse potential.
    float
        Equilibrium length (re) of Morse potential.
    """
    popt = (np.inf, 0.0, np.inf)
    if np.min(adsorption_energies) < 0.0:
        depth = max(abs(np.min(adsorption_energies)), 5.0)
        idx = np.argmin(adsorption_energies)
        re = distances[idx]
        if idx > 0 and idx < (len(distances) - 1):
            second_deriv = (
                adsorption_energies[idx + 1]
                + adsorption_energies[idx - 1]
                - (2.0 * adsorption_energies[idx])
            )
            second_deriv /= ((distances[idx + 1] - distances[idx - 1]) / 2.0) ** 2
            a = max(np.sqrt(second_deriv / (2.0 * depth)), 0.5)
        else:
            a = 1.3
        try:
            popt, _ = curve_fit(
                morse_potential,
                distances,
                adsorption_energies,
                p0=(depth, a, re),
                bounds=(0.0, np.inf),
            )
        except (ValueError, RuntimeError):
            pass
    return popt


@pytest.fixture
def processed_data() -> dict[str, list]:
    """
    Gather and process all data for all systems.

    Returns
    -------
    dict[str, list | dict[str, dict[str, dict[str, dict[str, list | tuple]]]]]
        Dictionary of all processed data.
    """
    results = {"distances": [], "ref": {}} | {model: {} for model in MODELS}
    ref_stored = False
    dist_stored = False

    for model in MODELS:
        model_dir = CALC_PATH / model
        if not model_dir.exists():
            continue

        for orientation in ORIENTATIONS:
            if not ref_stored:
                results["ref"][orientation] = {}
            results[model][orientation] = {}

            for strain in STRAINS:
                if not ref_stored:
                    results["ref"][orientation][strain] = {
                        "energies": [],
                    }
                results[model][orientation][strain] = {
                    "energies": [],
                }

                struct_write_dir = OUT_PATH / model / "structs"
                struct_write_dir.mkdir(parents=True, exist_ok=True)
                systems = ase.io.iread(
                    model_dir / f"{orientation}_{strain}.xyz",
                    index=":",
                    format="extxyz",
                )
                for atoms in systems:
                    dist = get_molecule_distance(atoms)
                    if not dist_stored:
                        results["distances"].append(dist)
                    if not ref_stored:
                        results["ref"][orientation][strain]["energies"].append(
                            atoms.info["ref_adsorption_energy"] * 1000.0
                        )
                    results[model][orientation][strain]["energies"].append(
                        atoms.info["mlip_adsorption_energy"] * 1000.0
                    )

                    ase.io.write(
                        struct_write_dir / f"{orientation}_{strain}_L{dist:.2f}.xyz",
                        atoms,
                        format="xyz",
                    )

                if not ref_stored:
                    results["ref"][orientation][strain]["params"] = (
                        get_binding_parameters(
                            results["distances"],
                            results["ref"][orientation][strain]["energies"],
                        )
                    )
                results[model][orientation][strain]["params"] = get_binding_parameters(
                    results["distances"],
                    results[model][orientation][strain]["energies"],
                )

                @plot_scatter(
                    filename=OUT_PATH / model / f"figure_{orientation}_{strain}.json",
                    title=f"{orientation} binding energy curve ({strain[1:5]}% strain)",
                    x_label="Distance / Å",
                    y_label="Adsorption energy / meV",
                    show_line=True,
                )
                def plot_model_binding_energy_curve(
                    model, orientation, strain
                ) -> dict[str, tuple[list[float], list[float]]]:
                    return {
                        "ref": (
                            results["distances"],
                            results["ref"][orientation][strain]["energies"],
                        ),
                        model: (
                            results["distances"],
                            results[model][orientation][strain]["energies"],
                        ),
                    }

                plot_model_binding_energy_curve(model, orientation, strain)
                dist_stored = True

        def get_all_hover_data() -> dict[str, list[str]]:
            hover_data = {"Orientation": [], "Strain": [], "Distance": []}
            for orientation in ORIENTATIONS:
                for strain in STRAINS:
                    for dist in results["distances"]:
                        hover_data["Orientation"].append(orientation)
                        hover_data["Strain"].append(strain[1:5] + "%")
                        hover_data["Distance"].append(f"{dist:.2f} Å")
            return hover_data

        @plot_parity(
            filename=OUT_PATH / model / "figure_all_parity.json",
            title="Adsorption energies",
            x_label="Predicted adsorption energy / meV",
            y_label="Reference adsorption energy / meV",
            hoverdata=get_all_hover_data(),
        )
        def plot_model_all_parity(model) -> dict[str, list[float]]:
            parity_data = {"ref": [], model: []}
            for orientation in ORIENTATIONS:
                for strain in STRAINS:
                    parity_data["ref"].extend(
                        results["ref"][orientation][strain]["energies"]
                    )
                    parity_data[model].extend(
                        results[model][orientation][strain]["energies"]
                    )
            return parity_data

        def get_binding_hover_data() -> dict[str, list[str]]:
            hover_data = {"Orientation": [], "Strain": []}
            for orientation in ORIENTATIONS:
                for strain in STRAINS:
                    hover_data["Orientation"].append(orientation)
                    hover_data["Strain"].append(strain[1:5] + "%")
            return hover_data

        @plot_parity(
            filename=OUT_PATH / model / "figure_binding_energies_parity.json",
            title="Binding energies",
            x_label="Predicted binding energy / meV",
            y_label="Reference binding energy / meV",
            hoverdata=get_binding_hover_data(),
        )
        def plot_model_binding_energies_parity(model) -> dict[str, list[float]]:
            parity_data = {"ref": [], model: []}
            for orientation in ORIENTATIONS:
                for strain in STRAINS:
                    parity_data["ref"].append(
                        results["ref"][orientation][strain]["params"][0]
                    )
                    parity_data[model].append(
                        np.nan_to_num(
                            results[model][orientation][strain]["params"][0],
                            nan=-1.0,
                            posinf=-1.0,
                            neginf=-1.0,
                        )
                    )
            return parity_data

        @plot_parity(
            filename=OUT_PATH / model / "figure_binding_lengths_parity.json",
            title="Binding lengths",
            x_label="Predicted binding length / Å",
            y_label="Reference binding length / Å",
            hoverdata=get_binding_hover_data(),
        )
        def plot_model_binding_lengths_parity(model) -> dict[str, list[float]]:
            parity_data = {"ref": [], model: []}
            for orientation in ORIENTATIONS:
                for strain in STRAINS:
                    parity_data["ref"].append(
                        results["ref"][orientation][strain]["params"][2]
                    )
                    parity_data[model].append(
                        np.nan_to_num(
                            results[model][orientation][strain]["params"][2],
                            nan=-1.0,
                            posinf=-1.0,
                            neginf=-1.0,
                        )
                    )
            return parity_data

        plot_model_all_parity(model)
        plot_model_binding_energies_parity(model)
        plot_model_binding_lengths_parity(model)
        ref_stored = True

    return results


@pytest.fixture
def all_adsorption_energies_mae(processed_data) -> dict[str, float]:
    """
    Get mean absolute error for all adsorption energies.

    Parameters
    ----------
    processed_data
        Dictionary of processed data.

    Returns
    -------
    dict[str, float]
        Dictionary of MAEs for all models.
    """
    results = {}
    for model in MODELS:
        deviations = []
        for orientation in ORIENTATIONS:
            for strain in STRAINS:
                for i in range(len(processed_data["distances"])):
                    deviations.append(
                        abs(
                            processed_data[model][orientation][strain]["energies"][i]
                            - processed_data["ref"][orientation][strain]["energies"][i]
                        )
                    )
        results[model] = np.mean(deviations)
    return results


@pytest.fixture
def binding_energies_mae(processed_data) -> dict[str, float]:
    """
    Get mean absolute error of binding energies across all orientations and strains.

    Parameters
    ----------
    processed_data
        Dictionary of processed data.

    Returns
    -------
    dict[str, float]
        Dictionary of binding energy MAEs for all models.
    """
    results = {}
    for model in MODELS:
        deviations = []
        for orientation in ORIENTATIONS:
            for strain in STRAINS:
                deviations.append(
                    abs(
                        processed_data[model][orientation][strain]["params"][0]
                        - processed_data["ref"][orientation][strain]["params"][0]
                    )
                )
        results[model] = np.nan_to_num(
            np.mean(deviations), nan=99999, posinf=99999, neginf=99999
        )
    return results


@pytest.fixture
def binding_lengths_mae(processed_data) -> dict[str, float]:
    """
    Get mean absolute error of binding lengths across all orientations and strains.

    Parameters
    ----------
    processed_data
        Dictionary of processed data.

    Returns
    -------
    dict[str, float]
        Dictionary of binding length MAEs for all models.
    """
    results = {}
    for model in MODELS:
        deviations = []
        for orientation in ORIENTATIONS:
            for strain in STRAINS:
                deviations.append(
                    abs(
                        processed_data[model][orientation][strain]["params"][2]
                        - processed_data["ref"][orientation][strain]["params"][2]
                    )
                )
        results[model] = np.nan_to_num(
            np.mean(deviations), nan=999, posinf=999, neginf=999
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "graphene_wetting_under_strain_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    all_adsorption_energies_mae: dict[str, float],
    binding_energies_mae: dict[str, float],
    binding_lengths_mae: dict[str, float],
) -> dict[str, dict]:
    """
    Get all graphene wetting metrics.

    Parameters
    ----------
    all_adsorption_energies_mae
        Mean absolute errors across all orientations, distances, and strains for all
        models.
    binding_energies_mae
        Mean absolute errors of binding energies across all orientations and strains
        for all models.
    binding_lengths_mae
        Mean absolute errors of binding lengths across all orientations and strains for
        all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "All Adsorption Energies MAE": all_adsorption_energies_mae,
        "Binding Energies MAE": binding_energies_mae,
        "Binding Lengths MAE": binding_lengths_mae,
    }


def test_graphene_wetting_under_strain(metrics: dict[str, dict]) -> None:
    """
    Run graphene wetting test.

    Parameters
    ----------
    metrics
        All graphene wetting metrics.
    """
    return
