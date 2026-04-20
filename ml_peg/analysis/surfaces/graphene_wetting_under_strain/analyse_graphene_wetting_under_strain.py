"""Analyse graphene wetting under strain benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import Atoms
import ase.io
import numpy as np
from plotly.colors import DEFAULT_PLOTLY_COLORS
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pytest
from scipy.optimize import curve_fit
import yaml

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config, mae
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


def mie_potential(
    r: np.ndarray, epsilon: float, re: float, n: float, m: float
) -> np.ndarray:
    """
    Compute Mie potential.

    Parameters
    ----------
    r
        Radial coordinates.
    epsilon
        Potential well depth.
    re
        Equilibrium length.
    n
        Exponent of repulsive potential.
    m
        Exponent of attractive potential.

    Returns
    -------
    NDArray
        Potentials at corresponding radii.
    """
    coeff = (n / (n - m)) * np.power((n / m), (m / (n - m)))
    x = re * np.power((m / n), 1.0 / (n - m)) / r
    return coeff * epsilon * (np.power(x, n) - np.power(x, m))


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
        Potential well depth.
    float
        Equilibrium length.
    """
    # Initial estimates
    depth = -np.min(adsorption_energies)
    re = distances[np.argmin(adsorption_energies)]
    dist_max = np.max(distances)

    # If depth is negative -- i.e. entire binding curve is positive -- then there is no
    # potential that can be fitted to the curve. Return parameters immediately.
    if depth <= 0.0:
        return depth, 0.0

    # Attempt to fit Morse potential, and return if the fit is valid
    first_deriv = np.gradient(adsorption_energies, distances)
    second_deriv = np.gradient(first_deriv, distances)
    second_deriv_at_min = second_deriv[np.argmin(adsorption_energies)]
    if second_deriv_at_min > 0.0:
        a = max(np.sqrt(second_deriv_at_min / (2.0 * depth)), 0.5)
    else:
        a = 1.3
    try:
        popt, _ = curve_fit(
            morse_potential,
            distances,
            adsorption_energies,
            p0=(depth, a, re),
            bounds=(0.0, (2.0 * depth, np.inf, dist_max)),
        )
        return popt[0], popt[2]
    except (ValueError, RuntimeError):
        pass

    # Attempt to fit Mie potential, and return if the fit is valid
    try:
        popt, _ = curve_fit(
            mie_potential,
            distances,
            adsorption_energies,
            p0=(depth, re, 12.0, 3.0),
            bounds=((0.0, 0.0, 6.0, 0.0), (2.0 * depth, dist_max, np.inf, 6.0)),
        )
        return popt[0], popt[1]
    except (ValueError, RuntimeError):
        pass

    # No fits -- just return numerical estimate
    return depth, re


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
    struct_write_dir = OUT_PATH / "structs"
    struct_write_dir.mkdir(parents=True, exist_ok=True)

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

                systems = ase.io.iread(
                    model_dir / f"{orientation}_{strain}.xyz",
                    index=":",
                    format="extxyz",
                )

                if not ref_stored:
                    (struct_write_dir / f"{orientation}_{strain}.xyz").unlink(
                        missing_ok=True
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

                    if not ref_stored:
                        ase.io.write(
                            struct_write_dir / f"{orientation}_{strain}.xyz",
                            atoms,
                            format="xyz",
                            append=True,
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

                dist_stored = True

        ref_stored = True

    return results


@pytest.fixture
def generate_plots_for_app(processed_data) -> None:
    """
    Generate plots for all models across all configurations.

    Parameters
    ----------
    processed_data
        Dictionary of processed data.
    """
    # These plots require a fairly low-level access to Plotly, in order to produce the
    # 3x3 grid of binding energy curves over 3 orientations and 3 strain conditions

    # First plot: 3x3 grid of adsorption energies across all orientations and strains
    subplot_titles = [
        f"{orientation}, {strain[1:5]}% strain"
        for strain in STRAINS
        for orientation in ORIENTATIONS
    ]
    fig = make_subplots(
        rows=3,
        cols=3,
        subplot_titles=subplot_titles,
    )
    for i, orientation in enumerate(ORIENTATIONS):
        for j, strain in enumerate(STRAINS):
            fig.add_trace(
                go.Scatter(
                    x=processed_data["distances"],
                    y=processed_data["ref"][orientation][strain]["energies"],
                    name="Reference",
                    legendgroup="Reference",
                    showlegend=((i + j) == 0),
                    mode="lines+markers",
                    line={"color": "#000000"},
                    hovertemplate="%{y:.1f} meV",
                ),
                row=(j + 1),
                col=(i + 1),
            )
    for iter, model in enumerate(MODELS):
        for i, orientation in enumerate(ORIENTATIONS):
            for j, strain in enumerate(STRAINS):
                color = DEFAULT_PLOTLY_COLORS[iter % len(DEFAULT_PLOTLY_COLORS)]
                fig.add_trace(
                    go.Scatter(
                        x=processed_data["distances"],
                        y=processed_data[model][orientation][strain]["energies"],
                        name=model,
                        legendgroup=model,
                        showlegend=((i + j) == 0),
                        mode="lines+markers",
                        line={"color": color},
                        hovertemplate="%{y:.1f} meV",
                    ),
                    row=(j + 1),
                    col=(i + 1),
                )
    fig.update_xaxes(
        title_text="Distance / Å",
        unifiedhovertitle_text="<b>Distance: </b>%{x:.2f} Å",
    )
    fig.update_yaxes(title_text="Energy / meV")
    fig.update_layout(
        title_text=(
            "Adsorption energy curve over various orientations and strain conditions"
        ),
        width=1500,
        height=1500,
        hovermode="x unified",
    )
    filename = OUT_PATH / "figure_adsorption_energies.json"
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    fig.write_json(filename)

    # Second plot: 1x3 grid of binding energies across all orientations
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=ORIENTATIONS,
    )
    strains_to_plot = [float(strain[1:5]) for strain in STRAINS]
    for i, orientation in enumerate(ORIENTATIONS):
        fig.add_trace(
            go.Scatter(
                x=strains_to_plot,
                y=[
                    processed_data["ref"][orientation][strain]["params"][0]
                    for strain in STRAINS
                ],
                name="Reference",
                legendgroup="Reference",
                showlegend=(i == 0),
                mode="lines+markers",
                line={"color": "#000000"},
                hovertemplate="%{y:.1f} meV",
            ),
            row=1,
            col=(i + 1),
        )
    for iter, model in enumerate(MODELS):
        color = DEFAULT_PLOTLY_COLORS[iter % len(DEFAULT_PLOTLY_COLORS)]
        for i, orientation in enumerate(ORIENTATIONS):
            fig.add_trace(
                go.Scatter(
                    x=strains_to_plot,
                    y=[
                        processed_data[model][orientation][strain]["params"][0]
                        for strain in STRAINS
                    ],
                    name=model,
                    legendgroup=model,
                    showlegend=(i == 0),
                    mode="lines+markers",
                    line={"color": color},
                    hovertemplate="%{y:.1f} meV",
                ),
                row=1,
                col=(i + 1),
            )
    fig.update_xaxes(
        title_text="Strain / %",
        unifiedhovertitle_text="<b>Strain: </b>%{x:+.2f} %",
    )
    fig.update_yaxes(title_text="Energy / meV")
    fig.update_layout(
        title_text="Binding energy against strain over various orientations",
        width=1500,
        height=500,
        hovermode="x unified",
    )
    filename = OUT_PATH / "figure_binding_energies.json"
    fig.write_json(filename)

    # Third plot: 1x3 grid of binding lengths across all orientations
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=ORIENTATIONS,
    )
    for i, orientation in enumerate(ORIENTATIONS):
        fig.add_trace(
            go.Scatter(
                x=strains_to_plot,
                y=[
                    processed_data["ref"][orientation][strain]["params"][1]
                    for strain in STRAINS
                ],
                name="Reference",
                legendgroup="Reference",
                showlegend=(i == 0),
                mode="lines+markers",
                line={"color": "#000000"},
                hovertemplate="%{y:.2f} Å",
            ),
            row=1,
            col=(i + 1),
        )
    for iter, model in enumerate(MODELS):
        color = DEFAULT_PLOTLY_COLORS[iter % len(DEFAULT_PLOTLY_COLORS)]
        for i, orientation in enumerate(ORIENTATIONS):
            fig.add_trace(
                go.Scatter(
                    x=strains_to_plot,
                    y=[
                        processed_data[model][orientation][strain]["params"][1]
                        for strain in STRAINS
                    ],
                    name=model,
                    legendgroup=model,
                    showlegend=(i == 0),
                    mode="lines+markers",
                    line={"color": color},
                    hovertemplate="%{y:.2f} Å",
                ),
                row=1,
                col=(i + 1),
            )
    fig.update_xaxes(
        title_text="Strain / %",
        unifiedhovertitle_text="<b>Strain: </b>%{x:+.2f} %",
    )
    fig.update_yaxes(title_text="Length / Å")
    fig.update_layout(
        title_text="Binding length against strain over various orientations",
        width=1500,
        height=500,
        hovermode="x unified",
    )
    filename = OUT_PATH / "figure_binding_lengths.json"
    fig.write_json(filename)

    return


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

    ref = []
    for orientation in ORIENTATIONS:
        for strain in STRAINS:
            for i in range(len(processed_data["distances"])):
                ref.append(processed_data["ref"][orientation][strain]["energies"][i])

    for model in MODELS:
        prediction = []
        for orientation in ORIENTATIONS:
            for strain in STRAINS:
                for i in range(len(processed_data["distances"])):
                    prediction.append(
                        processed_data[model][orientation][strain]["energies"][i]
                    )
        results[model] = mae(ref, prediction)

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

    ref = []
    for orientation in ORIENTATIONS:
        for strain in STRAINS:
            ref.append(processed_data["ref"][orientation][strain]["params"][0])

    for model in MODELS:
        prediction = []
        for orientation in ORIENTATIONS:
            for strain in STRAINS:
                prediction.append(
                    processed_data[model][orientation][strain]["params"][0]
                )
        if np.isinf(np.max(prediction)):
            results[model] = None
        else:
            results[model] = mae(ref, prediction)

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

    ref = []
    for orientation in ORIENTATIONS:
        for strain in STRAINS:
            ref.append(processed_data["ref"][orientation][strain]["params"][1])

    for model in MODELS:
        prediction = []
        for orientation in ORIENTATIONS:
            for strain in STRAINS:
                prediction.append(
                    processed_data[model][orientation][strain]["params"][1]
                )
        if np.isinf(np.max(prediction)):
            results[model] = None
        else:
            results[model] = mae(ref, prediction)

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


def test_graphene_wetting_under_strain(
    metrics: dict[str, dict], generate_plots_for_app: None
) -> None:
    """
    Run graphene wetting test.

    Parameters
    ----------
    metrics
        All graphene wetting metrics.
    generate_plots_for_app
        Hook for PyTest fixture.
    """
    return
