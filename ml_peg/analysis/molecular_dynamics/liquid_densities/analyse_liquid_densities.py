"""Analyse the liquid densities benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase import Atoms, units
from ase.data import atomic_numbers
from ase.io import write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "liquid_densities"
AU_TO_G_CM3 = 1e24 / units.mol
G_CM3_TO_AU = 1 / AU_TO_G_CM3

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

DATA_PATH = (
    download_s3_data(
        filename="liquid_densities.zip",
        key="inputs/molecular_dynamics/liquid_densities/liquid_densities.zip",
    )
    / "liquid_densities"
)


def labels() -> list:
    """
    Get list of system names.

    Returns
    -------
    list
        List of all system names.
    """
    with open(DATA_PATH / "liquid_densities.json") as f:
        data = json.load(f)
    return list(data.keys())


def get_atoms(fname):
    """
    Get atoms from the json file.

    Parameters
    ----------
    fname
        Path to the json file.

    Returns
    -------
    atoms
        ASE atoms object of the starting system.
    """
    with open(fname) as file:
        data = json.load(file)
    numbers = np.array([atomic_numbers[symbol] for symbol in data["elements"]])
    positions = np.array(data["coordinates"])
    cell = np.array(data["lattice"]).reshape(3, 3)
    atoms = Atoms(numbers=numbers, positions=positions)
    atoms.set_cell(cell)
    atoms.set_pbc(True)
    return atoms


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_liquid_densities.json",
    title="Densities",
    x_label="Predicted density / kcal/mol",
    y_label="Reference density / kcal/mol",
    hoverdata={
        "Labels": labels(),
    },
)
def liquid_densities() -> dict[str, list]:
    """
    Get liquid densities for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        for label in labels():
            atoms = get_atoms(
                DATA_PATH / f"equilibrated_structures/{label.replace(',', '_')}.json"
            )
            with open(DATA_PATH / "liquid_densities.json") as f:
                data = json.load(f)
                atoms.info["ref_density"] = data[label]["experiment"]
                atoms.info["model_density"] = data[label][model_name]
                results[model_name].append(data[label][model_name])
                if not ref_stored:
                    results["ref"].append(data[label]["experiment"])

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms)
        ref_stored = True
    return results


@pytest.fixture
def get_mae(liquid_densities) -> dict[str, float]:
    """
    Get mean absolute error for conformer energies.

    Parameters
    ----------
    liquid_densities
        Dictionary of reference and predicted conformer energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted conformer energies errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(liquid_densities["ref"], liquid_densities[model_name])
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "liquid_densities_metrics_table.json",
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


def test_liquid_densities(metrics: dict[str, dict]) -> None:
    """
    Run liquid densities test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
