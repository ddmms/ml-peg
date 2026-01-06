"""Analyse IONPI19 ion - pi system interaction benchmark. 10.1039/D1CP01333E."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_d3_name_map,
    load_metrics_config,
    mae,
    rmse,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

KCAL_TO_EV = units.kcal / units.mol
EV_TO_KCAL = 1 / KCAL_TO_EV
CALC_PATH = CALCS_ROOT / "non_covalent_interactions" / "ionpi19" / "outputs"
OUT_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "ionpi19"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

species = {
    1: ["1_AB", "1_A", "1_B"],
    2: ["2_AB", "2_A", "2_B"],
    3: ["3_AB", "3_A", "3_B"],
    4: ["4_AB", "4_A", "4_B"],
    5: ["5_AB", "5_A", "5_B"],
    6: ["6_AB", "6_A", "6_B"],
    7: ["7_AB", "7_A", "7_B"],
    8: ["8_AB", "8_A", "8_B"],
    9: ["9_AB", "9_A", "9_B"],
    10: ["10_AB", "10_A", "10_B"],
    11: ["11_AB", "11_A", "11_B"],
    12: ["12_AB", "12_A", "12_B"],
    13: ["13_AB", "13_A", "13_B"],
    14: ["14_AB", "14_A", "14_B"],
    15: ["15_AB", "15_A", "15_B"],
    16: ["16_AB", "15_A", "16_B"],
    17: ["17_AB", "17_A", "17_B"],
    18: ["18_A", "18_B"],
    19: ["19_A", "19_B"],
}

stoich = {
    1: [1, -1, -1],
    2: [1, -1, -1],
    3: [1, -1, -1],
    4: [1, -1, -1],
    5: [1, -1, -1],
    6: [1, -1, -1],
    7: [1, -1, -1],
    8: [1, -1, -1],
    9: [1, -1, -1],
    10: [1, -1, -1],
    11: [1, -1, -1],
    12: [1, -1, -1],
    13: [1, -1, -1],
    14: [1, -1, -1],
    15: [1, -1, -1],
    16: [1, -1, -1],
    17: [1, -1, -1],
    18: [1, -1],
    19: [1, -1],
}


def labels() -> list:
    """
    Get list of system identifiers.

    Returns
    -------
    list
        List of system identifiers (one per system, 19 total).
    """
    # Systems 1-17 have complexes (AB), show those labels
    # Systems 18-19 only have fragments, show system numbers
    labels_list = []
    for i in range(1, 18):
        labels_list.append(f"{i}_AB")
    labels_list.append("18")
    labels_list.append("19")
    return labels_list


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_ionpi19.json",
    title="Energies",
    x_label="Predicted energy / eV",
    y_label="Reference energy / eV",
    hoverdata={
        "Labels": labels(),
    },
)
def conformer_energies() -> dict[str, list]:
    """
    Get conformer energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        for system in range(1, 20):
            model_int_energy = 0
            for spec, stoic in zip(species[system], stoich[system], strict=False):
                label = spec
                atoms = read(CALC_PATH / model_name / f"{label}.xyz")
                model_int_energy += atoms.info["model_energy"] * stoic
                ref_int_energy = atoms.info["ref_int_energy"]

                # Write structures for app
                structs_dir = OUT_PATH / model_name
                structs_dir.mkdir(parents=True, exist_ok=True)
                write(structs_dir / f"{label}.xyz", atoms)
            results[model_name].append(model_int_energy)
            if not ref_stored:
                results["ref"].append(ref_int_energy)
        ref_stored = True
    return results


@pytest.fixture
def get_mae(conformer_energies) -> dict[str, float]:
    """
    Get mean absolute error for conformer energies.

    Parameters
    ----------
    conformer_energies
        Dictionary of reference and predicted conformer energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted conformer energies errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            conformer_energies["ref"], conformer_energies[model_name]
        )
    return results


@pytest.fixture
def get_rmse(conformer_energies) -> dict[str, float]:
    """
    Get root mean square error for conformer energies.

    Parameters
    ----------
    conformer_energies
        Dictionary of reference and predicted conformer energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted conformer energies errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = rmse(
            conformer_energies["ref"], conformer_energies[model_name]
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "ionpi19_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(get_mae: dict[str, float], get_rmse: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute errors for all models.

    get_rmse
        Root Mean Square Error for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_mae,
        "RMSE": get_rmse,
    }


def test_ionpi19(metrics: dict[str, dict]) -> None:
    """
    Run IONPI19 test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
