"""Analyse OC157 benchmark."""

from __future__ import annotations

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "surfaces" / "OC157" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "OC157"


def get_relative_energies(energies: list) -> list:
    """
    Get pairs of relative energies for one triplet.

    Parameters
    ----------
    energies
        List of three energies to find relative energies for.

    Returns
    -------
    list
        All combinations of relative energies.
    """
    return [
        energies[1] - energies[0],
        energies[2] - energies[0],
        energies[2] - energies[1],
    ]


def compositions() -> list:
    """
    Get list of compositions.

    Returns
    -------
    list
        List of all compositions.
    """
    all_compositions = []
    for model_name in MODELS:
        for system_path in (CALC_PATH / model_name).glob("*.xyz"):
            structs = read(system_path, index=":")
            compositions = [atoms.info["composition"] for atoms in structs]
            all_compositions.extend(compositions)
        break
    return all_compositions


def labels() -> list:
    """
    Get list of labels.

    Returns
    -------
    list
        List of all relative energy labels.
    """
    for model_name in MODELS:
        n_systems = len(list((CALC_PATH / model_name).glob("*.xyz")))
        break
    return ["E_2 - E_1", "E_3 - E_2", "E_3 - E_2"] * n_systems


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_rel_energies.json",
    title="Relative energies",
    x_label="Predicted relative energy / meV",
    y_label="Reference relative energy / meV",
    hoverdata={
        "Composition": compositions(),
        "Labels": labels(),
    },
)
def relative_energies() -> dict[str, list]:
    """
    Get energy differences for all triplets.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted relative energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False
    for model_name in MODELS:
        for i, system_path in enumerate((CALC_PATH / model_name).glob("*.xyz")):
            structs = read(system_path, index=":")
            pred_energies = [atoms.get_potential_energy() for atoms in structs]
            results[model_name].extend(get_relative_energies(pred_energies))
            if not ref_stored:
                ref_energies = [atoms.info["ref_energy"] for atoms in structs]
                results["ref"].extend(get_relative_energies(ref_energies))

            # Write structures in order as glob is unsorted
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{i}.xyz", structs)
        ref_stored = True
    return results


@pytest.fixture
def oc157_mae(relative_energies) -> dict[str, float]:
    """
    Get mean average error across all triplets.

    Parameters
    ----------
    relative_energies
        Dictionary of reference and predicted relative energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted relative energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            relative_energies["ref"], relative_energies[model_name]
        )
    return results


@pytest.fixture
def ranking_error(relative_energies: dict[str, list]) -> dict[str, float]:
    """
    Get ranking error across all triplets.

    Parameters
    ----------
    relative_energies
        Dictionary of reference and predicted relative energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted ranking errors for all models.
    """
    results = {}
    ref_min = []
    ref_max = []
    for i in range(len(relative_energies["ref"]) // 3):
        ref_energies = relative_energies["ref"][3 * i : 3 * i + 3]
        ref_min.append(np.argmin(ref_energies))
        ref_max.append(np.argmax(ref_energies))

    for model_name in MODELS:
        pred_min = []
        pred_max = []
        for i in range(len(relative_energies[model_name]) // 3):
            pred_energies = relative_energies[model_name][3 * i : 3 * i + 3]
            pred_min.append(np.argmin(pred_energies))
            pred_max.append(np.argmax(pred_energies))

        results[model_name] = (
            1
            - 0.5 * np.mean(np.array(ref_min) == np.array(pred_min))
            - 0.5 * np.mean(np.array(ref_max) == np.array(pred_max))
        )

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "oc157_metrics_table.json",
    metric_tooltips={
        "Model": "Name of the model",
        "MAE": "Mean Absolute Error (meV)",
        "Ranking Error": "Error in ranking stability across triplets",
    },
)
def metrics(
    oc157_mae: dict[str, float], ranking_error: dict[str, float]
) -> dict[str, dict]:
    """
    Get all OC157 metrics.

    Parameters
    ----------
    oc157_mae
        Mean absolute errors for all models.
    ranking_error
        Ranking errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": oc157_mae,
        "Ranking Error": ranking_error,
    }


def test_oc157(metrics: dict[str, dict]) -> None:
    """
    Run OC157 test.

    Parameters
    ----------
    metrics
        All OC157 metrics.
    """
    return
