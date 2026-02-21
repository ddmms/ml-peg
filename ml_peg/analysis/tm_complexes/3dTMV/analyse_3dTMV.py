"""Analyse the 3dTMV benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import (
    build_table,
    plot_parity,
)
from ml_peg.analysis.utils.utils import (
    build_d3_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

EV_TO_KCAL = units.mol / units.kcal
CALC_PATH = CALCS_ROOT / "tm_complexes" / "3dTMV" / "outputs"
OUT_PATH = APP_ROOT / "data" / "tm_complexes" / "3dTMV"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

SUBSETS = {
    1: "SR",
    2: "SR",
    3: "SR",
    4: "SR",
    5: "SR",
    6: "SR",
    7: "SR",
    8: "SR",
    9: "SR",
    10: "SR",
    11: "SR",
    12: "SR",
    13: "SR/MR",
    14: "SR/MR",
    15: "SR/MR",
    16: "SR/MR",
    17: "SR/MR",
    18: "SR/MR",
    19: "SR/MR",
    20: "SR/MR",
    21: "SR/MR",
    22: "SR/MR",
    23: "MR",
    24: "MR",
    25: "MR",
    26: "MR",
    27: "MR",
    28: "MR",
}


def labels():
    """
    Get complex ids.

    Returns
    -------
    list
        IDs of the complexes.
    """
    return list(range(1, 29))


@pytest.fixture
def interaction_energies() -> dict[str, list]:
    """
    Get interaction energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted interaction energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}

    ref_stored = False

    for model_name in MODELS:
        for label in range(1, 29):
            atoms = read(CALC_PATH / model_name / f"{label}.xyz")
            if not ref_stored:
                results["ref"].append(atoms.info["ref_ionization_energy"] * EV_TO_KCAL)

            results[model_name].append(
                atoms.info["model_ionization_energy"] * EV_TO_KCAL
            )

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms)

        ref_stored = True
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_3dtmv.json",
    title="Ionization energies",
    x_label="Predicted ionization energy / kcal/mol",
    y_label="Reference ionization energy / kcal/mol",
    hoverdata={
        "Labels": labels(),
    },
)
def ionization_energies() -> dict[str, list]:
    """
    Get ionization energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        for complex_id in labels():
            atoms = read(CALC_PATH / model_name / f"{complex_id}.xyz")
            model_ion_energy = atoms.info["model_ionization_energy"]
            ref_ion_energy = atoms.info["ref_ionization_energy"]
            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{complex_id}.xyz", atoms)
            results[model_name].append(model_ion_energy * EV_TO_KCAL)
            if not ref_stored:
                results["ref"].append(ref_ion_energy * EV_TO_KCAL)
        ref_stored = True
    return results


@pytest.fixture
def sr_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for SR subset.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted energy errors for all models.
    """
    results = {}

    for model_name in MODELS:
        subsampled_ref_e = [
            interaction_energies["ref"][i] for i in labels() if SUBSETS[i] == "SR"
        ]
        subsampled_model_e = [
            interaction_energies[model_name][i] for i in labels() if SUBSETS[i] == "SR"
        ]
        results[model_name] = mae(subsampled_ref_e, subsampled_model_e)
    return results


@pytest.fixture
def mr_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for MR subset.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted energy errors for all models.
    """
    results = {}

    for model_name in MODELS:
        subsampled_ref_e = [
            interaction_energies["ref"][i - 1] for i in labels() if SUBSETS[i] == "MR"
        ]
        subsampled_model_e = [
            interaction_energies[model_name][i - 1]
            for i in labels()
            if SUBSETS[i] == "MR"
        ]
        results[model_name] = mae(subsampled_ref_e, subsampled_model_e)
    return results


@pytest.fixture
def sr_mr_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for SR/MR subset.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted energy errors for all models.
    """
    results = {}

    for model_name in MODELS:
        subsampled_ref_e = [
            interaction_energies["ref"][i] for i in labels() if SUBSETS[i] == "SR/MR"
        ]
        subsampled_model_e = [
            interaction_energies[model_name][i]
            for i in labels()
            if SUBSETS[i] == "SR/MR"
        ]
        results[model_name] = mae(subsampled_ref_e, subsampled_model_e)
    return results


@pytest.fixture
def total_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for all conmplexes.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted energy errors for all models.
    """
    results = {}

    for model_name in MODELS:
        results[model_name] = mae(
            interaction_energies["ref"], interaction_energies[model_name]
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "3dtmv_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(
    total_mae: dict[str, float],
    sr_mae: dict[str, float],
    mr_mae: dict[str, float],
    sr_mr_mae: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    total_mae
        Mean absolute errors for all models, all complexes.
    sr_mae
        Mean absolute errors for all models, single-reference complexes.
    mr_mae
        Mean absolute errors for all models, multi-reference complexes.
    sr_mr_mae
        Mean absolute errors for all models, intermediate complexes.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Overall MAE": total_mae,
        "SR MAE": sr_mae,
        "MR MAE": mr_mae,
        "SR/MR MAE": sr_mr_mae,
    }


def test_3dtmv(metrics: dict[str, dict]) -> None:
    """
    Run 3dTMV test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
