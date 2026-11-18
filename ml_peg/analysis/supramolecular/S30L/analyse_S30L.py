"""Analyse S30L benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map()

CALC_PATH = CALCS_ROOT / "supramolecular" / "S30L" / "outputs"
OUT_PATH = APP_ROOT / "data" / "supramolecular" / "S30L"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Constants
EV_TO_KCAL_PER_MOL = units.mol / units.kcal

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    info_keys=["complex_charge"],
    include_filenames=True,
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
    per_file_info={
        "counts": lambda structs: len(structs[0]),
        "ref_int_energy": lambda structs: structs[0].info["E_int_ref"],
    },
)

# Reference values are taken from the mock calculation, which is always complete
REF_ENERGIES = [energy * EV_TO_KCAL_PER_MOL for energy in INFO["ref_int_energy"]]
# S30L systems indexed 0-29 in files, 1-30 in original data
INDICES = list(range(1, len(INFO["complex_charge"]) + 1))


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_interaction_energies.json",
    title="S30L Interaction Energies",
    x_label="Predicted interaction energy / kcal/mol",
    y_label="Reference interaction energy / kcal/mol",
    hoverdata={
        "System": INDICES,
        "Complex Atoms": INFO["counts"],
        "Charge": INFO["complex_charge"],
    },
)
def interaction_energies() -> dict[str, list]:
    """
    Get interaction energies for all S30L systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted interaction energies.
    """
    results = {"ref": list(REF_ENERGIES)} | {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        for label in INFO["filenames"]:
            xyz_file = CALC_PATH / model_name / f"{label}.xyz"
            if not xyz_file.exists():
                results[model_name].append(float("nan"))
                continue

            atoms = read(xyz_file)
            results[model_name].append(atoms.info["E_int_model"] * EV_TO_KCAL_PER_MOL)

    return results


@pytest.fixture
def s30l_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for interaction energies for all systems.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            interaction_energies["ref"], interaction_energies[model_name]
        )
    return results


@pytest.fixture
def s30l_charged_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for charged systems only.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for charged systems.
    """
    charged_indices = [
        i for i, charge in enumerate(INFO["complex_charge"]) if charge != 0
    ]

    results = {}
    for model_name in MODELS:
        ref_charged = [interaction_energies["ref"][i] for i in charged_indices]
        pred_charged = [interaction_energies[model_name][i] for i in charged_indices]
        results[model_name] = mae(ref_charged, pred_charged)
    return results


@pytest.fixture
def s30l_neutral_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for neutral systems only.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for neutral systems.
    """
    neutral_indices = [
        i for i, charge in enumerate(INFO["complex_charge"]) if charge == 0
    ]

    results = {}
    for model_name in MODELS:
        ref_neutral = [interaction_energies["ref"][i] for i in neutral_indices]
        pred_neutral = [interaction_energies[model_name][i] for i in neutral_indices]
        results[model_name] = mae(ref_neutral, pred_neutral)
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "s30l_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    s30l_mae: dict[str, float],
    s30l_charged_mae: dict[str, float],
    s30l_neutral_mae: dict[str, float],
) -> dict[str, dict]:
    """
    Get all S30L metrics.

    Parameters
    ----------
    s30l_mae
        Mean absolute errors for all systems.
    s30l_charged_mae
        Mean absolute errors for charged systems.
    s30l_neutral_mae
        Mean absolute errors for neutral systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Neutral MAE": s30l_neutral_mae,
        "Charged MAE": s30l_charged_mae,
        "Overall MAE": s30l_mae,
    }


@pytest.mark.framework("mace-multihead", "mace-polar-1")
def test_s30l(metrics: dict[str, dict]) -> None:
    """
    Run S30L test.

    Parameters
    ----------
    metrics
        All S30L metrics.
    """
    return
