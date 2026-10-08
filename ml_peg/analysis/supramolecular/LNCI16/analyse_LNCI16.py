"""Analyse LNCI16 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "supramolecular" / "LNCI16" / "outputs"
OUT_PATH = APP_ROOT / "data" / "supramolecular" / "LNCI16"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    info_keys=["system", "complex_charge"],
    include_filenames=True,
    per_file_info={
        "complex_atoms": lambda structs: len(structs[0]),
        "ref_int_energy": lambda structs: structs[0].info["E_int_ref_kcal"],
    },
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)

# Reference values are taken from the mock calculation, which is always complete
REF_ENERGIES = INFO["ref_int_energy"]


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_interaction_energies.json",
    title="LNCI16 Interaction Energies",
    x_label="Predicted interaction energy / kcal/mol",
    y_label="Reference interaction energy / kcal/mol",
    hoverdata={
        "System": INFO["system"],
        "Complex Atoms": INFO["complex_atoms"],
        "Charge": INFO["complex_charge"],
        "Charged": [charge != 0 for charge in INFO["complex_charge"]],
    },
)
def interaction_energies() -> dict[str, list]:
    """
    Get interaction energies for all LNCI16 systems.

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
            results[model_name].append(atoms.info["E_int_model_kcal"])

    return results


@pytest.fixture
def lnci16_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for interaction energies.

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
@build_table(
    filename=OUT_PATH / "lnci16_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(lnci16_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all LNCI16 metrics.

    Parameters
    ----------
    lnci16_mae
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": lnci16_mae,
    }


def test_lnci16(metrics: dict[str, dict]) -> None:
    """
    Run LNCI16 test.

    Parameters
    ----------
    metrics
        All LNCI16 metrics.
    """
    return
