"""Analyse LNCI16 benchmark."""

from __future__ import annotations

import pandas as pd
import pytest

from mlip_testing.analysis.utils.decorators import build_table, plot_parity
from mlip_testing.analysis.utils.utils import mae
from mlip_testing.app import APP_ROOT
from mlip_testing.calcs import CALCS_ROOT
from mlip_testing.calcs.models.models import MODELS

CALC_PATH = CALCS_ROOT / "supramolecular" / "LNCI16" / "outputs"
OUT_PATH = APP_ROOT / "data" / "supramolecular" / "LNCI16"


def get_system_names() -> list[str]:
    """
    Get list of LNCI16 system names.

    Returns
    -------
    list[str]
        List of system names from the first available model results.
    """
    system_names = []
    for model_name in MODELS:
        results_file = CALC_PATH / model_name / "lnci16_results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            if not df.empty:
                system_names = df["system"].tolist()
                break
    return system_names


def get_atom_counts() -> list[int]:
    """
    Get complex atom counts for LNCI16.

    Returns
    -------
    list[int]
        List of complex atom counts from the first available model results.
    """
    for model_name in MODELS:
        results_file = CALC_PATH / model_name / "lnci16_results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            if not df.empty:
                return df["complex_atoms"].tolist()
    return []


def get_charges() -> list[int]:
    """
    Get complex charges for LNCI16.

    Returns
    -------
    list[int]
        List of complex charges from the first available model results.
    """
    for model_name in MODELS:
        results_file = CALC_PATH / model_name / "lnci16_results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            if not df.empty:
                return df["complex_charge"].tolist()
    return []


def get_is_charged() -> list[bool]:
    """
    Get whether systems are charged for LNCI16.

    Returns
    -------
    list[bool]
        List of boolean values indicating if systems are charged.
    """
    for model_name in MODELS:
        results_file = CALC_PATH / model_name / "lnci16_results.csv"
        if results_file.exists():
            df = pd.read_csv(results_file)
            if not df.empty:
                return df["is_charged"].tolist()
    return []


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_interaction_energies.json",
    title="LNCI16 Interaction Energies",
    x_label="Predicted interaction energy / kcal/mol",
    y_label="Reference interaction energy / kcal/mol",
    hoverdata={
        "System": get_system_names(),
        "Complex Atoms": get_atom_counts(),
        "Charge": get_charges(),
        "Charged": get_is_charged(),
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
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        results_file = CALC_PATH / model_name / "lnci16_results.csv"

        if not results_file.exists():
            results[model_name] = []
            continue

        df = pd.read_csv(results_file)
        if df.empty:
            results[model_name] = []
            continue

        # Store predicted energies
        results[model_name] = df["E_int_model_kcal"].tolist()

        # Store reference energies (only once)
        if not ref_stored:
            results["ref"] = df["E_int_ref_kcal"].tolist()
            ref_stored = True

        # Copy individual structure files to app data directory
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)

        # Copy individual structure files
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            import shutil

            for i in range(16):  # LNCI16 has 16 systems
                struct_file = model_dir / f"{i}.xyz"
                if struct_file.exists():
                    shutil.copy(struct_file, structs_dir / f"{i}.xyz")

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
        if interaction_energies[model_name]:
            results[model_name] = mae(
                interaction_energies["ref"], interaction_energies[model_name]
            )
        else:
            results[model_name] = float("nan")
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "lnci16_metrics_table.json",
    metric_tooltips={
        "Model": "Name of the model",
        "MAE": "Mean Absolute Error for all systems (kcal/mol)",
    },
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
