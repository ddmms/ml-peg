"""Analyse S30L benchmark."""

from __future__ import annotations

import pytest

from mlip_testing.analysis.utils.decorators import build_table, plot_parity
from mlip_testing.analysis.utils.utils import mae
from mlip_testing.app import APP_ROOT
from mlip_testing.calcs import CALCS_ROOT
from mlip_testing.calcs.models.models import MODELS

CALC_PATH = CALCS_ROOT / "supramolecular" / "S30L" / "outputs"
OUT_PATH = APP_ROOT / "data" / "supramolecular" / "S30L"


def get_system_indices() -> list[int]:
    """
    Get list of S30L system indices.

    Returns
    -------
    list[int]
        List of system indices (1-30) from structure files.
    """
    system_indices = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = list(model_dir.glob("*.xyz"))
            if xyz_files:
                # S30L has 30 systems indexed 0-29 in files, 1-30 in original data
                system_indices = list(range(1, len(xyz_files) + 1))
                break
    return system_indices


def get_atom_counts() -> list[int]:
    """
    Get complex atom counts for S30L.

    Returns
    -------
    list[int]
        List of complex atom counts from structure files.
    """
    from ase.io import read

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                atom_counts = []
                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    atom_counts.append(len(atoms))
                return atom_counts
    return []


def get_charges() -> list[int]:
    """
    Get complex charges for S30L.

    Returns
    -------
    list[int]
        List of complex charges from structure files.
    """
    from ase.io import read

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                charges = []
                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    charges.append(atoms.info.get("complex_charge", 0))
                return charges
    return []


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_interaction_energies.json",
    title="S30L Interaction Energies",
    x_label="Predicted interaction energy / kcal/mol",
    y_label="Reference interaction energy / kcal/mol",
    hoverdata={
        "System": get_system_indices(),
        "Complex Atoms": get_atom_counts(),
        "Charge": get_charges(),
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
    from ase.io import read

    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            results[model_name] = []
            continue

        xyz_files = sorted(model_dir.glob("*.xyz"))
        if not xyz_files:
            results[model_name] = []
            continue

        model_energies = []
        ref_energies = []

        for xyz_file in xyz_files:
            atoms = read(xyz_file)
            model_energies.append(atoms.info["E_int_model_kcal"])
            if not ref_stored:
                ref_energies.append(atoms.info["E_int_ref_kcal"])

        results[model_name] = model_energies

        # Store reference energies (only once)
        if not ref_stored:
            results["ref"] = ref_energies
            ref_stored = True

        # Copy individual structure files to app data directory
        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)

        # Copy individual structure files
        import shutil

        for i, xyz_file in enumerate(xyz_files):
            shutil.copy(xyz_file, structs_dir / f"{i}.xyz")

    return results


@pytest.fixture
def s30l_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for interaction energies (overall).

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
    # Get charges for filtering
    charges = get_charges()
    charged_indices = [i for i, charge in enumerate(charges) if charge != 0]

    results = {}
    for model_name in MODELS:
        if interaction_energies[model_name] and charged_indices:
            ref_charged = [interaction_energies["ref"][i] for i in charged_indices]
            pred_charged = [
                interaction_energies[model_name][i] for i in charged_indices
            ]
            results[model_name] = mae(ref_charged, pred_charged)
        else:
            results[model_name] = float("nan")
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
    # Get charges for filtering
    charges = get_charges()
    neutral_indices = [i for i, charge in enumerate(charges) if charge == 0]

    results = {}
    for model_name in MODELS:
        if interaction_energies[model_name] and neutral_indices:
            ref_neutral = [interaction_energies["ref"][i] for i in neutral_indices]
            pred_neutral = [
                interaction_energies[model_name][i] for i in neutral_indices
            ]
            results[model_name] = mae(ref_neutral, pred_neutral)
        else:
            results[model_name] = float("nan")
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "s30l_metrics_table.json",
    metric_tooltips={
        "Model": "Name of the model",
        "MAE": "Mean Absolute Error for all systems (kcal/mol)",
        "Charged MAE": "Mean Absolute Error for charged systems (kcal/mol)",
        "Neutral MAE": "Mean Absolute Error for neutral systems (kcal/mol)",
    },
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
        "MAE": s30l_mae,
        "Charged MAE": s30l_charged_mae,
        "Neutral MAE": s30l_neutral_mae,
    }


def test_s30l(metrics: dict[str, dict]) -> None:
    """
    Run S30L test.

    Parameters
    ----------
    metrics
        All S30L metrics.
    """
    return
