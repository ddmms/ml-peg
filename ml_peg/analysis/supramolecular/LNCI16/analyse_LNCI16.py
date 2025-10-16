"""Analyse LNCI16 benchmark."""

from __future__ import annotations

from ase.io import read
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "supramolecular" / "LNCI16" / "outputs"
OUT_PATH = APP_ROOT / "data" / "supramolecular" / "LNCI16"


def get_system_names() -> list[str]:
    """
    Get list of LNCI16 system names.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """
    system_names = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    system_names.append(
                        atoms.info.get("system", f"system_{xyz_file.stem}")
                    )
                break
    return system_names


def get_atom_counts() -> list[int]:
    """
    Get complex atom counts for LNCI16.

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
    Get complex charges for LNCI16.

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


def get_is_charged() -> list[bool]:
    """
    Get whether systems are charged for LNCI16.

    Returns
    -------
    list[bool]
        List of boolean values indicating if systems are charged.
    """
    from ase.io import read

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                is_charged = []
                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    charge = atoms.info.get("complex_charge", 0)
                    is_charged.append(charge != 0)
                return is_charged
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
            results[model_name] = None
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
