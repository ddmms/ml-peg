"""Analyse PLA15 benchmark."""

from __future__ import annotations

import pytest

from mlip_testing.analysis.utils.decorators import build_table, plot_parity
from mlip_testing.analysis.utils.utils import mae
from mlip_testing.app import APP_ROOT
from mlip_testing.calcs import CALCS_ROOT
from mlip_testing.calcs.models.models import MODELS

CALC_PATH = CALCS_ROOT / "supramolecular" / "PLA15" / "outputs"
OUT_PATH = APP_ROOT / "data" / "supramolecular" / "PLA15"


def get_system_identifiers() -> list[str]:
    """
    Get list of PLA15 system identifiers.

    Returns
    -------
    list[str]
        List of system identifiers from structure files.
    """
    from ase.io import read

    system_identifiers = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    system_identifiers.append(
                        atoms.info.get("identifier", f"system_{xyz_file.stem}")
                    )
                break
    return system_identifiers


def get_atom_counts() -> list[int]:
    """
    Get complex atom counts for PLA15.

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
    Get complex charges for PLA15.

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


def get_protein_atom_counts() -> list[int]:
    """
    Get protein atom counts for PLA15.

    Returns
    -------
    list[int]
        List of protein atom counts from structure files.
    """
    from ase.io import read

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                protein_counts = []
                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    protein_counts.append(atoms.info.get("protein_atoms", 0))
                return protein_counts
    return []


def get_ligand_atom_counts() -> list[int]:
    """
    Get ligand atom counts for PLA15.

    Returns
    -------
    list[int]
        List of ligand atom counts from structure files.
    """
    from ase.io import read

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                ligand_counts = []
                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    ligand_counts.append(atoms.info.get("ligand_atoms", 0))
                return ligand_counts
    return []


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_interaction_energies.json",
    title="PLA15 Protein-Ligand Interaction Energies",
    x_label="Predicted interaction energy / kcal/mol",
    y_label="Reference interaction energy / kcal/mol",
    hoverdata={
        "System": get_system_identifiers(),
        "Complex Atoms": get_atom_counts(),
        "Protein Atoms": get_protein_atom_counts(),
        "Ligand Atoms": get_ligand_atom_counts(),
        "Charge": get_charges(),
    },
)
def interaction_energies() -> dict[str, list]:
    """
    Get interaction energies for all PLA15 systems.

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
def pla15_mae(interaction_energies) -> dict[str, float]:
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
    filename=OUT_PATH / "pla15_metrics_table.json",
    metric_tooltips={
        "Model": "Name of the model",
        "MAE": "Mean Absolute Error for all systems (kcal/mol)",
    },
)
def metrics(pla15_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all PLA15 metrics.

    Parameters
    ----------
    pla15_mae
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": pla15_mae,
    }


def test_pla15(metrics: dict[str, dict]) -> None:
    """
    Run PLA15 test.

    Parameters
    ----------
    metrics
        All PLA15 metrics.
    """
    return
