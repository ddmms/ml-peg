"""Analyse PLA15 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "supramolecular" / "PLA15" / "outputs"
OUT_PATH = APP_ROOT / "data" / "supramolecular" / "PLA15"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KCAL = units.mol / units.kcal


def get_info() -> dict[str, list[int]]:
    """
    Get dictionary of info for PLA15 structures.

    Returns
    -------
    dict[str, list[int]]
        Dictionary of info.
    """
    info = {
        "identifiers": [],
        "complex_atoms": [],
        "complex_charges": [],
        "ligand_atoms": [],
        "ligand_charges": [],
        "protein_atoms": [],
        "protein_charges": [],
        "interaction_types": [],
    }

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                for xyz_file in xyz_files:
                    atoms = read(xyz_file, index=":")
                    info["identifiers"].append(atoms[0].info["identifier"])
                    info["interaction_types"].append(atoms[0].info["interaction_type"])

                    info["complex_atoms"].append(len(atoms[0]))
                    info["complex_charges"].append(atoms[0].info["charge"])

                    info["ligand_atoms"].append(len(atoms[1]))
                    info["ligand_charges"].append(atoms[1].info["charge"])

                    info["protein_atoms"].append(len(atoms[2]))
                    info["protein_charges"].append(atoms[2].info["charge"])

                return info
    return info


INFO = get_info()


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_interaction_energies.json",
    title="PLA15 Protein-Ligand Interaction Energies",
    x_label="Predicted interaction energy / kcal/mol",
    y_label="Reference interaction energy / kcal/mol",
    hoverdata={
        "System": INFO["identifiers"],
        "Complex Atoms": INFO["complex_atoms"],
        "Ligand Atoms": INFO["ligand_atoms"],
        "Protein Atoms": INFO["protein_atoms"],
        "Complex Charge": INFO["complex_charges"],
        "Ligand Charge": INFO["ligand_charges"],
        "Protein Charge": INFO["protein_charges"],
        "Interaction Type": INFO["interaction_types"],
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
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            results[model_name] = []
            continue

        for struct_path in sorted(model_dir.glob("*.xyz")):
            atoms = read(struct_path, index=0)

            if not ref_stored:
                results["ref"].append(atoms.info["ref_int_energy"] * EV_TO_KCAL)

            results[model_name].append(atoms.info["model_int_energy"] * EV_TO_KCAL)

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{struct_path.stem}.xyz", atoms)

        ref_stored = True
    return results


@pytest.fixture
def pla15_r2(interaction_energies) -> dict[str, float]:
    """
    Get Pearson's r² for interaction energies.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of Pearson's r² values for all models.
    """
    from scipy.stats import pearsonr

    results = {}
    for model_name in MODELS:
        if interaction_energies[model_name]:
            r, _ = pearsonr(
                interaction_energies["ref"], interaction_energies[model_name]
            )
            results[model_name] = r**2
        else:
            results[model_name] = None
    return results


@pytest.fixture
def pla15_mae(interaction_energies) -> dict[str, float]:
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
            results[model_name] = None
    return results


@pytest.fixture
def pla15_ion_ion_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for ion-ion interactions.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for ion-ion systems.
    """
    # Get interaction types for filtering
    interaction_types = INFO["interaction_types"]
    ion_ion_indices = [
        i for i, itype in enumerate(interaction_types) if itype == "ion-ion"
    ]

    results = {}
    for model_name in MODELS:
        if interaction_energies[model_name] and ion_ion_indices:
            ref_ion_ion = [interaction_energies["ref"][i] for i in ion_ion_indices]
            pred_ion_ion = [
                interaction_energies[model_name][i] for i in ion_ion_indices
            ]
            results[model_name] = mae(ref_ion_ion, pred_ion_ion)
        else:
            results[model_name] = None
    return results


@pytest.fixture
def pla15_ion_neutral_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for ion-neutral interactions.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for ion-neutral systems.
    """
    # Get interaction types for filtering
    interaction_types = INFO["interaction_types"]
    ion_neutral_indices = [
        i for i, itype in enumerate(interaction_types) if itype == "ion-neutral"
    ]

    results = {}
    for model_name in MODELS:
        if interaction_energies[model_name] and ion_neutral_indices:
            ref_ion_neutral = [
                interaction_energies["ref"][i] for i in ion_neutral_indices
            ]
            pred_ion_neutral = [
                interaction_energies[model_name][i] for i in ion_neutral_indices
            ]
            results[model_name] = mae(ref_ion_neutral, pred_ion_neutral)
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "pla15_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(
    pla15_mae: dict[str, float],
    pla15_r2: dict[str, float],
    pla15_ion_ion_mae: dict[str, float],
    pla15_ion_neutral_mae: dict[str, float],
) -> dict[str, dict]:
    """
    Get all PLA15 metrics.

    Parameters
    ----------
    pla15_mae
        Mean absolute errors for all systems.
    pla15_r2
        R² values for all systems.
    pla15_ion_ion_mae
        Mean absolute errors for ion-ion interactions.
    pla15_ion_neutral_mae
        Mean absolute errors for ion-neutral interactions.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": pla15_mae,
        "R²": pla15_r2,
        "Ion-Ion MAE": pla15_ion_ion_mae,
        "Ion-Neutral MAE": pla15_ion_neutral_mae,
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
