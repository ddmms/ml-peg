"""
Analyse the QUID benchmark for ligand-pocket interactions.

Puleva, M., Medrano Sandonas, L., Lőrincz, B.D. et al,
Extending quantum-mechanical benchmark accuracy to biological ligand-pocket
interactions,
Nat Commun 16, 8583 (2025). https://doi.org/10.1038/s41467-025-63587-9
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "non_covalent_interactions" / "QUID" / "outputs"
OUT_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "QUID"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KCAL = units.mol / units.kcal


def labels() -> dict[str, list[int]]:
    """
    Get dictionary of info for QUID structures.

    Returns
    -------
    dict[str, list[int]]
        Dictionary of system indices, complex atom counts, and complex charges.
    """
    info = {"all": [], "equilibrium": [], "dissociation": []}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))

            info["all"] = [path.stem for path in xyz_files]
            info["equilibrium"] = [label for label in info["all"] if "_" not in label]
            info["dissociation"] = [label for label in info["all"] if "_" in label]

    return info


LABELS = labels()


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_quid.json",
    title="Interaction energies",
    x_label="Predicted energy / kcal/mol",
    y_label="Reference energy / kcal/mol",
    hoverdata={"Labels": LABELS["all"]},
)
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
        for label in LABELS["all"]:
            atoms = read(CALC_PATH / model_name / f"{label}.xyz", index=0)

            if not ref_stored:
                results["ref"].append(atoms.info["ref_int_energy"][()] * EV_TO_KCAL)

            results[model_name].append(atoms.info["model_int_energy"][()] * EV_TO_KCAL)

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms)

        ref_stored = True
    return results


@pytest.fixture
def total_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for energies for all systems.

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
def equilibrium_mae(interaction_energies) -> dict[str, float]:
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
    equilibrium_indices = [
        i for i, label in enumerate(LABELS["all"]) if label in LABELS["equilibrium"]
    ]

    results = {}
    for model_name in MODELS:
        if interaction_energies[model_name]:
            ref_equilibrium = [
                interaction_energies["ref"][i] for i in equilibrium_indices
            ]
            pred_equilibrium = [
                interaction_energies[model_name][i] for i in equilibrium_indices
            ]
            results[model_name] = mae(ref_equilibrium, pred_equilibrium)
        else:
            results[model_name] = None
    return results


@pytest.fixture
def dissociation_mae(interaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for dissociation systems only.

    Parameters
    ----------
    interaction_energies
        Dictionary of reference and predicted interaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted interaction energy errors for dissociation systems.
    """
    dissociation_indices = [
        i for i, label in enumerate(LABELS["all"]) if label in LABELS["dissociation"]
    ]

    results = {}
    for model_name in MODELS:
        if interaction_energies[model_name]:
            ref_dissociation = [
                interaction_energies["ref"][i] for i in dissociation_indices
            ]
            pred_dissociation = [
                interaction_energies[model_name][i] for i in dissociation_indices
            ]
            results[model_name] = mae(ref_dissociation, pred_dissociation)
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "quid_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    total_mae: dict[str, float],
    equilibrium_mae: dict[str, float],
    dissociation_mae: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    total_mae
        Mean absolute errors for all models.
    equilibrium_mae
        Mean absolute errors for all models for equilibrium systems.
    dissociation_mae
        Mean absolute errors for all models for dissociation systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Equilibrium MAE": equilibrium_mae,
        "Dissociation MAE": dissociation_mae,
        "Overall MAE": total_mae,
    }


def test_quid(metrics: dict[str, dict]) -> None:
    """
    Run QUID test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
