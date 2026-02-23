"""Analyse Elemental TM Vacancy Formation Energies benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "elemental_tm_vacancies" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "elemental_tm_vacancies"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def get_system_names() -> list[str]:
    """
    Get list of all system names.

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
                    system_names.append(atoms.info["system"])
                break
    return system_names


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_vacancy_formation_energies.json",
    title="Elemental TM Vacancy Formation Energies",
    x_label="Predicted Vacancy Formation Energy / eV",
    y_label="Reference Vacancy Formation Energy / eV",
    hoverdata={
        "System": get_system_names(),
    },
)
def vacancy_formation_energies() -> dict[str, list]:
    """
    Get vacancy formation energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted vacancy formation energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_files = sorted(model_dir.glob("*.xyz"))
        if not xyz_files:
            continue

        for xyz_file in xyz_files:
            structs = read(xyz_file, index=":")

            bulk_energy = structs[0].get_potential_energy()
            num_atoms = len(structs[0])
            system = structs[0].info["system"]
            vacancy_energy = structs[1].get_potential_energy()

            vacancy_formation_energy = (
                vacancy_energy - ((num_atoms - 1) / num_atoms) * bulk_energy
            )
            results[model_name].append(vacancy_formation_energy)

            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system}.xyz", structs)

            # Store reference energies (only once)
            if not ref_stored:
                results["ref"].append(structs[0].info["ref"])

        ref_stored = True

    return results


@pytest.fixture
def vacancy_formation_errors(vacancy_formation_energies) -> dict[str, float]:
    """
    Get mean absolute error for vacancy formation energies.

    Parameters
    ----------
    vacancy_formation_energies
        Dictionary of reference and predicted vacancy formation energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted vacancy formation energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if vacancy_formation_energies[model_name]:
            results[model_name] = mae(
                vacancy_formation_energies["ref"],
                vacancy_formation_energies[model_name],
            )
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "vacancy_formation_energies_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(vacancy_formation_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all Vacancy Formation Energies metrics.

    Parameters
    ----------
    vacancy_formation_errors
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": vacancy_formation_errors,
    }


def test_vacancy_formation_energies(metrics: dict[str, dict]) -> None:
    """
    Run test to assess a model on elemental TM vac. form. energies.

    Parameters
    ----------
    metrics
        All elemental TM vacancy formation energies metrics.
    """
    return
