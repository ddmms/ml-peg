"""Analyse X23 benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
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
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "molecular_crystal" / "X23" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_crystal" / "X23"
FILTER_DATA_PATH = OUT_PATH / "x23_element_filter_data.json"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    index="0",
    info_keys=["system"],
    out_path=OUT_PATH,
)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_lattice_energies.json",
    title="X23 Lattice Energies",
    x_label="Predicted lattice energy / kJ/mol",
    y_label="Reference lattice energy / kJ/mol",
    hoverdata={
        "System": INFO["system"],
    },
)
def lattice_energies() -> dict[str, list]:
    """
    Get lattice energies for all X23 systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted lattice energies.
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

            solid_energy = structs[0].get_potential_energy()
            num_molecules = structs[0].info["num_molecules"]
            system = structs[0].info["system"]
            molecule_energy = structs[1].get_potential_energy()

            lattice_energy = (solid_energy / num_molecules) - molecule_energy
            results[model_name].append(lattice_energy * EV_TO_KJ_PER_MOL)

            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system}.xyz", structs)

            # Store reference energies (only once)
            if not ref_stored:
                results["ref"].append(structs[0].info["ref"])

        ref_stored = True

    return results


def get_errors(lattice_energies: dict[str, list]) -> dict[str, float]:
    """
    Get mean absolute error for lattice energies.

    Parameters
    ----------
    lattice_energies
        Dictionary of reference and predicted lattice energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted lattice energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if lattice_energies.get(model_name):
            results[model_name] = mae(
                lattice_energies["ref"], lattice_energies[model_name]
            )
        else:
            results[model_name] = None
    return results


def get_metrics(lattice_energies: dict[str, list]) -> dict[str, dict]:
    """
    Get all X23 metrics.

    Parameters
    ----------
    lattice_energies
        Dictionary of reference and predicted lattice energies.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": get_errors(lattice_energies),
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "x23_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(lattice_energies: dict[str, list]) -> dict[str, dict]:
    """
    Get all X23 metrics.

    Parameters
    ----------
    lattice_energies
        Dictionary of reference and predicted lattice energies.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return get_metrics(lattice_energies)


def test_x23(metrics: dict[str, dict]) -> None:
    """
    Run X23 test.

    Parameters
    ----------
    metrics
        All X23 metrics.
    """
    return
