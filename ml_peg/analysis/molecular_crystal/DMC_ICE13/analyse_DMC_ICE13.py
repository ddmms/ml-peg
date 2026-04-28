"""Analyse DMC-ICE13 benchmark."""

from __future__ import annotations

from pathlib import Path

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
CALC_PATH = CALCS_ROOT / "molecular_crystal" / "DMC_ICE13" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_crystal" / "DMC_ICE13"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*_polymorph.xyz",
    info_keys=["polymorph"],
    out_path=OUT_PATH,
    write_structs=True,
)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_lattice_energies.json",
    title="DMC-ICE13 Lattice Energies",
    x_label="Predicted lattice energy / meV",
    y_label="Reference lattice energy / meV",
    hoverdata={
        "Polymorph": INFO["polymorph"],
    },
)
def lattice_energies() -> dict[str, list]:
    """
    Get lattice energies for all DMC-ICE13 polymorphs and plot as scatter.

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

        xyz_files = sorted(model_dir.glob("*_polymorph.xyz"))
        if not xyz_files:
            continue

        water = read(model_dir / "water.xyz")
        water_energy = water.get_potential_energy()

        for xyz_file in xyz_files:
            struct = read(xyz_file)
            solid_energy = struct.get_potential_energy()
            num_molecules = len(struct) / 3
            polymorph = struct.info["polymorph"]

            lattice_energy = (solid_energy / num_molecules) - water_energy
            results[model_name].append(lattice_energy * 1000)

            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{polymorph}.xyz", struct)

            # Store reference energies (only once)
            if not ref_stored:
                results["ref"].append(struct.info["ref"])

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
        if lattice_energies[model_name]:
            results[model_name] = mae(
                lattice_energies["ref"], lattice_energies[model_name]
            )
        else:
            results[model_name] = None
    return results


def get_metrics(lattice_energies: dict[str, list]) -> dict[str, dict]:
    """
    Get all DMC-ICE13 metrics.

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
    filename=OUT_PATH / "dmc_ice13_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(lattice_energies: dict[str, list]) -> dict[str, dict]:
    """
    Get all DMC-ICE13 metrics.

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


def test_dmc_ice13(metrics: dict[str, dict]) -> None:
    """
    Run DMC-ICE13 test.

    Parameters
    ----------
    metrics
        All DMC-ICE13 metrics.
    """
    return
