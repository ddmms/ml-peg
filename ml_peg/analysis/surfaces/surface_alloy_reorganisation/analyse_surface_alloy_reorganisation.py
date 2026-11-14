"""Analyse benchmark of surface alloy reorganisation barriers and energies."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import (
    get_struct_info,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
# DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "surfaces" / "surface_alloy_reorganisation" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "surface_alloy_reorganisation"

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
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_surface_alloy_energies.json",
    title="Surface Alloy Reorganisation Reaction Energies",
    x_label="Predicted reaction energy / kJ/mol",
    y_label="Reference reaction energy / kJ/mol",
    hoverdata={
        "System": INFO["system"],
    },
)
def reaction_energies() -> dict[str, list]:
    """
    Get reaction energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted reaction energies and barriers.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        initial_xyz_files = sorted(model_dir.glob("*initial*.xyz"))
        if not initial_xyz_files:
            continue

        for initial_xyz_file in initial_xyz_files:
            if not initial_xyz_file.exists():
                continue

            final_xyz_file = Path(
                str.replace(str(initial_xyz_file), "initial", "final")
            )
            transition_xyz_file = Path(
                str.replace(str(initial_xyz_file), "initial", "transition")
            )

            initial_struct = read(initial_xyz_file)
            final_struct = read(final_xyz_file)
            transition_struct = read(transition_xyz_file)

            initial_energy = initial_struct.get_potential_energy()
            final_energy = final_struct.get_potential_energy()

            initial_energy_ref = initial_struct.info["ref"]
            final_energy_ref = final_struct.info["ref"]

            reaction_energy = final_energy - initial_energy
            reaction_energy_ref = final_energy_ref - initial_energy_ref

            system = initial_struct.info["system"]

            if np.abs(reaction_energy_ref) > 10**-2:
                results[model_name].append(reaction_energy * EV_TO_KJ_PER_MOL)

            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(
                structs_dir / f"{system}_reaction_energy.xyz",
                [initial_struct, transition_struct, final_struct],
            )

            if not ref_stored:
                if np.abs(reaction_energy_ref) > 10**-2:
                    results["ref"].append(reaction_energy_ref * EV_TO_KJ_PER_MOL)

        ref_stored = True

    for model_name in MODELS:
        if not results[model_name]:
            results[model_name] = [np.nan] * len(results["ref"])

    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_surface_alloy_barriers.json",
    title="Surface Alloy Reorganisation Reaction Barriers",
    x_label="Predicted reaction barrier / kJ/mol",
    y_label="Reference reaction barrier / kJ/mol",
    hoverdata={
        "System": INFO["system"],
    },
)
def reaction_barriers() -> dict[str, list]:
    """
    Get reaction barriers for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted reaction energies and barriers.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        initial_xyz_files = sorted(model_dir.glob("*initial*.xyz"))
        if not initial_xyz_files:
            continue

        for initial_xyz_file in initial_xyz_files:
            if not initial_xyz_file.exists():
                continue

            final_xyz_file = Path(
                str.replace(str(initial_xyz_file), "initial", "final")
            )
            transition_xyz_file = Path(
                str.replace(str(initial_xyz_file), "initial", "transition")
            )

            initial_struct = read(initial_xyz_file)
            final_struct = read(final_xyz_file)
            transition_struct = read(transition_xyz_file)

            initial_energy = initial_struct.get_potential_energy()
            transition_state_energy = transition_struct.get_potential_energy()

            initial_energy_ref = initial_struct.info["ref"]
            transition_state_energy_ref = transition_struct.info["ref"]

            reaction_barrier = transition_state_energy - initial_energy
            reaction_barrier_ref = transition_state_energy_ref - initial_energy_ref

            results[model_name].append(reaction_barrier * EV_TO_KJ_PER_MOL)

            system = initial_struct.info["system"]

            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(
                structs_dir / f"{system}_reaction_energy.xyz",
                [initial_struct, transition_struct, final_struct],
            )

            if not ref_stored:
                results["ref"].append(reaction_barrier_ref * EV_TO_KJ_PER_MOL)

        ref_stored = True

    for model_name in MODELS:
        if not results[model_name]:
            results[model_name] = [np.nan] * len(results["ref"])

    return results


@pytest.fixture
def reaction_energy_errors(reaction_energies) -> dict[str, float]:
    """
    Get mean absolute error for reaction energies.

    Parameters
    ----------
    reaction_energies
        Dictionary of reference and predicted reaction energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted reaction energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if reaction_energies.get(model_name):
            results[model_name] = mae(
                reaction_energies["ref"], reaction_energies[model_name]
            )
        else:
            results[model_name] = np.nan

    return results


@pytest.fixture
def reaction_barrier_errors(reaction_barriers) -> dict[str, float]:
    """
    Get mean absolute error for reaction barriers.

    Parameters
    ----------
    reaction_barriers
        Dictionary of reference and predicted reaction barriers.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted reaction barrier errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if reaction_barriers.get(model_name):
            results[model_name] = mae(
                reaction_barriers["ref"], reaction_barriers[model_name]
            )
        else:
            results[model_name] = np.nan

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "surface_alloy_reorganisation_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=None,
)
def metrics(
    reaction_energy_errors: dict[str, float], reaction_barrier_errors: dict[str, float]
) -> dict[str, dict]:
    """
    Get all surface alloy reorganisation metrics.

    Parameters
    ----------
    reaction_energy_errors
        Mean absolute errors for the reaction energy of all systems.

    reaction_barrier_errors
        Mean absolute errors for the reaction barrier of all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE_energies": reaction_energy_errors,
        "MAE_barriers": reaction_barrier_errors,
    }


def test_surface_alloy_reorganisation(metrics: dict[str, dict]) -> None:
    """
    Run surface alloy reorganisation test.

    Parameters
    ----------
    metrics
        All surface alloy reorganisation metrics.
    """
    return
