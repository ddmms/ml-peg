"""Analyse benchmark of surface alloy reorganisation barriers and energies."""

from __future__ import annotations

from pathlib import Path
from warnings import warn

from ase import Atoms, units
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
CALC_PATH = CALCS_ROOT / "surfaces" / "surface_alloy_reorganisation" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "surface_alloy_reorganisation"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ

# Reference reaction energies below this magnitude (eV) are zero by symmetry
ZERO_ENERGY_TOL = 1e-2

STATES = ("initial_state", "transition_state", "final_state")

# Canonical reaction list, taken from the mock calculation, which is always complete.
# Only initial states are read so there is one entry per reaction.
INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*_initial_state_*.xyz",
    index="0",
    info_keys=["system"],
    include_filenames=True,
    write_info=True,
    write_structs=False,
    out_path=OUT_PATH,
)


def _state_file(model_dir: Path, initial_stem: str, state: str) -> Path:
    """
    Get path to the structure file for a given state of a reaction.

    Parameters
    ----------
    model_dir
        Directory containing a model's output structures.
    initial_stem
        Filename stem of the reaction's initial state.
    state
        State of the reaction.

    Returns
    -------
    Path
        Path to the structure file.
    """
    return model_dir / f"{initial_stem.replace('initial_state', state)}.xyz"


def _read_struct(path: Path) -> Atoms | None:
    """
    Read a structure, returning None if it is missing or unreadable.

    Parameters
    ----------
    path
        Path to structure file.

    Returns
    -------
    Atoms | None
        Structure, or None if it could not be read.
    """
    if not path.exists():
        warn(f"Missing structure file: {path}", stacklevel=2)
        return None
    try:
        return read(path)
    except Exception as exc:
        warn(f"Failed to read {path}: {exc}", stacklevel=2)
        return None


def _get_energy(struct: Atoms | None) -> float:
    """
    Get the predicted energy of a structure, or NaN if unavailable.

    Parameters
    ----------
    struct
        Structure to get energy from.

    Returns
    -------
    float
        Potential energy in eV, or NaN.
    """
    if struct is None:
        return np.nan
    try:
        return float(struct.get_potential_energy())
    except Exception:
        return float(struct.info.get("energy", np.nan))


def _get_ref(struct: Atoms | None) -> float:
    """
    Get the reference energy of a structure, or NaN if unavailable.

    Parameters
    ----------
    struct
        Structure to get reference energy from.

    Returns
    -------
    float
        Reference energy in eV, or NaN.
    """
    if struct is None:
        return np.nan
    return float(struct.info.get("ref", np.nan))


def _load_reference() -> tuple[list[float], list[float]]:
    """
    Load reference reaction energies and barriers from the mock outputs.

    Also writes initial, transition and final structures for each reaction as a
    trajectory for the app, named by reaction identifier.

    Returns
    -------
    tuple[list[float], list[float]]
        Reference reaction energies and barriers in eV, in canonical order.
    """
    mock_dir = CALC_PATH / "mock"
    structs_dir = OUT_PATH / "mock"
    structs_dir.mkdir(parents=True, exist_ok=True)

    energies, barriers = [], []
    for stem, system in zip(INFO["filenames"], INFO["system"], strict=True):
        structs = {
            state: _read_struct(_state_file(mock_dir, stem, state)) for state in STATES
        }
        refs = {state: _get_ref(struct) for state, struct in structs.items()}
        energies.append(refs["final_state"] - refs["initial_state"])
        barriers.append(refs["transition_state"] - refs["initial_state"])

        if all(struct is not None for struct in structs.values()):
            write(structs_dir / f"{system}.xyz", list(structs.values()))

    return energies, barriers


REF_ENERGIES_EV, REF_BARRIERS_EV = _load_reference()

# Reactions with (near) zero reference energy are excluded from the energy metric
ENERGY_MASK = [
    bool(np.isfinite(energy) and abs(energy) > ZERO_ENERGY_TOL)
    for energy in REF_ENERGIES_EV
]
ENERGY_SYSTEMS = [
    system for system, keep in zip(INFO["system"], ENERGY_MASK, strict=True) if keep
]
# Reactions without a reference barrier cannot be scored for any model
BARRIER_MASK = [bool(np.isfinite(barrier)) for barrier in REF_BARRIERS_EV]
BARRIER_SYSTEMS = [
    system for system, keep in zip(INFO["system"], BARRIER_MASK, strict=True) if keep
]


@pytest.fixture
def model_energies() -> dict[str, dict[str, list[float]]]:
    """
    Get predicted reaction energies and barriers for all models and reactions.

    Missing or failed structures give NaN for every observable depending on them,
    while preserving each reaction's position in the canonical list.

    Returns
    -------
    dict[str, dict[str, list[float]]]
        Predicted reaction energies and barriers in eV, for each model.
    """
    results = {}
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        energies, barriers = [], []
        for stem in INFO["filenames"]:
            if model_dir.exists():
                preds = {
                    state: _get_energy(_read_struct(_state_file(model_dir, stem, state)))
                    for state in STATES
                }
            else:
                preds = dict.fromkeys(STATES, np.nan)
            energies.append(preds["final_state"] - preds["initial_state"])
            barriers.append(preds["transition_state"] - preds["initial_state"])
        results[model_name] = {"energies": energies, "barriers": barriers}
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_surface_alloy_energies.json",
    title="Surface Alloy Reorganisation Reaction Energies",
    x_label="Predicted reaction energy / kJ/mol",
    y_label="Reference reaction energy / kJ/mol",
    hoverdata={
        "System": ENERGY_SYSTEMS,
    },
)
def reaction_energies(model_energies) -> dict[str, list]:
    """
    Get reaction energies for all systems.

    Parameters
    ----------
    model_energies
        Predicted reaction energies and barriers for all models.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted reaction energies.
    """
    results = {
        "ref": [
            energy * EV_TO_KJ_PER_MOL
            for energy, keep in zip(REF_ENERGIES_EV, ENERGY_MASK, strict=True)
            if keep
        ]
    }
    for model_name in MODELS:
        results[model_name] = [
            energy * EV_TO_KJ_PER_MOL
            for energy, keep in zip(
                model_energies[model_name]["energies"], ENERGY_MASK, strict=True
            )
            if keep
        ]
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_surface_alloy_barriers.json",
    title="Surface Alloy Reorganisation Reaction Barriers",
    x_label="Predicted reaction barrier / kJ/mol",
    y_label="Reference reaction barrier / kJ/mol",
    hoverdata={
        "System": BARRIER_SYSTEMS,
    },
)
def reaction_barriers(model_energies) -> dict[str, list]:
    """
    Get reaction barriers for all systems.

    Parameters
    ----------
    model_energies
        Predicted reaction energies and barriers for all models.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted reaction barriers.
    """
    results = {
        "ref": [
            barrier * EV_TO_KJ_PER_MOL
            for barrier, keep in zip(REF_BARRIERS_EV, BARRIER_MASK, strict=True)
            if keep
        ]
    }
    for model_name in MODELS:
        results[model_name] = [
            barrier * EV_TO_KJ_PER_MOL
            for barrier, keep in zip(
                model_energies[model_name]["barriers"], BARRIER_MASK, strict=True
            )
            if keep
        ]
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
    return {
        model_name: mae(reaction_energies["ref"], reaction_energies[model_name])
        for model_name in MODELS
    }


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
    return {
        model_name: mae(reaction_barriers["ref"], reaction_barriers[model_name])
        for model_name in MODELS
    }


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
