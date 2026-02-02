"""Analyse X23 benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
CALC_PATH = CALCS_ROOT / "molecular_crystal" / "X23" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_crystal" / "X23"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ
X23_METADATA: dict[str, dict[str, Any]] = {}
X23_SYSTEM_ORDER: list[str] = []
STRUCTURE_MODEL: str | None = None


def get_system_names() -> list[str]:
    """
    Get list of X23 system names.

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


def get_system_elements() -> list[str]:
    """
    Get list of X23 system elements.

    Returns
    -------
    list[str]
        List of system elements from structure files.
    """
    system_elements = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    symbols = sorted(set(atoms.get_chemical_symbols()))
                    system_elements.append(", ".join(symbols))
                break
    return system_elements


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_lattice_energies.json",
    title="X23 Lattice Energies",
    x_label="Predicted lattice energy / kJ/mol",
    y_label="Reference lattice energy / kJ/mol",
    hoverdata={
        "System": get_system_names(),
        "Elements": get_system_elements(),
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
    global STRUCTURE_MODEL
    X23_METADATA.clear()
    X23_SYSTEM_ORDER.clear()
    STRUCTURE_MODEL = None

    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_files = sorted(model_dir.glob("*.xyz"))
        if not xyz_files:
            continue
        if STRUCTURE_MODEL is None:
            STRUCTURE_MODEL = model_name

        for xyz_file in xyz_files:
            structs = read(xyz_file, index=":")

            solid_energy = structs[0].get_potential_energy()
            num_molecules = structs[0].info["num_molecules"]
            system = structs[0].info["system"]
            molecule_energy = structs[1].get_potential_energy()
            elements = sorted(set(structs[0].get_chemical_symbols()))
            ref_energy = structs[0].info["ref"]

            lattice_energy = (solid_energy / num_molecules) - molecule_energy
            converted_energy = lattice_energy * EV_TO_KJ_PER_MOL
            results[model_name].append(converted_energy)

            if system not in X23_METADATA:
                X23_SYSTEM_ORDER.append(system)
                X23_METADATA[system] = {
                    "system": system,
                    "elements": elements,
                    "ref": ref_energy,
                    "models": {},
                }
            X23_METADATA[system]["models"][model_name] = converted_energy

            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system}.xyz", structs)

            # Store reference energies (only once)
            if not ref_stored:
                results["ref"].append(ref_energy)

        ref_stored = True

    return results


@pytest.fixture
def x23_filter_payload(lattice_energies: dict[str, list]) -> dict[str, Any]:
    """
    Write X23 filtering payload with per-system metadata.

    Parameters
    ----------
    lattice_energies
        Fixture ensuring metadata has been populated and structures copied.

    Returns
    -------
    dict[str, Any]
        Payload written to disk for downstream filtering.
    """
    _ = lattice_energies  # ensure metadata populated
    payload = {
        "systems": [X23_METADATA[system] for system in X23_SYSTEM_ORDER],
        "structure_model": STRUCTURE_MODEL,
    }
    payload_path = OUT_PATH / "x23_filter_payload.json"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_text(json.dumps(payload, indent=2))
    return payload


@pytest.fixture
def x23_errors(lattice_energies) -> dict[str, float]:
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


@pytest.fixture
@build_table(
    filename=OUT_PATH / "x23_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(x23_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all X23 metrics.

    Parameters
    ----------
    x23_errors
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": x23_errors,
    }


def test_x23(
    metrics: dict[str, dict],
    x23_filter_payload: dict[str, Any],
) -> None:
    """
    Run X23 test.

    Parameters
    ----------
    metrics
        All X23 metrics.
    x23_filter_payload
        Filter payload generated alongside the metrics to support interactive
        element filtering in the Dash application.
    """
    return
