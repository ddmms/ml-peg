"""Analyse Metal_Surface_Reconstructions benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase import Atoms
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "surfaces" / "metal_surface_reconstructions" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "metal_surfaces"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Prefixes of structures defining elemental references, rather than data points
REF_PREFIXES = ("bulk", "gas_phase")

# Extract system metadata from mock calculation
ALL_INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    index="0",
    info_keys=["system"],
    write_info=False,
    write_structs=False,
)

# Reference structures are required for all slabs, and only contain elements also
# present in the slabs, so info is saved for the slabs alone, matching the metrics
REF_SYSTEMS = [
    system for system in ALL_INFO["system"] if system.startswith(REF_PREFIXES)
]
INFO = {
    key: [
        value
        for value, system in zip(values, ALL_INFO["system"], strict=True)
        if not system.startswith(REF_PREFIXES)
    ]
    for key, values in ALL_INFO.items()
}
SYSTEMS = INFO["system"]

OUT_PATH.mkdir(parents=True, exist_ok=True)
with (OUT_PATH / "info.json").open("w", encoding="utf8") as f:
    json.dump(INFO, f, indent=1)


def read_struct(model_name: str, system: str) -> Atoms | None:
    """
    Read structure calculated by a model for a given system.

    Parameters
    ----------
    model_name
        Name of model structure was calculated with.
    system
        Name of system to read.

    Returns
    -------
    Atoms | None
        Structure for the system, or `None` if the file is missing.
    """
    struct_path = CALC_PATH / model_name / f"{system}.xyz"
    if not struct_path.exists():
        warn(f"{struct_path} does not exist", stacklevel=2)
        return None
    return read(struct_path, index="0")


def get_energy(struct: Atoms) -> float:
    """
    Get energy calculated for a structure.

    Parameters
    ----------
    struct
        Structure to get energy of.

    Returns
    -------
    float
        Calculated energy, or NaN if unavailable.
    """
    try:
        return struct.get_potential_energy()
    except Exception as exc:
        warn(
            f"Unable to get energy for {struct.info.get('system')}: {exc}", stacklevel=2
        )
        return np.nan


def get_chemical_potentials(
    model_name: str, reference: bool = False
) -> dict[str, float]:
    """
    Get elemental chemical potentials from bulk and gas phase reference structures.

    Parameters
    ----------
    model_name
        Name of model structures were calculated with.
    reference
        Whether to use reference (DFT) energies, rather than predicted energies.
        Default is False.

    Returns
    -------
    dict[str, float]
        Energy per atom of each elemental reference structure.
    """
    chemical_potentials = {}
    for system in REF_SYSTEMS:
        struct = read_struct(model_name, system)
        if struct is None:
            continue
        energy = (
            struct.info.get("DFT_energy", np.nan) if reference else get_energy(struct)
        )
        chemical_potentials[struct.get_chemical_symbols()[0]] = energy / len(struct)

    return chemical_potentials


def get_surface_energy(
    struct: Atoms, energy: float, chemical_potentials: dict[str, float]
) -> float:
    """
    Get surface energy of a slab, relative to its elemental references.

    Parameters
    ----------
    struct
        Slab structure.
    energy
        Energy calculated for the slab.
    chemical_potentials
        Energy per atom of each elemental reference structure.

    Returns
    -------
    float
        Surface energy in meV/Å², or NaN if any reference is unavailable.
    """
    symbols = struct.get_chemical_symbols()
    if any(symbol not in chemical_potentials for symbol in symbols):
        warn(
            f"Missing elemental references for {struct.info.get('system')}",
            stacklevel=2,
        )
        return np.nan

    area = np.linalg.norm(np.cross(struct.cell[0], struct.cell[1]))

    return (
        (energy - np.sum([chemical_potentials[symbol] for symbol in symbols]))
        * 1000
        / area
    )


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "slab_energies.json",
    title="Metal Slab Energies",
    x_label="Predicted Surface Energy / meV/Å²",
    y_label="Reference Surface Energy / meV/Å²",
    hoverdata={
        "System": SYSTEMS,
    },
)
def slab_energies() -> dict[str, list]:
    """
    Get surface energies for all slabs systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted lattice energies.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}

    # Reference energies are stored with the structures calculated by all models
    ref_chemical_potentials = get_chemical_potentials("mock", reference=True)
    for system in SYSTEMS:
        struct = read_struct("mock", system)
        if struct is None:
            results["ref"].append(np.nan)
            continue

        results["ref"].append(
            get_surface_energy(
                struct, struct.info.get("DFT_energy", np.nan), ref_chemical_potentials
            )
        )

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            warn(f"{model_dir} does not exist", stacklevel=2)
            continue

        chemical_potentials = get_chemical_potentials(model_name)

        for system in SYSTEMS:
            struct = read_struct(model_name, system)
            if struct is None:
                results[model_name].append(np.nan)
                continue

            results[model_name].append(
                get_surface_energy(struct, get_energy(struct), chemical_potentials)
            )

            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system}.xyz", struct)

    return results


@pytest.fixture
def slab_displacements() -> dict[str, list]:
    """
    Get displacements of relaxed atoms from reference positions for all slab systems.

    Returns
    -------
    dict[str, list]
        Dictionary of predicted displacements of each relaxed atom, for all systems.
    """
    results = {mlip: [] for mlip in MODELS}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            warn(f"{model_dir} does not exist", stacklevel=2)
            continue

        for system in SYSTEMS:
            struct = read_struct(model_name, system)

            # Positions cannot be compared if the optimisation failed
            if (
                struct is None
                or np.isnan(get_energy(struct))
                or "DFT_positions" not in struct.arrays
            ):
                results[model_name].append(np.array([np.nan]))
                continue

            z_min = np.min(struct.positions[:, 2])
            moving = struct.positions[:, 2] > z_min + 0.1

            results[model_name].append(
                np.linalg.norm(
                    struct.positions[moving] - struct.arrays["DFT_positions"][moving],
                    axis=1,
                )
            )

    return results


@pytest.fixture
def ranking_error(slab_energies) -> dict[str, float]:
    """
    Get ranking error across all triplets.

    Parameters
    ----------
    slab_energies
        Dictionary of reference and predicted surface energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted ranking errors for all models.
    """
    results = {}

    ref = np.asarray(slab_energies["ref"], dtype=float)

    # Triplets with missing reference energies cannot be ranked
    triplets = [
        slice(3 * i, 3 * i + 3)
        for i in range(len(ref) // 3)
        if np.isfinite(ref[3 * i : 3 * i + 3]).all()
    ]

    ref_min = np.array([np.argmin(ref[triplet]) for triplet in triplets])
    ref_max = np.array([np.argmax(ref[triplet]) for triplet in triplets])

    for model_name in MODELS:
        pred = np.asarray(slab_energies[model_name], dtype=float)

        if not triplets or pred.size != ref.size or not np.isfinite(pred).all():
            results[model_name] = np.nan
            continue

        pred_min = np.array([np.argmin(pred[triplet]) for triplet in triplets])
        pred_max = np.array([np.argmax(pred[triplet]) for triplet in triplets])

        results[model_name] = float(
            1 - 0.5 * np.mean(ref_min == pred_min) - 0.5 * np.mean(ref_max == pred_max)
        )

    return results


@pytest.fixture
def metal_surfaces_errors(slab_energies) -> dict[str, float]:
    """
    Get mean absolute error for surface energies.

    Parameters
    ----------
    slab_energies
        Dictionary of reference and predicted surface energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted lattice energy errors for all models.
    """
    results = {}

    ref = np.asarray(slab_energies["ref"], dtype=float)
    valid = np.isfinite(ref)

    for model_name in MODELS:
        pred = np.asarray(slab_energies[model_name], dtype=float)

        if not valid.any() or pred.size != ref.size:
            results[model_name] = np.nan
            continue

        results[model_name] = mae(ref[valid], pred[valid])

    return results


@pytest.fixture
def metal_position_errors(slab_displacements) -> dict[str, float]:
    """
    Get mean absolute error for positions.

    Parameters
    ----------
    slab_displacements
        Dictionary of predicted displacements from reference postitons.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted lattice energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        displacements = slab_displacements[model_name]

        if not displacements:
            results[model_name] = np.nan
            continue

        results[model_name] = float(np.mean(np.concatenate(displacements)))

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "metal_surfaces_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=None,
)
def metrics(
    metal_surfaces_errors: dict[str, float],
    metal_position_errors: dict[str, float],
    ranking_error: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metal surface reconstructions metrics.

    Parameters
    ----------
    metal_surfaces_errors
        Mean absolute errors for all surface energies.
    metal_position_errors
        Mean absolute errors for all positions.
    ranking_error
        Mean ranking error for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": metal_surfaces_errors,
        "Displacement": metal_position_errors,
        "Ranking Error": ranking_error,
    }


def test_metal_surfaces(metrics: dict[str, dict]) -> None:
    """
    Run metal surface reconstructions test.

    Parameters
    ----------
    metrics
        All metal surface metrics.
    """
    return
