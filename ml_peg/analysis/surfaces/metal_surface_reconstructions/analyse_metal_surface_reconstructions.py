"""Analyse Metal_Surface_Reconstructions benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
print(MODELS)

CALC_PATH = CALCS_ROOT / "surfaces" / "metal_surface_reconstructions" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "metal_surfaces"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ


def get_system_names() -> list[str]:
    """
    Get list of metal surface reconstructions system names.

    Returns
    -------
    list[str]
        List of system names from structure files.
    """
    system_names = []
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        print(model_dir)
        if model_dir.exists():
            print("yes")
            xyz_files = sorted(model_dir.glob("*.xyz"))
            if xyz_files:
                for xyz_file in xyz_files:
                    atoms = read(xyz_file)
                    system_names.append(atoms.info["system"])

    return system_names


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "slab_energies.json",
    title="Metal Slab Energies",
    x_label="Predicted Surface Energy / meV/Å²",
    y_label="Reference Surface Energy / meV/Å²",
    hoverdata={
        "System": get_system_names(),
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
    ref_stored = False

    ref_mu = {}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        xyz_files = sorted(model_dir.glob("*.xyz"))
        if not xyz_files:
            continue

        model_mu = {}
        for xyz_file in xyz_files:
            name = xyz_file.name
            if name.startswith("bulk") or name.startswith("gas_phase"):
                structs = read(xyz_file, index=":")[0]
                model_mu[structs.get_chemical_symbols()[0]] = (
                    structs.get_potential_energy() / len(structs)
                )
                if not ref_stored:
                    ref_mu[structs.get_chemical_symbols()[0]] = structs.info[
                        "DFT_energy"
                    ] / len(structs)

        for xyz_file in xyz_files:
            name = xyz_file.name
            if not (name.startswith("bulk") or name.startswith("gas_phase")):
                structs = read(xyz_file, index=":")[0]
                system = structs.info["system"]
                symbols = structs.get_chemical_symbols()
                cell = structs.cell
                area = np.linalg.norm(np.cross(cell[0], cell[1]))

                results[model_name].append(
                    (
                        structs.get_potential_energy()
                        - np.sum([model_mu[s] for s in symbols])
                    )
                    * 1000
                    / area
                )

                # Copy individual structure files to app data directory
                structs_dir = OUT_PATH / model_name
                structs_dir.mkdir(parents=True, exist_ok=True)
                write(structs_dir / f"{system}.xyz", structs)

                # Store reference energies (only once)
                if not ref_stored:
                    results["ref"].append(
                        (
                            structs.info["DFT_energy"]
                            - np.sum([ref_mu[s] for s in symbols])
                        )
                        * 1000
                        / area
                    )

        ref_stored = True

    return results


@pytest.fixture
def slab_positions() -> dict[str, list]:
    """
    Get positions for all slabs systems.

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
            name = xyz_file.name
            if not (name.startswith("bulk") or name.startswith("gas_phase")):
                structs = read(xyz_file, index=":")[0]
                z_min = np.min(structs.positions[:, 2])
                moving = structs.positions[:, 2] > z_min + 0.1
                results[model_name].append(structs.positions[moving])

                # Store reference energies (only once)
                if not ref_stored:
                    results["ref"].append(structs.arrays["DFT_positions"][moving])

        ref_stored = True

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
    print(slab_energies.keys())
    results = {}
    ref_min = []
    ref_max = []
    for i in range(len(slab_energies["ref"]) // 3):
        ref_energies = slab_energies["ref"][3 * i : 3 * i + 3]
        ref_min.append(np.argmin(ref_energies))
        ref_max.append(np.argmax(ref_energies))

    for model_name in MODELS:
        if slab_energies[model_name]:
            pred_min = []
            pred_max = []
            for i in range(len(slab_energies[model_name]) // 3):
                pred_energies = slab_energies[model_name][3 * i : 3 * i + 3]
                pred_min.append(np.argmin(pred_energies))
                pred_max.append(np.argmax(pred_energies))

            results[model_name] = (
                1
                - 0.5 * np.mean(np.array(ref_min) == np.array(pred_min))
                - 0.5 * np.mean(np.array(ref_max) == np.array(pred_max))
            )
        else:
            results[model_name] = None

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
    for model_name in MODELS:
        if slab_energies[model_name]:
            results[model_name] = mae(slab_energies["ref"], slab_energies[model_name])
        else:
            results[model_name] = None
    return results


@pytest.fixture
def metal_position_errors(slab_positions) -> dict[str, float]:
    """
    Get mean absolute error for positions.

    Parameters
    ----------
    slab_positions
        Dictionary of reference and predicted postitons.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted lattice energy errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if slab_positions[model_name]:
            results[model_name] = np.mean(
                np.linalg.norm(
                    np.concatenate(slab_positions["ref"])
                    - np.concatenate(slab_positions[model_name]),
                    axis=1,
                )
            )
        else:
            results[model_name] = None
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
