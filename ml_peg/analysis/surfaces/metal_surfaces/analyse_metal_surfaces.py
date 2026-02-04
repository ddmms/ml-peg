
from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
print(MODELS)
#D3_MODEL_NAMES = build_d3_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "surfaces" / "metal_surfaces" / "outputs"
OUT_PATH = APP_ROOT / "data" / "surfaces" / "metal_surfaces"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ


def get_system_names() -> list[str]:
    """
    Get list of metal_surface system names.

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
            print('yes')
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
    x_label="Predicted Energy / eV",
    y_label="Reference Energy / eV",
    hoverdata={
        "System": get_system_names(),
    },
)
def slab_energies() -> dict[str, list]:
    """
    Get energies for all slabs systems.

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
            system = structs[0].info["system"]
            results[model_name].append(structs[0].get_potential_energy())



            # Copy individual structure files to app data directory
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{system}.xyz", structs)

            # Store reference energies (only once)
            if not ref_stored:
                results["ref"].append(structs[0].info["DFT_energy"])

        ref_stored = True

    return results



@pytest.fixture
def slab_positions() -> dict[str, list]:
    """
    Get energies for all slabs systems.

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
            z_min = np.min(structs[0].positions[:,2])
            moving = structs[0].positions[:,2]>z_min+0.1
            results[model_name].append(structs[0].positions[moving])



            # Store reference energies (only once)
            if not ref_stored:
                results["ref"].append(structs[0].arrays["DFT_posittions"][moving])

        ref_stored = True

    return results




@pytest.fixture
def metal_surfaces_errors(slab_energies) -> dict[str, float]:
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
        if slab_energies[model_name]:
            results[model_name] = mae(
                slab_energies["ref"], slab_energies[model_name]
            )
        else:
            results[model_name] = None
    return results



@pytest.fixture
def metal_position_errors(slab_positions) -> dict[str, float]:
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
        if slab_positions[model_name]:
            results[model_name] = (
                np.mean(np.linalg.norm(np.concat(slab_positions["ref"])-np.concat(slab_positions[model_name]),axis=1))
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
def metrics(metal_surfaces_errors: dict[str, float], metal_position_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all metal surface metrics.

    Parameters
    ----------
    metal_surfaces_errors
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": metal_surfaces_errors,
        "Displacement": metal_position_errors,
    }

def test_metal_surfaces(metrics: dict[str, dict]) -> None:
    """
    Run metal surface test.

    Parameters
    ----------
    metrics
        All metal surface metrics.
    """
    return
