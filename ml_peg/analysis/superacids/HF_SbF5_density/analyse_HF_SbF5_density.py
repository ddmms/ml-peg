"""Analyse HF/SbF5 density benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
DISPERSION_MODEL_NAMES = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "superacids" / "HF_SbF5_density" / "outputs"
OUT_PATH = APP_ROOT / "data" / "superacids" / "HF_SbF5_density"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Experimental reference densities
REF_DENSITIES = {
    "X_0": 0.989,
    "X_10": 1.677,
    "X_100": 3.141,
}


# amu to g conversion factor
AMU_TO_G = 1000 / units.kg
A3_TO_CM3 = 1e-24


def compute_density_from_volume_dat(volume_path: Path, atoms_path: Path) -> float:
    """
    Compute average density from volume.dat and atomic masses.

    Parameters
    ----------
    volume_path
        Path to volume.dat file (columns: step, volume_A3).
    atoms_path
        Path to any xyz file for this system (to get atomic masses).

    Returns
    -------
    float
        Average density in g/cm³.
    """
    # Read total mass from atoms
    atoms = read(atoms_path)
    total_mass_amu = np.sum(atoms.get_masses())

    # Read volume time series, skip header
    data = np.loadtxt(volume_path, comments="#")
    # Take second half as production (discard equilibration)
    n_points = len(data)
    production = data[n_points // 2 :, 1]  # column 1 = volume in ų
    avg_volume = np.mean(production)

    return (total_mass_amu * AMU_TO_G) / (avg_volume * A3_TO_CM3)


def get_system_names() -> list[str]:
    """
    Get list of system names.

    Returns
    -------
    list[str]
        List of system names from output directories.
    """
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            systems = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            if systems:
                return systems
    return []


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_density.json",
    title="HF/SbF5 Mixture Densities",
    x_label="Predicted density / g/cm³",
    y_label="Experimental density / g/cm³",
    hoverdata={
        "System": get_system_names(),
    },
)
def densities() -> dict[str, list]:
    """
    Get predicted and reference densities for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted densities.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            continue

        systems = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
        if not systems:
            continue

        for system in systems:
            system_dir = model_dir / system
            volume_path = system_dir / "volume.dat"
            atoms_path = system_dir / f"{system}.xyz"

            if not volume_path.exists() or not atoms_path.exists():
                continue

            density = compute_density_from_volume_dat(volume_path, atoms_path)
            results[model_name].append(density)

            if not ref_stored:
                results["ref"].append(REF_DENSITIES[system])

        ref_stored = True

    return results


@pytest.fixture
def density_errors(densities) -> dict[str, float]:
    """
    Get mean absolute percentage error for densities.

    Parameters
    ----------
    densities
        Dictionary of reference and predicted densities.

    Returns
    -------
    dict[str, float]
        Dictionary of density MAPE for all models.
    """
    results = {}
    for model_name in MODELS:
        if densities[model_name]:
            preds = np.array(densities[model_name])
            refs = np.array(densities["ref"])
            mape = np.mean(np.abs(preds - refs) / refs) * 100  # in %
            results[model_name] = mape
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "hf_sbf5_density_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_MODEL_NAMES,
)
def metrics(density_errors: dict[str, float]) -> dict[str, dict]:
    """
    Get all HF/SbF5 density metrics.

    Parameters
    ----------
    density_errors
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAPE": density_errors,
    }


def test_hf_sbf5_density(metrics: dict[str, dict]) -> None:
    """
    Run HF/SbF5 density test.

    Parameters
    ----------
    metrics
        All HF/SbF5 density metrics.
    """
    return
