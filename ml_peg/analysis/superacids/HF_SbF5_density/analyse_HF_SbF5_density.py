"""Analyse HF/SbF5 density benchmark."""

from __future__ import annotations

from pathlib import Path
from warnings import warn

from ase import units
from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity, plot_scatter
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
DISPERSION_MODEL_NAMES = build_dispersion_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "superacids" / "HF_SbF5_density" / "outputs"
OUT_PATH = APP_ROOT / "data" / "superacids" / "HF_SbF5_density"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Experimental reference densities, from Shair and Schurig,
# Ind. Eng. Chem. 43, 1624 (1951), https://doi.org/10.1021/ie50499a042
REF_DENSITIES = {
    "X_0": 0.989,
    "X_10": 1.677,
    "X_100": 3.141,
}

SYSTEMS = sorted(REF_DENSITIES)

# amu to g conversion factor
AMU_TO_G = 1000 / units.kg
A3_TO_CM3 = 1e-24

# Minimum number of production samples for a density to be considered valid
MIN_SAMPLES = 2

# MD timestep in fs and number of steps run, to convert steps of volume.dat
# into a time axis. These match the values set in the calculation.
DT_FS = 0.5
N_NPT_STEPS = 200000
FS_TO_PS = 1e-3

# Composition of each system, as the mol % of SbF5 in the mixture
SYSTEM_COMPOSITIONS = {
    "X_0": 0,
    "X_10": 10,
    "X_100": 100,
}


def compute_density(traj_path: Path, volume_path: Path) -> float:
    """
    Compute average density from volume.dat and atomic masses.

    Parameters
    ----------
    traj_path
        Path to trajectory of this system (to get atomic masses).
    volume_path
        Path to volume.dat file (columns: step, volume_A3).

    Returns
    -------
    float
        Average density in g/cm³, or NaN if there are too few samples.
    """
    # Read total mass from the first frame, so that runs in progress can be analysed
    atoms = read(traj_path, index=0)
    total_mass_amu = np.sum(atoms.get_masses())

    # Read volume time series, skip header
    data = np.loadtxt(volume_path, comments="#", ndmin=2)
    # Take second half as production (discard equilibration)
    production = data[len(data) // 2 :, 1] if data.size else np.array([])
    if production.size < MIN_SAMPLES:
        return np.nan
    avg_volume = np.mean(production)

    return (total_mass_amu * AMU_TO_G) / (avg_volume * A3_TO_CM3)


def compute_density_series(
    traj_path: Path, volume_path: Path
) -> tuple[list[float], list[float]]:
    """
    Compute the instantaneous density over the whole trajectory.

    Note that the average of this series is not exactly the density reported in
    the metrics table, which is computed from the average volume.

    Parameters
    ----------
    traj_path
        Path to trajectory of this system (to get atomic masses).
    volume_path
        Path to volume.dat file (columns: step, volume_A3).

    Returns
    -------
    tuple[list[float], list[float]]
        Time in ps, and density in g/cm³.
    """
    atoms = read(traj_path, index=0)
    total_mass_amu = np.sum(atoms.get_masses())

    data = np.loadtxt(volume_path, comments="#", ndmin=2)
    if not data.size:
        return [], []

    time_ps = data[:, 0] * DT_FS * FS_TO_PS
    densities = (total_mass_amu * AMU_TO_G) / (data[:, 1] * A3_TO_CM3)

    return time_ps.tolist(), densities.tolist()


def plot_density_series(system: str) -> None:
    """
    Plot the density of all models against time for one system.

    Parameters
    ----------
    system
        System identifier (X_0, X_10, X_100).
    """
    composition = SYSTEM_COMPOSITIONS[system]
    total_time_ps = N_NPT_STEPS * DT_FS * FS_TO_PS

    @plot_scatter(
        filename=OUT_PATH / f"figure_density_time_{system}.json",
        title=f"HF/SbF5 mixture with x = {composition}% SbF5",
        x_label="Time / ps",
        y_label="Density / g/cm³",
        show_line=True,
        show_markers=False,
        hlines={"Target": REF_DENSITIES[system]},
        highlight_range={"Production": [total_time_ps / 2, total_time_ps]},
    )
    def density_series() -> dict[str, list]:
        """
        Get the density of all models against time.

        Returns
        -------
        dict[str, list]
            Times and densities for all models with a trajectory.
        """
        results = {}

        for model_name in MODELS:
            system_dir = CALC_PATH / model_name / system
            traj_path = system_dir / f"{system}.traj"
            volume_path = system_dir / "volume.dat"

            # Missing systems are left out of the plot
            if not traj_path.exists() or not volume_path.exists():
                continue

            try:
                results[model_name] = list(
                    compute_density_series(traj_path, volume_path)
                )
            except Exception as exc:
                warn(
                    f"Error computing density series for {model_name} {system}: {exc}",
                    stacklevel=2,
                )

        return results

    density_series()


@pytest.fixture
def density_series() -> None:
    """Plot the density of all models against time, for every system."""
    for system in SYSTEMS:
        plot_density_series(system)


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_density.json",
    title="HF/SbF5 Mixture Densities",
    x_label="Predicted density / g/cm³",
    y_label="Experimental density / g/cm³",
    hoverdata={
        "System": SYSTEMS,
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
    results = {"ref": [REF_DENSITIES[system] for system in SYSTEMS]} | {
        mlip: [np.nan] * len(SYSTEMS) for mlip in MODELS
    }

    for model_name in MODELS:
        for index, system in enumerate(SYSTEMS):
            system_dir = CALC_PATH / model_name / system
            traj_path = system_dir / f"{system}.traj"
            volume_path = system_dir / "volume.dat"

            # Missing systems are left as NaN, to keep systems aligned with `ref`
            if not traj_path.exists() or not volume_path.exists():
                continue

            try:
                results[model_name][index] = compute_density(traj_path, volume_path)
            except Exception as exc:
                warn(
                    f"Error computing density for {model_name} {system}: {exc}",
                    stacklevel=2,
                )

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
    refs = np.array(densities["ref"], dtype=float)

    for model_name in MODELS:
        preds = np.array(densities[model_name], dtype=float)
        # Models missing any system are scored as None, as in `mae`
        if np.isnan(np.sum(preds)):
            results[model_name] = None
        else:
            results[model_name] = float(np.mean(np.abs(preds - refs) / refs) * 100)

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


def test_hf_sbf5_density(metrics: dict[str, dict], density_series: None) -> None:
    """
    Run HF/SbF5 density test.

    Parameters
    ----------
    metrics
        All HF/SbF5 density metrics.
    density_series
        Density against time plots for all systems.
    """
    # Elemental info for filtering, from the mock calculation
    get_struct_info(
        calc_path=CALC_PATH,
        glob_pattern="*/*.traj",
        index=0,
        include_dirs=True,
        write_structs=False,
        out_path=OUT_PATH,
    )
