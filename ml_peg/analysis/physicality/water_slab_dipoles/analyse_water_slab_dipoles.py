"""Analyse water slab dipole benchmark."""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read
import numpy as np
import pytest
from scipy.constants import e, epsilon_0

from ml_peg.analysis.utils.decorators import build_table, plot_hist
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)
CALC_PATH = CALCS_ROOT / "physicality" / "water_slab_dipoles" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "water_slab_dipoles"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Unit conversion
EV_TO_KJ_PER_MOL = units.mol / units.kJ

# We consider a dipole as bad if the expected band gap is <= 0
# The expected band gap is 4.50 V - |P_z_per_unit_area| / epsilon_0
# Hence "bad" is |P_z_per_unit_area| > 4.50 V * epsilon_0 / e * 10^(-10)
# epsilon_0 is in F/m = C/(V*m), so this gives it in e/(V*A)
DIPOLE_BAD_THRESHOLD = 4.50 * epsilon_0 / e * 10 ** (-10)


def get_dipoles() -> dict[str, np.ndarray]:
    """
    Get total dipole per unit area in z direction.

    Returns
    -------
    dict[str, np.ndarray]
        Dictionary with array of dipoles for each model.
    """
    results = {}
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if model_dir.exists():
            if (model_dir / "dipoles.npy").is_file():
                results[model_name] = np.load(model_dir / "dipoles.npy")
            else:
                atoms = read(model_dir / "slab.xyz", ":")
                dipoles = np.zeros(len(atoms))
                for i, struc in enumerate(atoms):
                    o_index = [atom.index for atom in struc if atom.number == 8]
                    h_index = [atom.index for atom in struc if atom.number == 1]
                    dipoles[i] = (
                        np.sum(struc.positions[o_index, 2]) * (-0.8476)
                        + np.sum(struc.positions[h_index, 2]) * 0.4238
                    )
                dipoles_unit_area = dipoles / atoms[0].cell[0, 0] / atoms[0].cell[1, 1]
                results[model_name] = dipoles_unit_area
                np.save(model_dir / "dipoles.npy", dipoles_unit_area)
    return results


def plot_distribution(model: str) -> None:
    """
    Plot Dipole Distribution.

    Parameters
    ----------
    model
        Name of MLIP.
    """

    @plot_hist(
        filename=OUT_PATH / f"figure_{model}_dipoledistr.json",
        title=f"Dipole Distribution {model}",
        x_label="Total z-Dipole per unit area [e/A]",
        good=-DIPOLE_BAD_THRESHOLD,
        bad=DIPOLE_BAD_THRESHOLD,
        bins=20,
    )
    def plot_distr() -> dict[str, np.ndarray]:
        """
        Plot a NEB and save the structure file.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary of array with all dipoles for each model.
        """
        return get_dipoles()  # {'sigma': get_dipoles(), 'n_bad': get_dipoles()}

    plot_distr()


@pytest.fixture
def dipole_std() -> dict[str, float]:
    """
    Get standard deviation of total z dipole per unit area (in e/A).

    Returns
    -------
    dict[str, float]
        Dictionary of standard deviation of dipole distribution for all models.
    """
    dipoles = get_dipoles()
    results = {}
    for model_name in MODELS:
        if model_name in dipoles.keys():
            plot_distribution(model_name)
            results[model_name] = np.std(dipoles[model_name])
        else:
            results[model_name] = None
    return results


@pytest.fixture
def n_bad() -> dict[str, float]:
    """
    Get fraction of dipoles that are bad.

    Returns
    -------
    dict[str, float]
        Dictionary of percentage of breakdown candidates for all models.
    """
    dipoles = get_dipoles()

    results = {}
    for model_name in MODELS:
        if model_name in dipoles.keys():
            plot_distribution(model_name)
            results[model_name] = (
                np.abs(dipoles[model_name]) > DIPOLE_BAD_THRESHOLD
            ).sum() / len(dipoles[model_name])
        else:
            results[model_name] = None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "water_slab_dipoles_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(dipole_std: dict[str, float], n_bad: dict[str, float]) -> dict[str, dict]:
    """
    Get all water slab dipoles metrics.

    Parameters
    ----------
    dipole_std
        Standard deviation of dipole distribution.
    n_bad
        Percentage of tested structures with dipole larger than water band gap.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "sigma": dipole_std,
        "Fraction Breakdown Candidates": n_bad,
    }


def test_water_slab_dipoles(metrics: dict[str, dict]) -> None:
    """
    Run water slab dipoles test.

    Parameters
    ----------
    metrics
        All water slab dipole metrics.
    """
    return
