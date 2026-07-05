"""Analyse water Cl2 cluster relaxation test."""

from __future__ import annotations

from pathlib import Path

from ase.io import Trajectory, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    get_struct_info,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "physicality" / "water_cl2_relaxation" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "water_cl2_relaxation"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    include_filenames=True,
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)


def plot_relaxation(model_name: str):
    """
    Plot relaxation paths and save all structure files.

    Parameters
    ----------
    model_name
        Name of MLIP.

    Returns
    -------
    list[list, list]
        List of optimization steps and Cl-Cl distances.
    """

    @plot_scatter(
        filename=OUT_PATH / model_name / "figure_water_cl2.json",
        title="Relaxation trajectory.",
        x_label="Optimization step",
        y_label="Cl-Cl distance, Å",
        show_line=True,
    )
    def cl_cl_distances() -> dict[str, list]:
        """
        Get Cl-Cl distances for all systems.

        Returns
        -------
        dict[str, list]
            Dictionary of all Cl-Cl distances.
        """
        results = {}
        traj = Trajectory(CALC_PATH / model_name / "relaxation.traj")
        distances = [atoms.get_distance(219, 220) for atoms in traj]
        steps = list(range(1, len(distances) + 1))
        results[model_name] = [steps, distances]

        structs_dir = OUT_PATH / model_name
        structs_dir.mkdir(parents=True, exist_ok=True)
        write(structs_dir / "relaxation.xyz", traj)

        return results

    return cl_cl_distances()[model_name]


@pytest.fixture
def get_cl2_stability() -> dict[str, float]:
    """
    Get Cl2 stability for all models.

    Returns
    -------
    dict[str, float]
        Dictionary of Cl2 stability (non-dissociation) for all models.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results = {}
    for model_name in MODELS:
        cl_cl_distances = plot_relaxation(model_name)
        final_distance = cl_cl_distances[1][-1]
        results[model_name] = bool(final_distance < 2.2)

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "water_cl2_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(get_cl2_stability: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_cl2_stability
        Cl2 stability (non-dissociation) for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Cl2_stability": get_cl2_stability,
    }


def test_water_cl2_relaxation(metrics: dict[str, dict]) -> None:
    """
    Run water Cl2 cluster relaxation test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
