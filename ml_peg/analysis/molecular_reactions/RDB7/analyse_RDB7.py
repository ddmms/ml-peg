"""
Analyse the RDB7 reaction barrier dataset.

Spiekermann, K., Pattanaik, L. & Green, W.H.
High accuracy barrier heights, enthalpies,
and rate coefficients for chemical reactions.
Sci Data 9, 417 (2022)
https://doi.org/10.1038/s41597-022-01529-6
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest
from tqdm import tqdm

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import (
    build_d3_name_map,
    load_metrics_config,
    mae,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

EV_TO_KCAL = units.mol / units.kcal
CALC_PATH = CALCS_ROOT / "molecular_reactions" / "RDB7" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "RDB7"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def labels() -> list:
    """
    Get list of system names.

    Returns
    -------
    list
        List of all system names.
    """
    for model_name in MODELS:
        labels_list = [
            path.stem.replace("_ts", "")
            for path in sorted((CALC_PATH / model_name).glob("*_ts.xyz"))
        ]
        break
    return labels_list


@pytest.fixture
def barrier_heights() -> dict[str, list]:
    """
    Get barrier heights for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted barrier heights.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        for label in tqdm(labels()):
            atoms = read(CALC_PATH / model_name / f"{label}_ts.xyz")
            results[model_name].append(atoms.info["model_forward_barrier"] * EV_TO_KCAL)
            if not ref_stored:
                results["ref"].append(atoms.info["ref_forward_barrier"] * EV_TO_KCAL)

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}_ts.xyz", atoms)
        ref_stored = True
    return results


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_barrier_density.json",
    title="Reaction barrier density plot",
    x_label="Reference reaction barrier / kcal/mol",
    y_label="Predicted reaction barrier / kcal/mol",
    annotation_metadata={"system_count": "Systems"},
)
def barrier_density(barrier_heights: dict[str, list]) -> dict[str, dict]:
    """
    Density scatter inputs for reaction barrier.

    Parameters
    ----------
    barrier_heights
        Aggregated barrier height data per model.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    ref_vals = barrier_heights["ref"]
    density_inputs: dict[str, dict] = {}
    for model_name in MODELS:
        preds = barrier_heights.get(model_name, [])
        density_inputs[model_name] = {
            "ref": ref_vals,
            "pred": preds,
            "meta": {"system_count": len([val for val in preds if val is not None])},
        }
    return density_inputs


@pytest.fixture
def get_mae(barrier_heights) -> dict[str, float]:
    """
    Get mean absolute error for barrier heights.

    Parameters
    ----------
    barrier_heights
        Dictionary of reference and predicted barrier heights.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted barrier height errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(barrier_heights["ref"], barrier_heights[model_name])
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "rdb7_barriers_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(get_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    get_mae
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {"MAE": get_mae}


def test_rdb7_barriers(
    metrics: dict[str, dict],
    barrier_density: dict[str, dict],
) -> None:
    """
    Run rdb7_barriers test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    barrier_density
        Density scatter inputs for reaction barrier.
    """
    return
