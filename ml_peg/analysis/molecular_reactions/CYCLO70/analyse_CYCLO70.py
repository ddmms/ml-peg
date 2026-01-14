"""
Analyse the CYCLO70 dataset for pericyclic reaction barriers.

CYCLO70: A New Challenging Pericyclic Benchmarking Set for Kinetics
and Thermochemistry Evaluation
Javier E. Alfonso-Ramos, Carlo Adamo, Éric Brémond, and Thijs Stuyver
Journal of Chemical Theory and Computation 2025 21 (18), 8907-8917
DOI: 10.1021/acs.jctc.5c00925
"""

from __future__ import annotations

from pathlib import Path

from ase import units
from ase.io import read, write
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_d3_name_map, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_d3_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "molecular_reactions" / "CYCLO70" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "CYCLO70"


METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

EV_TO_KCAL = units.mol / units.kcal


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
            path.stem for path in sorted((CALC_PATH / model_name).glob("TS_*.xyz"))
        ]
        break
    return labels_list


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_cyclo70_barriers.json",
    title="Reaction barriers",
    x_label="Predicted barrier / kcal/mol",
    y_label="Reference barrier / kcal/mol",
    hoverdata={
        "Labels": labels(),
    },
)
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
        for label in labels():
            atoms_ts = read(CALC_PATH / model_name / f"{label}.xyz")

            results[model_name].append(atoms_ts.info["model_forward_bh"] * EV_TO_KCAL)
            results[model_name].append(atoms_ts.info["model_reverse_bh"] * EV_TO_KCAL)

            if not ref_stored:
                results["ref"].append(atoms_ts.info["ref_forward_bh"] * EV_TO_KCAL)
                results["ref"].append(atoms_ts.info["ref_reverse_bh"] * EV_TO_KCAL)

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms_ts)
        ref_stored = True
    return results


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
    filename=OUT_PATH / "cyclo70_barriers_metrics_table.json",
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


def test_cyclo70_barriers(metrics: dict[str, dict]) -> None:
    """
    Run CYCLO70 barriers test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
