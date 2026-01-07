"""
Analyse the BH2O-36 benchmark for hydrolysis reaction barriers.

Journal of Chemical Theory and Computation 2023 19 (11), 3159-3171
DOI: 10.1021/acs.jctc.3c00176
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

KCAL_TO_EV = units.kcal / units.mol
EV_TO_KCAL = 1 / KCAL_TO_EV
CALC_PATH = CALCS_ROOT / "molecular_reactions" / "bh2o_36" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "bh2o_36"

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
    labels_list = []
    for model_name in MODELS:
        for system_path in sorted((CALC_PATH / model_name).glob("*ts.xyz")):
            labels_list.append(system_path.stem.replace("_ts", ""))
        break  # only need the first model to list available systems
    return labels_list


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_bh2o_36_barriers.json",
    title="Reaction barriers",
    x_label="Predicted barrier / eV",
    y_label="Reference barrier / eV",
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
            atoms_rct = read(CALC_PATH / model_name / f"{label}_rct.xyz")
            atoms_pro = read(CALC_PATH / model_name / f"{label}_pro.xyz")
            atoms_ts = read(CALC_PATH / model_name / f"{label}_ts.xyz")

            results[model_name].append(
                atoms_ts.info["pred_energy"] - atoms_rct.info["pred_energy"]
            )
            results[model_name].append(
                atoms_ts.info["pred_energy"] - atoms_pro.info["pred_energy"]
            )

            if not ref_stored:
                results["ref"].append(
                    atoms_ts.info["ref_energy"] - atoms_rct.info["ref_energy"]
                )
                results["ref"].append(
                    atoms_ts.info["ref_energy"] - atoms_pro.info["ref_energy"]
                )

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}_rct.xyz", atoms_rct)
            write(structs_dir / f"{label}_pro.xyz", atoms_pro)
            write(structs_dir / f"{label}_ts.xyz", atoms_ts)
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
    filename=OUT_PATH / "bh2o_36_barriers_metrics_table.json",
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
    return {
        "MAE": get_mae,
    }


def test_bh2o_36_barriers(metrics: dict[str, dict]) -> None:
    """
    Run bh2o_36_barriers test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
