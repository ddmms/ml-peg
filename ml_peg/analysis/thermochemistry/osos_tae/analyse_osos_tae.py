"""Analyse OSOS_TAE benchmark results."""

from __future__ import annotations

from pathlib import Path
import shutil

from ase import units
from ase.io import read
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "thermochemistry" / "osos_tae" / "outputs"
OUT_PATH = APP_ROOT / "data" / "thermochemistry" / "osos_tae"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)

EV_TO_KCAL_PER_MOL = units.mol / units.kcal

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    info_keys=["mol_id"],
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)

PRED_KEY = "tae_pred"
REF_KEY = "tae_ref"


@pytest.fixture
@plot_parity(
    title="Total Atomization Energy",
    x_label="Predicted TAE (kcal/mol)",
    y_label="Reference TAE (kcal/mol)",
    filename=OUT_PATH / "osos_tae_parity_plot.json",
    hoverdata={
        "Molecule": INFO["mol_id"],
    },
)
def tae() -> dict[str, list[float]]:
    """
    Get tae for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted tae.
    """
    # read reference from xyz files in mock directory.
    ref_files = sorted((CALC_PATH / "mock").glob("*.xyz"))
    results = {
        "ref": [float(read(f).info[REF_KEY]) * EV_TO_KCAL_PER_MOL for f in ref_files]
    }

    dest_dir = OUT_PATH / "mock"
    shutil.copytree(CALC_PATH / "mock", dest_dir, dirs_exist_ok=True)

    mol_ids = [f.stem for f in ref_files]

    for model_name in MODELS:
        model_path = CALC_PATH / model_name
        if not model_path.exists():
            results[model_name] = []
            continue

        preds = []
        for mol_id in mol_ids:
            xyz_file = model_path / f"{mol_id}.xyz"
            if not xyz_file.exists():
                results[model_name].append(np.nan)
                continue

            atoms = read(xyz_file)
            preds.append(float(atoms.info[PRED_KEY]) * EV_TO_KCAL_PER_MOL)

        results[model_name] = preds

    return results


@pytest.fixture
def tae_errors(tae) -> dict[str, float]:
    """
    Get mean absolute error for tae.

    Parameters
    ----------
    tae
        Dictionary of reference and predicted tae.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted tae errors for all models.
    """
    results = {}
    for model_name in MODELS:
        if tae.get(model_name):
            results[model_name] = mae(tae["ref"], tae[model_name])
        else:
            results[model_name] = np.nan
    return results


@pytest.fixture
@build_table(
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    filename=OUT_PATH / "osos_tae_table.json",
)
def metrics(tae_errors: dict[str, list]) -> dict[str, list]:
    """
    Get all osos_tae metrics.

    Parameters
    ----------
    tae_errors
        Mean absolute errors for all systems.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE": tae_errors,
    }


def test_osos_tae(metrics: dict[str, list]) -> None:
    """
    Run osos_tae analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        OSOS_TAE metric results provided by fixtures.
    """
    return
