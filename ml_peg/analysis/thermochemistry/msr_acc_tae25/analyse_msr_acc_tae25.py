"""Analyse msr_acc_tae25 benchmark results."""

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

CALC_PATH = CALCS_ROOT / "thermochemistry" / "msr_acc_tae25" / "outputs"
OUT_PATH = APP_ROOT / "data" / "thermochemistry" / "msr_acc_tae25"
MODELS = sorted(d.name for d in CALC_PATH.iterdir() if d.is_dir())

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)

EV_TO_KCAL_PER_MOL = units.mol / units.kcal

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    info_keys=["name"],
    write_info=True,
    write_structs=True,
    out_path=OUT_PATH,
)

PRED_KEY = "tae_pred"
REF_KEY = "tae_ref"


def _get_xyz_files(model_dir: Path) -> list[Path]:
    """
    Return `.xyz` files for a model sorted by filename index.

    Parameters
    ----------
    model_dir : Path
        Directory containing per-model Wiggle150 outputs.

    Returns
    -------
    list[Path]
        Sorted list of `.xyz` files.
    """
    return model_dir.glob("*.xyz")


@pytest.fixture
@plot_parity(
    title="Total Atomization Energy",
    x_label="Reference TAE (kcal/mol)",
    y_label="Predicted TAE (kcal/mol)",
    filename=OUT_PATH / "msr_acc_tae25_parity_plot.json",
    hoverdata={
        "Molecule": INFO["name"],
    },
)
def tae() -> dict[str, list]:
    """
    Get tae for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted tae.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        model_path = CALC_PATH / model_name
        if not model_path.exists():
            results[model_name] = []
            continue

        xyz_files = _get_xyz_files(model_path)
        if not xyz_files:
            results[model_name] = []
            continue

        for xyz_file in xyz_files:
            atoms = read(xyz_file)
            results[model_name].append(float(atoms.info[PRED_KEY]) * EV_TO_KCAL_PER_MOL)

            if not ref_stored:
                results["ref"].append(float(atoms.info[REF_KEY]) * EV_TO_KCAL_PER_MOL)

            dest_dir = OUT_PATH / model_name
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(xyz_file, dest_dir / xyz_file.name)

        ref_stored = True

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
    filename=OUT_PATH / "msr_acc_tae25_table.json",
)
def metrics(tae_errors: dict[str, list]) -> dict[str, list]:
    """
    Get all msr_acc_tae25 metrics.

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


def test_msr_acc_tae25(metrics: dict[str, list]) -> None:
    """
    Run msr_acc_tae25 analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        MSR_ACC_TAE25 metric results provided by fixtures.
    """
    return
