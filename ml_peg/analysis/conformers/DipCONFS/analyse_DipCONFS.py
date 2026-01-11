"""
Analyse DipCONFS conformation energy benchmark.

10.1021/acs.jctc.4c00801.
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

EV_TO_KCAL = units.mol / units.kcal
CALC_PATH = CALCS_ROOT / "conformers" / "DipCONFS" / "outputs"
OUT_PATH = APP_ROOT / "data" / "conformers" / "DipCONFS"

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
        labels_list = [path.stem for path in sorted((CALC_PATH / model_name).glob("*"))]
        break
    return labels_list


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_dipconfs.json",
    title="Energies",
    x_label="Predicted energy / kcal/mol",
    y_label="Reference energy / kcal/mol",
    hoverdata={
        "Labels": labels(),
    },
)
def conformer_energies() -> dict[str, list]:
    """
    Get conformer energies for all systems.

    Returns
    -------
    dict[str, list]
        Dictionary of all reference and predicted barrier heights.
    """
    results = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        for label in labels():
            atoms = read(CALC_PATH / model_name / f"{label}.xyz")

            results[model_name].append(atoms.info["model_rel_energy"] * EV_TO_KCAL)
            if not ref_stored:
                results["ref"].append(atoms.info["ref_energy"] * EV_TO_KCAL)

            # Write structures for app
            structs_dir = OUT_PATH / model_name
            structs_dir.mkdir(parents=True, exist_ok=True)
            write(structs_dir / f"{label}.xyz", atoms)
        ref_stored = True
    return results


@pytest.fixture
def get_mae(conformer_energies) -> dict[str, float]:
    """
    Get mean absolute error for conformer energies.

    Parameters
    ----------
    conformer_energies
        Dictionary of reference and predicted conformer energies.

    Returns
    -------
    dict[str, float]
        Dictionary of predicted conformer energies errors for all models.
    """
    results = {}
    for model_name in MODELS:
        results[model_name] = mae(
            conformer_energies["ref"], conformer_energies[model_name]
        )
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "dipconfs_metrics_table.json",
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


def test_dipconfs(metrics: dict[str, dict]) -> None:
    """
    Run DipCONFS test.

    Parameters
    ----------
    metrics
        All new benchmark metric names and dictionary of values for each model.
    """
    return
