"""Analyse Tautobase tautomer benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.calculators.calculator import Calculator
from mlipaudit.benchmarks.tautomers.tautomers import TautomersModelOutput
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegTautomersBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "molecular_reactions" / "tautomers" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "tautomers"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def labels() -> list:
    """
    Get the ordered list of tautomer pair IDs.

    Returns
    -------
    list
        List of all tautomer pair structure IDs.
    """
    for model_name in MODELS:
        path = CALC_PATH / model_name / "model_output.json"
        if path.exists():
            output = TautomersModelOutput.model_validate_json(path.read_text())
            return output.structure_ids
    return []


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``TautomersResult``.
    """
    data_input_dir = download_s3_data(
        key="inputs/molecular_reactions/tautomers/tautomers.zip",
        filename="tautomers.zip",
    )

    results = {}
    for model_name in MODELS:
        path = CALC_PATH / model_name / "model_output.json"
        if not path.exists():
            continue
        benchmark = MlPegTautomersBenchmark(
            force_field=Calculator(),
            data_input_dir=data_input_dir,
            run_mode="standard",
        )
        benchmark.model_output = TautomersModelOutput.model_validate_json(
            path.read_text()
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_tautomers.json",
    title="Tautomer reaction energies",
    x_label="Predicted reaction energy / kcal/mol",
    y_label="Reference reaction energy / kcal/mol",
    hoverdata={
        "Labels": labels(),
    },
)
def tautomer_energies(analyze_results) -> dict[str, list]:
    """
    Get predicted and reference tautomer reaction energies.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``TautomersResult``.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted reaction energies, aligned to the
        ordered tautomer pair IDs.
    """
    ids = labels()
    ref_map: dict[str, float] = {}
    pred_maps: dict[str, dict[str, float]] = {}

    for model_name, result in analyze_results.items():
        pred_maps[model_name] = {}
        for molecule in result.molecules:
            if molecule.failed:
                continue
            pred_maps[model_name][molecule.structure_id] = (
                molecule.predicted_energy_diff
            )
            ref_map.setdefault(molecule.structure_id, molecule.ref_energy_diff)

    results = {"ref": [ref_map.get(i) for i in ids]}
    for model_name in MODELS:
        model_preds = pred_maps.get(model_name, {})
        results[model_name] = [model_preds.get(i) for i in ids]
    return results


@pytest.fixture
def get_mae(analyze_results) -> dict[str, float]:
    """
    Get the reaction energy mean absolute error for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``TautomersResult``.

    Returns
    -------
    dict[str, float]
        Mean absolute error in kcal/mol for each model.
    """
    return {model_name: result.mae for model_name, result in analyze_results.items()}


@pytest.fixture
@build_table(
    filename=OUT_PATH / "tautomers_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(tautomer_energies, get_mae: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    tautomer_energies
        Reference and predicted reaction energies (triggers the parity plot).
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


def test_tautomers(metrics: dict[str, dict]) -> None:
    """
    Run tautomers analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Tautomers metric results provided by fixtures.
    """
