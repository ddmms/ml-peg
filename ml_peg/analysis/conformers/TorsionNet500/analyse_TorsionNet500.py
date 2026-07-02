"""Analyse TorsionNet500 dihedral scan benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.calculator import Calculator
from mlipaudit.benchmarks.dihedral_scan.dihedral_scan import DihedralScanModelOutput
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegDihedralScanBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "conformers" / "TorsionNet500" / "outputs"
OUT_PATH = APP_ROOT / "data" / "conformers" / "TorsionNet500"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def labels() -> list:
    """
    Get the ordered list of fragment names.

    Returns
    -------
    list
        List of all dihedral scan fragment names.
    """
    for model_name in MODELS:
        path = CALC_PATH / model_name / "model_output.json"
        if path.exists():
            output = DihedralScanModelOutput.model_validate_json(path.read_text())
            return [fragment.fragment_name for fragment in output.fragments]
    return []


def _barrier(profile: list[float]) -> float:
    """
    Get the torsion barrier height of an energy profile.

    Parameters
    ----------
    profile
        Energy profile along the dihedral scan in kcal/mol.

    Returns
    -------
    float
        Difference between the maximum and minimum energy in the profile.
    """
    return max(profile) - min(profile)


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``DihedralScanResult``.
    """
    data_input_dir = download_s3_data(
        key="inputs/conformers/TorsionNet500/TorsionNet500.zip",
        filename="TorsionNet500.zip",
    )

    results = {}
    for model_name in MODELS:
        path = CALC_PATH / model_name / "model_output.json"
        if not path.exists():
            continue
        benchmark = MlPegDihedralScanBenchmark(
            force_field=Calculator(),
            data_input_dir=data_input_dir,
            run_mode="standard",
        )
        benchmark.model_output = DihedralScanModelOutput.model_validate_json(
            path.read_text()
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
def struct_info() -> dict:
    """
    Write the combined element set to ``info.json`` for filtering.

    Returns
    -------
    dict
        Mapping with the sorted list of elements present in the dataset.
    """
    data_input_dir = download_s3_data(
        key="inputs/conformers/TorsionNet500/TorsionNet500.zip",
        filename="TorsionNet500.zip",
    )
    benchmark = MlPegDihedralScanBenchmark(
        force_field=Calculator(),
        data_input_dir=data_input_dir,
        run_mode="standard",
    )
    elements = sorted(
        {
            symbol
            for fragment in benchmark._torsion_net_500.values()
            for symbol in fragment.atom_symbols
        }
    )
    info = {"elements": elements}
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_text(json.dumps(info, indent=1))
    return info


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_torsionnet500.json",
    title="Torsion barrier heights",
    x_label="Predicted barrier height / kcal/mol",
    y_label="Reference barrier height / kcal/mol",
    hoverdata={
        "Fragment": labels(),
    },
)
def barrier_heights(analyze_results) -> dict[str, list]:
    """
    Get predicted and reference torsion barrier heights.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``DihedralScanResult``.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted barrier heights, aligned to the
        ordered fragment names.
    """
    ids = labels()
    ref_map: dict[str, float] = {}
    pred_maps: dict[str, dict[str, float]] = {}

    for model_name, result in analyze_results.items():
        pred_maps[model_name] = {}
        for fragment in result.fragments:
            if fragment.failed:
                continue
            pred_maps[model_name][fragment.fragment_name] = _barrier(
                fragment.predicted_energy_profile
            )
            ref_map.setdefault(
                fragment.fragment_name, _barrier(fragment.reference_energy_profile)
            )

    results = {"ref": [ref_map.get(i) for i in ids]}
    for model_name in MODELS:
        model_preds = pred_maps.get(model_name, {})
        results[model_name] = [model_preds.get(i) for i in ids]
    return results


@pytest.fixture
def get_mae(analyze_results) -> dict[str, float]:
    """
    Get the barrier height mean absolute error for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``DihedralScanResult``.

    Returns
    -------
    dict[str, float]
        Mean absolute barrier height error in kcal/mol for each model.
    """
    return {
        model_name: result.mae_barrier_height
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_score(analyze_results) -> dict[str, float]:
    """
    Get the mlipaudit benchmark score for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``DihedralScanResult``.

    Returns
    -------
    dict[str, float]
        The mlipaudit per-fragment soft-threshold score (0 to 1) for each model.
    """
    return {model_name: result.score for model_name, result in analyze_results.items()}


@pytest.fixture
@build_table(
    filename=OUT_PATH / "torsionnet500_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    barrier_heights, get_mae: dict[str, float], get_score: dict[str, float]
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    barrier_heights
        Reference and predicted barrier heights (triggers the parity plot).
    get_mae
        Mean absolute barrier height errors for all models.
    get_score
        The mlipaudit benchmark scores for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Barrier Height MAE": get_mae,
        "Torsion Score": get_score,
    }


def test_torsionnet500(
    metrics: dict[str, dict], barrier_heights, struct_info: dict
) -> None:
    """
    Run TorsionNet500 analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        TorsionNet500 metric results provided by fixtures.
    barrier_heights
        Reference and predicted barrier heights (triggers the parity plot).
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
