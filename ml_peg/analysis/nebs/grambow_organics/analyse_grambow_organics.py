"""Analyse the Grambow organics NEB convergence benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.calculator import Calculator
import numpy as np
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.nudged_elastic_band.nudged_elastic_band import (
    FINAL_CONVERGENCE_THRESHOLD,
    NEB_DATASET_FILENAME,
)
from mlipaudit.io import load_model_output_from_disk

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegGrambowOrganicsBenchmark
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegGrambowOrganicsBenchmark.name

CALC_PATH = CALCS_ROOT / "nebs" / "grambow_organics" / "outputs"
OUT_PATH = APP_ROOT / "data" / "nebs" / "grambow_organics"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def check_dataset() -> None:
    """
    Check the dataset saved by the calculation is available.

    The calculation copies and unzips the downloaded dataset into its outputs,
    so the analysis does not need to download the input data again.

    Raises
    ------
    ValueError
        If the dataset is missing from the calculation outputs.
    """
    dataset_path = CALC_PATH / BENCHMARK / NEB_DATASET_FILENAME
    if not dataset_path.exists():
        raise ValueError(f"{dataset_path} does not exist. Please run the calculation.")


@pytest.fixture
def convergence_rate() -> dict[str, float]:
    """
    Get the fraction of NEB simulations that converged for each model.

    A reaction counts as converged when the maximum final NEB force is below the
    convergence threshold. Failed simulations count as not converged.

    Returns
    -------
    dict[str, float]
        Fraction of converged reactions for each model.
    """
    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegGrambowOrganicsBenchmark
        )
        states = model_output.simulation_states
        if not states:
            continue
        n_converged = sum(
            state is not None
            and np.sqrt((state.forces**2).sum(axis=1).max())
            < FINAL_CONVERGENCE_THRESHOLD
            for state in states
        )
        results[model_name] = n_converged / len(states)
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
    check_dataset()

    benchmark = MlPegGrambowOrganicsBenchmark(
        force_field=Calculator(),
        data_input_dir=CALC_PATH,
        run_mode="standard",
    )
    elements = sorted(
        {
            symbol
            for reaction in benchmark._grambow_data.values()
            for molecule in (
                reaction.reactants,
                reaction.products,
                reaction.transition_state,
            )
            for symbol in molecule.atom_symbols
        }
    )
    info = {"elements": elements}
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_text(json.dumps(info, indent=1))
    return info


@pytest.fixture
@build_table(
    filename=OUT_PATH / "grambow_organics_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(convergence_rate: dict[str, float]) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    convergence_rate
        Fraction of converged NEB reactions for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Convergence Rate": convergence_rate,
    }


def test_grambow_organics(metrics: dict[str, dict], struct_info: dict) -> None:
    """
    Run Grambow organics NEB analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Grambow organics metric results provided by fixtures.
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
