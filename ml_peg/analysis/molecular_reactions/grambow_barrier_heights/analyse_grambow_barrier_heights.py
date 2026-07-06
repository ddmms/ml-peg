"""
Analyse the Grambow reaction barrier benchmark.

Grambow, C.A., Pattanaik, L. & Green, W.H.
Reactants, products, and transition states of elementary chemical reactions
based on quantum chemistry.
Sci Data 7, 137 (2020).
DOI: 10.1038/s41597-020-0460-4
"""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.calculator import Calculator
from mlipaudit.benchmarks.reactivity.reactivity import ReactivityModelOutput
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_density_scatter
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegGrambowBarrierHeightsBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "molecular_reactions" / "grambow_barrier_heights" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_reactions" / "grambow_barrier_heights"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``ReactivityResult``.
    """
    data_input_dir = download_s3_data(
        key="inputs/molecular_reactions/grambow_barrier_heights/grambow_barrier_heights.zip",
        filename="grambow_barrier_heights.zip",
    )

    results = {}
    for model_name in MODELS:
        path = CALC_PATH / model_name / "model_output.json"
        if not path.exists():
            continue
        benchmark = MlPegGrambowBarrierHeightsBenchmark(
            force_field=Calculator(),
            data_input_dir=data_input_dir,
            run_mode="standard",
        )
        benchmark.model_output = ReactivityModelOutput.model_validate_json(
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
        key="inputs/molecular_reactions/grambow_barrier_heights/grambow_barrier_heights.zip",
        filename="grambow_barrier_heights.zip",
    )
    benchmark = MlPegGrambowBarrierHeightsBenchmark(
        force_field=Calculator(),
        data_input_dir=data_input_dir,
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
@plot_density_scatter(
    filename=OUT_PATH / "figure_barrier_density.json",
    title="Reaction barrier density plot",
    x_label="Reference activation energy / kcal/mol",
    y_label="Predicted activation energy / kcal/mol",
    annotation_metadata={"system_count": "Reactions"},
)
def barrier_density(analyze_results) -> dict[str, dict]:
    """
    Density scatter inputs for the reaction barrier heights.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``ReactivityResult``.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    density_inputs: dict[str, dict] = {}
    for model_name, result in analyze_results.items():
        reactions = result.reaction_results.values()
        ref = [reaction.activation_energy_ref for reaction in reactions]
        pred = [reaction.activation_energy_pred for reaction in reactions]
        density_inputs[model_name] = {
            "ref": ref,
            "pred": pred,
            "meta": {"system_count": len(pred)},
        }
    return density_inputs


@pytest.fixture
def get_barrier_mae(analyze_results) -> dict[str, float]:
    """
    Get the activation energy (barrier height) MAE for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``ReactivityResult``.

    Returns
    -------
    dict[str, float]
        Mean absolute activation energy error in kcal/mol for each model.
    """
    return {
        model_name: result.mae_activation_energy
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_reaction_energy_mae(analyze_results) -> dict[str, float]:
    """
    Get the reaction energy (enthalpy) MAE for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``ReactivityResult``.

    Returns
    -------
    dict[str, float]
        Mean absolute enthalpy of reaction error in kcal/mol for each model.
    """
    return {
        model_name: result.mae_enthalpy_of_reaction
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_score(analyze_results) -> dict[str, float]:
    """
    Get the mlipaudit benchmark score for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``ReactivityResult``.

    Returns
    -------
    dict[str, float]
        The mlipaudit soft-threshold score (0 to 1) for each model.
    """
    return {model_name: result.score for model_name, result in analyze_results.items()}


@pytest.fixture
@build_table(
    filename=OUT_PATH / "grambow_barrier_heights_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    barrier_density,
    get_barrier_mae: dict[str, float],
    get_reaction_energy_mae: dict[str, float],
    get_score: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    barrier_density
        Density scatter inputs for the reaction barriers (triggers the plot).
    get_barrier_mae
        Activation energy MAEs for all models.
    get_reaction_energy_mae
        Reaction energy MAEs for all models.
    get_score
        The mlipaudit benchmark scores for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Barrier Height MAE": get_barrier_mae,
        "Reaction Energy MAE": get_reaction_energy_mae,
        "Grambow Score": get_score,
    }


def test_grambow_barrier_heights(
    metrics: dict[str, dict], barrier_density: dict[str, dict], struct_info: dict
) -> None:
    """
    Run Grambow barrier heights analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Grambow barrier heights metric results provided by fixtures.
    barrier_density : dict[str, dict]
        Density scatter inputs for the reaction barriers.
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
