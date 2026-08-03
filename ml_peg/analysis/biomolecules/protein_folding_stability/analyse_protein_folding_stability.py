"""Analyse the protein folding stability benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.calculator import Calculator
from ase.io import read
import numpy as np
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.folding_stability.folding_stability import STRUCTURE_NAMES
from mlipaudit.io import load_model_output_from_disk

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegFoldingStabilityBenchmark
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegFoldingStabilityBenchmark.name

CALC_PATH = CALCS_ROOT / "biomolecules" / "protein_folding_stability" / "outputs"
OUT_PATH = APP_ROOT / "data" / "biomolecules" / "protein_folding_stability"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def structure_xyz(structure_name: str) -> Path:
    """
    Get the path to a starting structure saved by the calculation.

    Parameters
    ----------
    structure_name
        Name of the structure.

    Returns
    -------
    Path
        Path to the structure's starting geometry.
    """
    return CALC_PATH / BENCHMARK / "starting_structures" / f"{structure_name}.xyz"


def check_dataset() -> None:
    """
    Check the input structures saved by the calculation are available.

    The calculation copies the downloaded input data into its outputs, so the
    analysis does not need to download it again.

    Raises
    ------
    ValueError
        If any starting structure is missing from the calculation outputs.
    """
    for structure_name in STRUCTURE_NAMES:
        if not structure_xyz(structure_name).exists():
            raise ValueError(
                f"{structure_xyz(structure_name)} does not exist. "
                "Please run the calculation."
            )


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``FoldingStabilityResult``.
    """
    check_dataset()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegFoldingStabilityBenchmark(
            force_field=Calculator(),
            data_input_dir=CALC_PATH,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegFoldingStabilityBenchmark
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
def struct_info() -> dict:
    """
    Write per-structure element info to ``info.json`` for filtering.

    Elements are stored as one list per structure, so individual structures can
    be excluded once partial filtering is supported. The order follows
    ``STRUCTURE_NAMES``.

    Returns
    -------
    dict
        Mapping with the per-structure lists of elements.
    """
    check_dataset()

    info = {
        "systems": list(STRUCTURE_NAMES),
        "elements": [
            sorted(set(read(structure_xyz(name)).get_chemical_symbols()))
            for name in STRUCTURE_NAMES
        ],
    }

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with (OUT_PATH / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)

    return info


@pytest.fixture
@plot_scatter(
    title="RMSD from reference structure along trajectory",
    x_label="Frame",
    y_label="RMSD / Å",
    show_line=True,
    show_markers=False,
    filename=str(OUT_PATH / "figure_rmsd_trajectory.json"),
)
def rmsd_trajectories(analyze_results) -> dict[str, tuple[list, list]]:
    """
    Get the RMSD trajectory averaged across structures for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``FoldingStabilityResult``.

    Returns
    -------
    dict[str, tuple[list, list]]
        Per-model ``(frame, mean RMSD)`` profiles across the trajectory.
    """
    results = {}
    for model_name, result in analyze_results.items():
        if result.failed:
            continue
        trajectories = [
            molecule.rmsd_trajectory
            for molecule in result.molecules
            if not molecule.failed and molecule.rmsd_trajectory is not None
        ]
        if not trajectories:
            continue
        num_frames = min(len(traj) for traj in trajectories)
        stacked = np.array([traj[:num_frames] for traj in trajectories])
        mean_rmsd = stacked.mean(axis=0)
        results[model_name] = (list(range(num_frames)), mean_rmsd.tolist())
    return results


@pytest.fixture
def get_avg_rmsd(analyze_results) -> dict[str, float]:
    """
    Get the average RMSD for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``FoldingStabilityResult``.

    Returns
    -------
    dict[str, float]
        Average RMSD from the reference structure, averaged across molecules.
    """
    return {
        model_name: (result.avg_rmsd if result.avg_rmsd is not None else np.nan)
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_avg_tm_score(analyze_results) -> dict[str, float]:
    """
    Get the average TM score for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``FoldingStabilityResult``.

    Returns
    -------
    dict[str, float]
        Average TM score against the reference structure, averaged across molecules.
    """
    return {
        model_name: (result.avg_tm_score if result.avg_tm_score is not None else np.nan)
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
def get_rgyr_deviation(analyze_results) -> dict[str, float]:
    """
    Get the maximum radius of gyration deviation for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``FoldingStabilityResult``.

    Returns
    -------
    dict[str, float]
        Maximum absolute deviation of the radius of gyration from the initial
        state, taken across molecules, in Angstrom.
    """
    return {
        model_name: (
            result.max_abs_deviation_radius_of_gyration
            if result.max_abs_deviation_radius_of_gyration is not None
            else np.nan
        )
        for model_name, result in analyze_results.items()
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "protein_folding_stability_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    rmsd_trajectories,
    get_avg_rmsd: dict[str, float],
    get_avg_tm_score: dict[str, float],
    get_rgyr_deviation: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    rmsd_trajectories
        Per-model averaged RMSD trajectories (triggers the RMSD line plot).
    get_avg_rmsd
        Average RMSD values for all models.
    get_avg_tm_score
        Average TM scores for all models.
    get_rgyr_deviation
        Maximum radius of gyration deviations for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "RMSD": get_avg_rmsd,
        "TM Score": get_avg_tm_score,
        "Rgyr Deviation": get_rgyr_deviation,
    }


def test_protein_folding_stability(metrics: dict[str, dict], struct_info: dict) -> None:
    """
    Run protein folding stability analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Protein folding stability metric results provided by fixtures.
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
