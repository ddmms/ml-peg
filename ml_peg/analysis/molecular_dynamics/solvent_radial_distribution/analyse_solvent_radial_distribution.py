"""Analyse the solvent radial distribution benchmark."""

from __future__ import annotations

import json
from pathlib import Path

from ase.calculators.calculator import Calculator
from ase.io import read
import numpy as np
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.benchmarks.solvent_radial_distribution.solvent_radial_distribution import (  # noqa: E501
    REFERENCE_MAXIMA,
)
from mlipaudit.io import load_model_output_from_disk

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegSolventRadialDistributionBenchmark
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

MODELS = load_models(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegSolventRadialDistributionBenchmark.name
SOLVENTS = ["CCl4", "methanol", "acetonitrile"]

CALC_PATH = (
    CALCS_ROOT / "molecular_dynamics" / "solvent_radial_distribution" / "outputs"
)
OUT_PATH = APP_ROOT / "data" / "molecular_dynamics" / "solvent_radial_distribution"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def solvent_pdb(solvent: str) -> Path:
    """
    Get the path to a solvent's equilibrated structure saved by the calculation.

    Parameters
    ----------
    solvent
        Name of the solvent.

    Returns
    -------
    Path
        Path to the solvent's equilibrated PDB file.
    """
    return CALC_PATH / BENCHMARK / f"{solvent}_eq.pdb"


def check_dataset() -> None:
    """
    Check the input structures saved by the calculation are available.

    The calculation copies the downloaded input data into its outputs, so the
    analysis does not need to download it again.

    Raises
    ------
    ValueError
        If any solvent structure is missing from the calculation outputs.
    """
    for solvent in SOLVENTS:
        if not solvent_pdb(solvent).exists():
            raise ValueError(
                f"{solvent_pdb(solvent)} does not exist. Please run the calculation."
            )


@pytest.fixture
def analyze_results() -> dict:
    """
    Run the mlipaudit analysis for each model.

    Returns
    -------
    dict
        Mapping of model name to its ``SolventRadialDistributionResult``.
    """
    check_dataset()

    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        benchmark = MlPegSolventRadialDistributionBenchmark(
            force_field=Calculator(),
            data_input_dir=CALC_PATH,
            run_mode="standard",
        )
        benchmark.model_output = load_model_output_from_disk(
            CALC_PATH / model_name, MlPegSolventRadialDistributionBenchmark
        )
        results[model_name] = benchmark.analyze()
    return results


@pytest.fixture
def struct_info() -> dict:
    """
    Write per-solvent element info to ``info.json`` for filtering.

    Elements are stored as one list per solvent, so individual solvents can be
    excluded once partial filtering is supported. The order follows ``SOLVENTS``.

    Returns
    -------
    dict
        Mapping with the per-solvent lists of elements.
    """
    check_dataset()

    info = {
        "systems": list(SOLVENTS),
        "elements": [
            sorted(set(read(solvent_pdb(solvent)).get_chemical_symbols()))
            for solvent in SOLVENTS
        ],
    }

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with (OUT_PATH / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)

    return info


def get_rdf_profiles(analyze_results: dict, solvent: str) -> dict:
    """
    Get the predicted radial distribution profiles of one solvent for each model.

    A vertical line at the experimental first solvent peak position is included as
    the reference trace.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SolventRadialDistributionResult``.
    solvent
        Name of the solvent.

    Returns
    -------
    dict
        Per-model ``(radii, g(r))`` profiles, and the reference peak position.
    """
    results = {}
    for model_name, result in analyze_results.items():
        if result.failed:
            continue
        for structure in result.structures:
            if (
                structure.structure_name != solvent
                or structure.failed
                or structure.radii is None
                or structure.rdf is None
            ):
                continue
            results[model_name] = (structure.radii, structure.rdf)

    ref_peak = REFERENCE_MAXIMA[solvent]["distance"]
    max_rdf = max((max(rdf) for _, rdf in results.values()), default=1.0)
    results["ref"] = ([ref_peak, ref_peak], [0.0, max_rdf])
    return results


@pytest.fixture
@plot_scatter(
    title="Carbon tetrachloride C-C radial distribution function",
    x_label="r / Å",
    y_label="g(r)",
    show_line=True,
    show_markers=False,
    filename=str(OUT_PATH / "figure_rdf_CCl4.json"),
)
def rdf_ccl4(analyze_results) -> dict[str, tuple[list, list]]:
    """
    Get the predicted carbon tetrachloride RDF profiles for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SolventRadialDistributionResult``.

    Returns
    -------
    dict[str, tuple[list, list]]
        Per-model ``(radii, g(r))`` profiles, and the reference peak position.
    """
    return get_rdf_profiles(analyze_results, "CCl4")


@pytest.fixture
@plot_scatter(
    title="Methanol O-O radial distribution function",
    x_label="r / Å",
    y_label="g(r)",
    show_line=True,
    show_markers=False,
    filename=str(OUT_PATH / "figure_rdf_methanol.json"),
)
def rdf_methanol(analyze_results) -> dict[str, tuple[list, list]]:
    """
    Get the predicted methanol RDF profiles for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SolventRadialDistributionResult``.

    Returns
    -------
    dict[str, tuple[list, list]]
        Per-model ``(radii, g(r))`` profiles, and the reference peak position.
    """
    return get_rdf_profiles(analyze_results, "methanol")


@pytest.fixture
@plot_scatter(
    title="Acetonitrile N-N radial distribution function",
    x_label="r / Å",
    y_label="g(r)",
    show_line=True,
    show_markers=False,
    filename=str(OUT_PATH / "figure_rdf_acetonitrile.json"),
)
def rdf_acetonitrile(analyze_results) -> dict[str, tuple[list, list]]:
    """
    Get the predicted acetonitrile RDF profiles for each model.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SolventRadialDistributionResult``.

    Returns
    -------
    dict[str, tuple[list, list]]
        Per-model ``(radii, g(r))`` profiles, and the reference peak position.
    """
    return get_rdf_profiles(analyze_results, "acetonitrile")


@pytest.fixture
def get_peak_deviations(analyze_results) -> dict[str, dict[str, float]]:
    """
    Get the first solvent peak deviation of each solvent for each model.

    If a solvent's simulation failed or is missing, its deviation is NaN, so the
    model's overall benchmark score is also NaN.

    Parameters
    ----------
    analyze_results
        Mapping of model name to its ``SolventRadialDistributionResult``.

    Returns
    -------
    dict[str, dict[str, float]]
        Per-solvent deviation of the first solvent peak from the reference, in
        Angstrom, for each model.
    """
    deviations = {solvent: {} for solvent in SOLVENTS}
    for model_name, result in analyze_results.items():
        structures = {
            structure.structure_name: structure for structure in result.structures
        }
        for solvent in SOLVENTS:
            structure = structures.get(solvent)
            if (
                structure is None
                or structure.failed
                or structure.peak_deviation is None
            ):
                deviations[solvent][model_name] = np.nan
            else:
                deviations[solvent][model_name] = structure.peak_deviation
    return deviations


@pytest.fixture
@build_table(
    filename=OUT_PATH / "solvent_radial_distribution_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    rdf_ccl4,
    rdf_methanol,
    rdf_acetonitrile,
    get_peak_deviations: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    rdf_ccl4
        Predicted carbon tetrachloride RDF profiles (triggers the RDF plot).
    rdf_methanol
        Predicted methanol RDF profiles (triggers the RDF plot).
    rdf_acetonitrile
        Predicted acetonitrile RDF profiles (triggers the RDF plot).
    get_peak_deviations
        Per-solvent first solvent peak deviations for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        f"Peak Deviation ({solvent})": get_peak_deviations[solvent]
        for solvent in SOLVENTS
    }


def test_solvent_radial_distribution(
    metrics: dict[str, dict], struct_info: dict
) -> None:
    """
    Run solvent radial distribution analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Solvent RDF metric results provided by fixtures.
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
