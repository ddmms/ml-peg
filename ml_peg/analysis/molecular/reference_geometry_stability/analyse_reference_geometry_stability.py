"""Analyse the reference geometry stability benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase import units
from ase.calculators.calculator import Calculator
import numpy as np
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
import mdtraj as md
from mlipaudit.benchmarks.reference_geometry_stability import (
    reference_geometry_stability as rgs,
)
from mlipaudit.io import load_model_output_from_disk
from mlipaudit.scoring import compute_benchmark_score
from mlipaudit.utils.stability import is_simulation_stable

from ml_peg.analysis.utils.decorators import build_table, plot_hist
from ml_peg.analysis.utils.utils import build_dispersion_name_map, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegReferenceGeometryStabilityBenchmark
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegReferenceGeometryStabilityBenchmark.name

CALC_PATH = CALCS_ROOT / "molecular" / "reference_geometry_stability" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "reference_geometry_stability"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


@pytest.fixture
def reference_benchmark():
    """
    Load both reference subsets from staged outputs or the benchmark archive.

    Returns
    -------
    MlPegReferenceGeometryStabilityBenchmark
        Benchmark configured with the available reference dataset.
    """
    filenames = (rgs.OPENFF_NEUTRAL_FILENAME, rgs.OPENFF_CHARGED_FILENAME)
    data_root = CALC_PATH
    if not all((data_root / BENCHMARK / name).is_file() for name in filenames):
        data_root = download_s3_data(
            key=f"inputs/molecular/{BENCHMARK}/{BENCHMARK}.zip",
            filename=f"{BENCHMARK}.zip",
        )
    return MlPegReferenceGeometryStabilityBenchmark(
        force_field=Calculator(), data_input_dir=data_root, run_mode="standard"
    )


def _subset_rmsds(dataset, records):
    """
    Align each converged geometry to its reference and retain individual failures.

    Parameters
    ----------
    dataset
        Reference molecules keyed by ID.
    records
        Model output for the same set of molecules.

    Returns
    -------
    list
        Heavy-atom RMSDs in angstrom, with None for failed molecules.
    """
    if (
        len(records) != len(dataset)
        or {r.molecule_name for r in records} != dataset.keys()
    ):
        raise ValueError("Archive molecule IDs do not match the reference dataset")
    values = []
    for record in records:
        if record.failed:
            values.append(None)
            continue
        state = record.simulation_state
        molecule = dataset[record.molecule_name]
        forces = np.asarray(state.forces[-1])
        if (
            not np.isfinite(forces).all()
            or np.linalg.norm(forces, axis=1).max()
            > rgs.SIMULATION_CONFIG["max_force_convergence_threshold"]
        ):
            raise ValueError(
                f"{record.molecule_name}: forces exceed convergence tolerance. "
                "Rerun the calculation."
            )
        if not is_simulation_stable(state):
            values.append(None)
            continue
        ref = rgs.create_mdtraj_trajectory_from_positions(
            np.asarray(molecule.coordinates), molecule.atom_symbols
        )
        pred = rgs.create_mdtraj_trajectory_from_positions(
            state.positions[-1], molecule.atom_symbols
        )
        rmsd_nm = md.rmsd(
            pred, ref, atom_indices=ref.top.select("not element H"), parallel=False
        )[0]
        values.append(float(rmsd_nm * units.nm / units.Angstrom))
    return values


@pytest.fixture
def analyze_results(reference_benchmark) -> dict:
    """
    Run the mlipaudit analysis for each model.

    Parameters
    ----------
    reference_benchmark
        Benchmark initialized with both OpenFF reference subsets.

    Returns
    -------
    dict
        Per-molecule RMSDs by model, or None for an invalid archive.
    """
    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        try:
            output = load_model_output_from_disk(
                CALC_PATH / model_name, MlPegReferenceGeometryStabilityBenchmark
            )
            results[model_name] = _subset_rmsds(
                reference_benchmark._openff_neutral_dataset, output.openff_neutral
            ) + _subset_rmsds(
                reference_benchmark._openff_charged_dataset, output.openff_charged
            )
        except Exception as exc:
            warn(f"Cannot score {model_name}: {exc}", stacklevel=2)
            results[model_name] = None
    return results


@pytest.fixture
def struct_info(reference_benchmark) -> dict:
    """
    Write the combined element set to ``info.json`` for filtering.

    Parameters
    ----------
    reference_benchmark
        Benchmark initialized with both OpenFF reference subsets.

    Returns
    -------
    dict
        Mapping with the sorted list of elements present in the dataset.
    """
    benchmark = reference_benchmark
    elements = sorted(
        {
            symbol
            for dataset in (
                benchmark._openff_neutral_dataset,
                benchmark._openff_charged_dataset,
            )
            for molecule in dataset.values()
            for symbol in molecule.atom_symbols
        }
    )
    info = {"elements": elements}
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_text(json.dumps(info, indent=1))
    return info


@pytest.fixture
@plot_hist(
    filename=OUT_PATH / "figure_rmsd_histogram.json",
    bins=40,
    title="Heavy-atom RMSD after minimization",
    x_label="RMSD / Å",
    y_label="Probability density",
)
def rmsd_histogram(analyze_results) -> dict[str, list]:
    """
    Get per-molecule heavy-atom RMSD values for each model.

    Parameters
    ----------
    analyze_results
        Per-molecule RMSDs by model, or None for an invalid archive.

    Returns
    -------
    dict[str, list]
        RMSD values (Angstrom) across both datasets for each model.
    """
    results = {}
    for model_name, values in analyze_results.items():
        successful = [value for value in values or [] if value is not None]
        if successful:
            results[model_name] = successful
    return results


@pytest.fixture
def get_avg_rmsd(analyze_results) -> dict[str, float]:
    """
    Get the average heavy-atom RMSD for each model.

    Parameters
    ----------
    analyze_results
        Per-molecule RMSDs by model, or None for an invalid archive.

    Returns
    -------
    dict[str, float]
        Average RMSD in Angstrom for each model.
    """
    return {
        model_name: float(np.mean(values)) if values and None not in values else np.nan
        for model_name, values in analyze_results.items()
    }


@pytest.fixture
def get_score(analyze_results) -> dict[str, float]:
    """
    Get the mlipaudit benchmark score for each model.

    Parameters
    ----------
    analyze_results
        Per-molecule RMSDs by model, or None for an invalid archive.

    Returns
    -------
    dict[str, float]
        The mlipaudit per-molecule soft-threshold RMSD score (0 to 1) for each model.
    """
    return {
        model_name: compute_benchmark_score([values], [rgs.RMSD_SCORE_THRESHOLD])
        if values
        else np.nan
        for model_name, values in analyze_results.items()
    }


@pytest.fixture
@build_table(
    filename=OUT_PATH / "reference_geometry_stability_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    rmsd_histogram,
    get_avg_rmsd: dict[str, float],
    get_score: dict[str, float],
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    rmsd_histogram
        Per-molecule RMSD values (triggers the histogram plot).
    get_avg_rmsd
        Average RMSDs for all models.
    get_score
        The mlipaudit benchmark scores for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "Avg RMSD": get_avg_rmsd,
        "Geometry Score": get_score,
    }


def test_reference_geometry_stability(
    metrics: dict[str, dict], struct_info: dict
) -> None:
    """
    Run reference geometry stability analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Reference geometry stability metric results provided by fixtures.
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    """
