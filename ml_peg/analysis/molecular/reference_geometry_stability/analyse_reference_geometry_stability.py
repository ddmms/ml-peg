"""Analyse the reference geometry stability benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase import Atoms, units
from ase.build.rotate import rotation_matrix_from_points
from ase.io import write
import numpy as np
import plotly.graph_objects as go
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
import mdtraj as md
from mlipaudit.benchmarks.reference_geometry_stability import (
    reference_geometry_stability as rgs,
)
from mlipaudit.io import load_model_output_from_disk
from mlipaudit.scoring import compute_benchmark_score
from mlipaudit.utils.stability import is_simulation_stable
from mlipaudit.utils.trajectory_helpers import create_mdtraj_trajectory_from_positions

from ml_peg.analysis.utils.decorators import build_table, merge_saved_traces, plot_hist
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    check_dataset,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegReferenceGeometryStabilityBenchmark
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

FORCE_THRESHOLD = rgs.SIMULATION_CONFIG["max_force_convergence_threshold"]


def converged(state) -> bool:
    """
    Check whether a minimization reached the force convergence threshold.

    The saved forces correspond to the saved geometry, so this also catches a state
    that never recorded the frame its minimization finished on.

    Parameters
    ----------
    state
        Simulation state saved for one molecule.

    Returns
    -------
    bool
        Whether the maximum force on the final saved frame is within the threshold.
    """
    if state.forces is None:
        return False
    return bool(np.linalg.norm(state.forces[-1], axis=1).max() <= FORCE_THRESHOLD)


@pytest.fixture
def reference_datasets() -> dict:
    """
    Read the OpenFF references staged by the calculation.

    Returns
    -------
    dict
        Reference molecules by subset and molecule identifier.
    """
    datasets = {}
    for subset, filename in (
        ("openff_neutral", rgs.OPENFF_NEUTRAL_FILENAME),
        ("openff_charged", rgs.OPENFF_CHARGED_FILENAME),
    ):
        path = CALC_PATH / BENCHMARK / filename
        check_dataset(path)
        datasets[subset] = rgs.Molecules.validate_json(path.read_text())
    return datasets


def analyze_model(model, model_output, datasets) -> tuple[dict, go.Scatter]:
    """
    Calculate geometry metrics and save the aligned structures for one model.

    Parameters
    ----------
    model
        Model identifier used in the output directory.
    model_output
        Raw model output containing saved minimization frames and failure flags.
    datasets
        OpenFF reference molecules keyed by subset and identifier.

    Returns
    -------
    tuple[dict, go.Scatter]
        Locally calculated metrics and the per-molecule RMSD plot trace.
    """
    labels, plotted, points, symbols = [], [], [], []
    rmsds = []
    subset_means, relaxed_means = [], []
    score_inputs = []
    audit_inputs = []
    complete = True
    total = sum(len(dataset) for dataset in datasets.values())
    n_converged = 0
    directory = OUT_PATH / "structures" / model
    directory.mkdir(parents=True, exist_ok=True)
    for subset, dataset in datasets.items():
        outputs = getattr(model_output, subset)
        names = {output.molecule_name for output in outputs}
        if len(names) != len(outputs) or not names.issubset(dataset):
            raise ValueError(f"Invalid molecule identifiers in {subset}")
        complete = complete and names == dataset.keys()
        subset_rmsds, relaxed_rmsds = [], []
        for index, output in enumerate(outputs):
            state = output.simulation_state
            if output.failed or state is None or not is_simulation_stable(state):
                score_inputs.append(None)
                audit_inputs.append(None)
                complete = False
                continue
            molecule = dataset[output.molecule_name]
            reference = Atoms(molecule.atom_symbols, positions=molecule.coordinates)
            positions = state.positions[-1]
            # Match the alignment, precision and unit conversion used by MLIP Audit.
            target = create_mdtraj_trajectory_from_positions(
                positions, molecule.atom_symbols
            )
            ref = create_mdtraj_trajectory_from_positions(
                reference.positions, molecule.atom_symbols
            )
            heavy_indices = ref.top.select("not element H")
            value = (
                float(md.rmsd(target, ref, atom_indices=heavy_indices)[0]) * units.nm
            )
            if not np.isfinite(value):
                score_inputs.append(None)
                audit_inputs.append(None)
                complete = False
                continue
            audit_inputs.append(value)
            subset_rmsds.append(value)
            is_converged = converged(state)
            if is_converged:
                n_converged += 1
                relaxed_rmsds.append(value)
                score_inputs.append(value)
                rmsds.append(value)
            else:
                # An unrelaxed geometry stays near the reference, so score it zero
                # rather than rewarding the stalled minimization.
                score_inputs.append(None)
            heavy = reference.numbers != 1
            reference.translate(-reference.positions[heavy].mean(axis=0))
            centered = positions - positions[heavy].mean(axis=0)
            rotation = rotation_matrix_from_points(
                centered[heavy].T, reference.positions[heavy].T
            )
            relaxed = Atoms(molecule.atom_symbols, positions=centered @ rotation.T)
            asset = f"{subset}_{index}"
            write(directory / f"{asset}.xyz", [reference, relaxed], format="xyz")
            labels.append(f"{subset.removeprefix('openff_')} / {output.molecule_name}")
            plotted.append(value)
            status = "Relaxed" if is_converged else "Unconverged (scored zero)"
            symbols.append("circle" if is_converged else "circle-open")
            # ``struct_from_scatter`` reads the asset from the last customdata entry.
            points.append([status, f"{model}/{asset}"])
        subset_means.append(np.mean(subset_rmsds) if subset_rmsds else np.nan)
        relaxed_means.append(np.mean(relaxed_rmsds) if relaxed_rmsds else np.nan)
    mean_rmsd = float(np.mean(subset_means))
    relaxed_mean = float(np.mean(relaxed_means))
    score = (
        compute_benchmark_score([score_inputs], [rgs.RMSD_SCORE_THRESHOLD])
        if score_inputs
        else np.nan
    )
    # MLIP Audit applies no convergence criterion, so scores the geometry as saved.
    audit_score = (
        compute_benchmark_score([audit_inputs], [rgs.RMSD_SCORE_THRESHOLD])
        if audit_inputs
        else np.nan
    )
    row = {
        "rmsds": rmsds,
        "Avg RMSD": mean_rmsd if complete else np.nan,
        "Avg RMSD (relaxed only)": relaxed_mean,
        "Geometry Score": score if complete else np.nan,
        "Geometry Score (MLIP Audit)": audit_score,
        # A rate over the molecules that happened to run is not comparable with one
        # over the full dataset, so a partial run reports nothing.
        "Convergence Rate": n_converged / total if complete and total else np.nan,
    }
    return row, go.Scatter(
        name=model,
        x=labels,
        y=plotted,
        customdata=points,
        mode="markers",
        marker={"symbol": symbols},
        hovertemplate=("%{x}<br>Heavy-atom RMSD: %{y:.4g} Å<br>%{customdata[0]}"),
    )


@pytest.fixture
def analyze_results(reference_datasets) -> dict:
    """
    Calculate metrics from saved model output and staged reference geometries.

    Parameters
    ----------
    reference_datasets
        Staged OpenFF reference molecules for both subsets.

    Returns
    -------
    dict
        Metrics and per-molecule RMSDs by model, or None if analysis failed.
    """
    results = {}
    figure = go.Figure()
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name / BENCHMARK
        if not (output_dir / "model_output.zip").exists():
            continue
        try:
            output = load_model_output_from_disk(
                CALC_PATH / model_name, MlPegReferenceGeometryStabilityBenchmark
            )
            row, trace = analyze_model(model_name, output, reference_datasets)
        except Exception as exc:
            warn(f"Cannot score {model_name}: {exc}", stacklevel=2)
            results[model_name] = None
            continue
        results[model_name] = row
        figure.add_trace(trace)
    figure.update_layout(
        title="Heavy-atom RMSD by molecule",
        xaxis_title="OpenFF subset / molecule",
        yaxis_title="RMSD / Å",
        xaxis={"nticks": 12},
    )
    path = OUT_PATH / "figure_molecule_rmsd.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    merge_saved_traces(figure, path).write_json(path)
    return results


@pytest.fixture
def struct_info(reference_datasets) -> dict:
    """
    Write the combined element set to ``info.json`` for filtering.

    Parameters
    ----------
    reference_datasets
        Staged OpenFF reference molecules for both subsets.

    Returns
    -------
    dict
        Mapping with the sorted list of elements present in the dataset.
    """
    elements = sorted(
        {
            symbol
            for dataset in reference_datasets.values()
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
        Metrics and per-molecule RMSDs by model, or None if analysis failed.

    Returns
    -------
    dict[str, list]
        RMSD values (Angstrom) across both datasets for each model.
    """
    return {
        model: result["rmsds"]
        for model, result in analyze_results.items()
        if result is not None and result["rmsds"]
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
    analyze_results: dict,
) -> dict[str, dict]:
    """
    Get all metrics.

    Parameters
    ----------
    rmsd_histogram
        Per-molecule RMSD values (triggers the histogram plot).
    analyze_results
        Calculated metrics and per-molecule RMSDs by model.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        metric: {
            model: result[metric] if result is not None else np.nan
            for model, result in analyze_results.items()
        }
        for metric in DEFAULT_THRESHOLDS
    }


@pytest.mark.framework("mlip_audit")
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
