"""Analyse the protein conformational sampling benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from shutil import copyfile
from warnings import warn

from ase.calculators.calculator import Calculator
from ase.io import read
import numpy as np
import plotly.graph_objects as go
import pytest

pytest.importorskip("mlipaudit", reason="Please install `mlipaudit` extra")
from mlipaudit.io import load_model_output_from_disk
from mlipaudit.utils.biomolecules import STRUCTURE_NAMES

from ml_peg.analysis.utils.decorators import build_table, merge_saved_traces
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    check_dataset,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.mlipaudit import MlPegSamplingBenchmark
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

BENCHMARK = MlPegSamplingBenchmark.name

# Directory the calculation copied the input structures to, matching the directory
# mlipaudit reads them from, i.e. ``{data_input_dir}/{data_name or name}``.
BENCHMARK_DATA_DIR = MlPegSamplingBenchmark.data_name or MlPegSamplingBenchmark.name

CALC_PATH = CALCS_ROOT / "biomolecules" / "protein_sampling" / "outputs"
OUT_PATH = APP_ROOT / "data" / "biomolecules" / "protein_sampling"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


@pytest.fixture
def dataset_dir() -> Path:
    """
    Locate the shared reference data staged by the calculation.

    Returns
    -------
    Path
        Directory containing the starting structures, topologies and angle libraries.
    """
    directory = CALC_PATH / BENCHMARK_DATA_DIR
    for name in STRUCTURE_NAMES:
        check_dataset(directory / f"{name}.xyz")
        check_dataset(directory / f"{name}.pdb")
    for name in ("backbone_reference_data.json", "sidechain_reference_data.json"):
        check_dataset(directory / name)
    return directory


@pytest.fixture
def analyze_results(dataset_dir: Path) -> dict:
    """
    Run the MLIP Audit analysis for each model.

    Parameters
    ----------
    dataset_dir
        Directory containing the staged benchmark data.

    Returns
    -------
    dict
        Mapping of model name to its sampling result, or None for an unreadable run.
    """
    benchmark = MlPegSamplingBenchmark(
        force_field=Calculator(), data_input_dir=dataset_dir.parent, run_mode="standard"
    )
    results = {}
    for model_name in MODELS:
        output_dir = CALC_PATH / model_name
        if not (output_dir / BENCHMARK / "model_output.zip").exists():
            continue
        try:
            benchmark.model_output = load_model_output_from_disk(
                output_dir, MlPegSamplingBenchmark
            )
            results[model_name] = benchmark.analyze()
        except Exception as exc:
            warn(f"Cannot score {model_name}: {exc}", stacklevel=2)
            results[model_name] = None
    return results


@pytest.fixture
def struct_info(dataset_dir: Path) -> dict:
    """
    Write per-structure element info to ``info.json`` for filtering.

    Elements are stored as one list per structure, so individual structures can
    be excluded once partial filtering is supported. The order follows
    ``STRUCTURE_NAMES``.

    Parameters
    ----------
    dataset_dir
        Directory containing the staged starting structures.

    Returns
    -------
    dict
        Mapping with the per-structure lists of elements.
    """
    elements = []
    for name in STRUCTURE_NAMES:
        path = dataset_dir / f"{name}.xyz"
        try:
            elements.append(sorted(set(read(path).get_chemical_symbols())))
        except Exception as exc:
            pytest.skip(f"Cannot read starting structure {path}: {exc}")

    info = {"systems": list(STRUCTURE_NAMES), "elements": elements}
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    for name in STRUCTURE_NAMES:
        copyfile(dataset_dir / f"{name}.xyz", OUT_PATH / f"{name}.xyz")
    with (OUT_PATH / "info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, indent=1)

    return info


@pytest.fixture
@build_table(
    filename=OUT_PATH / "protein_sampling_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(analyze_results: dict) -> dict[str, dict]:
    """
    Report upstream scores, backbone diagnostics and simulation coverage.

    Parameters
    ----------
    analyze_results
        Upstream sampling results by model, including failures.

    Returns
    -------
    dict[str, dict]
        Metric names mapped to per-model values.
    """
    rows = {name: {} for name in DEFAULT_THRESHOLDS}
    for model, result in analyze_results.items():
        for metric, attribute in (
            ("Sampling Score", "score"),
            ("Backbone Dihedral RMSD", "rmsd_backbone_total"),
            ("Backbone Hellinger Distance", "hellinger_distance_backbone_total"),
            ("Backbone Outliers Ratio", "outliers_ratio_backbone_total"),
        ):
            value = getattr(result, attribute, None)
            rows[metric][model] = value if value is not None else np.nan
        rows["Stable Proteins"][model] = (
            sum(not system.failed for system in result.systems)
            if result is not None
            else np.nan
        )
    return rows


@pytest.mark.framework("mlip_audit")
def test_protein_sampling(
    metrics: dict[str, dict], struct_info: dict, analyze_results: dict
) -> None:
    """
    Run protein sampling analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Protein sampling metric results provided by fixtures.
    struct_info : dict
        Element info written to ``info.json`` for filtering.
    analyze_results
        Upstream sampling results containing metrics for each residue type.
    """
    for metric, attribute in (
        ("Backbone Dihedral RMSD", "rmsd_backbone_dihedrals"),
        ("Backbone Hellinger Distance", "hellinger_distance_backbone_dihedrals"),
        ("Backbone Outliers Ratio", "outliers_ratio_backbone_dihedrals"),
    ):
        figure = go.Figure()
        for model, result in analyze_results.items():
            values = getattr(result, attribute, None) or {}
            residues = sorted(values)
            contributors = [
                [
                    system.structure_name
                    for system in result.systems
                    if not system.failed
                    and residue in (getattr(system, attribute) or {})
                ]
                for residue in residues
            ]
            figure.add_scatter(
                name=model,
                mode="markers",
                x=residues,
                y=[values[residue] for residue in residues],
                customdata=contributors,
                hovertemplate="Residue: %{x}<br>Value: %{y:.4g}",
            )
        figure.update_layout(
            title=f"{metric} by residue type",
            xaxis_title="Residue type",
            yaxis_title=metric,
        )
        path = OUT_PATH / f"figure_{metric.lower().replace(' ', '_')}.json"
        merge_saved_traces(figure, path).write_json(path)
