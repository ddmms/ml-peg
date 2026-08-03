"""Analyse carbon melt-quench stability simulations."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase import Atoms
from ase.io import read, write
import numpy as np
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import load_models

CATEGORY = "carbon"
BENCHMARK = "melt_quench_stability"

CALC_PATH = CALCS_ROOT / CATEGORY / BENCHMARK / "outputs"
OUT_PATH = APP_ROOT / "data" / CATEGORY / BENCHMARK

MODELS = load_models(current_models)
D3_MODEL_NAMES = build_dispersion_name_map(MODELS)

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

COMPOSITIONS = {
    "C": "Pure C",
    "CH": "C/H",
    "CHN": "C/H/N",
    "CHO": "C/H/O",
}
DENSITIES = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5)
TOTAL_TIME_PS = 13.0
EXPECTED_TRAJECTORIES = len(COMPOSITIONS) * len(DENSITIES)
METRIC = "Stable trajectories"


def load_status(model_name: str) -> list[dict]:
    """
    Load trajectory status records for one model.

    Parameters
    ----------
    model_name
        Registered model name.

    Returns
    -------
    list[dict]
        Status records, or an empty list when no results are available.
    """
    status_path = CALC_PATH / model_name / "status.json"
    if not status_path.exists():
        return []
    try:
        records = json.loads(status_path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        warn(f"Could not read {status_path}: {exc}", stacklevel=2)
        return []
    return records if isinstance(records, list) else []


def stable_percentage(records: list[dict]) -> float:
    """
    Calculate the percentage of trajectories that completed.

    Parameters
    ----------
    records
        Trajectory status records.

    Returns
    -------
    float
        Percentage stable, or NaN unless all expected trajectories are present.
    """
    expected_cases = {
        (composition, density) for composition in COMPOSITIONS for density in DENSITIES
    }
    available_cases = {
        (record.get("composition"), record.get("density_g_cm3")) for record in records
    }
    if available_cases != expected_cases or len(records) != EXPECTED_TRAJECTORIES:
        return np.nan
    return 100 * sum(bool(record.get("stable")) for record in records) / len(records)


def _records_by_case(records: list[dict]) -> dict[tuple[str, float], dict]:
    """
    Index status records by composition and density.

    Parameters
    ----------
    records
        Trajectory status records.

    Returns
    -------
    dict[tuple[str, float], dict]
        Records keyed by composition and density.
    """
    return {
        (record["composition"], float(record["density_g_cm3"])): record
        for record in records
        if "composition" in record and "density_g_cm3" in record
    }


def write_model_plot(model_name: str, records: list[dict]) -> None:
    """
    Write completed simulation time against density for one model.

    Parameters
    ----------
    model_name
        Registered model name.
    records
        Trajectory status records.
    """
    indexed = _records_by_case(records)
    figure = go.Figure()
    for composition, label in COMPOSITIONS.items():
        completed_times = []
        hover_status = []
        for density in DENSITIES:
            record = indexed.get((composition, density))
            completed_times.append(
                None if record is None else record.get("completed_time_ps")
            )
            if record is None:
                hover_status.append("Not run")
            elif record.get("stable"):
                hover_status.append("Completed")
            else:
                hover_status.append(str(record.get("failure") or "Failed"))

        figure.add_trace(
            go.Scatter(
                x=DENSITIES,
                y=completed_times,
                name=label,
                mode="lines+markers",
                customdata=hover_status,
                hovertemplate=(
                    "Density: %{x} g cm⁻³<br>"
                    "Completed: %{y:.3g} ps<br>"
                    "%{customdata}<extra>%{fullData.name}</extra>"
                ),
            )
        )

    figure.add_hline(
        y=TOTAL_TIME_PS,
        line_dash="dash",
        annotation_text="Complete",
        annotation_position="bottom right",
    )
    figure.update_layout(
        title=f"Carbon melt-quench stability — {model_name}",
        xaxis_title="Density / g cm⁻³",
        yaxis_title="Completed simulation time / ps",
        yaxis={"range": [0, TOTAL_TIME_PS * 1.05]},
    )
    model_dir = OUT_PATH / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    figure.write_json(model_dir / "figure_melt_quench_stability.json")


def _last_structure(run_dir: Path, completed_steps: int) -> Atoms | None:
    """
    Read the last valid structure produced for one trajectory.

    Parameters
    ----------
    run_dir
        Trajectory output directory.
    completed_steps
        Number of completed MD steps.

    Returns
    -------
    ase.Atoms | None
        Last valid structure, if available.
    """
    if completed_steps > 10_000:
        filenames = ("quench-traj.extxyz", "melt-traj.extxyz", "initial.extxyz")
    elif completed_steps > 0:
        filenames = ("melt-traj.extxyz", "initial.extxyz")
    else:
        filenames = ("initial.extxyz",)
    for filename in filenames:
        path = run_dir / filename
        if path.exists():
            try:
                return read(path, index=-1)
            except (OSError, ValueError, IndexError):
                continue
    return None


def write_model_structures(model_name: str, records: list[dict]) -> None:
    """
    Write the last valid structure for every available trajectory.

    Parameters
    ----------
    model_name
        Registered model name.
    records
        Trajectory status records.
    """
    model_dir = OUT_PATH / model_name
    for record in records:
        run_name = record.get("run_name")
        composition = record.get("composition")
        if not run_name or not composition:
            continue
        atoms = _last_structure(
            CALC_PATH / model_name / composition / run_name,
            int(record.get("completed_steps", 0)),
        )
        if atoms is None:
            continue
        atoms.calc = None
        structure_dir = model_dir / composition
        structure_dir.mkdir(parents=True, exist_ok=True)
        write(structure_dir / f"{run_name}.extxyz", atoms)


@pytest.fixture
def model_statuses() -> dict[str, list[dict]]:
    """
    Load status records and write app assets for every model.

    Returns
    -------
    dict[str, list[dict]]
        Status records keyed by model name.
    """
    statuses = {}
    for model_name in MODELS:
        records = load_status(model_name)
        statuses[model_name] = records
        write_model_plot(model_name, records)
        write_model_structures(model_name, records)
    return statuses


@pytest.fixture
@build_table(
    filename=OUT_PATH / "melt_quench_stability_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=D3_MODEL_NAMES,
)
def metrics(
    model_statuses: dict[str, list[dict]],
) -> dict[str, dict[str, float | None]]:
    """
    Build the melt-quench stability metric.

    Parameters
    ----------
    model_statuses
        Status records keyed by model name.

    Returns
    -------
    dict[str, dict[str, float | None]]
        Stable trajectory percentage for each model.
    """
    values = {}
    for model_name, records in model_statuses.items():
        value = stable_percentage(records)
        values[model_name] = float(value) if np.isfinite(value) else None
    return {METRIC: values}


def test_melt_quench_stability(
    metrics: dict[str, dict[str, float | None]],
) -> None:
    """
    Write benchmark element metadata.

    Parameters
    ----------
    metrics
        Generated stability metrics.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    (OUT_PATH / "info.json").write_text(
        json.dumps({"elements": ["C", "H", "N", "O"]}, indent=1) + "\n"
    )
