"""Analyse the MC500 molecular-crystal relaxation benchmark."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read, write
import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import (
    build_dispersion_name_map,
    load_metrics_config,
)
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
DISPERSION_NAME_MAP = build_dispersion_name_map(MODELS)

CALC_PATH = CALCS_ROOT / "molecular_crystal" / "MC500" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular_crystal" / "MC500"

RMSCD_THRESHOLD = 0.25

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def rms_cartesian_displacement(
    reference: Atoms, relaxed: Atoms
) -> tuple[float, float, np.ndarray]:
    """
    Calculate symmetric RMS Cartesian displacements between two crystal structures.

    The definition follows van de Streek and Neumann. For each atom, the fractional
    coordinate displacement is converted once with the reference cell and once with
    the relaxed cell. The Cartesian displacement is the mean of those two distances.

    Parameters
    ----------
    reference
        Reference crystal structure.
    relaxed
        Relaxed crystal structure with matching atom order and cell definition.

    Returns
    -------
    tuple[float, float, numpy.ndarray]
        All-atom RMSCD, non-hydrogen RMSCD, and per-atom Cartesian displacements,
        all in Angstrom.

    Raises
    ------
    ValueError
        If the structures do not contain the same atoms in the same order.
    """
    if len(reference) != len(relaxed) or not np.array_equal(
        reference.numbers, relaxed.numbers
    ):
        raise ValueError("Reference and relaxed structures have different atom order")

    fractional_delta = reference.get_scaled_positions(
        wrap=False
    ) - relaxed.get_scaled_positions(wrap=False)
    distance_reference_cell = np.linalg.norm(
        fractional_delta @ reference.cell.array, axis=1
    )
    distance_relaxed_cell = np.linalg.norm(
        fractional_delta @ relaxed.cell.array, axis=1
    )
    displacements = 0.5 * (distance_reference_cell + distance_relaxed_cell)

    rmscd = float(np.sqrt(np.mean(displacements**2)))
    non_hydrogen = reference.numbers != 1
    rmscd_no_h = (
        float(np.sqrt(np.mean(displacements[non_hydrogen] ** 2)))
        if non_hydrogen.any()
        else np.nan
    )
    return rmscd, rmscd_no_h, displacements


def _as_bool(value: str) -> bool:
    """
    Convert a CSV boolean field to ``bool``.

    Parameters
    ----------
    value
        CSV field to convert.

    Returns
    -------
    bool
        Whether the field contains ``"true"``, ignoring case and whitespace.
    """
    return value.strip().lower() == "true"


@pytest.fixture
def mc500_results() -> dict[str, dict[str, Any]]:
    """
    Load relaxation results, calculate RMSCD values, and stage app structures.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-model RMSCD values, labels, convergence flags, and total calculation
        counts.
    """
    results: dict[str, dict[str, Any]] = {}
    all_elements: set[str] = set()

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        csv_path = model_dir / "results.csv"
        if not csv_path.exists():
            continue

        with csv_path.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        model_results: dict[str, Any] = {
            "structure_ids": [],
            "refcodes": [],
            "rmscd": [],
            "rmscd_no_h": [],
            "converged": [],
            "steps": [],
            "max_force": [],
            "total": len(rows),
            "n_converged": sum(_as_bool(row["converged"]) for row in rows),
        }

        app_struct_dir = OUT_PATH / model_name / "structures"
        app_struct_dir.mkdir(parents=True, exist_ok=True)
        app_reference_dir = OUT_PATH / "reference"
        app_reference_dir.mkdir(parents=True, exist_ok=True)

        for row in rows:
            structure_id = row["structure_id"]
            trajectory_path = model_dir / "structures" / f"{structure_id}.xyz"
            if not trajectory_path.exists():
                continue

            reference, relaxed = read(trajectory_path, index=":")
            rmscd, rmscd_no_h, _ = rms_cartesian_displacement(reference, relaxed)

            model_results["structure_ids"].append(structure_id)
            model_results["refcodes"].append(row["refcode"])
            model_results["rmscd"].append(rmscd)
            model_results["rmscd_no_h"].append(rmscd_no_h)
            model_results["converged"].append(_as_bool(row["converged"]))
            model_results["steps"].append(int(row["steps"]))
            model_results["max_force"].append(float(row["max_force"]))
            all_elements.update(reference.get_chemical_symbols())

            # The reference is shared by every model, so it is written once to a
            # common directory, with only the metadata that identifies it.
            reference.info = {"structure_id": structure_id, "refcode": row["refcode"]}
            write(app_reference_dir / f"{structure_id}.xyz", reference)
            write(app_struct_dir / f"{structure_id}.xyz", relaxed)

        results[model_name] = model_results

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with (OUT_PATH / "info.json").open("w", encoding="utf-8") as f:
        json.dump({"elements": sorted(all_elements)}, f, indent=1)

    return results


@pytest.fixture
def rmscd_figures(mc500_results: dict[str, dict[str, Any]]) -> None:
    """
    Write one clickable RMSCD figure per model.

    Parameters
    ----------
    mc500_results
        Per-model MC500 results.
    """
    figures: dict[str, dict] = {}
    for model_name, result in mc500_results.items():
        if not result["rmscd_no_h"]:
            continue

        x_values = list(range(1, len(result["rmscd_no_h"]) + 1))
        customdata = [
            [refcode, converged, steps, max_force]
            for refcode, converged, steps, max_force in zip(
                result["refcodes"],
                result["converged"],
                result["steps"],
                result["max_force"],
                strict=True,
            )
        ]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=result["rmscd_no_h"],
                mode="markers",
                name="Excluding H",
                customdata=customdata,
                hovertemplate=(
                    "CSD refcode: %{customdata[0]}<br>"
                    "RMSCD excluding H: %{y:.4f} Å<br>"
                    "Converged: %{customdata[1]}<br>"
                    "Steps: %{customdata[2]}<br>"
                    "Max force: %{customdata[3]:.4f} eV/Å"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=result["rmscd"],
                mode="markers",
                name="All atoms",
                customdata=customdata,
                hovertemplate=(
                    "CSD refcode: %{customdata[0]}<br>"
                    "All-atom RMSCD: %{y:.4f} Å<br>"
                    "Converged: %{customdata[1]}<br>"
                    "Steps: %{customdata[2]}<br>"
                    "Max force: %{customdata[3]:.4f} eV/Å"
                    "<extra></extra>"
                ),
            )
        )
        fig.add_hline(
            y=RMSCD_THRESHOLD,
            line_dash="dash",
            line_color="#d62728",
            annotation_text=f"problem threshold = {RMSCD_THRESHOLD} Å",
            annotation_position="top right",
        )
        fig.update_layout(
            title=f"MC500 RMS Cartesian displacements – {model_name}",
            xaxis_title="Structure index",
            yaxis_title="RMSCD / Å",
        )
        figures[model_name] = fig.to_plotly_json()

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with (OUT_PATH / "figure_rmscd.json").open("w", encoding="utf-8") as f:
        json.dump(figures, f, cls=PlotlyJSONEncoder)


def _mean(values: list[float]) -> float:
    """
    Return the finite mean of a list, or NaN when it has no finite values.

    Parameters
    ----------
    values
        Values to average.

    Returns
    -------
    float
        Mean of the finite values, or NaN if none are finite.
    """
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else np.nan


@pytest.fixture
@build_table(
    filename=OUT_PATH / "mc500_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
    mlip_name_map=DISPERSION_NAME_MAP,
)
def metrics(
    mc500_results: dict[str, dict[str, Any]], rmscd_figures: None
) -> dict[str, dict[str, float]]:
    """
    Calculate the MC500 summary metrics.

    Parameters
    ----------
    mc500_results
        Per-model MC500 results.
    rmscd_figures
        Triggers creation of the interactive RMSCD figures.

    Returns
    -------
    dict[str, dict[str, float]]
        Metric values for each model.
    """
    del rmscd_figures

    mean_no_h = {}
    mean_all = {}
    within_threshold = {}
    convergence = {}

    for model_name, result in mc500_results.items():
        total = result["total"]
        values_no_h = result["rmscd_no_h"]
        mean_no_h[model_name] = _mean(values_no_h)
        mean_all[model_name] = _mean(result["rmscd"])
        within_threshold[model_name] = (
            100
            * sum(
                np.isfinite(value) and value <= RMSCD_THRESHOLD for value in values_no_h
            )
            / total
            if total
            else np.nan
        )
        convergence[model_name] = (
            100 * result["n_converged"] / total if total else np.nan
        )

    return {
        "Mean RMSCD excluding H": mean_no_h,
        "Mean RMSCD all atoms": mean_all,
        "Structures within threshold": within_threshold,
        "Convergence": convergence,
    }


def test_mc500(metrics: dict[str, dict[str, float]]) -> None:
    """
    Run the MC500 analysis.

    Parameters
    ----------
    metrics
        MC500 summary metrics.
    """
    return
