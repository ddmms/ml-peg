"""Analyse elasticity benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import (
    build_table,
    plot_density_scatter,
)
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "elasticity" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "elasticity"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

K_COLUMN = "K_vrh"
G_COLUMN = "G_vrh"


def _load_results(model_name: str) -> pd.DataFrame | None:
    """
    Load benchmark results for a model, if available.

    Parameters
    ----------
    model_name
        Name of the model whose results should be read.

    Returns
    -------
    pd.DataFrame | None
        Results dataframe if present, otherwise ``None``.
    """
    results_path = CALC_PATH / model_name / "moduli_results.csv"
    if not results_path.exists():
        return None
    return pd.read_csv(results_path)


def _filter_results(df: pd.DataFrame, model_name: str) -> tuple[pd.DataFrame, int]:
    """
    Filter outlier predictions and return remaining data with exclusion count.

    Parameters
    ----------
    df
        Dataframe containing raw benchmark results.
    model_name
        Model whose columns should be filtered.

    Returns
    -------
    tuple[pd.DataFrame, int]
        Filtered dataframe and number of excluded systems.
    """
    mask_bulk = df[f"{K_COLUMN}_{model_name}"].between(-50, 600)
    mask_shear = df[f"{G_COLUMN}_{model_name}"].between(-50, 600)
    valid = df[mask_bulk & mask_shear].copy()
    excluded = len(df) - len(valid)
    return valid, excluded


def _collect_model_data() -> dict[str, dict[str, Any]]:
    """
    Collect filtered results, MAEs, and exclusion counts per model.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of model name to bulk/shear data and exclusion metadata.
    """
    stats: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        df = _load_results(model_name)
        if df is None:
            stats[model_name] = {
                "bulk": None,
                "shear": None,
                "excluded": None,
            }
            continue
        filtered, excluded = _filter_results(df, model_name)
        if filtered.empty:
            stats[model_name] = {
                "bulk": None,
                "shear": None,
                "excluded": excluded,
            }
            continue
        stats[model_name] = {
            "bulk": {
                "ref": filtered[f"{K_COLUMN}_DFT"].tolist(),
                "pred": filtered[f"{K_COLUMN}_{model_name}"].tolist(),
            },
            "shear": {
                "ref": filtered[f"{G_COLUMN}_DFT"].tolist(),
                "pred": filtered[f"{G_COLUMN}_{model_name}"].tolist(),
            },
            "excluded": excluded,
        }
    return stats


def _density_inputs(property_key: str, model_stats: dict[str, dict[str, Any]]) -> dict:
    """
    Prepare mapping for density scatter decorator.

    Parameters
    ----------
    property_key
        Property key to extract (``"bulk"`` or ``"shear"``).
    model_stats
        Aggregated statistics for each model.

    Returns
    -------
    dict
        Mapping of model name to density-scatter inputs.
    """
    inputs = {}
    for model_name in MODELS:
        prop = model_stats.get(model_name, {}).get(property_key)
        excluded = model_stats.get(model_name, {}).get("excluded")
        if not prop:
            inputs[model_name] = {
                "ref": [],
                "pred": [],
                "meta": {"excluded": excluded},
            }
            continue
        ref_vals = prop["ref"]
        pred_vals = prop["pred"]
        inputs[model_name] = {
            "ref": ref_vals,
            "pred": pred_vals,
            "mae": mae(ref_vals, pred_vals) if ref_vals else None,
            "meta": {"excluded": excluded},
        }
    return inputs


@pytest.fixture
def elasticity_stats() -> dict[str, dict[str, Any]]:
    """
    Load and cache processed benchmark statistics per model.

    Returns
    -------
    dict[str, dict[str, Any]]
        Processed information per model (bulk, shear, exclusion counts).
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    return _collect_model_data()


@pytest.fixture
def bulk_mae(elasticity_stats: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """
    Mean absolute error for bulk modulus predictions.

    Parameters
    ----------
    elasticity_stats
        Aggregated bulk/shear data per model.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model (``None`` if no data).
    """
    results: dict[str, float | None] = {}
    for model_name in MODELS:
        prop = elasticity_stats.get(model_name, {}).get("bulk")
        if not prop:
            results[model_name] = None
            continue
        results[model_name] = mae(prop["ref"], prop["pred"])
    return results


@pytest.fixture
def shear_mae(elasticity_stats: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    """
    Mean absolute error for shear modulus predictions.

    Parameters
    ----------
    elasticity_stats
        Aggregated bulk/shear data per model.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model (``None`` if no data).
    """
    results: dict[str, float | None] = {}
    for model_name in MODELS:
        prop = elasticity_stats.get(model_name, {}).get("shear")
        if not prop:
            results[model_name] = None
            continue
        results[model_name] = mae(prop["ref"], prop["pred"])
    return results


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_bulk_density.json",
    title="Bulk modulus density",
    x_label="Reference bulk modulus / GPa",
    y_label="Predicted bulk modulus / GPa",
)
def bulk_density(elasticity_stats: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """
    Density scatter inputs for bulk modulus.

    Parameters
    ----------
    elasticity_stats
        Aggregated bulk/shear data per model.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    return _density_inputs("bulk", elasticity_stats)


@pytest.fixture
@plot_density_scatter(
    filename=OUT_PATH / "figure_shear_density.json",
    title="Shear modulus density",
    x_label="Reference shear modulus / GPa",
    y_label="Predicted shear modulus / GPa",
)
def shear_density(elasticity_stats: dict[str, dict[str, Any]]) -> dict[str, dict]:
    """
    Density scatter inputs for shear modulus.

    Parameters
    ----------
    elasticity_stats
        Aggregated bulk/shear data per model.

    Returns
    -------
    dict[str, dict]
        Mapping of model name to density-scatter data.
    """
    return _density_inputs("shear", elasticity_stats)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "elasticity_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    bulk_mae: dict[str, float | None],
    shear_mae: dict[str, float | None],
) -> dict[str, dict]:
    """
    All elasticity metrics.

    Parameters
    ----------
    bulk_mae
        Bulk modulus MAE per model.
    shear_mae
        Shear modulus MAE per model.

    Returns
    -------
    dict[str, dict]
        Mapping of metric name to model-value dictionaries.
    """
    return {
        "Bulk modulus MAE": bulk_mae,
        "Shear modulus MAE": shear_mae,
    }


def test_elasticity(
    metrics: dict[str, dict],
    bulk_density: dict[str, dict],
    shear_density: dict[str, dict],
) -> None:
    """
    Run elasticity analysis.

    Parameters
    ----------
    metrics
        Benchmark metric values.
    bulk_density
        Density scatter inputs for bulk modulus.
    shear_density
        Density scatter inputs for shear modulus.
    """
    return
