"""Analyse quasiharmonic benchmark results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "quasiharmonic" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "quasiharmonic"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def _make_hoverdata() -> dict[str, list]:
    """
    Create empty hoverdata dictionary for QHA plots.

    Returns
    -------
    dict[str, list]
        Empty hoverdata with Material, Temperature, and Pressure keys.
    """
    return {
        "Material": [],
        "Temperature (K)": [],
        "Pressure (GPa)": [],
    }


# Hover data for interactive plots (populated during fixture execution)
HOVERDATA_VOLUME = _make_hoverdata()


def _load_results(model_name: str) -> pd.DataFrame:
    """
    Load results for a given model.

    Parameters
    ----------
    model_name
        Name of the model.

    Returns
    -------
    pd.DataFrame
        Results dataframe or empty dataframe if missing.
    """
    path = CALC_PATH / model_name / "results.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    # Filter to only successful calculations
    if "status" in df.columns:
        df = df[df["status"] == "success"]
    return df.sort_values(["material", "temperature_K", "pressure_GPa"]).reset_index(
        drop=True
    )


def _gather_parity_data(
    ref_col: str,
    pred_col: str,
    hoverdata: dict[str, list],
) -> dict[str, list]:
    """
    Gather reference and predicted values for parity plotting.

    Parameters
    ----------
    ref_col
        Column name for reference values.
    pred_col
        Column name for predicted values.
    hoverdata
        Hoverdata dictionary to populate (mutated in place).

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted values per model.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        df = _load_results(model_name)
        if df.empty:
            continue

        if ref_col not in df.columns or pred_col not in df.columns:
            continue

        valid_df = df.dropna(subset=[pred_col, ref_col])
        results[model_name] = valid_df[pred_col].tolist()

        if not ref_stored and not valid_df.empty:
            results["ref"] = valid_df[ref_col].tolist()
            hoverdata["Material"] = valid_df["material"].tolist()
            hoverdata["Temperature (K)"] = valid_df["temperature_K"].tolist()
            hoverdata["Pressure (GPa)"] = valid_df["pressure_GPa"].tolist()
            ref_stored = True

    return results


def _calculate_mae(parity_data: dict[str, list]) -> dict[str, float | None]:
    """
    Calculate MAE for each model from parity data.

    Parameters
    ----------
    parity_data
        Dictionary with 'ref' key and model name keys containing lists.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model, None if data is missing or mismatched.
    """
    results: dict[str, float | None] = {}
    ref = parity_data.get("ref", [])
    for model_name in MODELS:
        pred = parity_data.get(model_name, [])
        if not ref or not pred or len(ref) != len(pred):
            results[model_name] = None
        else:
            results[model_name] = mae(ref, pred)
    return results


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_qha_volume_per_atom.json",
    title="QHA Volume per Atom",
    x_label="Predicted volume per atom / Å³/atom",
    y_label="Reference volume per atom / Å³/atom",
    hoverdata=HOVERDATA_VOLUME,
)
def qha_volume_per_atom() -> dict[str, list]:
    """
    Gather reference and predicted equilibrium volume per atom.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted values per model.
    """
    return _gather_parity_data(
        "ref_volume_per_atom", "pred_volume_per_atom", HOVERDATA_VOLUME
    )


@pytest.fixture
def volume_per_atom_mae(
    qha_volume_per_atom: dict[str, list],
) -> dict[str, float | None]:
    """
    Mean absolute error for equilibrium volume per atom.

    Parameters
    ----------
    qha_volume_per_atom
        Reference and predicted volumes per atom.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model.
    """
    return _calculate_mae(qha_volume_per_atom)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "quasiharmonic_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    volume_per_atom_mae: dict[str, float | None],
) -> dict[str, dict[str, float | None]]:
    """
    Build metrics dictionary for quasiharmonic benchmark.

    Each metric is computed over (structure, temperature, pressure) data points.

    Parameters
    ----------
    volume_per_atom_mae
        Volume per atom MAE per model.

    Returns
    -------
    dict[str, dict[str, float | None]]
        Metrics dictionary.
    """
    return {
        "Volume per atom MAE": volume_per_atom_mae,
    }


def test_quasiharmonic(
    metrics: dict[str, dict[str, Any]],
    qha_volume_per_atom: dict[str, list],
) -> None:
    """
    Run quasiharmonic analysis test.

    Parameters
    ----------
    metrics
        Metrics dictionary.
    qha_volume_per_atom
        Volume per atom parity data.
    """
    return
