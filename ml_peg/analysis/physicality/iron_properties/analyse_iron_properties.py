"""
Analyse BCC iron properties benchmark.

This analysis combines EOS, elastic, Bain path, defect, surface, and stacking fault
properties.

Reference
----------
Zhang, L., Csányi, G., van der Giessen, E., & Maresca, F. (2023).
Efficiency, Accuracy, and Transferability of Machine Learning Potentials:
Application to Dislocations and Cracks in Iron.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "iron_properties" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "iron_properties"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, _ = load_metrics_config(METRICS_CONFIG_PATH)

# DFT reference values
DFT_REFERENCE = {
    # EOS properties
    "a0": 2.831,  # Lattice parameter (Å)
    "B0": 178.0,  # Bulk modulus (GPa)
    "E_bcc_fcc": 83.5,  # BCC-FCC energy difference (meV/atom)
    # Defect properties
    "E_vac": 2.02,  # Vacancy formation energy (eV)
    "gamma_100": 2.41,  # Surface energy (J/m²)
    "gamma_110": 2.37,
    "gamma_111": 2.58,
    "gamma_112": 2.48,
    "gamma_us_110": 0.75,  # Unstable SFE (J/m²)
    "gamma_us_112": 1.12,
    # Traction-separation properties
    "max_traction_100": 35.0,  # Max traction for (100) cleavage (GPa)
    "max_traction_110": 30.0,  # Max traction for (110) cleavage (GPa)
}


# Curve file mapping for CSV loading
CURVE_FILES = {
    "eos": "eos_curve.csv",
    "bain": "bain_path.csv",
    "sfe_110": "sfe_110_curve.csv",
    "sfe_112": "sfe_112_curve.csv",
    "ts_100": "ts_100_curve.csv",
    "ts_110": "ts_110_curve.csv",
}


def load_model_results(model_name: str) -> dict[str, Any] | None:
    """
    Load iron properties results for a model.

    Parameters
    ----------
    model_name : str
        Name of the model to load results for.

    Returns
    -------
    dict[str, Any] | None
        Dictionary of results, or None if file does not exist.
    """
    json_path = CALC_PATH / model_name / "results.json"
    if not json_path.exists():
        return None
    return json.loads(json_path.read_text())


def load_curve(model_name: str, curve_type: str) -> pd.DataFrame:
    """
    Load curve data for a model.

    Parameters
    ----------
    model_name : str
        Name of the model to load curve for.
    curve_type : str
        Type of curve to load (e.g., 'eos', 'bain', 'sfe_110').

    Returns
    -------
    pd.DataFrame
        Curve data, or empty DataFrame if file does not exist.
    """
    filename = CURVE_FILES.get(curve_type)
    if not filename:
        return pd.DataFrame()
    csv_path = CALC_PATH / model_name / filename
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def compute_metrics(results: dict[str, Any]) -> dict[str, float]:
    """
    Compute metrics from model results.

    Parameters
    ----------
    results : dict[str, Any]
        Dictionary containing model calculation results.

    Returns
    -------
    dict[str, float]
        Dictionary mapping metric names to computed values.
    """
    metrics: dict[str, float] = {}

    # ==========================================================================
    # EOS metrics
    # ==========================================================================
    eos = results.get("eos", {})
    if "a0" in eos:
        a0_mlip = eos["a0"]
        a0_error = abs(a0_mlip - DFT_REFERENCE["a0"]) / DFT_REFERENCE["a0"] * 100
        metrics["a0 error (%)"] = a0_error

    if "B0" in eos:
        B0_mlip = eos["B0"]  # noqa: N806
        B0_error = abs(B0_mlip - DFT_REFERENCE["B0"]) / DFT_REFERENCE["B0"] * 100  # noqa: N806
        metrics["B0 error (%)"] = B0_error

    # ==========================================================================
    # Bain path metrics
    # ==========================================================================
    bain = results.get("bain_path", {})
    if "delta_E_meV" in bain:
        E_bcc_fcc_mlip = bain["delta_E_meV"]  # noqa: N806
        E_bcc_fcc_error = abs(E_bcc_fcc_mlip - DFT_REFERENCE["E_bcc_fcc"])  # noqa: N806
        metrics["BCC-FCC ΔE error (meV)"] = E_bcc_fcc_error

    # ==========================================================================
    # Elastic constants metrics
    # ==========================================================================
    elastic = results.get("elastic", {})
    if "C11" in elastic:
        C11_error = (  # noqa: N806
            abs(elastic["C11"] - DFT_REFERENCE["C11"]) / DFT_REFERENCE["C11"] * 100
        )
        metrics["C11 error (%)"] = C11_error
    if "C12" in elastic:
        C12_error = (  # noqa: N806
            abs(elastic["C12"] - DFT_REFERENCE["C12"]) / DFT_REFERENCE["C12"] * 100
        )
        metrics["C12 error (%)"] = C12_error
    if "C44" in elastic:
        C44_error = (  # noqa: N806
            abs(elastic["C44"] - DFT_REFERENCE["C44"]) / DFT_REFERENCE["C44"] * 100
        )
        metrics["C44 error (%)"] = C44_error

    # ==========================================================================
    # Vacancy metrics
    # ==========================================================================
    vacancy = results.get("vacancy", {})
    if "E_vac" in vacancy:
        E_vac_mlip = vacancy["E_vac"]  # noqa: N806
        E_vac_error = (  # noqa: N806
            abs(E_vac_mlip - DFT_REFERENCE["E_vac"]) / DFT_REFERENCE["E_vac"] * 100
        )
        metrics["E_vac error (%)"] = E_vac_error

    # ==========================================================================
    # Surface energy metrics
    # ==========================================================================
    surfaces = results.get("surfaces", {})
    surface_errors = []

    for surface in ["100", "110", "111", "112"]:
        key_mlip = f"gamma_{surface}"
        if key_mlip in surfaces:
            gamma_mlip = surfaces[key_mlip]
            gamma_dft = DFT_REFERENCE[key_mlip]
            error = abs(gamma_mlip - gamma_dft)
            surface_errors.append(error)

    if surface_errors:
        metrics["Surface MAE (J/m²)"] = np.mean(surface_errors)

    # ==========================================================================
    # Stacking fault metrics
    # ==========================================================================
    sfe_110 = results.get("sfe_110", {})
    if "max_sfe" in sfe_110:
        max_sfe_110_mlip = sfe_110["max_sfe"]
        max_sfe_110_error = (
            abs(max_sfe_110_mlip - DFT_REFERENCE["gamma_us_110"])
            / DFT_REFERENCE["gamma_us_110"]
            * 100
        )
        metrics["Max SFE 110 error (%)"] = max_sfe_110_error

    sfe_112 = results.get("sfe_112", {})
    if "max_sfe" in sfe_112:
        max_sfe_112_mlip = sfe_112["max_sfe"]
        max_sfe_112_error = (
            abs(max_sfe_112_mlip - DFT_REFERENCE["gamma_us_112"])
            / DFT_REFERENCE["gamma_us_112"]
            * 100
        )
        metrics["Max SFE 112 error (%)"] = max_sfe_112_error

    # ==========================================================================
    # Traction-separation metrics
    # ==========================================================================
    ts_100 = results.get("ts_100", {})
    if "max_traction" in ts_100:
        traction_100_error = (
            abs(ts_100["max_traction"] - DFT_REFERENCE["max_traction_100"])
            / DFT_REFERENCE["max_traction_100"]
            * 100
        )
        metrics["Max traction (100) error (%)"] = traction_100_error

    ts_110 = results.get("ts_110", {})
    if "max_traction" in ts_110:
        traction_110_error = (
            abs(ts_110["max_traction"] - DFT_REFERENCE["max_traction_110"])
            / DFT_REFERENCE["max_traction_110"]
            * 100
        )
        metrics["Max traction (110) error (%)"] = traction_110_error

    return metrics


def _load_all_results() -> dict[str, dict[str, Any]]:
    """
    Load results for all models.

    Returns
    -------
    dict[str, dict[str, Any]]
        Dictionary mapping model names to their results.
    """
    all_results: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        results = load_model_results(model_name)
        if results is not None:
            all_results[model_name] = results
    return all_results


def _load_curves_for_all_models(curve_type: str) -> dict[str, pd.DataFrame]:
    """
    Load curves of given type for all models.

    Parameters
    ----------
    curve_type : str
        Type of curve to load (e.g., 'eos', 'bain', 'sfe_110').

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping model names to their curve DataFrames.
    """
    curves: dict[str, pd.DataFrame] = {}
    for model_name in MODELS:
        curve = load_curve(model_name, curve_type)
        if not curve.empty:
            curves[model_name] = curve
    return curves


@pytest.fixture
def iron_eos_curves() -> dict[str, pd.DataFrame]:
    """
    Load EOS curves for all models.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping model names to their EOS curve DataFrames.
    """
    return _load_curves_for_all_models("eos")


@pytest.fixture
def iron_bain_curves() -> dict[str, pd.DataFrame]:
    """
    Load Bain path curves for all models.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping model names to their Bain path curve DataFrames.
    """
    return _load_curves_for_all_models("bain")


@pytest.fixture
def iron_sfe_110_curves() -> dict[str, pd.DataFrame]:
    """
    Load SFE 110 curves for all models.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping model names to their SFE 110 curve DataFrames.
    """
    return _load_curves_for_all_models("sfe_110")


@pytest.fixture
def iron_sfe_112_curves() -> dict[str, pd.DataFrame]:
    """
    Load SFE 112 curves for all models.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping model names to their SFE 112 curve DataFrames.
    """
    return _load_curves_for_all_models("sfe_112")


@pytest.fixture
def iron_ts_100_curves() -> dict[str, pd.DataFrame]:
    """
    Load T-S (100) curves for all models.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping model names to their T-S (100) curve DataFrames.
    """
    return _load_curves_for_all_models("ts_100")


@pytest.fixture
def iron_ts_110_curves() -> dict[str, pd.DataFrame]:
    """
    Load T-S (110) curves for all models.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping model names to their T-S (110) curve DataFrames.
    """
    return _load_curves_for_all_models("ts_110")


def collect_metrics() -> pd.DataFrame:
    """
    Gather metrics for all models.

    Returns
    -------
    pd.DataFrame
        DataFrame containing metrics for all models.
    """
    metrics_rows: list[dict[str, float | str]] = []

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    all_results = _load_all_results()

    for model_name, results in all_results.items():
        model_metrics = compute_metrics(results)
        row = {"Model": model_name} | model_metrics
        metrics_rows.append(row)

    columns = ["Model"] + list(DEFAULT_THRESHOLDS.keys())

    return pd.DataFrame(metrics_rows).reindex(columns=columns)


@pytest.fixture
def iron_properties_collection() -> pd.DataFrame:
    """
    Collect iron properties metrics across all models.

    Returns
    -------
    pd.DataFrame
        DataFrame containing iron properties metrics for all models.
    """
    return collect_metrics()


@pytest.fixture
def iron_properties_metrics_dataframe(
    iron_properties_collection: pd.DataFrame,
) -> pd.DataFrame:
    """
    Provide the aggregated iron properties metrics dataframe.

    Parameters
    ----------
    iron_properties_collection : pd.DataFrame
        Collection of iron properties metrics.

    Returns
    -------
    pd.DataFrame
        The aggregated iron properties metrics DataFrame.
    """
    return iron_properties_collection


@pytest.fixture
@build_table(
    filename=OUT_PATH / "iron_properties_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=None,
)
def metrics(
    iron_properties_metrics_dataframe: pd.DataFrame,
) -> dict[str, dict]:
    """
    Compute iron properties metrics for all models.

    Parameters
    ----------
    iron_properties_metrics_dataframe
        Aggregated per-model metrics.

    Returns
    -------
    dict[str, dict]
        Mapping of metric names to per-model results.
    """
    metrics_df = iron_properties_metrics_dataframe
    metrics_dict: dict[str, dict[str, float | None]] = {}
    for column in metrics_df.columns:
        if column == "Model":
            continue
        values = [
            value if pd.notna(value) else None for value in metrics_df[column].tolist()
        ]
        metrics_dict[column] = dict(zip(metrics_df["Model"], values, strict=False))
    return metrics_dict


def test_iron_properties(metrics: dict[str, dict]) -> None:
    """
    Run iron properties analysis.

    Parameters
    ----------
    metrics : dict[str, dict]
        Dictionary of iron properties metrics from the metrics fixture.
    """
    return
