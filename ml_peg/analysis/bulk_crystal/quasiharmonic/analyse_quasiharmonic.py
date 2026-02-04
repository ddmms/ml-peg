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
HOVERDATA_LATTICE = _make_hoverdata()
HOVERDATA_VOLUME = _make_hoverdata()
HOVERDATA_THERMAL_EXPANSION = _make_hoverdata()
HOVERDATA_BULK_MODULUS = _make_hoverdata()
HOVERDATA_HEAT_CAPACITY = _make_hoverdata()


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
    filename=OUT_PATH / "figure_qha_lattice_constants.json",
    title="QHA Lattice Constants",
    x_label="Predicted lattice constant / \u00c5",
    y_label="Reference lattice constant / \u00c5",
    hoverdata=HOVERDATA_LATTICE,
)
def qha_lattice_constants() -> dict[str, list]:
    """
    Gather reference and predicted lattice constants for parity plotting.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted values per model.
    """
    return _gather_parity_data("ref_lattice_a", "pred_lattice_a", HOVERDATA_LATTICE)


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
@plot_parity(
    filename=OUT_PATH / "figure_qha_thermal_expansion.json",
    title="QHA Thermal Expansion",
    x_label="Predicted thermal expansion / 10\u207b\u2076 K\u207b\u00b9",
    y_label="Reference thermal expansion / 10\u207b\u2076 K\u207b\u00b9",
    hoverdata=HOVERDATA_THERMAL_EXPANSION,
)
def qha_thermal_expansion() -> dict[str, list]:
    """
    Gather reference and predicted thermal expansion coefficients.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted values per model.
    """
    return _gather_parity_data(
        "ref_thermal_expansion_1e6_K",
        "pred_thermal_expansion_1e6_K",
        HOVERDATA_THERMAL_EXPANSION,
    )


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_qha_bulk_modulus.json",
    title="QHA Bulk Modulus",
    x_label="Predicted bulk modulus / GPa",
    y_label="Reference bulk modulus / GPa",
    hoverdata=HOVERDATA_BULK_MODULUS,
)
def qha_bulk_modulus() -> dict[str, list]:
    """
    Gather reference and predicted bulk modulus values.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted values per model.
    """
    return _gather_parity_data(
        "ref_bulk_modulus_GPa", "pred_bulk_modulus_GPa", HOVERDATA_BULK_MODULUS
    )


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_qha_heat_capacity.json",
    title="QHA Heat Capacity",
    x_label="Predicted heat capacity / J/(mol·K)",
    y_label="Reference heat capacity / J/(mol·K)",
    hoverdata=HOVERDATA_HEAT_CAPACITY,
)
def qha_heat_capacity() -> dict[str, list]:
    """
    Gather reference and predicted heat capacity values.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted values per model.
    """
    return _gather_parity_data(
        "ref_heat_capacity_J_mol_K",
        "pred_heat_capacity_J_mol_K",
        HOVERDATA_HEAT_CAPACITY,
    )


@pytest.fixture
def lattice_constant_mae(
    qha_lattice_constants: dict[str, list],
) -> dict[str, float | None]:
    """
    Mean absolute error for lattice constants.

    Parameters
    ----------
    qha_lattice_constants
        Reference and predicted lattice constants.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model.
    """
    return _calculate_mae(qha_lattice_constants)


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
def thermal_expansion_mae(
    qha_thermal_expansion: dict[str, list],
) -> dict[str, float | None]:
    """
    Mean absolute error for thermal expansion coefficient.

    Parameters
    ----------
    qha_thermal_expansion
        Reference and predicted thermal expansion.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model.
    """
    return _calculate_mae(qha_thermal_expansion)


@pytest.fixture
def bulk_modulus_mae(qha_bulk_modulus: dict[str, list]) -> dict[str, float | None]:
    """
    Mean absolute error for bulk modulus.

    Parameters
    ----------
    qha_bulk_modulus
        Reference and predicted bulk modulus.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model.
    """
    return _calculate_mae(qha_bulk_modulus)


@pytest.fixture
def heat_capacity_mae(qha_heat_capacity: dict[str, list]) -> dict[str, float | None]:
    """
    Mean absolute error for heat capacity at constant pressure.

    Parameters
    ----------
    qha_heat_capacity
        Reference and predicted heat capacity.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model.
    """
    return _calculate_mae(qha_heat_capacity)


@pytest.fixture
@build_table(
    filename=OUT_PATH / "quasiharmonic_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    lattice_constant_mae: dict[str, float | None],
    volume_per_atom_mae: dict[str, float | None],
    thermal_expansion_mae: dict[str, float | None],
    bulk_modulus_mae: dict[str, float | None],
    heat_capacity_mae: dict[str, float | None],
) -> dict[str, dict[str, float | None]]:
    """
    Build metrics dictionary for quasiharmonic benchmark.

    Each metric is computed over (structure, temperature, pressure) data points.

    Parameters
    ----------
    lattice_constant_mae
        Lattice constant MAE per model.
    volume_per_atom_mae
        Volume per atom MAE per model.
    thermal_expansion_mae
        Thermal expansion MAE per model.
    bulk_modulus_mae
        Bulk modulus MAE per model.
    heat_capacity_mae
        Heat capacity MAE per model.

    Returns
    -------
    dict[str, dict[str, float | None]]
        Metrics dictionary.
    """
    return {
        "Lattice constant MAE": lattice_constant_mae,
        "Volume per atom MAE": volume_per_atom_mae,
        "Thermal expansion MAE": thermal_expansion_mae,
        "Bulk modulus MAE": bulk_modulus_mae,
        "Heat capacity MAE": heat_capacity_mae,
    }


def test_quasiharmonic(
    metrics: dict[str, dict[str, Any]],
    qha_lattice_constants: dict[str, list],
    qha_volume_per_atom: dict[str, list],
    qha_thermal_expansion: dict[str, list],
    qha_bulk_modulus: dict[str, list],
    qha_heat_capacity: dict[str, list],
) -> None:
    """
    Run quasiharmonic analysis test.

    Parameters
    ----------
    metrics
        Metrics dictionary.
    qha_lattice_constants
        Lattice constants parity data.
    qha_volume_per_atom
        Volume per atom parity data.
    qha_thermal_expansion
        Thermal expansion parity data.
    qha_bulk_modulus
        Bulk modulus parity data.
    qha_heat_capacity
        Heat capacity parity data.
    """
    return
