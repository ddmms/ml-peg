"""Analyse quasiharmonic benchmark results."""

from __future__ import annotations

from pathlib import Path

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

# Hover data for interactive plots
HOVERDATA_LATTICE = {
    "Material": [],
    "Temperature (K)": [],
    "Pressure (GPa)": [],
}

HOVERDATA_VOLUME = {
    "Material": [],
    "Temperature (K)": [],
    "Pressure (GPa)": [],
}

HOVERDATA_THERMAL_EXPANSION = {
    "Material": [],
    "Temperature (K)": [],
    "Pressure (GPa)": [],
}

HOVERDATA_BULK_MODULUS = {
    "Material": [],
    "Temperature (K)": [],
    "Pressure (GPa)": [],
}


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
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        df = _load_results(model_name)
        if df.empty:
            continue

        if "pred_lattice_a" not in df.columns:
            continue

        # Filter valid predictions
        valid_df = df.dropna(subset=["pred_lattice_a", "ref_lattice_a"])

        results[model_name] = valid_df["pred_lattice_a"].tolist()

        if not ref_stored and not valid_df.empty:
            results["ref"] = valid_df["ref_lattice_a"].tolist()
            HOVERDATA_LATTICE["Material"] = valid_df["material"].tolist()
            HOVERDATA_LATTICE["Temperature (K)"] = valid_df["temperature_K"].tolist()
            HOVERDATA_LATTICE["Pressure (GPa)"] = valid_df["pressure_GPa"].tolist()
            ref_stored = True

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
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        df = _load_results(model_name)
        if df.empty:
            continue

        ref_col = "ref_volume_per_atom"
        pred_col = "pred_volume_per_atom"

        if ref_col not in df.columns or pred_col not in df.columns:
            continue

        valid_df = df.dropna(subset=[pred_col, ref_col])
        results[model_name] = valid_df[pred_col].tolist()

        if not ref_stored and not valid_df.empty:
            results["ref"] = valid_df[ref_col].tolist()
            HOVERDATA_VOLUME["Material"] = valid_df["material"].tolist()
            HOVERDATA_VOLUME["Temperature (K)"] = valid_df["temperature_K"].tolist()
            HOVERDATA_VOLUME["Pressure (GPa)"] = valid_df["pressure_GPa"].tolist()
            ref_stored = True

    return results


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
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        df = _load_results(model_name)
        if df.empty:
            continue

        ref_col = "ref_thermal_expansion_1e6_K"
        pred_col = "pred_thermal_expansion_1e6_K"

        if ref_col not in df.columns or pred_col not in df.columns:
            continue

        valid_df = df.dropna(subset=[ref_col, pred_col])
        results[model_name] = valid_df[pred_col].tolist()

        if not ref_stored and not valid_df.empty:
            results["ref"] = valid_df[ref_col].tolist()
            HOVERDATA_THERMAL_EXPANSION["Material"] = valid_df["material"].tolist()
            HOVERDATA_THERMAL_EXPANSION["Temperature (K)"] = valid_df[
                "temperature_K"
            ].tolist()
            HOVERDATA_THERMAL_EXPANSION["Pressure (GPa)"] = valid_df[
                "pressure_GPa"
            ].tolist()
            ref_stored = True

    return results


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
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    results: dict[str, list] = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        df = _load_results(model_name)
        if df.empty:
            continue

        ref_col = "ref_bulk_modulus_GPa"
        pred_col = "pred_bulk_modulus_GPa"

        if ref_col not in df.columns or pred_col not in df.columns:
            continue

        valid_df = df.dropna(subset=[ref_col, pred_col])
        results[model_name] = valid_df[pred_col].tolist()

        if not ref_stored and not valid_df.empty:
            results["ref"] = valid_df[ref_col].tolist()
            HOVERDATA_BULK_MODULUS["Material"] = valid_df["material"].tolist()
            HOVERDATA_BULK_MODULUS["Temperature (K)"] = valid_df[
                "temperature_K"
            ].tolist()
            HOVERDATA_BULK_MODULUS["Pressure (GPa)"] = valid_df["pressure_GPa"].tolist()
            ref_stored = True

    return results


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
    results: dict[str, float | None] = {}
    ref = qha_lattice_constants.get("ref", [])
    for model_name in MODELS:
        pred = qha_lattice_constants.get(model_name, [])
        if not ref or not pred or len(ref) != len(pred):
            results[model_name] = None
        else:
            results[model_name] = mae(ref, pred)
    return results


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
    results: dict[str, float | None] = {}
    ref = qha_volume_per_atom.get("ref", [])
    for model_name in MODELS:
        pred = qha_volume_per_atom.get(model_name, [])
        if not ref or not pred or len(ref) != len(pred):
            results[model_name] = None
        else:
            results[model_name] = mae(ref, pred)
    return results


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
    results: dict[str, float | None] = {}
    ref = qha_thermal_expansion.get("ref", [])
    for model_name in MODELS:
        pred = qha_thermal_expansion.get(model_name, [])
        if not ref or not pred or len(ref) != len(pred):
            results[model_name] = None
        else:
            results[model_name] = mae(ref, pred)
    return results


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
    results: dict[str, float | None] = {}
    ref = qha_bulk_modulus.get("ref", [])
    for model_name in MODELS:
        pred = qha_bulk_modulus.get(model_name, [])
        if not ref or not pred or len(ref) != len(pred):
            results[model_name] = None
        else:
            results[model_name] = mae(ref, pred)
    return results


@pytest.fixture
def heat_capacity_mae() -> dict[str, float | None]:
    """
    Mean absolute error for heat capacity at constant pressure.

    Returns
    -------
    dict[str, float | None]
        MAE values for each model.
    """
    results: dict[str, float | None] = {}

    # Collect reference and predictions
    ref_values: list[float] = []
    pred_values: dict[str, list[float]] = {m: [] for m in MODELS}
    ref_collected = False

    for model_name in MODELS:
        df = _load_results(model_name)
        if df.empty:
            continue

        ref_col = "ref_heat_capacity_J_mol_K"
        pred_col = "pred_heat_capacity_J_mol_K"

        if ref_col not in df.columns or pred_col not in df.columns:
            continue

        valid_df = df.dropna(subset=[ref_col, pred_col])

        if not ref_collected and not valid_df.empty:
            ref_values = valid_df[ref_col].tolist()
            ref_collected = True

        pred_values[model_name] = valid_df[pred_col].tolist()

    for model_name in MODELS:
        pred = pred_values.get(model_name, [])
        if not ref_values or not pred or len(ref_values) != len(pred):
            results[model_name] = None
        else:
            results[model_name] = mae(ref_values, pred)

    return results


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
    metrics: dict[str, dict],
    qha_lattice_constants: dict[str, list],
    qha_volume_per_atom: dict[str, list],
    qha_thermal_expansion: dict[str, list],
    qha_bulk_modulus: dict[str, list],
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
    """
    return
