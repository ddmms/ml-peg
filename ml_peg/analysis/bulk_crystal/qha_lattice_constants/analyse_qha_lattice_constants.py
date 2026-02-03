"""Analyse QHA lattice constants benchmark."""

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
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "qha_lattice_constants" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "qha_lattice_constants"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

HOVERDATA = {
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
    return df.sort_values(["material", "temperature_K", "pressure_GPa"]).reset_index(
        drop=True
    )


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_qha_lattice_constants.json",
    title="QHA lattice constants",
    x_label="Predicted lattice constant / Å",
    y_label="Reference lattice constant / Å",
    hoverdata=HOVERDATA,
)
def qha_lattice_constants() -> dict[str, list]:
    """
    Gather reference and predicted lattice constants for parity plotting.

    Returns
    -------
    dict[str, list]
        Dictionary of reference and predicted values per model.
    """
    results: dict[str, list] = {"ref": []} | {mlip: [] for mlip in MODELS}
    ref_stored = False

    for model_name in MODELS:
        df = _load_results(model_name)
        if df.empty:
            continue

        results[model_name] = df["pred_lattice_a"].tolist()

        if not ref_stored:
            results["ref"] = df["ref_lattice_a"].tolist()
            HOVERDATA["Material"] = df["material"].tolist()
            HOVERDATA["Temperature (K)"] = df["temperature_K"].tolist()
            HOVERDATA["Pressure (GPa)"] = df["pressure_GPa"].tolist()
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
        if not ref or not pred:
            results[model_name] = None
        else:
            results[model_name] = mae(ref, pred)
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "qha_lattice_constants_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(lattice_constant_mae: dict[str, float | None]) -> dict[str, dict]:
    """
    Build metrics dictionary for QHA lattice constants.

    Parameters
    ----------
    lattice_constant_mae
        MAE per model.

    Returns
    -------
    dict[str, dict]
        Metrics dictionary.
    """
    return {"Lattice constant MAE (QHA)": lattice_constant_mae}


def test_qha_lattice_constants(
    metrics: dict[str, dict],
    qha_lattice_constants: dict[str, list],
) -> None:
    """
    Run QHA lattice constants analysis test.

    Parameters
    ----------
    metrics
        Metrics dictionary for QHA lattice constants.
    qha_lattice_constants
        Parity plot data.
    """
    return
