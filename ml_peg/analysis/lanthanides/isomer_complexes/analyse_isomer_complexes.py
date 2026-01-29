"""Analyse lanthanide isomer complex benchmark."""

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
CALC_PATH = CALCS_ROOT / "lanthanides" / "isomer_complexes" / "outputs"
OUT_PATH = APP_ROOT / "data" / "lanthanides" / "isomer_complexes"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# r2SCAN-3c references (kcal/mol) from Table S4 (lanthanides only)
R2SCAN_REF: dict[str, dict[str, float]] = {
    "Lu_ff6372": {"iso1": 2.15, "iso2": 12.96, "iso3": 0.00, "iso4": 2.08},
    "Ce_ff6372": {"iso1": 2.47, "iso2": 7.13, "iso3": 0.00, "iso4": 2.17},
    "Ce_1d271a": {"iso1": 0.00, "iso2": 2.20},
    "Sm_ed79e8": {"iso1": 2.99, "iso2": 0.00},
    "La_f1a50d": {"iso1": 0.00, "iso2": 3.11},
    "Eu_ff6372": {"iso1": 0.00, "iso2": 6.74},
    "Nd_c5f44a": {"iso1": 0.00, "iso2": 1.61},
}


def _load_isomer_dataframe() -> pd.DataFrame:
    """
    Load isomer energies from per-model outputs.

    Returns
    -------
    pandas.DataFrame
        Loaded dataframe, or an empty dataframe if no data are found.
    """
    combined_path = CALC_PATH / "isomer_energies.csv"
    if combined_path.exists():
        return pd.read_csv(combined_path)

    csv_paths = sorted(CALC_PATH.glob("*/isomer_energies.csv"))
    frames = [pd.read_csv(path) for path in csv_paths]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _build_reference_df() -> pd.DataFrame:
    """
    Build a reference dataframe from the r2SCAN-3c table.

    Returns
    -------
    pandas.DataFrame
        Dataframe with columns: system, isomer, ref.
    """
    records = []
    for system, iso_map in R2SCAN_REF.items():
        for iso, ref in iso_map.items():
            records.append({"system": system, "isomer": iso, "ref": ref})
    return pd.DataFrame.from_records(records)


REFERENCE_DF = (
    _build_reference_df().sort_values(["system", "isomer"]).reset_index(drop=True)
)
REFERENCE_INDEX = pd.MultiIndex.from_frame(REFERENCE_DF[["system", "isomer"]])
REFERENCE_HOVERDATA = {
    "System": REFERENCE_DF["system"].tolist(),
    "Isomer": REFERENCE_DF["isomer"].tolist(),
}


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_isomer_complexes.json",
    title="Lanthanide isomer relative energies",
    x_label="Model Delta E (kcal/mol)",
    y_label="r2SCAN-3c Delta E (kcal/mol)",
    hoverdata=REFERENCE_HOVERDATA,
)
def isomer_relative_energies() -> dict[str, list]:
    """
    Build parity data for lanthanide isomer complexes benchmark.

    Returns
    -------
    dict[str, list]
        Reference and per-model relative energies.
    """
    df = _load_isomer_dataframe()
    df = (
        df.merge(REFERENCE_DF, on=["system", "isomer"], how="inner")
        if not df.empty
        else df
    )

    prediction_table = pd.DataFrame(index=REFERENCE_INDEX)
    if not df.empty:
        prediction_table = df.pivot_table(
            index=["system", "isomer"],
            columns="model",
            values="rel_energy_kcal",
            aggfunc="first",
        ).reindex(REFERENCE_INDEX)

    results: dict[str, list] = {"ref": REFERENCE_DF["ref"].tolist()}
    for model in MODELS:
        if model in prediction_table.columns:
            series = prediction_table[model]
            results[model] = series.where(series.notna(), None).tolist()
        else:
            results[model] = [None] * len(results["ref"])

    return results


@pytest.fixture
def isomer_complex_outputs(
    isomer_relative_energies: dict[str, list],
) -> dict[str, float | None]:
    """
    Build outputs for lanthanide isomer complexes benchmark.

    Parameters
    ----------
    isomer_relative_energies
        Reference and per-model relative energies.

    Returns
    -------
    dict[str, float | None]
        Mean absolute errors by model.
    """
    ref_vals = isomer_relative_energies["ref"]
    mae_by_model: dict[str, float | None] = {}
    for model in MODELS:
        preds = isomer_relative_energies[model]
        pairs = [
            (ref, pred)
            for ref, pred in zip(ref_vals, preds, strict=True)
            if pred is not None
        ]
        if not pairs:
            mae_by_model[model] = None
            continue
        ref, pred = zip(*pairs, strict=True)
        mae_by_model[model] = mae(list(ref), list(pred))
    return mae_by_model


@pytest.fixture
@build_table(
    filename=OUT_PATH / "isomer_complexes_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(isomer_complex_outputs: dict[str, float | None]) -> dict[str, dict]:
    """
    Collect metrics for lanthanide isomer complexes.

    Parameters
    ----------
    isomer_complex_outputs
        Mean absolute errors for all models.

    Returns
    -------
    dict[str, dict]
        Metrics keyed by name for all models.
    """
    return {"MAE": isomer_complex_outputs}


def test_isomer_complexes(metrics: dict[str, dict]) -> None:
    """
    Run lanthanide isomer complexes benchmark analysis.

    Parameters
    ----------
    metrics
        All lanthanide isomer complex metrics.
    """
    return
