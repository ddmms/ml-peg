"""Analyse lanthanide isomer complex benchmark."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

from dash import dash_table
import pandas as pd
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.utils import calc_table_scores, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT

CSV_ENV_VAR = "ML_PEG_LANTHANIDE_CSV"
STRUCT_ENV_VAR = "ML_PEG_LANTHANIDE_STRUCTURES"

CALC_PATH = CALCS_ROOT / "lanthanides" / "isomer_complexes" / "outputs"
OUT_PATH = APP_ROOT / "data" / "lanthanides" / "isomer_complexes"
STRUCT_OUT_PATH = OUT_PATH / "structures"

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


def _resolve_csv_path() -> Path | None:
    """
    Resolve the source CSV path for isomer energies.

    Returns
    -------
    Path | None
        CSV path if found, otherwise ``None``.
    """
    env_path = os.environ.get(CSV_ENV_VAR)
    if env_path:
        return Path(env_path).expanduser()
    csv_path = CALC_PATH / "isomer_energies.csv"
    return csv_path if csv_path.exists() else None


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


def _copy_structures(struct_root: Path, reference_df: pd.DataFrame) -> dict[tuple, str]:
    """
    Copy reference structures into the app assets directory.

    Parameters
    ----------
    struct_root
        Root directory containing isomer structures.
    reference_df
        Dataframe of systems/isomers to copy.

    Returns
    -------
    dict[tuple, str]
        Mapping of (system, isomer) to asset path.
    """
    struct_map: dict[tuple, str] = {}
    for _, row in reference_df.iterrows():
        system = row["system"]
        iso = row["isomer"]
        src = struct_root / system / iso / "orca.xyz"
        if not src.exists():
            continue
        dest_dir = STRUCT_OUT_PATH / system
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{iso}.xyz"
        shutil.copyfile(src, dest)
        struct_map[(system, iso)] = (
            f"assets/lanthanides/isomer_complexes/structures/{system}/{iso}.xyz"
        )
    return struct_map


def _build_table(
    mae_by_model: dict[str, float | None],
    model_order: list[str],
) -> None:
    """
    Build the metrics table JSON for the app.

    Parameters
    ----------
    mae_by_model
        MAE values keyed by model name.
    model_order
        Ordered list of model names to include.
    """
    metrics_data = []
    for model in model_order:
        metrics_data.append(
            {"MLIP": model, "MAE": mae_by_model.get(model), "id": model}
        )

    metrics_data = calc_table_scores(
        metrics_data,
        thresholds=DEFAULT_THRESHOLDS,
        weights=DEFAULT_WEIGHTS,
    )

    metrics_columns = (
        {"name": "MLIP", "id": "MLIP"},
        {"name": "MAE", "id": "MAE"},
        {"name": "Score", "id": "Score"},
    )

    summary_tooltips = {
        "MLIP": "Model identifier, hover for configuration details.",
        "Score": "Weighted score across metrics, Higher is better (normalised 0 to 1).",
    }
    tooltip_header = DEFAULT_TOOLTIPS | summary_tooltips

    model_configs = {model: {} for model in model_order}
    model_levels = dict.fromkeys(model_order)
    metric_levels = {
        metric_name: DEFAULT_THRESHOLDS.get(metric_name, {}).get("level_of_theory")
        for metric_name in DEFAULT_THRESHOLDS
    }

    model_name_map = {model: model for model in model_order}

    table = dash_table.DataTable(
        metrics_data,
        list(metrics_columns),
        id="metrics",
        tooltip_header=tooltip_header,
    )

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH / "isomer_complexes_metrics_table.json", "w") as fp:
        json.dump(
            {
                "data": table.data,
                "columns": table.columns,
                "tooltip_header": tooltip_header,
                "thresholds": DEFAULT_THRESHOLDS,
                "weights": DEFAULT_WEIGHTS,
                "model_levels_of_theory": model_levels,
                "metric_levels_of_theory": metric_levels,
                "model_configs": model_configs,
                "model_name_map": model_name_map,
            },
            fp,
        )


@pytest.fixture
def isomer_complex_outputs() -> dict[str, float | None]:
    """
    Build outputs for lanthanide isomer complexes benchmark.

    Returns
    -------
    dict[str, float | None]
        Mean absolute errors by model.
    """
    csv_path = _resolve_csv_path()
    if csv_path is None:
        pytest.skip(
            "No lanthanide isomer CSV found. "
            "Set ML_PEG_LANTHANIDE_CSV or run calc to stage outputs."
        )

    df = pd.read_csv(csv_path)
    if df.empty:
        pytest.skip("Lanthanide isomer CSV is empty.")

    reference_df = _build_reference_df()
    df = df.merge(reference_df, on=["system", "isomer"], how="inner")
    if df.empty:
        pytest.skip("No overlap between CSV entries and r2SCAN-3c reference data.")

    struct_map: dict[tuple, str] = {}
    struct_root_env = os.environ.get(STRUCT_ENV_VAR)
    if struct_root_env:
        struct_root = Path(struct_root_env).expanduser()
        if struct_root.exists():
            struct_map = _copy_structures(struct_root, reference_df)

    models = sorted(df["model"].unique().tolist())

    mae_by_model: dict[str, float | None] = {}
    fig = go.Figure()

    for model in models:
        sub = df[df["model"] == model]
        if sub.empty:
            mae_by_model[model] = None
            continue

        mae_by_model[model] = mae(
            sub["ref"].tolist(),
            sub["rel_energy_kcal"].tolist(),
        )

        customdata = []
        for _, row in sub.iterrows():
            struct_path = struct_map.get((row["system"], row["isomer"]), "")
            customdata.append([struct_path, row["system"], row["isomer"]])

        fig.add_trace(
            go.Scatter(
                x=sub["ref"],
                y=sub["rel_energy_kcal"],
                mode="markers",
                name=model,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[1]}</b> %{customdata[2]}<br>"
                    "r2SCAN-3c: %{x:.2f} kcal/mol<br>"
                    "Model: %{y:.2f} kcal/mol"
                    "<extra></extra>"
                ),
            )
        )

    min_val = min(df["ref"].min(), df["rel_energy_kcal"].min())
    max_val = max(df["ref"].max(), df["rel_energy_kcal"].max())
    pad = 0.5
    min_val -= pad
    max_val += pad

    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            showlegend=False,
            line={"color": "#7f7f7f", "dash": "dash"},
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Lanthanide isomer relative energies",
        xaxis_title="r2SCAN-3c Delta E (kcal/mol)",
        yaxis_title="Model Delta E (kcal/mol)",
        plot_bgcolor="#ffffff",
    )

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    fig.write_json(OUT_PATH / "figure_isomer_complexes.json")
    _build_table(mae_by_model, models)

    return mae_by_model


def test_isomer_complexes(isomer_complex_outputs: dict[str, float | None]) -> None:
    """
    Run lanthanide isomer complexes benchmark analysis.

    Parameters
    ----------
    isomer_complex_outputs
        Mean absolute errors for all models.
    """
    return
