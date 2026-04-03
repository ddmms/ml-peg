"""Analyse density benchmark for LiTFSI/H2O electrolyte."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.utils import load_metrics_config

# --- Paths -------------------------------------------------------------------

CALCS_ROOT = Path(__file__).resolve().parents[3] / "calcs"
CALC_PATH = CALCS_ROOT / "wise_electrolytes" / "density" / "outputs"
APP_ROOT = Path(__file__).resolve().parents[3] / "app"
OUT_PATH = APP_ROOT / "data" / "wise_electrolytes" / "density"

MODELS = [
    "matpes-r2scan",
    "mace-mpa-0-medium",
    "mace-omat-0-medium",
    "mace-mp-0b3",
    "mace-mh-1-omat",
    "mace-mh-1-omol",
]

RHO_EXP = 1.7126  # g/cm³  (Gilbert et al., JCED 2017, DOI: 10.1021/acs.jced.7b00135)

# Map benchmark model names → ml-peg registry names (for app summary table)
MODEL_NAME_MAP = {
    "matpes-r2scan": "mace-matpes-r2scan",
    "mace-mpa-0-medium": "mace-mpa-0",
    "mace-omat-0-medium": "mace-omat-0",
}

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


# --- Helpers -----------------------------------------------------------------


def load_density_results() -> dict[str, dict]:
    """
    Load density.json for each model.

    Returns
    -------
    dict[str, dict]
        Per-model density results keyed by model name.
    """
    results = {}
    for model in MODELS:
        json_path = CALC_PATH / model / "density.json"
        if json_path.exists():
            with open(json_path) as f:
                results[model] = json.load(f)
    return results


def normalize_metric(value: float, good: float, bad: float) -> float:
    """
    Normalize metric to [0, 1] where 0=good, 1=bad.

    Parameters
    ----------
    value : float
        Raw metric value.
    good : float
        Value corresponding to score 0.
    bad : float
        Value corresponding to score 1.

    Returns
    -------
    float
        Normalized score clipped to [0, 1].
    """
    if bad == good:
        return 0.0 if value <= good else 1.0
    score = (value - good) / (bad - good)
    return max(0.0, min(1.0, score))


# --- Plot builders -----------------------------------------------------------


def build_density_bar_chart(data: dict[str, dict]) -> go.Figure:
    """
    Build bar chart of density per model.

    Parameters
    ----------
    data : dict[str, dict]
        Per-model density results.

    Returns
    -------
    go.Figure
        Plotly bar chart figure.
    """
    fig = go.Figure()
    for model in MODELS:
        if model not in data:
            continue
        fig.add_trace(
            go.Bar(
                x=[model],
                y=[data[model]["rho_mean"]],
                error_y={"type": "data", "array": [data[model]["rho_std"]]},
                name=model,
            )
        )

    fig.add_hline(
        y=RHO_EXP,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Exp. ({RHO_EXP} g/cm\u00b3, Gilbert 2017)",
    )
    fig.update_layout(
        title="Electrolyte Density (LiTFSI/H2O, 21 m)",
        yaxis_title="Density / g cm⁻³",
        showlegend=False,
    )
    return fig


def build_density_timeseries(data: dict[str, dict]) -> go.Figure:
    """
    Build density vs time plot for all models.

    Parameters
    ----------
    data : dict[str, dict]
        Per-model density results.

    Returns
    -------
    go.Figure
        Plotly timeseries figure.
    """
    fig = go.Figure()
    for model in MODELS:
        if model not in data:
            continue
        d = data[model]
        fig.add_trace(
            go.Scatter(
                x=d["time_full"],
                y=d["density_full"],
                mode="lines",
                name=model,
                opacity=0.8,
            )
        )

    fig.add_hline(
        y=RHO_EXP,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Exp. ({RHO_EXP} g/cm\u00b3)",
    )
    fig.update_layout(
        title="NPT Density vs Time (LiTFSI/H2O)",
        xaxis_title="Time / ps",
        yaxis_title="Density / g cm⁻³",
    )
    return fig


# --- Metrics table -----------------------------------------------------------


def build_metrics_table(data: dict[str, dict]) -> dict:
    """
    Build metrics table JSON compatible with ml-peg app format.

    Parameters
    ----------
    data : dict[str, dict]
        Per-model density results.

    Returns
    -------
    dict
        Table with data, columns, tooltips, and thresholds.
    """
    rows = []
    for model in MODELS:
        if model not in data:
            continue
        d = data[model]
        abs_err = d["rho_abs_error"]
        pct_err = abs(d["rho_error_pct"])

        score_abs = normalize_metric(
            abs_err,
            DEFAULT_THRESHOLDS["Density Error"]["good"],
            DEFAULT_THRESHOLDS["Density Error"]["bad"],
        )
        score_pct = normalize_metric(
            pct_err,
            DEFAULT_THRESHOLDS["Density Error (%)"]["good"],
            DEFAULT_THRESHOLDS["Density Error (%)"]["bad"],
        )

        w_abs = DEFAULT_WEIGHTS.get("Density Error", 1.0)
        w_pct = DEFAULT_WEIGHTS.get("Density Error (%)", 1.0)
        total_w = w_abs + w_pct
        overall = (
            1.0 - (score_abs * w_abs + score_pct * w_pct) / total_w
            if total_w > 0
            else 0.0
        )

        rows.append(
            {
                "MLIP": model,
                "id": MODEL_NAME_MAP.get(model, model),
                "Density (g/cm3)": round(d["rho_mean"], 4),
                "Density Error": round(abs_err, 4),
                "Density Error (%)": round(pct_err, 2),
                "Score": round(overall, 3),
            }
        )

    rows.sort(key=lambda r: r["Score"], reverse=True)

    columns = [
        {"name": c, "id": c}
        for c in [
            "MLIP",
            "Density Error",
            "Density Error (%)",
            "Score",
        ]
    ]

    return {
        "data": rows,
        "columns": columns,
        "tooltip_header": DEFAULT_TOOLTIPS,
        "thresholds": {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()},
        "weights": dict(DEFAULT_WEIGHTS),
        "model_name_map": MODEL_NAME_MAP,
    }


# --- Pytest interface --------------------------------------------------------


@pytest.fixture
def density_analysis():
    """
    Run full density analysis: table + plots.

    Returns
    -------
    dict
        Metrics table dictionary.
    """
    data = load_density_results()
    if not data:
        pytest.skip("No density data found")

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    table = build_metrics_table(data)
    with open(OUT_PATH / "density_metrics_table.json", "w") as f:
        json.dump(table, f, indent=2)

    fig_bar = build_density_bar_chart(data)
    with open(OUT_PATH / "figure_density_bar.json", "w") as f:
        f.write(fig_bar.to_json())

    fig_ts = build_density_timeseries(data)
    with open(OUT_PATH / "figure_density_timeseries.json", "w") as f:
        f.write(fig_ts.to_json())

    return table


def test_density(density_analysis) -> None:
    """
    Run density benchmark — generates table and plots.

    Parameters
    ----------
    density_analysis : dict
        Fixture providing the metrics table.
    """
    table = density_analysis
    assert len(table["data"]) > 0, "No models produced density data"
