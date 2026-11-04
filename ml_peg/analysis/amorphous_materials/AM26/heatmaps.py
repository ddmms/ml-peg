"""Render stored AM26 system diagnostics without recalculating scientific metrics."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from ml_peg.analysis.amorphous_materials.AM26.metrics import SYSTEM_LABELS, SYSTEM_ORDER
from ml_peg.analysis.utils.utils import load_metrics_config

THRESHOLDS, _, _ = load_metrics_config(Path(__file__).with_name("metrics.yml"))
FIGURE_FILENAMES = {
    "Energy RelRMSE": "figure_energy_relrmse_heatmap.json",
    "Force RelRMSE": "figure_force_relrmse_heatmap.json",
}


def build_heatmap(bundle: dict[str, Any], metric: str) -> go.Figure:
    """
    Lay out stored model-by-system percentages, preserving out-of-range values.

    Parameters
    ----------
    bundle
        The merged Phase 3 system diagnostics, keyed by model under ``models``.
    metric
        ``Energy RelRMSE`` or ``Force RelRMSE``; no metrics are calculated here.

    Returns
    -------
    go.Figure
        Diagnostic heatmap with a threshold-based colour range and explicit gaps.
        Values above the bad threshold retain their raw z values and labels.
    """
    if metric not in FIGURE_FILENAMES:
        raise ValueError(f"Unsupported AM26 heatmap metric: {metric}")
    limits = THRESHOLDS[f"Mean {metric}"]
    good, bad = limits["good"], limits["bad"]
    columns = [SYSTEM_LABELS[system] for system in SYSTEM_ORDER]
    models = bundle.get("models", {})
    names = list(models)
    values, labels = [], []
    missing_x, missing_y = [], []
    for model, data in models.items():
        row, text = [], []
        systems = (data or {}).get("systems", {})
        for system, column in zip(SYSTEM_ORDER, columns, strict=True):
            value = (systems.get(system) or {}).get(metric)
            value = float(value) if value is not None else None
            if value is not None and not math.isfinite(value):
                value = None
            row.append(value)
            text.append(f"{value:.4g}" if value is not None else "")
            if value is None:
                missing_x.append(column)
                missing_y.append(model)
        values.append(row)
        labels.append(text)

    fig = go.Figure()
    if names:
        # Follow ML-PEG's EOS heatmaps: threshold-limited Viridis_r colours,
        # with the original values retained for hover rather than clipped z data.
        fig.add_trace(
            go.Heatmap(
                z=values,
                x=columns,
                y=names,
                text=labels,
                texttemplate="%{text}",
                colorscale="Viridis_r",
                zmin=good,
                zmax=bad,
                connectgaps=False,
                hoverongaps=False,
                xgap=1,
                ygap=1,
                colorbar={
                    "title": {"text": "RelRMSE [%]"},
                    "tickvals": [0, 25, 50, 75, 100],
                    "ticktext": ["0", "25", "50", "75", "≥100"],
                },
                hovertemplate=(
                    "Model: %{y}<br>Material system: %{x}<br>"
                    f"{metric}: %{{z:.15g}}%<extra></extra>"
                ),
            )
        )
    else:
        fig.add_annotation(
            text="No AM26 system results available.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
        )
    if missing_x:
        # A gap cannot display Heatmap text. A text-only overlay labels gaps
        # without inventing numeric values or adding any interactive callbacks.
        fig.add_trace(
            go.Scatter(
                x=missing_x,
                y=missing_y,
                mode="text",
                text=["N/A"] * len(missing_x),
                textfont={"color": "#666666", "size": 11},
                showlegend=False,
                hovertemplate=(
                    "Model: %{y}<br>Material system: %{x}<br>"
                    f"{metric}: unavailable or failed<extra></extra>"
                ),
            )
        )
    fig.update_layout(
        title={"text": f"AM26 {metric} [%]"},
        xaxis={
            "title": "Material system",
            "categoryorder": "array",
            "categoryarray": columns,
        },
        yaxis={
            "title": "Model",
            "autorange": "reversed",
            "categoryorder": "array",
            "categoryarray": names,
        },
        height=max(300, 28 * len(names) + 160),
        margin={"l": 180, "r": 40, "t": 70, "b": 110},
        plot_bgcolor="#eeeeee",
        meta={
            "source": "system_metrics.json",
            "metric": metric,
            "system_order": list(SYSTEM_ORDER),
        },
    )
    fig.add_annotation(
        text=(
            "Colour scale: 0–100%; ≥100% uses the upper-end colour. "
            "Values are not truncated.<br>N/A: unavailable or failed."
        ),
        x=0,
        y=0,
        xref="paper",
        yref="paper",
        xanchor="left",
        yanchor="top",
        yshift=-70,
        showarrow=False,
        font={"size": 11},
    )
    return fig


def write_heatmaps(bundle: dict[str, Any], out_path: Path) -> None:
    """
    Serialize both diagnostic figures from the already merged analysis bundle.

    Parameters
    ----------
    bundle
        Phase 3 diagnostics, including models retained by an incremental update.
    out_path
        Benchmark analysis output directory under ``ml_peg/app/data``.
    """
    out_path.mkdir(parents=True, exist_ok=True)
    for metric, filename in FIGURE_FILENAMES.items():
        build_heatmap(bundle, metric).write_json(out_path / filename)
