"""Render stored AM26 system diagnostics without recalculating scientific metrics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import plotly.graph_objects as go

from ml_peg.analysis.amorphous_materials.AM26.metrics import SYSTEM_LABELS, SYSTEM_ORDER
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app.utils.plot_helpers import build_annotated_heatmap

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
    values = []
    for data in models.values():
        systems = (data or {}).get("systems", {})
        values.append(
            [(systems.get(system) or {}).get(metric) for system in SYSTEM_ORDER]
        )

    fig = build_annotated_heatmap(
        values,
        x_labels=columns,
        y_labels=names,
        zmin=good,
        zmax=bad,
        colorscale="Viridis_r",
        colorbar={
            "title": {"text": "RelRMSE [%]"},
            "tickvals": [0, 25, 50, 75, 100],
            "ticktext": ["0", "25", "50", "75", "≥100"],
        },
        hovertemplate=(
            "Model: %{y}<br>Material system: %{x}<br>"
            f"{metric}: %{{z:.15g}}%<extra></extra>"
        ),
        missing_hovertemplate=(
            "Model: %{y}<br>Material system: %{x}<br>"
            f"{metric}: unavailable or failed<extra></extra>"
        ),
    )
    if not names:
        fig.add_annotation(
            text="No AM26 system results available.",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
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
