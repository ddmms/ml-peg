"""Reusable Plotly/Dash helpers for ML-PEG apps."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from dash import dcc, html
import numpy as np
import plotly.graph_objects as go


def figure_from_dict(
    figure_dict: Mapping | None, fallback_title: str | None = None
) -> go.Figure:
    """
    Convert a serialized Plotly figure dictionary into a ``go.Figure``.

    Parameters
    ----------
    figure_dict
        JSON-serialisable figure dictionary, typically saved during analysis.
    fallback_title
        Title used when ``figure_dict`` is missing.

    Returns
    -------
    go.Figure
        Reconstructed figure with a placeholder title when necessary.
    """
    if not figure_dict:
        fig = go.Figure()
        if fallback_title:
            fig.update_layout(title=fallback_title)
        return fig
    return go.Figure(figure_dict)


def build_violin_distribution(
    values: Sequence[float],
    labels: Sequence[str],
    *,
    title: str,
    yaxis_title: str,
    hovertemplate: str,
    color: str = "#636EFA",
) -> go.Figure:
    """
    Build a violin plot showing a distribution of scalar errors.

    Parameters
    ----------
    values
        Numeric samples to plot.
    labels
        Labels associated with each sample, included in hover text.
    title
        Figure title.
    yaxis_title
        Axis label describing the plotted quantity.
    hovertemplate
        Template applied to violin samples.
    color
        Violin and line colour. Default corresponds to Plotly blue.

    Returns
    -------
    go.Figure
        Configured violin plot (or placeholder when ``values`` is empty).
    """
    fig = go.Figure()
    if not values:
        fig.update_layout(title=title or "No data available.")
        return fig
    customdata = [[label] for label in labels]
    fig.add_trace(
        go.Violin(
            y=values,
            text=labels,
            customdata=customdata,
            points="all",
            jitter=0.05,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            line_color=color,
            opacity=0.6,
            hovertemplate=hovertemplate,
        )
    )
    fig.update_layout(title=title, yaxis_title=yaxis_title, plot_bgcolor="#ffffff")
    return fig


def build_confusion_heatmap(
    matrix: Sequence[Sequence[float]],
    *,
    x_labels: Sequence[str],
    y_labels: Sequence[str],
    title: str,
    colorscale: str = "Blues",
    hovertemplate: str | None = None,
) -> go.Figure:
    """
    Build a confusion-matrix heatmap with annotated counts.

    Parameters
    ----------
    matrix
        2D array-like of counts.
    x_labels
        Labels shown along the x-axis.
    y_labels
        Labels shown along the y-axis.
    title
        Figure title.
    colorscale
        Plotly colourscale applied to the heatmap.
    hovertemplate
        Optional template overriding the default hover text.

    Returns
    -------
    go.Figure
        Heatmap with value annotations.
    """
    data = np.asarray(matrix, dtype=float)
    fig = go.Figure()
    if data.size == 0:
        fig.update_layout(title=title or "No data available.", plot_bgcolor="#ffffff")
        return fig
    template = (
        hovertemplate or "Pred %{y}<br>Ref %{x}<br>Count: %{z:.0f}<extra></extra>"
    )
    fig.add_trace(
        go.Heatmap(
            z=data,
            x=x_labels,
            y=y_labels,
            colorscale=colorscale,
            showscale=True,
            hovertemplate=template,
        )
    )
    max_val = data.max(initial=0.0)
    total = data.sum()
    for y_idx, y_label in enumerate(y_labels):
        for x_idx, x_label in enumerate(x_labels):
            cell_val = data[y_idx, x_idx]
            pct = (cell_val / total * 100) if total else 0.0
            color = "#ffffff" if max_val and cell_val >= 0.6 * max_val else "#111111"
            fig.add_annotation(
                x=x_label,
                y=y_label,
                text=f"{cell_val:.0f} ({pct:.1f}%)",
                showarrow=False,
                font={"color": color, "size": 14},
            )
    fig.update_layout(title=title, plot_bgcolor="#ffffff")
    return fig


def build_classified_parity_scatter(
    points: Sequence[Mapping[str, float]],
    *,
    title: str,
    xaxis_title: str,
    yaxis_title: str,
    class_key: str = "class",
    ref_key: str = "ref",
    pred_key: str = "pred",
    label_key: str = "label",
    hovertemplate: str,
    colours: Mapping[str, str] | None = None,
) -> go.Figure:
    """
    Build a scatter plot grouping points by classification labels.

    Parameters
    ----------
    points
        Sequence of metadata dictionaries containing reference/prediction values.
    title
        Figure title.
    xaxis_title
        Label for the x-axis.
    yaxis_title
        Label for the y-axis.
    class_key
        Key storing the class/group identifier. Default ``"class"``.
    ref_key
        Key storing reference values. Default ``"ref"``.
    pred_key
        Key storing predicted values. Default ``"pred"``.
    label_key
        Key used for hover labels. Default ``"label"``.
    hovertemplate
        Plotly hover template string.
    colours
        Optional mapping of class label -> marker colour.

    Returns
    -------
    go.Figure
        Scatter parity plot grouped by class.
    """
    fig = go.Figure()
    if not points:
        fig.update_layout(title=title or "No data available.")
        return fig
    default_colours = {
        "TP": "#2ca02c",
        "TN": "#1f77b4",
        "FP": "#ff7f0e",
        "FN": "#d62728",
    }
    colour_map = colours or default_colours
    for label, colour in colour_map.items():
        subset = [point for point in points if point.get(class_key) == label]
        if not subset:
            continue
        refs = [point[ref_key] for point in subset]
        preds = [point[pred_key] for point in subset]
        hover = [point.get(label_key) or point.get("id") for point in subset]
        custom = [[point.get("id") or idx] for idx, point in enumerate(subset)]
        fig.add_trace(
            go.Scatter(
                x=refs,
                y=preds,
                mode="markers",
                name=label,
                marker={"color": colour},
                text=hover,
                customdata=custom,
                hovertemplate=hovertemplate,
            )
        )
    refs_all = [point[ref_key] for point in points]
    preds_all = [point[pred_key] for point in points]
    lower = min(min(refs_all), min(preds_all))
    upper = max(max(refs_all), max(preds_all))
    fig.add_trace(
        go.Scatter(
            x=[lower, upper],
            y=[lower, upper],
            mode="lines",
            showlegend=False,
            line={"color": "#8c8c8c", "dash": "dash"},
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        plot_bgcolor="#ffffff",
    )
    return fig


def build_serialized_scatter_content(
    model_display: str,
    column_id: str,
    *,
    models_data: Mapping[str, Any],
    label_map: Mapping[str, str],
    scatter_id: str,
    instructions: str,
):
    """
    Render a stored scatter figure referenced by a table column.

    Parameters
    ----------
    model_display
        Table entry label for the selected model.
    column_id
        Column identifier originating from the Dash DataTable.
    models_data
        Interactive dataset keyed by model name.
    label_map
        Mapping of column labels to metric keys.
    scatter_id
        Graph identifier used for the scatter component.
    instructions
        Instructional text displayed above the scatter plot.

    Returns
    -------
    tuple | None
        ``(content, meta)`` pair consumed by the shared callback helper.
    """
    metric_key = label_map.get(column_id)
    if metric_key is None:
        return None
    figure = figure_from_dict(
        models_data.get(model_display, {}).get("figures", {}).get(metric_key),
        fallback_title=f"No data for {model_display}",
    )
    graph = dcc.Graph(id=scatter_id, figure=figure)
    meta = {"model": model_display, "type": "metric", "metric": metric_key}
    content = html.Div([html.P(instructions), graph])
    return content, meta


def build_classification_panel(
    model_display: str,
    column_id: str,
    *,
    models_data: Mapping[str, Any],
    scatter_id: str,
    confusion_id: str,
    instructions: str,
    scatter_hovertemplate: str,
    xaxis_title: str,
    yaxis_title: str,
    confusion_axes: Sequence[Sequence[str]],
    classification_key: str = "stability",
):
    """
    Build parity scatter and confusion matrix content for classifications.

    Parameters
    ----------
    model_display
        Table entry label for the selected model.
    column_id
        Column identifier (unused but kept for signature parity).
    models_data
        Interactive dataset keyed by model name.
    scatter_id
        Graph identifier used for the classification scatter.
    confusion_id
        Graph identifier used for the confusion matrix.
    instructions
        Instructional text displayed above the plots.
    scatter_hovertemplate
        Plotly hover template string used for the parity scatter.
    xaxis_title
        Scatter x-axis label.
    yaxis_title
        Scatter y-axis label.
    confusion_axes
        Pair of ``(x_labels, y_labels)`` for the confusion matrix.
    classification_key
        Key in ``models_data`` storing the classification metadata.

    Returns
    -------
    tuple
        ``(content, meta)`` pair consumed by the shared callback helper.
    """
    points = (
        models_data.get(model_display, {}).get(classification_key, {}).get("points", [])
    )
    scatter = build_classified_parity_scatter(
        points,
        title=f"{model_display} – Classification scatter",
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        hovertemplate=scatter_hovertemplate,
    )
    confusion_data = (
        models_data.get(model_display, {}).get(classification_key, {}).get("confusion")
        or []
    )
    confusion = build_confusion_heatmap(
        confusion_data,
        x_labels=confusion_axes[0],
        y_labels=confusion_axes[1],
        title=f"{model_display} – Classification confusion matrix",
    )
    scatter_graph = dcc.Graph(id=scatter_id, figure=scatter)
    confusion_graph = dcc.Graph(id=confusion_id, figure=confusion)
    meta = {"model": model_display, "type": classification_key}
    content = html.Div([html.P(instructions), scatter_graph, confusion_graph])
    return content, meta


def resolve_scatter_selection(
    point_data: Mapping[str, Any],
    scatter_meta: Mapping[str, Any],
    *,
    models_data: Mapping[str, Any],
    system_lookup,
):
    """
    Resolve the clicked scatter point into its backing metadata entry.

    Parameters
    ----------
    point_data
        Plotly point dictionary from ``clickData``.
    scatter_meta
        Metadata describing the currently active scatter context.
    models_data
        Interactive dataset keyed by model name.
    system_lookup
        Callable ``(model_entry, point_id) -> dict | None`` used for custom lookups.

    Returns
    -------
    dict | None
        Selection context containing ``model`` and ``selection`` keys.
    """
    custom = point_data.get("customdata") or []
    point_id = (
        custom[0] if custom else point_data.get("text") or point_data.get("label")
    )
    if point_id is None:
        return None
    model_display = scatter_meta.get("model")
    meta_type = scatter_meta.get("type")
    if model_display is None or meta_type is None:
        return None
    model_entry = models_data.get(model_display, {})
    selected = None
    if meta_type == "metric":
        metric_key = scatter_meta.get("metric")
        metric_points = (
            model_entry.get("metrics", {}).get(metric_key, {}).get("points", [])
        )
        selected = next(
            (entry for entry in metric_points if entry.get("id") == point_id),
            None,
        )
        if selected is None and isinstance(point_id, int):
            if 0 <= point_id < len(metric_points):
                selected = metric_points[point_id]
    elif meta_type == "stability":
        metric_points = model_entry.get("stability", {}).get("points", [])
        selected = next(
            (entry for entry in metric_points if entry.get("id") == point_id),
            None,
        )
    else:
        selected = system_lookup(model_entry, point_id)
    if not selected:
        return None
    return {"model": model_display, "selection": selected}
