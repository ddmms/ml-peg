"""Helpers to create callbacks and reusable plots for Dash apps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

from dash import Input, Output, State, callback, callback_context, html
from dash.dcc import Graph
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from dash.html import Div, Iframe
import numpy as np
import plotly.graph_objects as go

from ml_peg.app.utils.weas import generate_weas_html


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
    matrix: Iterable[Iterable[float]],
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


def plot_from_table_column(
    table_id: str, plot_id: str, column_to_plot: dict[str, Graph]
) -> None:
    """
    Attach callback to show plot when a table column is clicked.

    Parameters
    ----------
    table_id
        ID for Dash table being clicked.
    plot_id
        ID for Dash plot placeholder Div.
    column_to_plot
        Dictionary relating table headers (keys) and plot to show (values).
    """

    @callback(Output(plot_id, "children"), Input(table_id, "active_cell"))
    def show_plot(active_cell) -> Div:
        """
        Register callback to show plot when a table column is clicked.

        Parameters
        ----------
        active_cell
            Clicked cell in Dash table.

        Returns
        -------
        Div
            Message explaining interactivity, or plot on table click.
        """
        if not active_cell:
            return Div("Click on a metric to view plot.")
        column_id = active_cell.get("column_id", None)
        if column_id:
            if column_id in column_to_plot:
                return Div(column_to_plot[column_id])
            raise PreventUpdate
        raise ValueError("Invalid column_id")


def plot_from_table_cell(
    table_id: str,
    plot_id: str,
    cell_to_plot: dict[str, dict[Graph]],
) -> None:
    """
    Attach callback to show plot when a table cell is clicked.

    Parameters
    ----------
    table_id
        ID for Dash table being clicked.
    plot_id
        ID for Dash plot placeholder Div.
    cell_to_plot
        Nested dictionary of model names, column names, and plot to show.
    """

    @callback(Output(plot_id, "children"), Input(table_id, "active_cell"))
    def show_plot(active_cell) -> Div:
        """
        Register callback to show plot when a table cell is clicked.

        Parameters
        ----------
        active_cell
            Clicked cell in Dash table.

        Returns
        -------
        Div
            Message explaining interactivity, or plot on cell click.
        """
        if not active_cell:
            return Div("Click on a metric to view plot.")
        column_id = active_cell.get("column_id", None)
        row_id = active_cell.get("row_id", None)

        if row_id in cell_to_plot and column_id in cell_to_plot[row_id]:
            return Div(cell_to_plot[row_id][column_id])
        return Div("Click on a metric to view plot.")


def struct_from_scatter(
    scatter_id: str,
    struct_id: str,
    structs: str | list[str],
    mode: Literal["struct", "traj"] = "struct",
) -> None:
    """
    Attach callback to show a structure when a scatter point is clicked.

    Parameters
    ----------
    scatter_id
        ID for Dash scatter being clicked.
    struct_id
        ID for Dash plot placeholder Div where structures will be visualised.
    structs
        List of structure filenames in same order as scatter data to be visualised.
    mode
        Whether to display a single structure ("struct"), or trajectory from an initial
        image ("traj"). Default is "struct".
    """

    @callback(
        Output(struct_id, "children", allow_duplicate=True),
        Input(scatter_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_struct(click_data):
        """
        Register callback to show structure when a scatter point is clicked.

        Parameters
        ----------
        click_data
            Clicked data point in scatter plot.

        Returns
        -------
        Div
            Visualised structure on plot click.
        """
        if not click_data:
            return None
        idx = click_data["points"][0]["pointNumber"]

        if isinstance(structs, str):
            struct = structs
            index = idx
        else:
            struct = structs[idx]
            index = 0

        return Div(
            Iframe(
                srcDoc=generate_weas_html(struct, mode, index),
                style={
                    "height": "550px",
                    "width": "100%",
                    "border": "1px solid #ddd",
                    "borderRadius": "5px",
                },
            )
        )


def struct_from_table(
    table_id: str,
    struct_id: str,
    column_to_struct: dict[str, str],
    mode: Literal["struct", "traj"] = "struct",
) -> None:
    """
    Attach callback to show a structure when a table is clicked.

    Parameters
    ----------
    table_id
        ID for Dash table being clicked.
    struct_id
        ID for Dash plot placeholder Div where structures will be visualised.
    column_to_struct
        Dictionary of structure filenames indexed by table column.
    mode
        Whether to display a single structure ("struct"), or trajectory from an initial
        image ("traj"). Default is "struct".
    """

    @callback(
        Output(struct_id, "children", allow_duplicate=True),
        Input(table_id, "active_cell"),
        prevent_initial_call="initial_duplicate",
    )
    def show_struct(active_cell):
        """
        Register callback to show structure when a table is clicked.

        Parameters
        ----------
        active_cell
            Clicked cell in Dash table.

        Returns
        -------
        Div
            Visualised structure on plot click.
        """
        if not active_cell:
            return Div("Click on a metric to view the structure.")

        column_id = active_cell.get("column_id", None)
        if column_id:
            if column_id in column_to_struct:
                struct = column_to_struct[column_id]

                return Div(
                    Iframe(
                        srcDoc=generate_weas_html(struct, mode),
                        style={
                            "height": "550px",
                            "width": "100%",
                            "border": "1px solid #ddd",
                            "borderRadius": "5px",
                        },
                    )
                )
        return Div("Click on a metric to view the structure.")


def register_table_plot_callbacks(
    *,
    table_id: str,
    table_data: list[dict],
    plot_container_id: str,
    scatter_meta_store_id: str,
    last_cell_store_id: str,
    column_handlers: dict[str, Callable[[str, str], tuple[Component, dict] | None]],
    default_handler: Callable[[str, str], tuple[Component, dict] | None] | None = None,
    refresh_message: str = "Click on a metric to view scatter plots.",
    model_key: str = "MLIP",
) -> None:
    """
    Register callbacks that map table cell clicks to plot content.

    Parameters
    ----------
    table_id
        Dash table identifier emitting ``active_cell`` callbacks.
    table_data
        Pre-rendered table rows used to look up the clicked model.
    plot_container_id
        Div ID hosting the rendered plot content.
    scatter_meta_store_id
        Store component ID that tracks the latest plot metadata.
    last_cell_store_id
        Store component ID used to reset when the same cell is clicked twice.
    column_handlers
        Mapping of column identifiers to callables returning ``(content, meta)``.
    default_handler
        Fallback callable invoked when ``column_handlers`` has no entry.
    refresh_message
        Message displayed when a user re-clicks the active cell.
    model_key
        Key in ``table_data`` used to look up the model display name.
    """

    @callback(
        Output(plot_container_id, "children"),
        Output(scatter_meta_store_id, "data"),
        Output(last_cell_store_id, "data"),
        Input(table_id, "active_cell"),
        State(last_cell_store_id, "data"),
        prevent_initial_call=True,
    )
    def update_plot(active_cell, last_cell):
        """
        Map table cells to plot content.

        Parameters
        ----------
        active_cell
            Dash ``active_cell`` data dict from the metrics table.
        last_cell
            Previously clicked cell stored in ``last_cell_store_id``.

        Returns
        -------
        tuple
            Plot container children, scatter meta, and new ``last_cell`` value.
        """
        if not active_cell:
            raise PreventUpdate
        if last_cell and last_cell == active_cell:
            return html.Div(refresh_message), None, None
        row = active_cell.get("row")
        column = active_cell.get("column_id")
        if row is None or column is None or row < 0 or row >= len(table_data):
            raise PreventUpdate
        model_display = table_data[row].get(model_key)
        if not model_display:
            raise PreventUpdate
        handler = column_handlers.get(column)
        if handler is None:
            handler = default_handler
        if handler is None:
            raise PreventUpdate
        result = handler(model_display, column)
        if not result:
            raise PreventUpdate
        content, meta = result
        return content, meta, active_cell


def register_scatter_asset_callbacks(
    *,
    scatter_id: str,
    meta_store_id: str,
    asset_container_id: str,
    data_lookup: Callable[[dict, dict], dict | None],
    asset_renderer: Callable[[dict, dict], Component | None],
    empty_message: str,
    missing_message: str,
) -> None:
    """
    Register callbacks that map scatter clicks to rendered assets.

    Parameters
    ----------
    scatter_id
        Graph ID emitting ``clickData`` event dicts.
    meta_store_id
        Store component describing the currently active scatter context.
    asset_container_id
        Div ID where rendered assets will be displayed.
    data_lookup
        Callable receiving ``(point_data, scatter_meta)`` and returning metadata.
    asset_renderer
        Callable that converts lookup results to Dash components.
    empty_message
        Message shown when scatter metadata changes (before a click).
    missing_message
        Message shown when no asset can be produced for the click event.
    """

    @callback(
        Output(asset_container_id, "children"),
        Input(scatter_id, "clickData"),
        Input(meta_store_id, "data"),
        prevent_initial_call=True,
    )
    def display_asset(click_data, scatter_meta):
        """
        Render the requested asset when a scatter point is clicked.

        Parameters
        ----------
        click_data
            Plotly ``clickData`` event data.
        scatter_meta
            Metadata describing the active scatter context.

        Returns
        -------
        dash.html.Div | Component
            Rendered asset container or an informational message.
        """
        trigger = callback_context.triggered_id
        if trigger is None:
            raise PreventUpdate
        if trigger == meta_store_id:
            return html.Div(empty_message)
        if trigger != scatter_id or not scatter_meta:
            raise PreventUpdate
        if not click_data or not click_data.get("points"):
            raise PreventUpdate
        point_data = click_data["points"][0]
        asset_data = data_lookup(point_data, scatter_meta)
        if not asset_data:
            return html.Div(missing_message)
        rendered = asset_renderer(asset_data, scatter_meta)
        if rendered is None:
            return html.Div(missing_message)
        return rendered
