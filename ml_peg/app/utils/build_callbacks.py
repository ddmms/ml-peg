"""Helpers to create callbaclks for Dash app."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from dash import (
    Input,
    Output,
    State,
    callback,
    callback_context,
    clientside_callback,
    html,
)
from dash.dcc import Graph
from dash.development.base_component import Component
from dash.exceptions import PreventUpdate
from dash.html import Div, Iframe

from ml_peg.app.utils.build_components import build_plot_download_controls
from ml_peg.app.utils.plot_helpers import INSTRUCTION_STYLE, POINT_HINT, TABLE_HINT
from ml_peg.app.utils.register_callbacks import register_plot_download_callbacks
from ml_peg.app.utils.weas import generate_weas_html


def plot_with_download_controls(graph: Graph) -> Div:
    """
    Wrap a Plotly graph with CSV/PNG/SVG/HTML download controls.

    Parameters
    ----------
    graph
        Dash graph component.

    Returns
    -------
    Div
        Graph with download controls above it.
    """
    graph_id = getattr(graph, "id", None)
    if not isinstance(graph_id, str):
        return Div(graph)
    return Div(
        [
            build_plot_download_controls(graph_id),
            Div(POINT_HINT, style=INSTRUCTION_STYLE),
            graph,
        ]
    )


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
    register_plot_download_callbacks()

    @callback(
        Output(plot_id, "children"),
        Output(table_id, "active_cell"),
        Input(table_id, "active_cell"),
    )
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
            return Div(TABLE_HINT, style=INSTRUCTION_STYLE), None
        column_id = active_cell.get("column_id", None)
        if column_id:
            if column_id in column_to_plot:
                return plot_with_download_controls(column_to_plot[column_id]), None
            raise PreventUpdate
        raise ValueError("Invalid column_id")


def plot_from_table_cell(
    table_id: str,
    plot_id: str,
    cell_to_plot: dict[str, dict[Graph]],
    table_data: list[dict] | None = None,
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
    table_data
        Optional table data to check for None/missing values. If provided,
        cells with None values will show "No data available" message.
    """
    register_plot_download_callbacks()

    @callback(
        Output(plot_id, "children"),
        Output(table_id, "active_cell"),
        Input(table_id, "active_cell"),
        State(table_id, "data"),
    )
    def show_plot(active_cell, current_table_data) -> Div:
        """
        Register callback to show plot when a table cell is clicked.

        Parameters
        ----------
        active_cell
            Clicked cell in Dash table.
        current_table_data
            Current table data (includes live updates from callbacks).

        Returns
        -------
        Div
            Message explaining interactivity, or plot on cell click.
        """
        if not active_cell:
            return Div(TABLE_HINT, style=INSTRUCTION_STYLE), None
        column_id = active_cell.get("column_id", None)
        row_id = active_cell.get("row_id", None)
        row_index = active_cell.get("row", None)

        # Check if cell value is None (no data for this model)
        if current_table_data and row_index is not None:
            try:
                cell_value = current_table_data[row_index].get(column_id)
                if cell_value is None:
                    return Div("No data available for this model."), None
            except (IndexError, KeyError, TypeError):
                pass  # Fall through to normal handling

        if row_id in cell_to_plot and column_id in cell_to_plot[row_id]:
            return plot_with_download_controls(cell_to_plot[row_id][column_id]), None
        return Div(TABLE_HINT, style=INSTRUCTION_STYLE), None


# A scatter can be passed to more than one helper, but its highlight callback
# should only be registered once. Store the first frame-follow setting so later
# registrations can be skipped or rejected when they disagree.
registered_highlights: dict[str, bool] = {}


def _register_point_highlight(scatter_id: str, follow_frames: bool = True) -> None:
    """
    Ring the most recently clicked point of a scatter plot.

    Registered once per ``scatter_id`` and runs entirely client-side, so it adds
    no server load and works for any scatter wired for click interactions. On
    each click a single ring marker (a transparent-fill ``__clicked_point__``
    trace) replaces the previous one. Its type and axes are copied from the
    clicked trace so the ring sits on the same render layer (svg vs WebGL) and
    subplot as the point. Frame following can be disabled for viewers whose
    frames do not correspond one-to-one with scatter points, such as a density
    cell containing multiple structures.

    Parameters
    ----------
    scatter_id
        ID of the Dash ``Graph`` whose clicked point should be highlighted.
    follow_frames
        Whether a WEAS frame-change event moves the highlight to the point at
        the same index. Default is True.
    """
    existing_follow_frames = registered_highlights.get(scatter_id)
    if existing_follow_frames is not None:
        if existing_follow_frames != follow_frames:
            raise ValueError(
                f"Conflicting frame-follow settings for scatter {scatter_id!r}."
            )
        return
    registered_highlights[scatter_id] = follow_frames

    clientside_callback(
        """
        function(clickData, figure) {
            const dc = window.dash_clientside;
            if (!clickData || !figure || !figure.data) { return dc.no_update; }
            const pt = clickData.points[0];
            const src = figure.data[pt.curveNumber] || {};
            const data = figure.data.filter(
                (t) => t.name !== '__clicked_point__'
            );
            data.push({
                x: [pt.x],
                y: [pt.y],
                type: src.type || 'scatter',
                mode: 'markers',
                name: '__clicked_point__',
                xaxis: src.xaxis,
                yaxis: src.yaxis,
                hoverinfo: 'skip',
                showlegend: false,
                cliponaxis: false,
                marker: {
                    size: 16,
                    color: 'rgba(0,0,0,0)',
                    line: {color: '#ff1493', width: 3},
                },
            });
            // Record the clicked curve so a playing WEAS trajectory can move
            // this ring to the matching frame's point (weas_frame_follow.js).
            window.__mlPegActiveTraj = {
                scatterId: "__SCATTER_ID__",
                x: src.x || [],
                y: src.y || [],
                followFrames: __FOLLOW_FRAMES__,
            };
            const out = Object.assign({}, figure, {data: data});
            // Pin the axes to the live rendered range so adding the ring (a large
            // marker) can't autorange and resize the plot. Reading _fullLayout
            // (not the stale State figure) also preserves any current user zoom.
            const gd = document.getElementById("__SCATTER_ID__");
            const plot = gd && gd.querySelector('.js-plotly-plot');
            if (plot && plot._fullLayout) {
                out.layout = Object.assign({}, figure.layout);
                // Preserve current legend visibility: adding the ring trace must
                // not flip Plotly's auto-legend on for plots that have none.
                out.layout.showlegend = plot._fullLayout.showlegend;
                const xa = (src.xaxis || 'x').replace('x', 'xaxis');
                const ya = (src.yaxis || 'y').replace('y', 'yaxis');
                const flx = plot._fullLayout[xa], fly = plot._fullLayout[ya];
                if (flx && fly) {
                    out.layout[xa] = Object.assign(
                        {}, out.layout[xa], {range: flx.range.slice(), autorange: false}
                    );
                    out.layout[ya] = Object.assign(
                        {}, out.layout[ya], {range: fly.range.slice(), autorange: false}
                    );
                }
            }
            return out;
        }
        """.replace("__SCATTER_ID__", scatter_id).replace(
            "__FOLLOW_FRAMES__", str(follow_frames).lower()
        ),
        Output(scatter_id, "figure", allow_duplicate=True),
        Input(scatter_id, "clickData"),
        State(scatter_id, "figure"),
        prevent_initial_call=True,
    )


def plot_from_scatter(
    scatter_id: str,
    plot_id: str,
    plots_list: list[Graph],
) -> None:
    """
    Attach callback to show plot when a scatter point is clicked.

    Parameters
    ----------
    scatter_id
        ID for Dash scatter being clicked.
    plot_id
        ID for Dash plot placeholder Div where new plot will be rendered.
    plots_list
        List of plots to show, in same order as scatter data.
    """
    register_plot_download_callbacks()
    _register_point_highlight(scatter_id)

    @callback(
        Output(plot_id, "children", allow_duplicate=True),
        Input(scatter_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_plot(click_data) -> Div:
        """
        Register callback to show plot when a scatter point is clicked.

        Parameters
        ----------
        click_data
            Clicked data point in scatter plot.

        Returns
        -------
        Div
            Plot on scatter click.
        """
        if not click_data:
            return Div(POINT_HINT, style=INSTRUCTION_STYLE)
        idx = click_data["points"][0]["pointNumber"]

        if idx >= 0 and idx < len(plots_list):
            return plot_with_download_controls(plots_list[idx])
        return Div(POINT_HINT, style=INSTRUCTION_STYLE)


def struct_from_scatter(
    scatter_id: str,
    struct_id: str,
    structs: str | list[str],
    mode: Literal["struct", "traj"] = "struct",
    follow_frames: bool | None = None,
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
    follow_frames
        Whether stepping through a WEAS trajectory moves the scatter highlight to
        the point with the same index. If None (default), following is enabled
        only for a single shared trajectory file. This is for e.g. a NEB, whereas a list
        of per-point files, including density-cell structure collections, keeps
        the highlight on the clicked point. Set explicitly to override.
    """
    if follow_frames is None:
        follow_frames = mode == "traj" and isinstance(structs, str)
    _register_point_highlight(scatter_id, follow_frames=follow_frames)

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
            return Div()
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


def struct_from_multi_scatters(
    scatter_id: str,
    struct_id: str,
    structs: list[str] | list[list[str]],
    mode: Literal["struct", "traj"] = "struct",
    follow_frames: bool = True,
) -> None:
    """
    Attach callback to show a structure when a multiline scatter point is clicked.

    Unlike `struct_from_scatter`, which accepts a single traj file or single list of
    struct files and renders a struct based on the clicked point index, this callback
    instead accepts a list of traj files or a list of list of struct files which is
    rendered based on the clicked curve number and then point index.

    Parameters
    ----------
    scatter_id
        ID for Dash scatter being clicked.
    struct_id
        ID for Dash plot placeholder Div where structures will be visualised.
    structs
        List of list of structure filenames, with outer list in same order as curves to
        be visualised, and inner list in same order as scatter data to be visualised.
    mode
        Whether to display a single structure ("struct"), or trajectory from an initial
        image ("traj"). Default is "struct".
    follow_frames
        Whether stepping through a WEAS trajectory moves the scatter highlight to
        the point with the same index. Default is True.

    Examples
    --------
    >>> struct_from_multi_scatters(
    >>>     scatter_id="test-figure",
    >>>     struct_id="test-placeholder",
    >>>     structs=[["config-i-j.xyz", ...], ...],
    >>>     mode="struct",
    >>> )

    When the `i`th data point of the `j`th curve of "test-figure" is clicked,
    `structs[j][i]` will be rendered in the "test-placeholder" Div.
    """
    _register_point_highlight(scatter_id, follow_frames=follow_frames)

    @callback(
        Output(struct_id, "children", allow_duplicate=True),
        Input(scatter_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_struct(click_data):
        """
        Register callback to show structure when a multiline scatter point is clicked.

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
            return Div()
        curve_number = click_data["points"][0]["curveNumber"]
        idx = click_data["points"][0]["pointNumber"]

        if isinstance(structs[curve_number], str):
            struct = structs[curve_number]
            index = idx
        else:
            struct = structs[curve_number][idx]
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
        Output(table_id, "active_cell"),
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
            return (
                Div(TABLE_HINT, style=INSTRUCTION_STYLE),
                None,
            )

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
                ), None

            raise PreventUpdate
        raise ValueError("Invalid column_id")


def scatter_and_assets_from_table(
    *,
    table_id: str,
    table_data: list[dict],
    plot_container_id: str,
    scatter_metadata_store_id: str,
    last_cell_store_id: str,
    column_handlers: dict[str, Callable[[str, str], tuple[Component, dict] | None]],
    scatter_id: str,
    default_handler: Callable[[str, str], tuple[Component, dict] | None] | None = None,
    model_key: str = "MLIP",
) -> None:
    """
    Render scatter content and persist model-specific metadata for asset callbacks.

    Parameters
    ----------
    table_id
        Dash table identifier emitting ``active_cell`` callbacks.
    table_data
        Default table rows.
    plot_container_id
        Div ID hosting the rendered plot content.
    scatter_metadata_store_id
        Store component ID that tracks the latest plot metadata.
    last_cell_store_id
        Store component ID used to reset when the same cell is clicked twice.
    column_handlers
        Mapping of column identifiers to callables returning ``(content, metadata)``.
    scatter_id
        Graph ID used for the rendered scatter plot. Handlers must return content
        containing a ``dcc.Graph`` with this ID so the generic plot-download
        callback can export the active plot.
    default_handler
        Fallback callable invoked when ``column_handlers`` has no entry.
    model_key
        Key in ``table_data`` used to look up the model display name.
    """
    register_plot_download_callbacks()

    @callback(
        Output(plot_container_id, "children"),
        Output(scatter_metadata_store_id, "data"),
        Output(last_cell_store_id, "data"),
        Output(table_id, "active_cell"),
        Input(table_id, "active_cell"),
        State(last_cell_store_id, "data"),
        prevent_initial_call=True,
    )
    def _update_scatter_and_metadata(active_cell, last_cell):
        """
        Update scatter content and metadata for the selected table cell.

        Map table cells to plot content and store scatter metadata describing the
        active model/metric (used by asset callbacks such as dispersion plots).

        Parameters
        ----------
        active_cell
            Dash ``active_cell`` data dict from the metrics table.
        last_cell
            Previously clicked cell stored in ``last_cell_store_id``.

        Returns
        -------
        tuple
            Plot container children, scatter metadata, and new ``last_cell`` value.
        """
        if not active_cell:
            raise PreventUpdate
        if last_cell == active_cell:
            raise PreventUpdate

        row = active_cell.get("row")
        row_id = active_cell.get("row_id")
        column = active_cell.get("column_id")
        if column is None:
            raise PreventUpdate

        selected_row = None
        if row_id is not None:
            for entry in table_data:
                if entry.get("id") == row_id:
                    selected_row = entry
                    break

        if selected_row is None:
            if row is None or row < 0 or row >= len(table_data):
                raise PreventUpdate
            selected_row = table_data[row]

        model_display = selected_row.get(model_key)
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
        content, metadata = result

        content = Div([build_plot_download_controls(scatter_id), content])

        return content, metadata, active_cell, None


def model_asset_from_scatter(
    *,
    scatter_id: str,
    metadata_store_id: str,
    asset_container_id: str,
    data_lookup: Callable[[dict, dict], dict | None],
    asset_renderer: Callable[[dict], Component | None],
    empty_message: str,
    missing_message: str,
) -> None:
    """
    Render a model-specific asset whenever the active scatter point is clicked.

    Parameters
    ----------
    scatter_id
        Graph ID emitting ``clickData`` event dicts.
    metadata_store_id
        Store component describing the currently active scatter context.
    asset_container_id
        Div ID where rendered assets will be displayed.
    data_lookup
        Callable receiving ``(point_data, scatter_metadata)`` and returning metadata.
    asset_renderer
        Callable that converts lookup results to Dash components.
    empty_message
        Message shown when scatter metadata changes (before a click).
    missing_message
        Message shown when no asset can be produced for the click event.
    """
    _register_point_highlight(scatter_id)

    @callback(
        Output(asset_container_id, "children"),
        Input(scatter_id, "clickData"),
        Input(metadata_store_id, "data"),
        prevent_initial_call=True,
    )
    def _display_asset(click_data, scatter_metadata):
        """
        Render the requested asset when a scatter point is clicked.

        Parameters
        ----------
        click_data
            Plotly ``clickData`` event data.
        scatter_metadata
            Stored metadata describing the active scatter context.

        Returns
        -------
        dash.html.Div | Component
            Rendered asset container or an informational message.
        """
        trigger = callback_context.triggered_id
        if trigger is None:
            raise PreventUpdate
        if trigger == metadata_store_id:
            return html.Div(empty_message)
        if trigger != scatter_id or not scatter_metadata:
            raise PreventUpdate
        if not click_data or not click_data.get("points"):
            raise PreventUpdate
        point_data = click_data["points"][0]
        asset_data = data_lookup(point_data, scatter_metadata)
        if not asset_data:
            return html.Div(missing_message)
        rendered = asset_renderer(asset_data)
        if rendered is None:
            return html.Div(missing_message)
        return rendered
