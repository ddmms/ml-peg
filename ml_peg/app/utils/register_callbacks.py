"""Register callbacks relating to interative weights."""

from __future__ import annotations

from dash import (
    ClientsideFunction,
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    ctx,
    dcc,
    html,
)
from dash.exceptions import PreventUpdate

from ml_peg.analysis.utils.utils import (
    calc_ranks,
    calc_scores,
    get_table_style,
    normalize_metric,
)


def register_summary_table_callbacks() -> None:
    """Register callbacks to update summary table."""

    @callback(
        Output("summary-table", "data"),
        Output("summary-table", "style_data_conditional"),
        Input("all-tabs", "value"),
        Input("summary-table-weight-store", "data"),
        State("summary-table-scores-store", "data"),
        State("summary-table", "data"),
        prevent_initial_call=False,
    )
    def update_summary_table(
        tabs_value: str,
        stored_weights: dict[str, float],
        stored_scores: dict[str, dict[str, float]],
        summary_data: list[dict],
    ) -> list[dict]:
        """
        Update summary table when scores change.

        Parameters
        ----------
        tabs_value
            Value of selected tab. Parameter unused, but required to register Input.
        stored_weights
            Stored summary weights dictionary.
        stored_scores
            Stored scores for table scores.
        summary_data
            Data from summary table to be updated.

        Returns
        -------
        list[dict]
            Updated summary table data.
        """
        # Update table from stored scores
        if stored_scores:
            for row in summary_data:
                for tab, values in stored_scores.items():
                    row[tab] = values[row["MLIP"]]

        # Update table contents
        summary_data = calc_scores(summary_data, stored_weights)
        summary_data = calc_ranks(summary_data)
        style = get_table_style(summary_data)

        return summary_data, style


def register_tab_table_callbacks(table_id) -> None:
    """
    Register callback to update table scores/rankings when stored values change.

    Parameters
    ----------
    table_id
        ID for table to update.
    """

    @callback(
        Output(table_id, "data"),
        Output(table_id, "style_data_conditional"),
        Input(f"{table_id}-weight-store", "data"),
        State(table_id, "data"),
        prevent_initial_call=True,
    )
    def update_table_scores(
        stored_weights: dict[str, float], table_data: list[dict]
    ) -> list[dict]:
        """
        Update scores table score and rankings when data store updates.

        Parameters
        ----------
        stored_weights
            Stored weight values for `table_id`.
        table_data
            Data from `table_id` to be updated.

        Returns
        -------
        list[dict]
            Updated table data.
        """
        table_data = calc_scores(table_data, stored_weights)
        table_data = calc_ranks(table_data)
        style = get_table_style(table_data)

        return table_data, style

    @callback(
        Output("summary-table-scores-store", "data", allow_duplicate=True),
        Input(f"{table_id}-weight-store", "data"),
        State(table_id, "data"),
        State("summary-table-scores-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_scores_store(
        stored_weights: dict[str, float],
        table_data: list[dict],
        scores_data: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """
        Update stored scores values when weights update.

        Parameters
        ----------
        stored_weights
            Stored weight values for `table_id`.
        table_data
            Data from `table_id` to be updated.
        scores_data
            Dictionary of scores for each tab.

        Returns
        -------
        dict[str, dict[str, float]]
            List of scoress indexed by table_id.
        """
        if not scores_data:
            scores_data = {}
        # Update scores store. Category table IDs are of form [category]-summary-table
        scores_data[table_id.removesuffix("-summary-table")] = {
            row["MLIP"]: row["Score"] for row in table_data
        }
        return scores_data


def register_weight_callbacks(input_id: str, table_id: str, column: str) -> None:
    """
    Register all callbacks for weight inputs.

    Parameters
    ----------
    input_id
        ID prefix for slider and input box.
    table_id
        ID for table. Also used to identify reset button and weight store.
    column
        Column header corresponding to slider and input box.
    """
    default_weight = 1.0

    @callback(
        Output(f"{table_id}-weight-store", "data", allow_duplicate=True),
        Input(f"{input_id}-slider", "value"),
        Input(f"{input_id}-input", "value"),
        Input(f"{table_id}-reset-button", "n_clicks"),
        State(f"{table_id}-weight-store", "data"),
        prevent_initial_call=True,
    )
    def store_slider_value(
        slider_weight: float,
        input_weight: float,
        n_clicks: int,
        stored_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Store weight values from slider and text input.

        Parameters
        ----------
        slider_weight
            Weight value from slider.
        input_weight
            Weight value from input box.
        n_clicks
            Number of clicks. Variable unused, but Input is required to reset weights.
        stored_weights
            Stored weights dictionary.

        Returns
        -------
        dict[str, float]
            Stored weights for each slider.
        """
        trigger_id = ctx.triggered_id

        if trigger_id == f"{input_id}-slider":
            stored_weights[column] = slider_weight
        elif trigger_id == f"{input_id}-input":
            if input_weight is not None:
                stored_weights[column] = input_weight
            else:
                raise PreventUpdate
        elif trigger_id == f"{table_id}-reset-button":
            stored_weights.update((key, default_weight) for key in stored_weights)
            stored_weights[column] = default_weight
        else:
            raise PreventUpdate

        return stored_weights

    @callback(
        Output(f"{input_id}-input", "value"),
        Output(f"{input_id}-slider", "value"),
        Input(f"{table_id}-weight-store", "data"),
        Input("all-tabs", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_slider_inputs(
        stored_weights: dict[str, float], tabs_value: str
    ) -> tuple[float, float]:
        """
        Sync weight values between slider and text input via Store.

        Parameters
        ----------
        stored_weights
            Stored weight values for each column.
        tabs_value
            Tab name. Variable unused, but required as input to trigger on tab change.

        Returns
        -------
        tuple[float, float]
            Weights to set slider value and text input value.
        """
        return stored_weights[column], stored_weights[column]


def register_metric_weight_box_callbacks(
    input_id: str, table_id: str, metric: str
) -> None:
    """
    Register callbacks for metric weight text boxes (no slider).

    Parameters
    ----------
    input_id
        ID for the numeric input box.
    table_id
        ID for table. Also used to identify reset button and weight store.
    metric
        Metric name corresponding to input.
    """
    default_weight = 1.0

    @callback(
        Output(f"{table_id}-weight-store", "data", allow_duplicate=True),
        Input(input_id, "value"),
        Input(f"{table_id}-reset-button", "n_clicks"),
        State(f"{table_id}-weight-store", "data"),
        prevent_initial_call=True,
    )
    def store_weight_value(
        input_weight: float, n_clicks: int, stored_weights: dict[str, float]
    ):
        trigger_id = ctx.triggered_id
        if trigger_id == input_id:
            if input_weight is None:
                raise PreventUpdate
            stored_weights[metric] = input_weight
        elif trigger_id == f"{table_id}-reset-button":
            stored_weights.update((key, default_weight) for key in stored_weights)
            stored_weights[metric] = default_weight
        else:
            raise PreventUpdate
        return stored_weights

    @callback(
        Output(input_id, "value"),
        Input(f"{table_id}-weight-store", "data"),
        Input("all-tabs", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_weight_input(stored_weights: dict[str, float] | None, _tabs_value: str):
        """Sync the numeric input value from the store (e.g., after Reset)."""
        if not stored_weights:
            raise PreventUpdate
        return stored_weights.get(metric, default_weight)


def register_normalization_callbacks(
    table_id: str,
    metrics: list[str],
    default_ranges: dict[str, tuple[float, float]] | None = None,
    input_suffix: str = "threshold",
) -> None:
    """
    Register callbacks for normalization threshold controls.

    Parameters
    ----------
    table_id
        ID for table to update.
    metrics
        List of metric names that have normalization thresholds.
    """
    # Per-metric store callbacks (simpler and reliable)
    for metric in metrics:

        @callback(
            Output(f"{table_id}-normalization-store", "data", allow_duplicate=True),
            Input(f"{table_id}-{metric}-good-{input_suffix}", "value"),
            Input(f"{table_id}-{metric}-bad-{input_suffix}", "value"),
            Input(f"{table_id}-reset-thresholds-button", "n_clicks"),
            State(f"{table_id}-normalization-store", "data"),
            prevent_initial_call=True,
        )
        def store_threshold_values(
            good_val, bad_val, n_clicks, stored_ranges, metric=metric
        ):
            """Update normalization ranges store for one metric or reset all."""
            trigger_id = ctx.triggered_id
            if stored_ranges is None:
                stored_ranges = (default_ranges or {}).copy()

            if trigger_id == f"{table_id}-reset-thresholds-button":
                return default_ranges or stored_ranges

            # Ensure key exists
            cur_x, cur_y = stored_ranges.get(metric, (0.0, 1.0))

            if trigger_id == f"{table_id}-{metric}-good-{input_suffix}":
                if good_val is None:
                    raise PreventUpdate
                stored_ranges[metric] = (good_val, cur_y)
            elif trigger_id == f"{table_id}-{metric}-bad-{input_suffix}":
                if bad_val is None:
                    raise PreventUpdate
                stored_ranges[metric] = (cur_x, bad_val)
            else:
                raise PreventUpdate

            return stored_ranges

    # Simple callback to toggle display between raw and normalized values (no score calculation)
    @callback(
        Output(f"{table_id}", "data"),
        Output(f"{table_id}", "style_data_conditional"),
        Input(f"{table_id}-normalized-toggle", "value"),
        State(f"{table_id}-raw-data-store", "data"),
        State(f"{table_id}-normalization-store", "data"),
        prevent_initial_call=True,
    )
    def toggle_normalized_display(
        show_normalized: list[str] | None,
        raw_data: list[dict],
        normalization_ranges: dict[str, tuple[float, float]] | None,
    ) -> tuple[list[dict], list[dict]]:
        """Toggle between showing raw and normalized metric values (display only)."""
        if not raw_data:
            raise PreventUpdate

        show_norm_flag = bool(show_normalized) and ("norm" in show_normalized)
        if show_norm_flag and normalization_ranges:
            # Show normalized values
            display_rows = []
            for row in raw_data:
                new_row = row.copy()
                for metric, (x_t, y_t) in normalization_ranges.items():
                    if metric in row:
                        new_row[metric] = normalize_metric(row[metric], x_t, y_t)
                display_rows.append(new_row)
        else:
            # Show raw values
            display_rows = [row.copy() for row in raw_data]

        style = get_table_style(display_rows)
        return display_rows, style

    # Register individual threshold input sync callbacks
    for metric in metrics:

        @callback(
            [
                Output(f"{table_id}-{metric}-good-{input_suffix}", "value"),
                Output(f"{table_id}-{metric}-bad-{input_suffix}", "value"),
            ],
            Input(f"{table_id}-normalization-store", "data"),
            prevent_initial_call=True,
        )
        def sync_threshold_inputs(normalization_ranges, metric=metric):
            """Sync threshold input values with stored ranges."""
            if normalization_ranges and metric in normalization_ranges:
                good_val, bad_val = normalization_ranges[metric]
                return good_val, bad_val
            return 0.0, 1.0


def register_control_table_width_sync(
    results_table_id: str, controls_table_id: str
) -> None:
    """
    Keep the controls table column widths aligned with the results table by
    measuring header widths and applying them to the controls table.
    """
    clientside_callback(
        ClientsideFunction(namespace="mlip", function_name="apply_widths_to_control"),
        [
            Output(controls_table_id, "style_cell_conditional"),
            Output(controls_table_id, "style_table"),
        ],
        [
            Input(results_table_id, "columns"),
            Input(results_table_id, "data_timestamp"),
        ],
        [
            State(controls_table_id, "columns"),
            State(results_table_id, "id"),
            State(controls_table_id, "id"),
        ],
    )


def register_overlay_callbacks(
    table_id: str, metrics: list[str], enable_weight_overlay: bool = False
) -> None:
    """Place overlay Good/Bad inputs at the visual centers of metric columns.

    If enable_weight_overlay is True, also render per-metric weight inputs as overlays.
    """
    # 1) Measure centers clientside and write to centers store
    clientside_callback(
        ClientsideFunction(namespace="mlip", function_name="measure_col_centers"),
        Output(f"{table_id}-centers-store", "data"),
        [
            Input(table_id, "columns"),
            Input(table_id, "data_timestamp"),
        ],
        [
            State(f"{table_id}-metrics-store", "data"),
            State(table_id, "id"),
            State(f"{table_id}-thresholds-overlay", "id"),
        ],
    )

    # 2) Render threshold overlay inputs positioned using centers
    @callback(
        Output(f"{table_id}-thresholds-overlay", "children"),
        Input(f"{table_id}-centers-store", "data"),
        Input(f"{table_id}-normalization-store", "data"),
        prevent_initial_call=True,
    )
    def render_threshold_overlay(centers: dict | None, norm_store: dict | None):
        if not centers:
            raise PreventUpdate

        # Extract grid dimensions for proper centering
        grid_height = centers.get("__gridHeight", 0)
        overlay_offset_top = centers.get("__overlayOffsetTop", 0)

        # print(f"DEBUG: grid_height={grid_height}, overlay_offset_top={overlay_offset_top}")

        # Calculate the center position relative to the grid container
        if grid_height > 0:
            # Calculate correct offset to reach grid center
            grid_center_from_top = grid_height / 2  # Center position from grid top
            distance_to_move_up = (
                overlay_offset_top - grid_center_from_top
            )  # How far to move up from overlay
            center_offset_px = -distance_to_move_up
            # print(f"DEBUG: grid_center_from_top={grid_center_from_top}, distance_to_move_up={distance_to_move_up}, center_offset_px={center_offset_px}")

            top_percent = f"{center_offset_px}px"
            # print(f"DEBUG: final top_percent={top_percent}")
        else:
            # Fallback: position above overlay anchor
            top_percent = "-30px"
            print(f"DEBUG: Using fallback top_percent={top_percent}")

        children = []
        for m in metrics:
            lp = centers.get(m)
            if lp is None:
                continue
            good_def, bad_def = 0.0, 1.0
            if norm_store and m in norm_store:
                good_def, bad_def = norm_store[m]
            grp = html.Div(
                [
                    html.Div(
                        [
                            html.Label(
                                "Good:",
                                style={
                                    "fontSize": "11px",
                                    "width": "40px",
                                    "textAlign": "right",
                                    "marginRight": "4px",
                                    "color": "lightseagreen",
                                },
                            ),
                            dcc.Input(
                                id=f"{table_id}-{m}-good-overlay",
                                type="number",
                                step=0.001,
                                value=good_def,
                                style={"width": "64px", "fontSize": "11px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "alignItems": "center",
                            "marginBottom": "2px",
                        },
                    ),
                    html.Div(
                        [
                            html.Label(
                                "Bad:",
                                style={
                                    "fontSize": "11px",
                                    "width": "40px",
                                    "textAlign": "right",
                                    "marginRight": "4px",
                                    "color": "#dc3545",
                                },
                            ),
                            dcc.Input(
                                id=f"{table_id}-{m}-bad-overlay",
                                type="number",
                                step=0.001,
                                value=bad_def,
                                style={"width": "64px", "fontSize": "11px"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                ],
                style={
                    "position": "absolute",
                    "left": f"{lp}%",
                    "transform": "translate(-50%, -50%)",
                    "top": top_percent,  # Dynamically calculated center position
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-end",  # Align input boxes to the right edge of container
                    "width": "108px",  # Fixed container width so centering is predictable
                    "pointerEvents": "auto",
                },
            )
            children.append(grp)
        return children

    # 3) Render weight overlay inputs positioned using centers (optional)
    if enable_weight_overlay:

        @callback(
            Output(f"{table_id}-weights-overlay", "children"),
            Input(f"{table_id}-centers-store", "data"),
            Input(f"{table_id}-weight-store", "data"),
            prevent_initial_call=True,
        )
        def render_weight_overlay(centers: dict | None, weight_store: dict | None):
            if not centers:
                raise PreventUpdate

            # Extract grid dimensions for proper centering (same calculation as thresholds)
            grid_height = centers.get("__gridHeight", 0)
            overlay_offset_top = centers.get("__overlayOffsetTop", 0)

            # print(f"WEIGHT DEBUG: grid_height={grid_height}, overlay_offset_top={overlay_offset_top}")

            # Calculate the center position relative to the grid container
            if grid_height > 0:
                # Calculate correct offset to reach grid center
                grid_center_from_top = grid_height / 2  # Center position from grid top
                distance_to_move_up = (
                    overlay_offset_top - grid_center_from_top
                )  # How far to move up from overlay
                # Adjust for single weight box vs stacked threshold boxes
                # Weight box needs additional offset to be truly centered
                weight_adjustment = 11  # Same as thresholds to start
                center_offset_px = -distance_to_move_up + weight_adjustment
                top_percent = f"{center_offset_px}px"
                # print(f"WEIGHT DEBUG: distance_to_move_up={distance_to_move_up}, center_offset_px={center_offset_px}, top_percent={top_percent}")
            else:
                # Fallback: position above overlay anchor
                top_percent = "-30px"
                print(f"WEIGHT DEBUG: Using fallback top_percent={top_percent}")

            children = []
            for m in metrics:
                lp = centers.get(m)
                if lp is None:
                    continue
                w_def = 1.0
                if weight_store and m in weight_store:
                    w_def = weight_store[m]
                grp = html.Div(
                    [
                        html.Div(
                            [
                                dcc.Input(
                                    id=f"{table_id}-{m}-w-overlay",
                                    type="number",
                                    step=0.01,
                                    value=w_def,
                                    style={"width": "64px", "fontSize": "11px"},
                                ),
                            ],
                            style={
                                "display": "flex",
                                "alignItems": "center",
                                "justifyContent": "center",
                            },
                        ),
                    ],
                    style={
                        "position": "absolute",
                        "left": f"{lp}%",
                        "transform": "translate(-50%, -50%)",
                        "top": top_percent,  # Dynamically calculated center position
                        "display": "flex",
                        "flexDirection": "column",
                        "alignItems": "flex-end",
                        "width": "108px",
                        "pointerEvents": "auto",
                    },
                )
                children.append(grp)
            return children

        # 3b) Reset all weights to defaults when Reset is pressed (overlay mode)
        @callback(
            Output(f"{table_id}-weight-store", "data", allow_duplicate=True),
            Input(f"{table_id}-reset-button", "n_clicks"),
            State(f"{table_id}-weight-store", "data"),
            prevent_initial_call=True,
        )
        def reset_weights_overlay(n_clicks, w_store):
            if not n_clicks:
                raise PreventUpdate
            # Reset all existing keys to 1.0
            if not w_store:
                return dict.fromkeys(metrics, 1.0)
            return dict.fromkeys(w_store.keys(), 1.0)

    # 4) Write overlay inputs back to normalization store
    for m in metrics:

        @callback(
            Output(f"{table_id}-normalization-store", "data", allow_duplicate=True),
            Input(f"{table_id}-{m}-good-overlay", "value"),
            Input(f"{table_id}-{m}-bad-overlay", "value"),
            State(f"{table_id}-normalization-store", "data"),
            prevent_initial_call=True,
        )
        def sync_norm_store(good_val, bad_val, store, metric=m):
            if store is None:
                store = {}
            if good_val is None and bad_val is None:
                raise PreventUpdate
            cur_good, cur_bad = store.get(metric, (0.0, 1.0))
            new_good = cur_good if good_val is None else good_val
            new_bad = cur_bad if bad_val is None else bad_val
            store[metric] = (new_good, new_bad)
            return store

        if enable_weight_overlay:

            @callback(
                Output(f"{table_id}-weight-store", "data", allow_duplicate=True),
                Input(f"{table_id}-{m}-w-overlay", "value"),
                State(f"{table_id}-weight-store", "data"),
                prevent_initial_call=True,
            )
            def sync_weight_overlay(w_val, w_store, metric=m):
                if w_store is None:
                    w_store = {}
                if w_val is None:
                    raise PreventUpdate
                w_store[metric] = w_val
                return w_store
