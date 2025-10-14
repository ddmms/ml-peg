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
    calc_normalized_scores,
    calc_ranks,
    calc_scores,
    get_table_style,
    normalize_metric,
)


def _coerce_weights(raw_weights: dict[str, float] | None) -> dict[str, float]:
    """
    Convert potentially non-numeric weight values into floats.

    Parameters
    ----------
    raw_weights
        Mapping from metric name to supplied weight.

    Returns
    -------
    dict[str, float]
        Dictionary containing only numeric weight values.
    """
    if not raw_weights:
        return {}
    coerced: dict[str, float] = {}
    for metric, value in raw_weights.items():
        try:
            coerced[metric] = float(value)
        except (TypeError, ValueError):
            continue
    return coerced


def _coerce_threshold_map(
    raw_thresholds: dict[str, dict[str, float] | list[float] | tuple[float, float]]
    | None,
) -> dict[str, tuple[float, float]]:
    """
    Convert raw threshold metadata into ``(good, bad)`` float tuples.

    Parameters
    ----------
    raw_thresholds
        Mapping supplied by Dash stores containing threshold info.

    Returns
    -------
    dict[str, tuple[float, float]]
        Cleaned threshold mapping keyed by metric name.
    """
    if not raw_thresholds:
        return {}

    cleaned: dict[str, tuple[float, float]] = {}
    for metric, bounds in raw_thresholds.items():
        try:
            if isinstance(bounds, dict):
                good_val = float(bounds["good"])
                bad_val = float(bounds["bad"])
            elif isinstance(bounds, list | tuple) and len(bounds) == 2:
                good_val = float(bounds[0])
                bad_val = float(bounds[1])
            else:
                continue
        except (KeyError, TypeError, ValueError):
            continue

        cleaned[metric] = (good_val, bad_val)

    return cleaned


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


def register_tab_table_callbacks(
    table_id: str, use_threshold_store: bool = False
) -> None:
    """
    Register callback to update table scores/rankings when stored values change.

    Parameters
    ----------
    table_id
        ID for table to update.
    use_threshold_store
        If True, also watch the per-metric normalization store and recompute
        scores using the configured thresholds.
    """
    if use_threshold_store:

        @callback(
            Output(table_id, "data", allow_duplicate=True),
            Output(table_id, "style_data_conditional", allow_duplicate=True),
            Input(f"{table_id}-weight-store", "data"),
            Input(f"{table_id}-normalization-store", "data"),
            Input("all-tabs", "value"),
            Input(f"{table_id}-normalized-toggle", "value"),
            State(f"{table_id}-raw-data-store", "data"),
            prevent_initial_call="initial_duplicate",
        )
        def update_table_scores_with_thresholds(
            stored_weights: dict[str, float] | None,
            threshold_store: dict | None,
            _tabs_value: str,
            toggle_value: list[str] | None,
            raw_data: list[dict] | None,
        ) -> tuple[list[dict], list[dict]]:
            if not raw_data:
                raise PreventUpdate

            weights = _coerce_weights(stored_weights)
            threshold_pairs = _coerce_threshold_map(threshold_store)
            base_rows = [row.copy() for row in raw_data]

            if threshold_pairs:
                scored_rows = calc_normalized_scores(
                    base_rows, threshold_pairs, weights
                )
            else:
                scored_rows = calc_scores(base_rows, weights)

            scored_rows = calc_ranks(scored_rows)

            show_normalized = bool(toggle_value) and ("norm" in toggle_value)
            if show_normalized and threshold_pairs:
                score_map = {
                    row.get("MLIP"): row
                    for row in scored_rows
                    if row.get("MLIP") is not None
                }
                display_rows: list[dict] = []
                for row in raw_data:
                    display_row = row.copy()
                    mlip = display_row.get("MLIP")
                    for metric, (x_t, y_t) in threshold_pairs.items():
                        if metric in display_row:
                            try:
                                metric_val = float(display_row[metric])
                            except (TypeError, ValueError):
                                continue
                            display_row[metric] = normalize_metric(metric_val, x_t, y_t)
                    if mlip in score_map:
                        display_row["Score"] = score_map[mlip].get("Score")
                        display_row["Rank"] = score_map[mlip].get("Rank")
                    display_rows.append(display_row)
                style = get_table_style(display_rows)
                return display_rows, style

            style = get_table_style(scored_rows)
            return scored_rows, style

    else:

        @callback(
            Output(table_id, "data", allow_duplicate=True),
            Output(table_id, "style_data_conditional", allow_duplicate=True),
            Input(f"{table_id}-weight-store", "data"),
            Input("all-tabs", "value"),
            State(table_id, "data"),
            prevent_initial_call="initial_duplicate",
        )
        def update_table_scores(
            stored_weights: dict[str, float] | None,
            _tabs_value: str,
            table_data: list[dict] | None,
        ) -> tuple[list[dict], list[dict]]:
            if not table_data:
                raise PreventUpdate

            weights = _coerce_weights(stored_weights)
            data_copy = calc_scores([row.copy() for row in table_data], weights)
            data_copy = calc_ranks(data_copy)
            style = get_table_style(data_copy)
            return data_copy, style

    @callback(
        Output("summary-table-scores-store", "data", allow_duplicate=True),
        Input(table_id, "data"),
        State("summary-table-scores-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_scores_store(
        table_data: list[dict],
        scores_data: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """
        Update stored scores values when weights update.

        Parameters
        ----------
        table_data
            Data from `table_id` to be updated.
        scores_data
            Dictionary of scores for each tab.

        Returns
        -------
        dict[str, dict[str, float]]
            List of scoress indexed by table_id.
        """
        # Only category summary tables should write to the global store
        if not table_id.endswith("-summary-table"):
            return scores_data

        if not scores_data:
            scores_data = {}
        # Update scores store. Category table IDs are of form [category]-summary-table
        scores_data[table_id.removesuffix("-summary-table")] = {
            row["MLIP"]: row["Score"] for row in table_data
        }
        return scores_data


def register_benchmark_to_category_callback(
    benchmark_table_id: str, category_table_id: str, benchmark_column: str
) -> None:
    """
    Propagate a benchmark table's Score into its category summary table column.

    Parameters
    ----------
    benchmark_table_id
        ID of the benchmark test table (e.g., "OC157-table").
    category_table_id
        ID of the category summary table (e.g., "Surfaces-summary-table").
    benchmark_column
        Column name in the category summary table corresponding to the benchmark.
    """

    @callback(
        Output(category_table_id, "data", allow_duplicate=True),
        Output(category_table_id, "style_data_conditional", allow_duplicate=True),
        Input(benchmark_table_id, "data"),
        Input(f"{benchmark_table_id}-weight-store", "data"),
        Input("all-tabs", "value"),
        State(category_table_id, "data"),
        State(f"{category_table_id}-weight-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_category_from_benchmark(
        benchmark_data: list[dict] | None,
        benchmark_weights: dict[str, float] | None,
        tabs_value: str,
        category_data: list[dict],
        category_weights: dict[str, float] | None,
    ) -> tuple[list[dict], list[dict]]:
        """
        Update category summary from a benchmark table.

        Parameters
        ----------
        benchmark_data
            Latest data rows emitted by the benchmark table.
        benchmark_weights
            Metric weights configured for the benchmark table.
        tabs_value
            Currently selected tab value (unused, used as a trigger).
        category_data
            Existing category summary rows to update.
        category_weights
            Metric weights configured for the category summary.

        Returns
        -------
        tuple[list[dict], list[dict]]
            Updated category table data and style tuple.
        """
        # if not benchmark_data or not category_data:
        #     raise PreventUpdate

        if not benchmark_data or not category_data:
            raise PreventUpdate

        triggered = ctx.triggered_id

        if triggered == f"{benchmark_table_id}-weight-store":
            weights = _coerce_weights(benchmark_weights)
            recomputed = calc_scores([row.copy() for row in benchmark_data], weights)
            benchmark_scores = {
                row.get("MLIP"): row.get("Score")
                for row in recomputed
                if row.get("MLIP") is not None
            }
        else:
            benchmark_scores = {
                row.get("MLIP"): row.get("Score")
                for row in benchmark_data
                if row.get("MLIP") is not None
            }

        # Inject into the appropriate column for each MLIP
        updated_category = []
        for row in category_data:
            mlip = row.get("MLIP")
            new_row = row.copy()
            if mlip in benchmark_scores and benchmark_scores[mlip] is not None:
                new_row[benchmark_column] = benchmark_scores[mlip]
            updated_category.append(new_row)

        # Recompute category Score and Rank using its existing weights
        weights = _coerce_weights(category_weights)
        updated_category = calc_scores(updated_category, weights)
        updated_category = calc_ranks(updated_category)
        style = get_table_style(updated_category)

        return updated_category, style


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
        Input(f"{input_id}-input", "value"),
        Input(f"{table_id}-reset-button", "n_clicks"),
        State(f"{table_id}-weight-store", "data"),
        prevent_initial_call=True,
    )
    def store_input_value(
        input_weight: float | None,
        n_clicks: int,
        stored_weights: dict[str, float],
    ) -> dict[str, float]:
        """
        Store weight values from the text input.

        Parameters
        ----------
        input_weight
            Weight value from input box.
        n_clicks
            Number of clicks. Variable unused, but Input is required to reset weights.
        stored_weights
            Stored weights dictionary.

        Returns
        -------
        dict[str, float]
            Stored weights for each input box.
        """
        trigger_id = ctx.triggered_id

        if trigger_id == f"{input_id}-input":
            if input_weight is None:
                raise PreventUpdate
            stored_weights[column] = input_weight
        elif trigger_id == f"{table_id}-reset-button":
            stored_weights.update((key, default_weight) for key in stored_weights)
        else:
            raise PreventUpdate

        return stored_weights

    @callback(
        Output(f"{input_id}-input", "value"),
        Input(f"{table_id}-weight-store", "data"),
        Input("all-tabs", "value"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_inputs(stored_weights: dict[str, float], tabs_value: str) -> float:
        """
        Sync weight values between the text input and Store.

        Parameters
        ----------
        stored_weights
            Stored weight values for each column.
        tabs_value
            Tab name. Variable unused, but required as input to trigger on tab change.

        Returns
        -------
        float
            Weight to set text input value.
        """
        return stored_weights[column]


def register_normalization_callbacks(
    table_id: str,
    metrics: list[str],
    default_ranges: dict[str, tuple[float, float]] | None = None,
    input_suffix: str = "threshold",
    register_toggle: bool = True,
) -> None:
    """
    Register callbacks for normalization threshold controls.

    Parameters
    ----------
    table_id
        ID for table to update.
    metrics
        List of metric names that have normalization thresholds.
    default_ranges
        Optional default threshold mapping used for resets.
    input_suffix
        Suffix used when constructing per-metric input IDs.
    register_toggle
        When True, register the raw/normalized display toggle callback.
    """
    if default_ranges:
        default_ranges = {
            metric: (float(bounds[0]), float(bounds[1]))
            for metric, bounds in default_ranges.items()
        }

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
                if default_ranges:
                    return {
                        key: (float(bounds[0]), float(bounds[1]))
                        for key, bounds in default_ranges.items()
                    }
                return stored_ranges

            # Ensure key exists
            cur_x, cur_y = stored_ranges.get(metric, (0.0, 1.0))

            if trigger_id == f"{table_id}-{metric}-good-{input_suffix}":
                if good_val is None:
                    raise PreventUpdate
                stored_ranges[metric] = (float(good_val), float(cur_y))
            elif trigger_id == f"{table_id}-{metric}-bad-{input_suffix}":
                if bad_val is None:
                    raise PreventUpdate
                stored_ranges[metric] = (float(cur_x), float(bad_val))
            else:
                raise PreventUpdate

            return stored_ranges

    if register_toggle:
        # Toggle display between raw and normalized values without recomputing scores
        @callback(
            Output(f"{table_id}", "data", allow_duplicate=True),
            Output(f"{table_id}", "style_data_conditional", allow_duplicate=True),
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
            """Toggle between raw and normalized metric values for display only."""
            if not raw_data:
                raise PreventUpdate

            show_norm_flag = bool(show_normalized) and ("norm" in show_normalized)
            ranges = _coerce_threshold_map(normalization_ranges)

            if show_norm_flag and ranges:
                # Show normalized values
                display_rows = []
                for row in raw_data:
                    new_row = row.copy()
                    for metric, (x_t, y_t) in ranges.items():
                        if metric in row:
                            try:
                                metric_val = float(row[metric])
                            except (TypeError, ValueError):
                                continue
                            new_row[metric] = normalize_metric(metric_val, x_t, y_t)
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
    Keep a controls table aligned with its companion results table.

    Parameters
    ----------
    results_table_id
        Component ID of the primary results table.
    controls_table_id
        Component ID of the companion controls table.
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
    """
    Place overlay Good/Bad inputs at the visual centers of metric columns.

    Parameters
    ----------
    table_id
        ID of the results table that owns the overlays.
    metrics
        Ordered list of metric column identifiers.
    enable_weight_overlay
        When True, render per-metric weight inputs alongside thresholds.
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
        """
        Render overlay threshold inputs aligned to column centers.

        Parameters
        ----------
        centers
            Mapping of metrics to their percentage center positions.
        norm_store
            Stored normalization ranges for the table.

        Returns
        -------
        list[html.Div]
            Overlay components positioned above the table.
        """
        if not centers:
            raise PreventUpdate

        # Extract grid dimensions for proper centering
        grid_height = centers.get("__gridHeight", 0)
        overlay_offset_top = centers.get("__overlayOffsetTop", 0)

        # Calculate the center position relative to the grid container
        if grid_height > 0:
            # Calculate correct offset to reach grid center
            grid_center_from_top = grid_height / 2  # Center position from grid top
            distance_to_move_up = (
                overlay_offset_top - grid_center_from_top
            )  # How far to move up from overlay
            center_offset_px = -distance_to_move_up
            top_percent = f"{center_offset_px}px"
        else:
            # Fallback: position above overlay anchor
            top_percent = "-30px"

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
                    "top": top_percent,
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-end",
                    "width": "108px",
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
            """
            Render overlay weight inputs aligned to column centers.

            Parameters
            ----------
            centers
                Mapping of metrics to their percentage center positions.
            weight_store
                Stored per-metric weight values.

            Returns
            -------
            list[html.Div]
                Overlay components positioned above the table.
            """
            if not centers:
                raise PreventUpdate

            # Extract grid dimensions for proper centering (same as thresholds)
            grid_height = centers.get("__gridHeight", 0)
            overlay_offset_top = centers.get("__overlayOffsetTop", 0)

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
            else:
                # Fallback: position above overlay anchor
                top_percent = "-30px"

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
