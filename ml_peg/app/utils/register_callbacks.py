"""Register callbacks relating to interative weights."""

from __future__ import annotations

from dash import Input, Output, State, callback, ctx
from dash.exceptions import PreventUpdate

from ml_peg.analysis.utils.utils import (
    calc_ranks,
    calc_scores,
    get_table_style,
    normalize_metric,
)
from ml_peg.app.utils.utils import clean_thresholds, clean_weights


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
        If `True`, also watch the per-metric normalization store and recompute
        scores using the configured thresholds. This should only be used for benchmark
        tables.
    """

    def _calc_scored_rows(
        raw_rows: list[dict],
        weights: dict[str, float],
        threshold_pairs: dict[str, tuple[float, float]] | None = None,
    ) -> list[dict]:
        """
        Return scored rows after applying weights and optional thresholds.

        Parameters
        ----------
        raw_rows : list[dict]
            Metric rows to be scored (each row mutated on a copy).
        weights : dict[str, float]
            Mapping of metric name to weight value.
        threshold_pairs : dict[str, tuple[float, float]] | None, optional
            Optional normalisation thresholds keyed by metric name.

        Returns
        -------
        list[dict]
            Rows with fresh ``Score`` and ``Rank`` entries.
        """
        working = [row.copy() for row in raw_rows]
        working = calc_scores(
            metrics_data=working, thresholds=threshold_pairs, weights=weights
        )
        return calc_ranks(working)

    def _materialize_display_rows(
        raw_rows: list[dict],
        scored_rows: list[dict],
        threshold_pairs: dict[str, tuple[float, float]] | None,
        toggle_value: list[str] | None,
    ) -> list[dict]:
        """
        Build table display rows, normalising metrics when requested.

        Parameters
        ----------
        raw_rows : list[dict]
            Original raw metric values.
        scored_rows : list[dict]
            Rows produced by :func:`_calc_scored_rows`.
        threshold_pairs : dict[str, tuple[float, float]] | None
            Normalisation thresholds, or ``None`` for raw metrics.
        toggle_value : list[str] | None
            Current state of the “Show normalized values” toggle.

        Returns
        -------
        list[dict]
            Rows to render in the DataTable.
        """
        show_normalized = bool(toggle_value) and ("norm" in (toggle_value or []))
        if not (show_normalized and threshold_pairs):
            return [row.copy() for row in scored_rows]

        score_map = {
            row.get("MLIP"): row for row in scored_rows if row.get("MLIP") is not None
        }
        display_rows: list[dict] = []
        for row in raw_rows:
            display_row = row.copy()
            mlip = display_row.get("MLIP")
            for metric, (good, bad) in threshold_pairs.items():
                if metric not in display_row:
                    continue
                try:
                    metric_val = float(display_row[metric])
                except (TypeError, ValueError):
                    continue
                display_row[metric] = normalize_metric(metric_val, good, bad)
            if mlip in score_map:
                display_row["Score"] = score_map[mlip].get("Score")
                display_row["Rank"] = score_map[mlip].get("Rank")
            display_rows.append(display_row)
        return display_rows

    if use_threshold_store:

        @callback(
            Output(table_id, "data", allow_duplicate=True),
            Output(table_id, "style_data_conditional", allow_duplicate=True),
            Output(f"{table_id}-computed-store", "data", allow_duplicate=True),
            Input(f"{table_id}-weight-store", "data"),
            Input(f"{table_id}-thresholds-store", "data"),
            Input("all-tabs", "value"),
            Input(f"{table_id}-normalized-toggle", "value"),
            State(f"{table_id}-raw-data-store", "data"),
            State(f"{table_id}-computed-store", "data"),
            prevent_initial_call="initial_duplicate",
        )
        def update_table_scores_with_thresholds(
            stored_weights: dict[str, float] | None,
            threshold_store: dict | None,
            _tabs_value: str,
            toggle_value: list[str] | None,
            raw_data: list[dict] | None,
            computed_store: list[dict] | None,
        ) -> tuple[list[dict], list[dict], list[dict]]:
            if not raw_data:
                raise PreventUpdate

            threshold_pairs = clean_thresholds(threshold_store)
            weights = clean_weights(stored_weights)
            trigger_id = ctx.triggered_id

            if (
                trigger_id in ("all-tabs", f"{table_id}-normalized-toggle")
                and computed_store
            ):
                # Tab switches and toggle flips reuse the cached scored rows rather than
                # recalculating scores, we only re-score when weights/thresholds change.
                display_rows = _materialize_display_rows(
                    raw_data, computed_store, threshold_pairs, toggle_value
                )
                style = get_table_style(display_rows)
                return display_rows, style, computed_store

            scored_rows = _calc_scored_rows(raw_data, weights, threshold_pairs)
            display_rows = _materialize_display_rows(
                raw_data, scored_rows, threshold_pairs, toggle_value
            )
            style = get_table_style(display_rows)
            return display_rows, style, scored_rows

    else:

        @callback(
            Output(table_id, "data", allow_duplicate=True),
            Output(table_id, "style_data_conditional", allow_duplicate=True),
            Output(f"{table_id}-computed-store", "data", allow_duplicate=True),
            Input(f"{table_id}-weight-store", "data"),
            Input("all-tabs", "value"),
            State(table_id, "data"),
            State(f"{table_id}-computed-store", "data"),
            prevent_initial_call="initial_duplicate",
        )
        def update_table_scores(
            stored_weights: dict[str, float] | None,
            _tabs_value: str,
            table_data: list[dict] | None,
            computed_store: list[dict] | None,
        ) -> tuple[list[dict], list[dict], list[dict]]:
            trigger_id = ctx.triggered_id

            if trigger_id == "all-tabs" and computed_store:
                # When returning to the tab we show the last scored rows instantly.
                display_rows = [row.copy() for row in computed_store]
                style = get_table_style(display_rows)
                return display_rows, style, computed_store

            if not table_data:
                raise PreventUpdate

            weights = clean_weights(stored_weights)
            scored_rows = _calc_scored_rows(table_data, weights, None)
            style = get_table_style(scored_rows)
            return scored_rows, style, scored_rows

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
    benchmark_table_id: str,
    category_table_id: str,
    benchmark_column: str,
    use_threshold_store: bool = False,
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
    use_threshold_store
        Whether the benchmark table exposes a normalization store for metrics.
    """
    _ = use_threshold_store  # cached rows handle normalization
    # flag kept for compatibility with existing call sites

    @callback(
        Output(category_table_id, "data", allow_duplicate=True),
        Output(category_table_id, "style_data_conditional", allow_duplicate=True),
        Output(f"{category_table_id}-computed-store", "data", allow_duplicate=True),
        Input(f"{benchmark_table_id}-computed-store", "data"),
        Input("all-tabs", "value"),
        State(category_table_id, "data"),
        State(f"{category_table_id}-weight-store", "data"),
        State(f"{category_table_id}-computed-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_category_from_benchmark(
        benchmark_computed_store: list[dict] | None,
        _tabs_value: str,
        category_data: list[dict] | None,
        category_weights: dict[str, float] | None,
        category_computed_store: list[dict] | None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Update a category summary table from cached benchmark scores.

        Parameters
        ----------
        benchmark_computed_store
            Latest scored benchmark rows emitted by the benchmark table.
        _tabs_value
            Current tab identifier (unused, required to trigger on tab change).
        category_data
            Existing category table rows shown to the user.
        category_weights
            Stored weights for the category summary metrics.
        category_computed_store
            Cached scored rows for the category summary.

        Returns
        -------
        tuple[list[dict], list[dict], list[dict]]
            Updated table data, updated style, and refreshed cached rows.
        """
        if category_data is None:
            raise PreventUpdate

        triggered = ctx.triggered_id
        base_rows = category_computed_store or category_data
        if triggered == "all-tabs":
            if not base_rows:
                raise PreventUpdate
            display_rows = [row.copy() for row in base_rows]
            style = get_table_style(display_rows)
            return display_rows, style, display_rows

        if benchmark_computed_store is None:
            raise PreventUpdate

        benchmark_scores = {
            row.get("MLIP"): row.get("Score")
            for row in benchmark_computed_store
            if row.get("MLIP") is not None and row.get("Score") is not None
        }

        working_rows = [row.copy() for row in base_rows]
        for row in working_rows:
            mlip = row.get("MLIP")
            if mlip in benchmark_scores:
                row[benchmark_column] = benchmark_scores[mlip]

        weights = clean_weights(category_weights)
        scored_rows = calc_scores(working_rows, weights)
        scored_rows = calc_ranks(scored_rows)
        style = get_table_style(scored_rows)
        return scored_rows, style, scored_rows


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
    default_thresholds: dict[str, tuple[float, float]] | None = None,
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
    default_thresholds
        Default threshold mapping used for resets. Default is `None`.
    register_toggle
        Whether to register the raw/normalized display toggle callback. Default is
        `True`.
    """
    input_suffix = "threshold"

    if default_thresholds:
        default_thresholds = {
            metric: (float(bounds[0]), float(bounds[1]))
            for metric, bounds in default_thresholds.items()
        }

    # Per-metric store callbacks (simpler and reliable)
    for metric in metrics:

        @callback(
            Output(f"{table_id}-thresholds-store", "data", allow_duplicate=True),
            Input(f"{table_id}-{metric}-good-{input_suffix}", "value"),
            Input(f"{table_id}-{metric}-bad-{input_suffix}", "value"),
            Input(f"{table_id}-reset-thresholds-button", "n_clicks"),
            State(f"{table_id}-thresholds-store", "data"),
            prevent_initial_call=True,
        )
        def store_threshold_values(
            good_val, bad_val, n_clicks, stored_thresholds, metric=metric
        ):
            """Update normalization thresholds store for one metric or reset all."""
            trigger_id = ctx.triggered_id
            if stored_thresholds is None:
                stored_thresholds = (default_thresholds or {}).copy()

            # Reset to defaults is specified via reset button
            if trigger_id == f"{table_id}-reset-thresholds-button":
                if default_thresholds:
                    return {
                        key: (float(bounds[0]), float(bounds[1]))
                        for key, bounds in default_thresholds.items()
                    }
                return stored_thresholds

            # Ensure key exists
            good_threshhold, bad_threshold = stored_thresholds.get(metric, (None, None))

            # Update thresholds from input boxes
            if trigger_id == f"{table_id}-{metric}-good-{input_suffix}":
                if good_val is None or bad_threshold is None:
                    raise PreventUpdate
                stored_thresholds[metric] = (float(good_val), float(bad_threshold))

            elif trigger_id == f"{table_id}-{metric}-bad-{input_suffix}":
                if bad_val is None or good_threshhold is None:
                    raise PreventUpdate
                stored_thresholds[metric] = (float(good_threshhold), float(bad_val))
            else:
                raise PreventUpdate

            return stored_thresholds

    if register_toggle:
        # Toggle display between raw and normalized values without recomputing scores
        @callback(
            Output(f"{table_id}", "data", allow_duplicate=True),
            Output(f"{table_id}", "style_data_conditional", allow_duplicate=True),
            Input(f"{table_id}-normalized-toggle", "value"),
            State(f"{table_id}-raw-data-store", "data"),
            State(f"{table_id}-thresholds-store", "data"),
            prevent_initial_call=True,
        )
        def toggle_normalized_display(
            show_normalized: list[str] | None,
            raw_data: list[dict],
            thresholds: dict[str, tuple[float, float]] | None,
        ) -> tuple[list[dict], list[dict]]:
            """Toggle between raw and normalised metric values for display only."""
            if not raw_data:
                raise PreventUpdate

            show_norm_flag = bool(show_normalized) and ("norm" in show_normalized)
            thresholds = clean_thresholds(thresholds)

            if show_norm_flag and thresholds:
                # Show normalied values
                display_rows = []
                for row in raw_data:
                    new_row = row.copy()
                    for metric, (x_t, y_t) in thresholds.items():
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
            Output(f"{table_id}-{metric}-good-{input_suffix}", "value"),
            Output(f"{table_id}-{metric}-bad-{input_suffix}", "value"),
            Input(f"{table_id}-thresholds-store", "data"),
            prevent_initial_call=True,
        )
        def sync_threshold_inputs(thresholds, metric=metric):
            """Sync threshold input values with stored thresholds."""
            if thresholds and metric in thresholds:
                good_val, bad_val = thresholds[metric]
                return good_val, bad_val
            raise PreventUpdate
