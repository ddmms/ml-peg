"""Register callbacks relating to interative weights."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from dash import Input, Output, State, callback, ctx
from dash.exceptions import PreventUpdate

from ml_peg.analysis.utils.utils import (
    calc_metric_scores,
    calc_table_scores,
    get_table_style,
    update_score_style,
)
from ml_peg.app.utils.utils import (
    Thresholds,
    build_level_of_theory_warnings,
    clean_thresholds,
    filter_rows_by_models,
    format_metric_columns,
    format_tooltip_headers,
    get_scores,
)


def apply_level_of_theory_warnings(
    rows: list[dict[str, Any]],
    base_style: list[dict[str, Any]],
    model_levels: dict[str, str | None] | None = None,
    metric_levels: dict[str, str | None] | None = None,
    model_configs: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Append level-of-theory warnings and tooltip rows to existing table styles.

    Parameters
    ----------
    rows
        Table rows currently being displayed.
    base_style
        Existing conditional style rules for those rows.
    model_levels
        Mapping from model name to its level-of-theory metadata.
    metric_levels
        Mapping from metric column name to its benchmark level-of-theory metadata.
    model_configs
        Optional configuration metadata for each model.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        Augmented style rules and tooltip rows.
    """
    warning_styles, tooltip_rows = build_level_of_theory_warnings(
        rows, model_levels, metric_levels, model_configs
    )
    style_with_warnings = base_style + warning_styles
    tooltip_data = tooltip_rows if tooltip_rows else [{} for _ in rows]
    return style_with_warnings, tooltip_data


def register_summary_table_callbacks(
    initial_rows: list[dict] | None = None,
    model_levels: dict[str, str | None] | None = None,
    metric_levels: dict[str, str | None] | None = None,
    model_configs: dict[str, Any] | None = None,
) -> None:
    """
    Register callbacks to update summary table.

    Parameters
    ----------
    initial_rows
        Starting full summary rows. These are used to seed the summary-table
        cache before any callback has written updated rows to it.
    model_levels
        Mapping from model name to its level of theory badge text.
    metric_levels
        Mapping from metric column name to its level of theory badge text.
    model_configs
        Optional metadata/configuration dictionary for each model.
    """
    default_rows = deepcopy(initial_rows) if initial_rows else []

    @callback(
        Output("summary-table-computed-store", "data", allow_duplicate=True),
        Input("summary-table-scores-store", "data"),
        Input("summary-table-weight-store", "data"),
        State("summary-table-computed-store", "data"),
        prevent_initial_call=True,
    )
    def update_summary_computed_store(
        stored_scores: dict[str, dict[str, float]] | None,
        stored_weights: dict[str, float],
        computed_store: list[dict] | None,
    ) -> list[dict]:
        """
        Update cached summary rows when category scores or summary weights change.

        Parameters
        ----------
        stored_scores
            Latest category-summary scores, grouped by category column.
        stored_weights
            Current weights applied to the overall summary table.
        computed_store
            Cached full summary rows. When available, these are updated and
            reused as the source of truth.

        Returns
        -------
        list[dict]
            Updated unfiltered rows written back to the cached summary store.
        """
        source_data = deepcopy(computed_store or default_rows)
        if not source_data:
            raise PreventUpdate

        # Update table from stored scores
        if stored_scores:
            for row in source_data:
                for tab, values in stored_scores.items():
                    if row["MLIP"] in values:
                        row[tab] = values[row["MLIP"]]

        updated_rows, _ = update_score_style(source_data, stored_weights)
        return updated_rows

    @callback(
        Output("summary-table", "data", allow_duplicate=True),
        Output("summary-table", "style_data_conditional", allow_duplicate=True),
        Output("summary-table", "tooltip_data", allow_duplicate=True),
        Input("selected-models-store", "data"),
        Input("summary-table-computed-store", "data"),
        Input("app-location", "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_summary_table(
        selected_models: list[str] | None,
        computed_store: list[dict] | None,
        _pathname: str,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Sync the visible summary table from cached unfiltered rows.

        Parameters
        ----------
        selected_models
            Models currently selected in the global model filter.
        computed_store
            Cached full summary rows for the overall summary table.
        _pathname
            Current pathname. Included so the visible table refreshes when the
            summary page is opened.

        Returns
        -------
        tuple[list[dict], list[dict], list[dict]]
            Filtered rows, style rules, and tooltip rows for the visible table.
        """
        if not computed_store:
            raise PreventUpdate

        filtered_rows = filter_rows_by_models(computed_store, selected_models)
        base_style = get_table_style(filtered_rows) if filtered_rows else []
        style_with_warnings, tooltip_data = apply_level_of_theory_warnings(
            filtered_rows,
            base_style,
            model_levels=model_levels,
            metric_levels=metric_levels,
            model_configs=model_configs,
        )
        return filtered_rows, style_with_warnings, tooltip_data


def register_category_table_callbacks(
    table_id: str,
    use_thresholds: bool = False,
    model_levels: dict[str, str | None] | None = None,
    metric_levels: dict[str, str | None] | None = None,
    model_configs: dict[str, Any] | None = None,
) -> None:
    """
    Register callback to update table scores when stored values change.

    Parameters
    ----------
    table_id
        ID for table to update.
    use_thresholds
        If `True`, also watch the per-metric normalization store and recompute
        scores using the configured thresholds. This should only be used for benchmark
        tables.
    model_levels
        Mapping of model name -> level of theory metadata.
    metric_levels
        Mapping of metric name -> level of theory metadata.
    model_configs
        Optional configuration metadata for each model.
    """
    # Benchmark tables
    if use_thresholds:

        @callback(
            Output(table_id, "data", allow_duplicate=True),
            Output(table_id, "style_data_conditional", allow_duplicate=True),
            Output(table_id, "tooltip_data", allow_duplicate=True),
            Output(table_id, "columns", allow_duplicate=True),
            Output(table_id, "tooltip_header", allow_duplicate=True),
            Output(f"{table_id}-computed-store", "data", allow_duplicate=True),
            Output(f"{table_id}-raw-data-store", "data"),
            Input(f"{table_id}-weight-store", "data"),
            Input(f"{table_id}-thresholds-store", "data"),
            Input("app-location", "pathname"),
            Input(f"{table_id}-normalized-toggle", "value"),
            Input("selected-models-store", "data"),
            State(f"{table_id}-raw-data-store", "data"),
            State(f"{table_id}-computed-store", "data"),
            State(f"{table_id}-raw-tooltip-store", "data"),
            State(table_id, "columns"),
            prevent_initial_call="initial_duplicate",
        )
        def update_benchmark_table_scores(
            stored_weights: dict[str, float] | None,
            stored_threshold: dict | None,
            _pathname: str,
            toggle_value: list[str] | None,
            selected_models: list[str] | None,
            stored_raw_data: list[dict] | None,
            stored_computed_data: list[dict] | None,
            raw_tooltips: dict[str, str] | None,
            current_columns: list[dict] | None,
        ) -> tuple[
            list[dict],
            list[dict],
            list[dict],
            list[dict],
            dict[str, str] | None,
            list[dict],
            list[dict],
        ]:
            """
            Update table when stored weights/threshold change, or page is changed.

            Parameters
            ----------
            stored_weights
                Stored weights dictionary for table metrics.
            stored_threshold
                Stored thresholds dictionary for table metric thresholds.
            _pathname
                Current URL path. Unused, required to trigger on path change.
            toggle_value
                Value of toggle to show normalised values.
            selected_models
                List of model names currently selected in the model filter.
            stored_raw_data
                Table data.
            stored_computed_data
                Latest table rows emitted by the table.
            """
            if not stored_raw_data or current_columns is None:
                raise PreventUpdate

            thresholds = clean_thresholds(stored_threshold)
            show_normalized = bool(toggle_value) and toggle_value[0] == "norm"
            trigger_id = ctx.triggered_id

            # Page changes and toggle flips reuse the cached scored rows rather than
            # recalculating scores, we only re-score when weights/thresholds change.
            if (
                trigger_id in ("app-location", f"{table_id}-normalized-toggle")
                and stored_computed_data
            ):
                display_rows = get_scores(
                    stored_raw_data, stored_computed_data, thresholds, toggle_value
                )
                scored_rows = calc_metric_scores(stored_raw_data, thresholds=thresholds)
                filtered_rows = filter_rows_by_models(display_rows, selected_models)
                filtered_scores = filter_rows_by_models(scored_rows, selected_models)
                style = (
                    get_table_style(filtered_rows, scored_data=filtered_scores)
                    if filtered_rows
                    else []
                )
                style, tooltip_data = apply_level_of_theory_warnings(
                    filtered_rows,
                    style,
                    model_levels=model_levels,
                    metric_levels=metric_levels,
                    model_configs=model_configs,
                )
                columns = format_metric_columns(
                    current_columns, thresholds, show_normalized
                )
                tooltips = format_tooltip_headers(
                    raw_tooltips, thresholds, show_normalized
                )
                return (
                    filtered_rows,
                    style,
                    tooltip_data,
                    columns,
                    tooltips,
                    stored_computed_data,
                    stored_raw_data,
                )

            # Update overall table score for new weights and thresholds
            metrics_data = calc_table_scores(
                stored_raw_data, stored_weights, thresholds
            )
            # Update stored scores per metric
            scored_rows = calc_metric_scores(stored_raw_data, thresholds)
            # Select between unitful and unitless data
            display_rows = get_scores(
                metrics_data, scored_rows, thresholds, toggle_value
            )
            filtered_rows = filter_rows_by_models(display_rows, selected_models)
            filtered_scores = filter_rows_by_models(scored_rows, selected_models)
            style = (
                get_table_style(filtered_rows, scored_data=filtered_scores)
                if filtered_rows
                else []
            )
            style, tooltip_data = apply_level_of_theory_warnings(
                filtered_rows,
                style,
                model_levels=model_levels,
                metric_levels=metric_levels,
                model_configs=model_configs,
            )
            columns = format_metric_columns(
                current_columns, thresholds, show_normalized
            )
            tooltips = format_tooltip_headers(raw_tooltips, thresholds, show_normalized)
            return (
                filtered_rows,
                style,
                tooltip_data,
                columns,
                tooltips,
                scored_rows,
                metrics_data,
            )

    else:

        @callback(
            Output(table_id, "data", allow_duplicate=True),
            Output(table_id, "style_data_conditional", allow_duplicate=True),
            Output(table_id, "tooltip_data", allow_duplicate=True),
            Output(f"{table_id}-computed-store", "data", allow_duplicate=True),
            Input(f"{table_id}-weight-store", "data"),
            Input("selected-models-store", "data"),
            Input("app-location", "pathname"),
            State(table_id, "data"),
            State(f"{table_id}-computed-store", "data"),
            prevent_initial_call="initial_duplicate",
        )
        def update_table_scores(
            stored_weights: dict[str, float] | None,
            selected_models: list[str] | None,
            _pathname: str,
            table_data: list[dict] | None,
            computed_store: list[dict] | None,
        ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
            # Always use computed_store (full unfiltered rows) as the source so
            # that re-selecting a model restores it. Fall back to table_data only
            # on first load before the store is populated.
            source_data = computed_store or table_data
            if not source_data:
                raise PreventUpdate

            trigger_id = ctx.triggered_id

            if trigger_id == "app-location":
                filtered_rows = filter_rows_by_models(source_data, selected_models)
                style = get_table_style(filtered_rows) if filtered_rows else []
                style, tooltip_data = apply_level_of_theory_warnings(
                    filtered_rows,
                    style,
                    model_levels=model_levels,
                    metric_levels=metric_levels,
                    model_configs=model_configs,
                )
                return filtered_rows, style, tooltip_data, source_data

            scored_rows, _ = update_score_style(source_data, stored_weights)
            filtered_rows = filter_rows_by_models(scored_rows, selected_models)
            style = get_table_style(filtered_rows) if filtered_rows else []
            style, tooltip_data = apply_level_of_theory_warnings(
                filtered_rows,
                style,
                model_levels=model_levels,
                metric_levels=metric_levels,
                model_configs=model_configs,
            )
            return filtered_rows, style, tooltip_data, scored_rows

        @callback(
            Output(table_id, "data", allow_duplicate=True),
            Output(table_id, "style_data_conditional", allow_duplicate=True),
            Output(table_id, "tooltip_data", allow_duplicate=True),
            Input(f"{table_id}-computed-store", "data"),
            Input("selected-models-store", "data"),
            Input("app-location", "pathname"),
            prevent_initial_call="initial_duplicate",
        )
        def sync_table_from_computed_store(
            computed_store: list[dict] | None,
            selected_models: list[str] | None,
            _pathname: str,
        ) -> tuple[list[dict], list[dict], list[dict]]:
            """
            Sync the visible category table from its cached unfiltered rows.

            Parameters
            ----------
            computed_store
                Cached unfiltered rows for the category summary.
            selected_models
                Currently selected model names.
            _pathname
                Current pathname. Unused, required so the callback hydrates when the
                category page is mounted.

            Returns
            -------
            tuple[list[dict], list[dict], list[dict]]
                Filtered rows, style rules, and tooltip rows for the visible table.
            """
            if not computed_store:
                raise PreventUpdate

            filtered_rows = filter_rows_by_models(computed_store, selected_models)
            style = get_table_style(filtered_rows) if filtered_rows else []
            style, tooltip_data = apply_level_of_theory_warnings(
                filtered_rows,
                style,
                model_levels=model_levels,
                metric_levels=metric_levels,
                model_configs=model_configs,
            )
            return filtered_rows, style, tooltip_data

    @callback(
        Output("summary-table-scores-store", "data", allow_duplicate=True),
        Input(f"{table_id}-computed-store", "data"),
        State("summary-table-scores-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_scores_store(
        computed_rows: list[dict] | None,
        scores_data: dict[str, dict[str, float]],
    ) -> dict[str, dict[str, float]]:
        """
        Update stored scores values when weights update.

        Parameters
        ----------
        computed_rows
            Cached full rows for the category summary table.
        scores_data
            Stored overall-summary inputs, keyed by category score column.

        Returns
        -------
        dict[str, dict[str, float]]
            Updated summary-score mapping.
        """
        # Only category summary tables should write to the global store
        if not table_id.endswith("-summary-table"):
            return scores_data

        if not computed_rows:
            return scores_data

        if not scores_data:
            scores_data = {}
        # Update scores store. Category table IDs are of form "[category]-summary-table"
        # Table headings are of the form "[category] Score"
        scores_data[table_id.removesuffix("-summary-table") + " Score"] = {
            row["MLIP"]: row["Score"] for row in computed_rows if row.get("MLIP")
        }
        return scores_data


def register_benchmark_to_category_callback(
    benchmark_table_id: str,
    category_table_id: str,
    benchmark_column: str,
    use_threshold_store: bool = False,
    model_name_map: dict[str, str] | None = None,
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
    model_name_map
        Optional mapping of displayed benchmark MLIP names -> original model names.
    """
    _ = use_threshold_store  # cached rows handle normalization
    # flag kept for compatibility with existing call sites
    name_map = dict(model_name_map or {})

    @callback(
        Output(f"{category_table_id}-computed-store", "data", allow_duplicate=True),
        Input(f"{benchmark_table_id}-computed-store", "data"),
        State(f"{category_table_id}-weight-store", "data"),
        State(f"{category_table_id}-computed-store", "data"),
        prevent_initial_call=True,
    )
    def update_category_from_benchmark(
        benchmark_computed_store: list[dict] | None,
        category_weights: dict[str, float] | None,
        category_computed_store: list[dict] | None,
    ) -> list[dict]:
        """
        Update cached category summary rows from a benchmark's cached scores.

        Parameters
        ----------
        benchmark_computed_store
            Latest scored benchmark rows emitted by the benchmark table.
        category_weights
            Stored weights for the category summary metrics.
        category_computed_store
            Cached scored rows for the category summary.

        Returns
        -------
        list[dict]
            Refreshed cached rows for the category summary table.
        """
        if not category_computed_store:
            raise PreventUpdate
        if not benchmark_computed_store:
            raise PreventUpdate
        category_rows = deepcopy(category_computed_store)

        benchmark_scores: dict[str, float] = {}
        for row in benchmark_computed_store:
            display_name = row.get("MLIP")
            original_name = name_map.get(display_name, display_name)
            score = row.get("Score")
            if display_name is None or original_name is None or score is None:
                continue
            benchmark_scores[original_name] = score

        for row in category_rows:
            mlip = row.get("MLIP")
            if mlip in benchmark_scores:
                row[benchmark_column] = benchmark_scores[mlip]

        category_rows, _ = update_score_style(category_rows, category_weights)
        return category_rows


def register_weight_callbacks(
    input_id: str, table_id: str, column: str, default_weights: dict[str, float]
) -> None:
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
    default_weights
        Optional weights for each metric, usually set during analysis. Default is
        `None`, which sets all weights to 1.
    """
    default_weights = default_weights if default_weights else {}

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
            stored_weights.update(
                (key, default_weights.get(key, 1.0)) for key in stored_weights
            )
        else:
            raise PreventUpdate

        return stored_weights

    @callback(
        Output(f"{input_id}-input", "value"),
        Input(f"{table_id}-weight-store", "data"),
        Input("app-location", "pathname"),
        prevent_initial_call="initial_duplicate",
    )
    def sync_inputs(stored_weights: dict[str, float], _pathname: str) -> float:
        """
        Sync weight values between the text input and Store.

        Parameters
        ----------
        stored_weights
            Stored weight values for each column.
        _pathname
            Current pathname. Variable unused, but required as input to trigger on
            path change.

        Returns
        -------
        float
            Weight to set text input value.
        """
        return stored_weights[column]


def register_normalization_callbacks(
    table_id: str,
    metrics: list[str],
    default_thresholds: Thresholds,
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
        Default threshold mapping used for resets.
    register_toggle
        Whether to register the raw/normalized display toggle callback. Default is
        `True`.
    """
    cleaned_defaults = clean_thresholds(default_thresholds)

    # Per-metric store callbacks (simpler and reliable)
    for metric in metrics:

        @callback(
            Output(f"{table_id}-thresholds-store", "data", allow_duplicate=True),
            Input(f"{table_id}-{metric}-good-threshold", "value"),
            Input(f"{table_id}-{metric}-bad-threshold", "value"),
            Input(f"{table_id}-reset-thresholds-button", "n_clicks"),
            State(f"{table_id}-thresholds-store", "data"),
            prevent_initial_call=True,
        )
        def store_threshold_values(
            good_val, bad_val, n_clicks, stored_thresholds, metric=metric
        ):
            """Update normalization thresholds store for one metric or reset all."""
            trigger_id = ctx.triggered_id
            cleaned_store = clean_thresholds(stored_thresholds) or {}

            # Reset to defaults is specified via reset button
            if trigger_id == f"{table_id}-reset-thresholds-button":
                if cleaned_defaults:
                    return deepcopy(cleaned_defaults)
                return cleaned_store

            # Ensure key exists
            if metric not in cleaned_store:
                cleaned_store[metric] = {
                    "good": None,
                    "bad": None,
                    "unit": (
                        cleaned_defaults.get(metric, {}).get("unit")
                        if cleaned_defaults
                        else None
                    ),
                }

            entry = cleaned_store[metric]
            good_threshold = entry.get("good")
            bad_threshold = entry.get("bad")

            # Update thresholds from input boxes
            if trigger_id == f"{table_id}-{metric}-good-threshold":
                if good_val is None or bad_threshold is None:
                    raise PreventUpdate
                entry["good"] = float(good_val)

            elif trigger_id == f"{table_id}-{metric}-bad-threshold":
                if bad_val is None or good_threshold is None:
                    raise PreventUpdate
                entry["bad"] = float(bad_val)
            else:
                raise PreventUpdate

            return cleaned_store

    if register_toggle:
        # Toggle display between raw and normalized values without recomputing scores
        @callback(
            Output(f"{table_id}", "data", allow_duplicate=True),
            Output(f"{table_id}", "style_data_conditional", allow_duplicate=True),
            Output(f"{table_id}", "columns", allow_duplicate=True),
            Output(f"{table_id}", "tooltip_header", allow_duplicate=True),
            Input(f"{table_id}-normalized-toggle", "value"),
            State(f"{table_id}-raw-data-store", "data"),
            State(f"{table_id}-thresholds-store", "data"),
            State(f"{table_id}-raw-tooltip-store", "data"),
            State(f"{table_id}", "columns"),
            prevent_initial_call=True,
        )
        def toggle_normalized_display(
            show_normalized: list[str] | None,
            raw_data: list[dict],
            thresholds: dict[str, Any] | None,
            raw_tooltips: dict[str, str] | None,
            current_columns: list[dict] | None,
        ) -> tuple[list[dict], list[dict], list[dict], dict[str, str] | None]:
            """Toggle between raw and normalised metric values for display only."""
            if not raw_data or current_columns is None:
                raise PreventUpdate

            cleaned_thresholds = clean_thresholds(thresholds)
            normalized_active = bool(show_normalized) and show_normalized[0] == "norm"

            # Get metric scores to display
            scored_rows = calc_metric_scores(raw_data, cleaned_thresholds)
            display_rows = get_scores(
                raw_data, scored_rows, cleaned_thresholds, show_normalized
            )
            style = get_table_style(display_rows, scored_data=scored_rows)
            columns = format_metric_columns(
                current_columns, cleaned_thresholds, normalized_active
            )
            tooltips = format_tooltip_headers(
                raw_tooltips, cleaned_thresholds, normalized_active
            )
            return display_rows, style, columns, tooltips

    # Register individual threshold input sync callbacks
    for metric in metrics:

        @callback(
            Output(f"{table_id}-{metric}-good-threshold", "value"),
            Output(f"{table_id}-{metric}-bad-threshold", "value"),
            Input(f"{table_id}-thresholds-store", "data"),
            prevent_initial_call=True,
        )
        def sync_threshold_inputs(thresholds, metric=metric):
            """Sync threshold input values with stored thresholds."""
            cleaned_thresholds = clean_thresholds(thresholds)
            if cleaned_thresholds and metric in cleaned_thresholds:
                entry = cleaned_thresholds[metric]
                return entry.get("good"), entry.get("bad")
            raise PreventUpdate
