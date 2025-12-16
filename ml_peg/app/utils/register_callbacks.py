"""Register callbacks relating to interative weights."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from dash import (
    Input,
    Output,
    State,
    callback,
    clientside_callback,
    ctx,
    dcc,
    no_update,
)
from dash.exceptions import PreventUpdate
import pandas as pd

from ml_peg.analysis.utils.utils import (
    calc_metric_scores,
    calc_table_scores,
    get_table_style,
    update_score_style,
)
from ml_peg.app.utils.download_helpers import DOWNLOAD_CLIENTSIDE_HANDLER
from ml_peg.app.utils.utils import (
    Thresholds,
    build_level_of_theory_warnings,
    clean_thresholds,
    format_metric_columns,
    format_tooltip_headers,
    get_scores,
)


def register_summary_table_callbacks(
    model_levels: dict[str, str | None] | None = None,
    metric_levels: dict[str, str | None] | None = None,
    model_configs: dict[str, Any] | None = None,
) -> None:
    """
    Register callbacks to update summary table.

    Parameters
    ----------
    model_levels
        Mapping from model name to its level of theory badge text.
    metric_levels
        Mapping from metric column name to its level of theory badge text.
    model_configs
        Optional metadata/configuration dictionary for each model.
    """

    @callback(
        Output("summary-table", "data"),
        Output("summary-table", "style_data_conditional"),
        Output(
            "summary-table", "tooltip_data"
        ),  # Needed to display model config & level of theory tooltips
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
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Update summary table when scores/weights change, and sync on tab change.

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
        tuple[list[dict], list[dict], list[dict]]
            Updated rows, conditional styling rules, and tooltip rows.
        """
        # Update table from stored scores
        if stored_scores:
            for row in summary_data:
                for tab, values in stored_scores.items():
                    row[tab] = values[row["MLIP"]]

        # Update table contents
        updated_rows, base_style = update_score_style(summary_data, stored_weights)

        warning_styles, tooltip_rows = build_level_of_theory_warnings(
            updated_rows, model_levels, metric_levels, model_configs
        )
        style_with_warnings = base_style + warning_styles
        return updated_rows, style_with_warnings, tooltip_rows


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
            Input("all-tabs", "value"),
            Input(f"{table_id}-normalized-toggle", "value"),
            State(f"{table_id}-raw-data-store", "data"),
            State(f"{table_id}-computed-store", "data"),
            State(f"{table_id}-raw-tooltip-store", "data"),
            State(table_id, "columns"),
            prevent_initial_call="initial_duplicate",
        )
        def update_benchmark_table_scores(
            stored_weights: dict[str, float] | None,
            stored_threshold: dict | None,
            _tabs_value: str,
            toggle_value: list[str] | None,
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
            Update table when stored weights/threshold change, or tab is changed.

            Parameters
            ----------
            stored_weights
                Stored weights dictionary for table metrics.
            stored_threshold
                Stored thresholds dictionary for table metric thresholds.
            _tabs_value
                Current tab identifier (unused, required to trigger on tab change).
            toggle_value
                Value of toggle to show normalised values.
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

            def apply_levels_of_theory(
                rows: list[dict], base_style: list[dict]
            ) -> tuple[list[dict], list[dict]]:
                warning_styles, tooltip_rows = build_level_of_theory_warnings(
                    rows, model_levels, metric_levels, model_configs
                )
                combined_style = base_style + warning_styles
                tooltip_data = tooltip_rows if tooltip_rows else [{} for _ in rows]
                return combined_style, tooltip_data

            # Tab switches and toggle flips reuse the cached scored rows rather than
            # recalculating scores, we only re-score when weights/thresholds change.
            if (
                trigger_id in ("all-tabs", f"{table_id}-normalized-toggle")
                and stored_computed_data
            ):
                display_rows = get_scores(
                    stored_raw_data, stored_computed_data, thresholds, toggle_value
                )
                scored_rows = calc_metric_scores(stored_raw_data, thresholds=thresholds)
                style = get_table_style(display_rows, scored_data=scored_rows)
                style, tooltip_data = apply_levels_of_theory(display_rows, style)
                columns = format_metric_columns(
                    current_columns, thresholds, show_normalized
                )
                tooltips = format_tooltip_headers(
                    raw_tooltips, thresholds, show_normalized
                )
                return (
                    display_rows,
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
            style = get_table_style(display_rows, scored_data=scored_rows)
            style, tooltip_data = apply_levels_of_theory(display_rows, style)
            columns = format_metric_columns(
                current_columns, thresholds, show_normalized
            )
            tooltips = format_tooltip_headers(raw_tooltips, thresholds, show_normalized)
            return (
                display_rows,
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
        ) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
            trigger_id = ctx.triggered_id

            def apply_levels(
                rows: list[dict], base_style: list[dict]
            ) -> tuple[list[dict], list[dict]]:
                warning_styles, tooltip_rows = build_level_of_theory_warnings(
                    rows, model_levels, metric_levels, model_configs
                )
                combined_style = base_style + warning_styles
                tooltips = tooltip_rows if tooltip_rows else [{} for _ in rows]
                return combined_style, tooltips

            if trigger_id == "all-tabs" and computed_store:
                style = get_table_style(computed_store)
                style, tooltip_data = apply_levels(computed_store, style)
                return computed_store, style, tooltip_data, computed_store

            if not table_data:
                raise PreventUpdate

            scored_rows, style = update_score_style(table_data, stored_weights)
            style, tooltip_data = apply_levels(scored_rows, style)
            return scored_rows, style, tooltip_data, scored_rows

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
        # Update scores store. Category table IDs are of form "[category]-summary-table"
        # Table headings are of the form "[category] Score"
        scores_data[table_id.removesuffix("-summary-table") + " Score"] = {
            row["MLIP"]: row["Score"] for row in table_data
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
        Optional mapping of displayed benchmark MLIP names -> canonical model names.
    """
    _ = use_threshold_store  # cached rows handle normalization
    # flag kept for compatibility with existing call sites
    name_map = (model_name_map or {}).copy()

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
        # Default to pre-computed category data to avoid multiple updates on tab change
        category_rows = category_computed_store or category_data
        if not category_rows:
            raise PreventUpdate

        benchmark_scores: dict[str, float] = {}
        for row in benchmark_computed_store:
            display_name = row.get("MLIP")
            canonical_name = name_map.get(display_name, display_name)
            score = row.get("Score")
            if display_name is None or canonical_name is None or score is None:
                continue
            benchmark_scores[canonical_name] = score

        for row in category_rows:
            mlip = row.get("MLIP")
            if mlip in benchmark_scores:
                row[benchmark_column] = benchmark_scores[mlip]

        category_rows, style = update_score_style(category_rows, category_weights)
        return category_rows, style, category_rows


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


def register_download_callbacks(table_id: str) -> None:
    """
    Attach download controls for a table.

    Parameters
    ----------
    table_id
        Identifier of the DataTable to link with the download controls.
    """

    @callback(
        Output(f"{table_id}-download", "data", allow_duplicate=True),
        Output(f"{table_id}-download-request", "data"),
        Input(f"{table_id}-download-button", "n_clicks"),
        State(f"{table_id}-download-format", "value"),
        State(table_id, "data"),
        State(table_id, "columns"),
        prevent_initial_call=True,
    )
    def download_table(
        n_clicks: int,
        fmt: str,
        table_data: list[dict],
        columns: list[dict],
    ):
        """
        Send the currently displayed table as CSV/PNG/SVG.

        Parameters
        ----------
        n_clicks : int
            Number of button clicks; ignored until > 0.
        fmt : str
            Selected download format (csv/png/svg).
        table_data : list[dict]
            Rows currently displayed in the DataTable.
        columns : list[dict]
            Column metadata for the table.

        Returns
        -------
        tuple
            Pair of (CSV download payload, image download payload).
        """
        if not n_clicks or not table_data or not columns:
            raise PreventUpdate

        fmt = (fmt or "csv").lower()
        filename_base = table_id.replace("_", "-")

        if fmt == "csv":
            df = pd.DataFrame(table_data)
            if "id" in df.columns:
                df = df.drop(columns=["id"])
            csv_payload = dcc.send_data_frame(
                df.to_csv,
                filename=f"{filename_base}.csv",
                index=False,
            )
            return csv_payload, no_update

        if fmt in {"png", "svg"}:
            return (
                no_update,
                {
                    "element_id": table_id,
                    "format": fmt,
                    "filename": f"{filename_base}.{fmt}",
                    "request_id": n_clicks,
                },
            )

        raise PreventUpdate

    clientside_callback(
        DOWNLOAD_CLIENTSIDE_HANDLER,
        Output(f"{table_id}-download", "data", allow_duplicate=True),
        Input(f"{table_id}-download-request", "data"),
        prevent_initial_call=True,
    )
