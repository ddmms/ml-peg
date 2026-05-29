"""Register callbacks relating to interative weights."""

from __future__ import annotations

import base64
from copy import deepcopy
from typing import Any, Literal

import dash
from dash import (
    MATCH,
    ClientsideFunction,
    Dash,
    Input,
    Output,
    Patch,
    State,
    callback,
    clientside_callback,
    ctx,
    dcc,
    no_update,
)
from dash.dash_table import DataTable
from dash.exceptions import PreventUpdate
import pandas as pd

from ml_peg.analysis.utils.utils import (
    calc_metric_scores,
    calc_table_scores,
    get_table_style,
    update_score_style,
)
from ml_peg.app.utils.utils import (
    Thresholds,
    build_level_of_theory_warnings,
    build_threshold_input_style,
    clean_thresholds,
    filter_rows_by_models,
    format_metric_columns,
    format_tooltip_headers,
    get_scores,
    get_threshold_colours,
)

THRESHOLD_INPUT_STEP = 0.0001
THRESHOLD_ROUND_DIGITS = 10


def enforce_threshold_direction(
    *,
    edited_field: Literal["good", "bad"],
    good: float,
    bad: float,
    default_good: float,
    default_bad: float,
    min_gap: float = THRESHOLD_INPUT_STEP,
) -> tuple[float, float]:
    """
    Preserve the original good/bad threshold direction after a user edit.

    Parameters
    ----------
    edited_field
        Which threshold input the user changed.
    good
        Candidate good threshold value after the edit.
    bad
        Candidate bad threshold value after the edit.
    default_good
        Original good threshold from benchmark metadata.
    default_bad
        Original bad threshold from benchmark metadata.
    min_gap
        Minimum allowed separation between the two thresholds.

    Returns
    -------
    tuple[float, float]
        Corrected ``(good, bad)`` thresholds.
    """
    if default_good == default_bad:
        return round(good, THRESHOLD_ROUND_DIGITS), round(bad, THRESHOLD_ROUND_DIGITS)

    good_is_higher = default_good > default_bad
    if good_is_higher and good <= bad:
        if edited_field == "good":
            bad = good - min_gap
        else:
            good = bad + min_gap
    elif not good_is_higher and good >= bad:
        if edited_field == "good":
            bad = good + min_gap
        else:
            good = bad - min_gap

    return round(good, THRESHOLD_ROUND_DIGITS), round(bad, THRESHOLD_ROUND_DIGITS)


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
        Input("cmap-store", "data"),
        prevent_initial_call="initial_duplicate",
        optional=True,
    )
    def sync_summary_table(
        selected_models: list[str] | None,
        computed_store: list[dict] | None,
        cmap_name: str | None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Sync the visible summary table from cached unfiltered rows.

        Parameters
        ----------
        selected_models
            Models currently selected in the global model filter.
        computed_store
            Cached full summary rows for the overall summary table.
        cmap_name
            Matplotlib colormap name from the cmap store.

        Returns
        -------
        tuple[list[dict], list[dict], list[dict]]
            Filtered rows, style rules, and tooltip rows for the visible table.
        """
        if not computed_store:
            raise PreventUpdate

        filtered_rows = filter_rows_by_models(computed_store, selected_models)
        base_style = (
            get_table_style(filtered_rows, cmap_name=cmap_name or "viridis_r")
            if filtered_rows
            else []
        )
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

    @callback(
        Output(table_id, "data", allow_duplicate=True),
        Output(table_id, "style_data_conditional", allow_duplicate=True),
        Input(f"{table_id}-raw-data-store", "data"),
        State(f"{table_id}-computed-store", "data"),
        State(f"{table_id}-weight-store", "data"),
        State(f"{table_id}-thresholds-store", "data"),
        State(f"{table_id}-normalized-toggle", "value"),
        State("selected-models-store", "data"),
        State("cmap-store", "data"),
        State(f"{table_id}-raw-tooltip-store", "data"),
        State(table_id, "columns"),
        prevent_initial_call=True,
        optional=True,
    )
    def update_table_from_store(
        stored_raw_data: list[dict] | None,
        stored_computed_data: list[dict] | None,
        weights: dict[str, float] | None,
        thresholds: dict | None,
        toggle_value: list[str] | None,
        selected_models: list[str] | None,
        cmap_name: str | None,
        raw_tooltips: dict[str, str] | None,
        current_columns: list[dict] | None,
    ) -> list[dict]:
        """
        Update visible table from cached data when the raw data store changes.

        Parameters
        ----------
        stored_raw_data
            Stored raw table data.
        stored_computed_data
            Stored computed table data.
        weights
            Stored weights for the table.
        thresholds
            Stored thresholds for the table.
        toggle_value
            Value of toggle to show normalised values.
        selected_models
            List of model names currently selected in the model filter.
        cmap_name
            Colourmap name from the cmap store.
        raw_tooltips
            Stored raw tooltip text for the table.
        current_columns
            Current table columns.

        Returns
        -------
        list[dict]
            Updated rows for the visible table.
        """
        display_rows = get_scores(
            stored_raw_data, stored_computed_data, thresholds, toggle_value
        )
        scored_rows = calc_metric_scores(stored_raw_data, thresholds=thresholds)
        filtered_rows = filter_rows_by_models(display_rows, selected_models)
        filtered_scores = filter_rows_by_models(scored_rows, selected_models)
        style = (
            get_table_style(
                filtered_rows,
                scored_data=filtered_scores,
                cmap_name=cmap_name or "viridis_r",
            )
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
        return filtered_rows, style

    # Benchmark tables
    if use_thresholds:

        @callback(
            Output(table_id, "data", allow_duplicate=True),
            Output(table_id, "style_data_conditional", allow_duplicate=True),
            Output(table_id, "tooltip_data", allow_duplicate=True),
            Output(table_id, "columns", allow_duplicate=True),
            Output(table_id, "tooltip_header", allow_duplicate=True),
            Output(f"{table_id}-computed-store", "data", allow_duplicate=True),
            Output(f"{table_id}-raw-data-store", "data", allow_duplicate=True),
            Input(f"{table_id}-weight-store", "data"),
            Input(f"{table_id}-thresholds-store", "data"),
            Input(f"{table_id}-normalized-toggle", "value"),
            Input("selected-models-store", "data"),
            Input("cmap-store", "data"),
            State(f"{table_id}-raw-data-store", "data"),
            State(f"{table_id}-computed-store", "data"),
            State(f"{table_id}-raw-tooltip-store", "data"),
            State(table_id, "columns"),
            prevent_initial_call="initial_duplicate",
            optional=True,
        )
        def update_benchmark_table_scores(
            stored_weights: dict[str, float] | None,
            stored_threshold: dict | None,
            toggle_value: list[str] | None,
            selected_models: list[str] | None,
            cmap_name: str | None,
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
                trigger_id in (f"{table_id}-normalized-toggle", "cmap-store")
                and stored_computed_data
            ):
                display_rows = get_scores(
                    stored_raw_data, stored_computed_data, thresholds, toggle_value
                )
                scored_rows = calc_metric_scores(stored_raw_data, thresholds=thresholds)
                filtered_rows = filter_rows_by_models(display_rows, selected_models)
                filtered_scores = filter_rows_by_models(scored_rows, selected_models)
                style = (
                    get_table_style(
                        filtered_rows,
                        scored_data=filtered_scores,
                        cmap_name=cmap_name or "viridis_r",
                    )
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
                get_table_style(
                    filtered_rows,
                    scored_data=filtered_scores,
                    cmap_name=cmap_name or "viridis_r",
                )
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
            Input("cmap-store", "data"),
            State(table_id, "data"),
            State(f"{table_id}-computed-store", "data"),
            prevent_initial_call="initial_duplicate",
            optional=True,
        )
        def update_table_scores(
            stored_weights: dict[str, float] | None,
            selected_models: list[str] | None,
            cmap_name: str | None,
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

            # Recompute scores only when weights changed
            if trigger_id == f"{table_id}-weight-store":
                scored_rows, _ = update_score_style(source_data, stored_weights)
                updated_store = scored_rows

            else:
                scored_rows = source_data
                updated_store = no_update

            filtered_rows = filter_rows_by_models(scored_rows, selected_models)
            style = (
                get_table_style(filtered_rows, cmap_name=cmap_name or "viridis_r")
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

            if not table_data or len(filtered_rows) != len(table_data):
                return filtered_rows, style, tooltip_data, scored_rows

            patch = Patch()
            rows_changed = False

            for row_index, (old_row, new_row) in enumerate(
                zip(table_data, filtered_rows, strict=True)
            ):
                for key, new_value in new_row.items():
                    if old_row.get(key) != new_value:
                        patch[row_index][key] = new_value
                        rows_changed = True

            # No visual change
            if not rows_changed:
                return no_update, style, tooltip_data, updated_store

            return patch, style, tooltip_data, updated_store

        @callback(
            Output(table_id, "data", allow_duplicate=True),
            Output(table_id, "style_data_conditional", allow_duplicate=True),
            Output(table_id, "tooltip_data", allow_duplicate=True),
            Input(f"{table_id}-computed-store", "data"),
            Input("selected-models-store", "data"),
            Input("cmap-store", "data"),
            prevent_initial_call="initial_duplicate",
            optional=True,
        )
        def sync_table_from_computed_store(
            computed_store: list[dict] | None,
            selected_models: list[str] | None,
            cmap_name: str | None,
        ) -> tuple[list[dict], list[dict], list[dict]]:
            """
            Sync the visible category table from its cached unfiltered rows.

            Parameters
            ----------
            computed_store
                Cached unfiltered rows for the category summary.
            selected_models
                Currently selected model names.

            Returns
            -------
            tuple[list[dict], list[dict], list[dict]]
                Filtered rows, style rules, and tooltip rows for the visible table.
            """
            if not computed_store:
                raise PreventUpdate

            filtered_rows = filter_rows_by_models(computed_store, selected_models)
            style = (
                get_table_style(filtered_rows, cmap_name=cmap_name or "viridis_r")
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
            raise PreventUpdate

        if not computed_rows:
            raise PreventUpdate

        # Category table IDs are of form "[category]-summary-table"
        category_key = table_id.removesuffix("-summary-table") + " Score"

        new_scores = {
            row["MLIP"]: row["Score"] for row in computed_rows if row.get("MLIP")
        }
        current_scores = (scores_data or {}).get(category_key)

        if current_scores == new_scores:
            return no_update

        patch = Patch()
        patch[category_key] = new_scores

        return patch


def register_benchmark_to_category_callback(
    all_tables: dict[str, dict[str, DataTable]], category_to_title: dict[str, str]
) -> None:
    """
    Propagate a benchmark table's Score into its category summary table column.

    Parameters
    ----------
    all_tables
        Tables for all tests, grouped by category.
    category_to_title
        Dictionary mapping category directory names to their display titles/table IDs.
    """
    all_info = {}
    for category, tables in all_tables.items():
        all_info[category] = {}
        for test_name, benchmark_table in tables.items():
            all_info[category][test_name] = {
                "benchmark_table_id": benchmark_table.id,
                "benchmark_column": test_name + " Score",
                "model_name_map": getattr(benchmark_table, "model_name_map", {}),
            }

    outputs = []
    inputs = []
    for category, category_info in sorted(all_info.items()):
        category_table_id = f"{category_to_title[category]}-summary-table"
        outputs.append(
            Output(f"{category_table_id}-computed-store", "data", allow_duplicate=True)
        )

        inputs.extend(
            [
                State(f"{category_table_id}-weight-store", "data"),
                State(f"{category_table_id}-computed-store", "data"),
            ]
        )
        inputs.extend(
            [
                Input(f"{table_info['benchmark_table_id']}-computed-store", "data")
                for _, table_info in sorted(category_info.items())
            ]
        )

    @callback(outputs, inputs, prevent_initial_call=True)
    def update_category_from_benchmark(*args) -> list[list[dict]]:
        """
        Update cached category summary rows from all benchmarks' cached scores.

        Parameters
        ----------
        *args
            States and Inputs for all category summary tables and benchmark tables.
            Ordered by category. For each category, the weights, computed store, and
            benchmark computed stores are listed sequentially.

        Returns
        -------
        list[list[dict]]
            Refreshed cached rows for each category summary table.
        """
        # Rebuild inputs for each category
        iterator = iter(args)
        patched_outputs = []

        for _category, category_info in sorted(all_info.items()):
            category_weights = next(iterator)
            current_rows = next(iterator)

            updated_rows = []
            for row in current_rows:
                updated_row = row.copy()
                updated_rows.append(updated_row)

            updated_by_mlip = {row["MLIP"]: row for row in updated_rows}

            benchmark_changed = False

            for _test_name, table_info in sorted(category_info.items()):
                benchmark_rows = next(iterator)

                name_map = table_info["model_name_map"]
                benchmark_column = table_info["benchmark_column"]

                for row in benchmark_rows:
                    display_name = row.get("MLIP")
                    original_name = name_map.get(display_name, display_name)
                    if original_name not in updated_by_mlip:
                        continue

                    new_score = row.get("Score")
                    target_row = updated_by_mlip[original_name]

                    if target_row.get(benchmark_column) != new_score:
                        target_row[benchmark_column] = new_score
                        benchmark_changed = True

            if not benchmark_changed:
                patched_outputs.append(no_update)
                continue

            # Recompute overall category scores using existing utility
            rescored_rows, _ = update_score_style(updated_rows, category_weights)

            patch = Patch()
            score_changed = False

            for idx, (old_row, new_row) in enumerate(
                zip(current_rows, rescored_rows, strict=True)
            ):
                # Patch benchmark columns
                for key, value in new_row.items():
                    if key in {"MLIP", "Score"}:
                        continue

                    if old_row.get(key) != value:
                        patch[idx][key] = value
                        score_changed = True

                # Patch overall score
                if old_row.get("Score") != new_row.get("Score"):
                    patch[idx]["Score"] = new_row.get("Score")
                    score_changed = True

            if score_changed:
                patched_outputs.append(patch)
            else:
                patched_outputs.append(no_update)

        return patched_outputs


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
        optional=True,
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
            Number of clicks.
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
        elif trigger_id == f"{table_id}-reset-button" and n_clicks > 0:
            stored_weights.update(
                (key, default_weights.get(key, 1.0)) for key in stored_weights
            )
        else:
            raise PreventUpdate

        return stored_weights

    @callback(
        Output(f"{input_id}-input", "value"),
        Input(f"{table_id}-weight-store", "data"),
        prevent_initial_call="initial_duplicate",
        optional=True,
    )
    def sync_inputs(stored_weights: dict[str, float]) -> float:
        """
        Sync weight values between the text input and Store.

        Parameters
        ----------
        stored_weights
            Stored weight values for each column.

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
            optional=True,
        )
        def store_threshold_values(
            good_val, bad_val, n_clicks, stored_thresholds, metric=metric
        ):
            """Update normalization thresholds store for one metric or reset all."""
            trigger_id = ctx.triggered_id
            cleaned_store = clean_thresholds(stored_thresholds) or {}

            # Reset to defaults is specified via reset button
            if trigger_id == f"{table_id}-reset-thresholds-button":
                if not n_clicks:
                    raise PreventUpdate
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
                default_entry = cleaned_defaults.get(metric)
                if default_entry is None:
                    raise PreventUpdate

                good, bad = enforce_threshold_direction(
                    edited_field="good",
                    good=float(good_val),
                    bad=float(bad_threshold),
                    default_good=default_entry["good"],
                    default_bad=default_entry["bad"],
                )
                entry["good"] = good
                entry["bad"] = bad

            elif trigger_id == f"{table_id}-{metric}-bad-threshold":
                if bad_val is None or good_threshold is None:
                    raise PreventUpdate
                default_entry = cleaned_defaults.get(metric)
                if default_entry is None:
                    raise PreventUpdate

                good, bad = enforce_threshold_direction(
                    edited_field="bad",
                    good=float(good_threshold),
                    bad=float(bad_val),
                    default_good=default_entry["good"],
                    default_bad=default_entry["bad"],
                )
                entry["good"] = good
                entry["bad"] = bad
            else:
                raise PreventUpdate

            return cleaned_store

    if metrics:
        threshold_style_outputs = [
            output
            for metric in metrics
            for output in (
                Output(f"{table_id}-{metric}-good-threshold", "style"),
                Output(f"{table_id}-{metric}-bad-threshold", "style"),
            )
        ]

        @callback(
            *threshold_style_outputs,
            Input("cmap-store", "data"),
            prevent_initial_call=False,
            optional=True,
        )
        def sync_threshold_input_styles(
            cmap_name: str | None,
        ) -> tuple[dict[str, str], ...]:
            """
            Colour threshold input borders to match the selected table colour scale.

            Parameters
            ----------
            cmap_name
                Current table colormap name from the shared colour-scheme store.

            Returns
            -------
            tuple[dict[str, str], ...]
                Alternating good/bad input styles for each metric.
            """
            colours = get_threshold_colours(cmap_name)
            styles: list[dict[str, str]] = []
            for _metric in metrics:
                styles.extend(
                    [
                        build_threshold_input_style(colours["good"]),
                        build_threshold_input_style(colours["bad"]),
                    ]
                )
            return tuple(styles)

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
            State("cmap-store", "data"),
            prevent_initial_call=True,
            optional=True,
        )
        def toggle_normalized_display(
            show_normalized: list[str] | None,
            raw_data: list[dict],
            thresholds: dict[str, Any] | None,
            raw_tooltips: dict[str, str] | None,
            current_columns: list[dict] | None,
            cmap_name: str | None,
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
            style = get_table_style(
                display_rows,
                scored_data=scored_rows,
                cmap_name=cmap_name or "viridis_r",
            )
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
            optional=True,
        )
        def sync_threshold_inputs(
            thresholds: Thresholds | None, metric: str = metric
        ) -> tuple[float | None, float | None]:
            """
            Sync threshold input values with stored thresholds.

            Parameters
            ----------
            thresholds
                Stored threshold values.
            metric
                Metric name corresponding to the threshold inputs.
            """
            cleaned_thresholds = clean_thresholds(thresholds)
            if cleaned_thresholds and metric in cleaned_thresholds:
                entry = cleaned_thresholds[metric]
                return entry.get("good"), entry.get("bad")
            raise PreventUpdate


def register_plot_download_callbacks() -> None:
    """Register one generic plot download callback once per Dash app."""
    app = dash.get_app()
    output = Output({"type": "plot-download", "index": MATCH}, "data")
    if str(output) in app.callback_map:
        return

    app.clientside_callback(
        ClientsideFunction(
            namespace="plot_download",
            function_name="downloadPlot",
        ),
        output,
        Input({"type": "plot-download-button", "index": MATCH}, "n_clicks"),
        State({"type": "plot-download-format", "index": MATCH}, "value"),
        State({"type": "plot-download-target", "index": MATCH}, "data"),
        prevent_initial_call=True,
    )


def register_image_download_callbacks() -> None:
    """
    Register one generic image download callback once per Dash app.

    Unlike the table download (which asks the browser to capture the live DOM),
    this callback decodes a pre-rendered image already stored as a base64 data
    URI in a ``dcc.Store``. The phonon dispersion plot is rendered server-side
    via kaleido at analysis time, so the full-resolution export is available
    without re-rendering in the browser.
    """
    app = dash.get_app()
    output = Output({"type": "image-download", "index": MATCH}, "data")
    if str(output) in app.callback_map:
        return

    @callback(
        output,
        Input({"type": "image-download-button", "index": MATCH}, "n_clicks"),
        State({"type": "image-download-format", "index": MATCH}, "value"),
        State({"type": "image-download-target", "index": MATCH}, "data"),
        prevent_initial_call=True,
        optional=True,
    )
    def _download_image(n_clicks, fmt, uris):
        """
        Decode the stored data URI and trigger a browser file download.

        Parameters
        ----------
        n_clicks
            Number of button clicks.
        fmt
            Selected download format (``"png"``, ``"svg"``, or ``"json"``).
        uris
            Mapping of format keys to base64 data URIs.

        Returns
        -------
        dict
            Dash ``dcc.send_bytes`` payload for the Download component.
        """
        if not n_clicks or not uris or not fmt:
            raise PreventUpdate
        uri = uris.get(fmt)
        if not uri:
            raise PreventUpdate
        data = base64.b64decode(uri.split(",")[1])
        mime = {
            "png": "image/png",
            "svg": "image/svg+xml",
            "json": "application/json",
        }.get(fmt, "application/octet-stream")
        return dcc.send_bytes(data, f"phonon_dispersion.{fmt}", type=mime)


def register_download_callbacks(table_id: str) -> None:
    """
    Register minimal table download callbacks for CSV, PNG, and SVG.

    CSV exports are generated from the table's Dash data payload. PNG/SVG exports
    are different: Python sends a small request to the browser, then the browser
    captures the table exactly as it has already been drawn on the page. That is
    what preserves conditional colours, warning styles, current column headers, and
    the table's CSS layout.

    Parameters
    ----------
    table_id
        ID of table to export.
    """

    @callback(
        Output(f"{table_id}-download", "data", allow_duplicate=True),
        Output(f"{table_id}-download-request", "data"),
        Input(f"{table_id}-download-button", "n_clicks"),
        State(f"{table_id}-download-format", "value"),
        State(table_id, "data"),
        State(table_id, "columns"),
        prevent_initial_call=True,
        optional=True,
    )
    def download_table(
        n_clicks: int,
        download_format: str,
        table_data: list[dict] | None,
        columns: list[dict] | None,
    ) -> tuple[dict | Any, dict | Any]:
        """
        Dispatch table download request.

        Parameters
        ----------
        n_clicks
            Number of clicks on the download button.
        download_format
            Requested format, one of ``csv``, ``png``, or ``svg``.
        table_data
            Currently visible table rows.
        columns
            Current table column metadata.

        Returns
        -------
        tuple[dict | Any, dict | Any]
            Pair of payloads for ``download`` and ``download-request`` stores.
            For CSV, the first item is a Dash download payload and the second is
            ``no_update``. For PNG/SVG, the first item is ``no_update`` and the
            second item is the client-side capture request.
        """
        if not n_clicks or not columns:
            raise PreventUpdate

        fmt = (download_format or "csv").lower()
        filename_base = table_id.replace("_", "-")
        column_ids = [col["id"] for col in columns if isinstance(col.get("id"), str)]
        export_cols = [col for col in column_ids if col != "id"]

        if fmt == "csv":
            if table_data:
                frame = pd.DataFrame(table_data)
                frame = frame.reindex(columns=export_cols)
            else:
                frame = pd.DataFrame(columns=export_cols)
            return (
                dcc.send_data_frame(
                    frame.to_csv,
                    filename=f"{filename_base}.csv",
                    index=False,
                ),
                no_update,
            )

        if fmt in {"png", "svg"}:
            # Image exports need the already-rendered table, not a new table recreated
            # from raw values. Send the target table id to the browser-side asset.
            return (
                no_update,
                {
                    "element_id": table_id,
                    "format": fmt,
                    "filename": f"{filename_base}.{fmt}",
                },
            )

        raise PreventUpdate

    clientside_callback(
        ClientsideFunction(
            namespace="table_download",
            function_name="captureTable",
        ),
        Output(f"{table_id}-download", "data", allow_duplicate=True),
        Input(f"{table_id}-download-request", "data"),
        prevent_initial_call=True,
    )


def register_filter_tables_callback(apps: dict[str, Dash]) -> None:
    """
    Update all tables when filter dropdown value changes.

    Parameters
    ----------
    apps
        Dictionary of test apps to register callbacks for.
    """
    app_entries = []
    for app in apps.values():
        app_entries.append(
            {
                "app": app,
                "weight_state": State(f"{app.table_id}-weight-store", "data"),
                "threshold_state": State(f"{app.table_id}-thresholds-store", "data"),
            }
        )

    outputs = []
    for entry in sorted(app_entries):
        app = entry["app"]
        outputs.extend(
            [
                Output(f"{app.table_id}-computed-store", "data"),
                Output(f"{app.table_id}-raw-data-store", "data", allow_duplicate=True),
            ]
        )

    states = []
    for entry in sorted(app_entries):
        states.extend([entry["weight_state"], entry["threshold_state"]])

    @callback(
        outputs, Input("element-filter", "value"), states, prevent_initial_call=True
    )
    def recompute_tables(elements, *args):
        """
        Recompute all benchmark tables when element filter is applied.

        Parameters
        ----------
        elements
            List of selected elements to filter by.
        *args
            Weight and threshold states for each app.

        Returns
        -------
        list[list[dict]]
            Updated rows for each app's computed store and raw data stores.
        """
        # Rebuild inputs for each app
        per_app_state = {}
        iterator = iter(args)

        for entry in sorted(app_entries):
            app = entry["app"]
            per_app_state[app.table_id] = {
                "weights": next(iterator),
                "thresholds": next(iterator),
            }

        results = []

        for entry in sorted(app_entries):
            app = entry["app"]
            state = per_app_state[app.table_id]
            weights = state["weights"]
            thresholds = state["thresholds"]

            updated_data = app.filter_table(elements)

            # Update overall table score for new weights and thresholds
            metrics_data = calc_table_scores(updated_data, weights, thresholds)

            # Update stored scores per metric
            scored_rows = calc_metric_scores(updated_data, thresholds)

            results.extend([scored_rows, metrics_data])

        return results
