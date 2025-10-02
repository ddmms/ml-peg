"""Register callbacks relating to interative weights."""

from __future__ import annotations

from dash import Input, Output, State, callback, ctx
from dash.exceptions import PreventUpdate

from ml_peg.analysis.utils.utils import calc_ranks, calc_scores, get_table_style


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
        Input("all-tabs", "value"),
        State(table_id, "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_table_scores(
        stored_weights: dict[str, float], tabs_value: str, table_data: list[dict]
    ) -> list[dict]:
        """
        Update scores table score and rankings when data store updates.

        Parameters
        ----------
        stored_weights
            Stored weight values for `table_id`.
        tabs_value
            Selected tab value (unused; triggers recompute on tab change).
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
        Input(f"{benchmark_table_id}-weight-store", "data"),
        Input("all-tabs", "value"),
        State(benchmark_table_id, "data"),
        State(category_table_id, "data"),
        State(f"{category_table_id}-weight-store", "data"),
        prevent_initial_call="initial_duplicate",
    )
    def update_category_from_benchmark(
        benchmark_weights: dict[str, float] | None,
        tabs_value: str,
        benchmark_data: list[dict],
        category_data: list[dict],
        category_weights: dict[str, float] | None,
    ) -> list[dict]:
        """
        Update category summary from a benchmark table.

        Parameters
        ----------
        benchmark_weights
            Metric weight mapping for the benchmark table.
        tabs_value
            Selected tab value (unused; triggers recompute on tab switch).
        benchmark_data
            Rows from the benchmark table containing metric columns and Score.
        category_data
            Current rows for the category summary table.
        category_weights
            Weight mapping for category columns used to recompute Score.

        Returns
        -------
        list[dict]
            Updated category table data and style tuple.
        """
        # Only handle metric-weight updates; ignore tab-change mounts/renders
        if ctx.triggered_id != f"{benchmark_table_id}-weight-store":
            raise PreventUpdate

        # Compute MLIP -> Score using latest metric weights for deterministic update
        b_weights = benchmark_weights if benchmark_weights else {}
        recomputed = calc_scores([row.copy() for row in benchmark_data], b_weights)
        benchmark_scores = {row["MLIP"]: row.get("Score") for row in recomputed}

        # Inject into the appropriate column for each MLIP
        for row in category_data:
            mlip = row["MLIP"]
            if mlip in benchmark_scores and benchmark_scores[mlip] is not None:
                row[benchmark_column] = benchmark_scores[mlip]

        # Recompute category Score and Rank using its existing weights
        weights = category_weights if category_weights else {}
        category_data = calc_scores(category_data, weights)
        category_data = calc_ranks(category_data)
        style = get_table_style(category_data)

        return category_data, style


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
