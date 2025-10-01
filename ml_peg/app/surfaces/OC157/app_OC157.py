"""Run OC157 app."""

from __future__ import annotations

import json

from dash import Dash
from dash.html import Div
import numpy as np

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.build_components import (
    build_metric_weight_components,
    build_threshold_inputs_under_table,
)
from ml_peg.app.utils.load import (
    get_metric_columns_from_json,
    read_plot,
)
from ml_peg.calcs.models.models import MODELS

BENCHMARK_NAME = "OC157"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "OC157"


class OC157App(BaseApp):
    """OC157 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_rel_energies.json", id=f"{BENCHMARK_NAME}-figure"
        )

        structs_dir = DATA_PATH / list(MODELS.keys())[0]

        # Assets dir will be parent directory
        structs = list(
            np.repeat(
                [
                    f"assets/surfaces/OC157/{list(MODELS.keys())[0]}/{i}.xyz"
                    for i in range(len(list(structs_dir.glob("*.xyz"))))
                ],
                3,
            )
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter, "Ranking Error": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="traj",
        )

        # Inline threshold/weight components register their own callbacks


def get_app() -> OC157App:
    """
    Get OC157 benchmark app layout and callback registration.

    Returns
    -------
    OC157App
        Benchmark layout and callback registration.
    """
    # Load normalization ranges from the normalized table JSON
    normalized_table_path = DATA_PATH / "oc157_normalized_metrics_table.json"
    normalization_ranges = {}

    try:
        with open(normalized_table_path) as f:
            normalized_table_data = json.load(f)
            normalization_ranges = normalized_table_data.get("normalization_ranges", {})
    except FileNotFoundError:
        print(
            f"Warning: {normalized_table_path} not found. Using empty normalization ranges."
        )

    # Get metric order from JSON; let results table size responsively
    metric_columns = get_metric_columns_from_json(
        DATA_PATH / "oc157_normalized_metrics_table.json"
    )

    # Embedded controls: handled directly inside the table, so omit external rows

    # Build controls table component
    threshold_inputs = build_threshold_inputs_under_table(
        table_columns=metric_columns,
        normalization_ranges=normalization_ranges,
        table_id=f"{BENCHMARK_NAME}-table",
        column_widths=None,
        use_overlay=True,
        enable_weight_overlay=True,
    )
    metric_weight_inputs = build_metric_weight_components(
        header="Metric weights",
        metrics=metric_columns,
        table_id=f"{BENCHMARK_NAME}-table",
        register_table_callbacks=False,
        column_widths=None,
        use_overlay=True,
    )

    return OC157App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting relative energies between 3 structures for 157 "
            "molecule-surface combinations. Use the normalized table below for advanced scoring."
        ),
        table_path=DATA_PATH
        / "oc157_normalized_metrics_table.json",  # Use normalized table
        extra_components=[
            threshold_inputs,
            metric_weight_inputs,
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        column_widths=None,  # let results table size responsively
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    oc157_app = get_app()
    full_app.layout = oc157_app.layout
    oc157_app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
