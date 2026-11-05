"""Analysis of bulk water benchmark."""

from __future__ import annotations

import json

from dash import Input, Output, callback, dcc
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
)
from ml_peg.app.utils.water_comparison import render_subplot_component
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Bulk Water"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/aqueous_solutions/bulk_water.html"
DATA_PATH = APP_ROOT / "data" / "aqueous_solutions" / "bulk_water"
INFO_PATH = DATA_PATH / "info.json"
METRICS = ["rdf_score", "vdos_score", "vacf_score"]


class BulkWaterApp(BaseApp):
    """Bulk Water benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Assets dir will be parent directory - individual files for each system

        cell_plots = {}
        interactive_data = {}

        # Load the complex interactive data from your @cell_to_bar decorator

        for metric in METRICS:
            try:
                with open(DATA_PATH / f"{metric}_bar.json") as f:
                    interactive_data[metric] = json.load(f)
            except FileNotFoundError:
                interactive_data[metric] = {"models": {}, "metrics": {}}

            # Extract pre-generated figures for each model-metric pair

            for model_name, model_data in (
                interactive_data[metric].get("models", {}).items()
            ):
                if model_name not in cell_plots:
                    cell_plots[model_name] = {}

                figures = model_data.get("figures", {})
                for metric_key, figure_data in figures.items():
                    cell_plots[model_name][metric_key] = dcc.Graph(
                        id=f"{BENCHMARK_NAME}-figure-bar-{model_name}",
                        figure=figure_data,
                    )

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_plots,
        )

        # Add click callbacks for each model's bar charts
        for model_name in cell_plots.keys():

            @callback(
                Output(
                    f"{BENCHMARK_NAME}-figure-plot-container-{model_name}", "children"
                ),
                Input(f"{BENCHMARK_NAME}-figure-bar-{model_name}", "clickData"),
                prevent_initial_call=True,
            )
            def show_comparison_plot(click_data):
                return render_subplot_component(click_data)


def get_app() -> BulkWaterApp:
    """
    Get Bulk Water benchmark app layout and callback registration.

    Returns
    -------
    BulkWaterApp
        Benchmark layout and callback registration.
    """
    extra_components = [
        Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
    ]
    for model_name in MODELS:
        extra_components.append(
            Div(id=f"{BENCHMARK_NAME}-figure-plot-container-{model_name}")
        )

    return BulkWaterApp(
        name=BENCHMARK_NAME,
        description="Bulk Water score.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "bulk_water_metrics_table.json",
        info_path=INFO_PATH,
        extra_components=extra_components,
    )
