"""Analysis of bulk water benchmark."""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import json

from dash import Input, Output, callback, dcc, html
from dash.html import Div
import matplotlib.pyplot as plt

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
)
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Bulk Water"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/aqueous_solutions/bulk_water.html"
DATA_PATH = APP_ROOT / "data" / "aqueous_solutions" / "bulk_water"
METRICS = ["rdf_score", "vdos_score", "vacf_score"]


def render_subplot_component(click_data: dict) -> html.Div | None:
    """
    Render comparison plot for clicked bar.

    Parameters
    ----------
    click_data
        Click data from bar chart containing data.

    Returns
    -------
    html.Div | None
        Component containing comparison plot.
    """
    if not click_data or "points" not in click_data:
        return html.Div("Click on a bar to see comparison.")

    point = click_data["points"][0]
    if "customdata" not in point or len(point["customdata"]) < 2:
        return html.Div(f"No data available: {point.keys()}")

    rdf_data = point["customdata"][1]  # Get RDF data from customdata
    pair_name = point["customdata"][0]  # Get pair name

    # Create matplotlib figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax1, ax2 = axes

    # Load data
    x_values = rdf_data["x_values"]
    ref_values = rdf_data["ref"]
    pred_values = rdf_data["pred"]
    error = rdf_data["error"]
    xlabel = rdf_data["xlabel"]
    ylabel = rdf_data["ylabel"]

    # Plot Reference and Predicted RDFs
    ax1.plot(x_values, ref_values, "b-", label="Reference", linewidth=2)
    ax1.plot(x_values, pred_values, "r--", label="Predicted", linewidth=2)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel)
    ax1.set_title(f"Comparison: {pair_name}")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Error
    ax2.plot(x_values, error, "k-", label="Error", linewidth=2)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(rf"${{\Delta}}${ylabel}")
    ax2.set_title("Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Check if xlim is provided
    if "xlim" in rdf_data:
        ax1.set_xlim(rdf_data["xlim"])
        ax2.set_xlim(rdf_data["xlim"])

    # Convert to base64 for display
    buffer = BytesIO()
    plt.tight_layout()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode()
    plt.close()

    return html.Div(
        [
            html.H4(f"Comparison: {pair_name}"),
            html.Img(
                src=f"data:image/png;base64,{image_base64}",
                style={"maxWidth": "100%", "border": "1px solid #ccc"},
            ),
        ]
    )


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

        print("Cell plots prepared:", cell_plots.keys())
        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_plots,
        )
        print(cell_plots.keys())

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
        Div(
            id=f"{BENCHMARK_NAME}-figure-bar",
            children="Click on a metric to see bar plot.",
        ),
        Div(
            id=f"{BENCHMARK_NAME}-figure-plot-container",
            children="Click on a bar to see comparison.",
        ),
    ]
    for model_name in MODELS:
        extra_components.append(Div(id=f"{BENCHMARK_NAME}-figure-bar-{model_name}"))
        extra_components.append(Div(id=f"{BENCHMARK_NAME}-figure-bar-{model_name}"))
        extra_components.append(
            Div(id=f"{BENCHMARK_NAME}-figure-plot-container-{model_name}")
        )

    return BulkWaterApp(
        name=BENCHMARK_NAME,
        description="Bulk Water score.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "bulk_water_metrics_table.json",
        extra_components=extra_components,
    )
