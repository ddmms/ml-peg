"""Analysis of copper water interface benchmark."""

from __future__ import annotations

import base64
from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import json

from dash import Input, Output, State, callback, dcc, html, no_update
from dash.html import Div
import matplotlib.pyplot as plt

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_with_download_controls
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.plot_helpers import INSTRUCTION_STYLE, TABLE_HINT
from ml_peg.app.utils.register_callbacks import register_plot_download_callbacks
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Copper Water Interface"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces/copper_water_interface.html"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "copper_water_interface"
INFO_PATH = DATA_PATH / "info.json"
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


class CopperWaterInterfaceApp(BaseApp):
    """Copper Water Interface benchmark app layout and callbacks."""

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

        # Dipole histogram shown when a dipole column is clicked.
        dipole_hist = read_plot(
            DATA_PATH / "figure_hist_dipoles.json",
            id=f"{BENCHMARK_NAME}-figure-hist-dipoles",
        )
        column_to_dipole_plot = {
            "stdev_dipole_z_deviation": dipole_hist,
            "Fraction Breakdown Candidates": dipole_hist,
        }

        # A single callback owns the table's `active_cell` (it inputs and resets it)
        # and drives BOTH the main bar-plot placeholder and the dipole-histogram
        # placeholder. It must not be split into two callbacks: Dash chains a
        # callback whose input is another callback's output, so a separate
        # `active_cell`-input callback would only ever run *after* this one reset
        # `active_cell` to None and would never see the clicked cell. Outputs left
        # unchanged use `no_update` so clicking one column never clears the other
        # placeholder.
        register_plot_download_callbacks()

        @callback(
            Output(f"{BENCHMARK_NAME}-figure-placeholder", "children"),
            Output(f"{BENCHMARK_NAME}-figure-placeholder-dipole", "children"),
            Output(self.table_id, "active_cell"),
            Input(self.table_id, "active_cell"),
            State(self.table_id, "data"),
        )
        def show_cell_plot(active_cell, current_table_data):
            """
            Render bar-plot / dipole-histogram content for the clicked table cell.

            Parameters
            ----------
            active_cell
                Clicked cell in Dash table.
            current_table_data
                Current table data (includes live updates from callbacks).

            Returns
            -------
            tuple
                Main placeholder children, dipole placeholder children, and the reset
                ``active_cell`` value. Unchanged placeholders use ``no_update``.
            """
            hint = Div(TABLE_HINT, style=INSTRUCTION_STYLE)
            if not active_cell:
                return hint, no_update, None

            column_id = active_cell.get("column_id", None)
            row_id = active_cell.get("row_id", None)
            row_index = active_cell.get("row", None)

            # Dipole columns drive the dipole placeholder only.
            if column_id in column_to_dipole_plot:
                dipole_plot = plot_with_download_controls(
                    column_to_dipole_plot[column_id]
                )
                return no_update, dipole_plot, None

            # Bar-metric columns drive the main placeholder only.
            if current_table_data and row_index is not None:
                try:
                    cell_value = current_table_data[row_index].get(column_id)
                    if cell_value is None:
                        return Div("No data available for this model."), no_update, None
                except (IndexError, KeyError, TypeError):
                    pass

            if row_id in cell_plots and column_id in cell_plots[row_id]:
                bar_plot = plot_with_download_controls(cell_plots[row_id][column_id])
                return bar_plot, no_update, None
            return hint, no_update, None

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


def get_app() -> CopperWaterInterfaceApp:
    """
    Get Copper Water Interface benchmark app layout and callback registration.

    Returns
    -------
    CopperWaterInterfaceApp
        Benchmark layout and callback registration.
    """
    extra_components = [
        Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        Div(
            id=f"{BENCHMARK_NAME}-figure-placeholder-dipole",
            children=Div(TABLE_HINT, style=INSTRUCTION_STYLE),
        ),
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

    return CopperWaterInterfaceApp(
        name=BENCHMARK_NAME,
        description="Copper Water Interface score.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "copper_water_interface_metrics_table.json",
        info_path=INFO_PATH,
        extra_components=extra_components,
    )
