"""Run Thermal thermal_Conductivity app."""

from __future__ import annotations

import json

from dash.dcc import Graph
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Thermal Conductivity"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/"
    "benchmarks/bulk_crystal.html#thermal-conductivity"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "thermal_conductivity"
INFO_PATH = DATA_PATH / "info.json"


class ThermalConductivityApp(BaseApp):
    """Thermal Conductivity benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Shared log-log parity plots (all models on one figure): full and fast.
        scatter = read_plot(
            DATA_PATH / "figure_thermal_conductivity.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        fast_scatter = read_plot(
            DATA_PATH / "figure_fast_thermal_conductivity.json",
            id=f"{BENCHMARK_NAME}-fast-figure",
        )

        # Per-model figures: kSRME / fast kSRME violins, and a stability-status
        # parity for Instability/Failure (points coloured stable/unstable).
        violin_file = DATA_PATH / "figure_ksrme_violin.json"
        fast_violin_file = DATA_PATH / "figure_fast_ksrme_violin.json"
        status_file = DATA_PATH / "figure_status_parity.json"
        violin_data = (
            json.loads(violin_file.read_text()) if violin_file.exists() else {}
        )
        fast_violin_data = (
            json.loads(fast_violin_file.read_text())
            if fast_violin_file.exists()
            else {}
        )
        status_data = (
            json.loads(status_file.read_text()) if status_file.exists() else {}
        )

        # Per-plot-type figure IDs are shared across models (only one is shown at a
        # time), so a single structure-viewer callback per type suffices.
        violin_id = f"{BENCHMARK_NAME}-ksrme-violin"
        fast_violin_id = f"{BENCHMARK_NAME}-fast-ksrme-violin"
        status_id = f"{BENCHMARK_NAME}-status"

        cell_to_plot: dict[str, dict] = {}
        for model in MODELS:
            plots = {"kSRE": scatter, "Fast kSRE": fast_scatter}
            if model in violin_data:
                plots["kSRME"] = Graph(id=violin_id, figure=violin_data[model])
            if model in fast_violin_data:
                plots["Fast kSRME"] = Graph(
                    id=fast_violin_id, figure=fast_violin_data[model]
                )
            if model in status_data:
                status_graph = Graph(id=status_id, figure=status_data[model])
                plots["Instability"] = status_graph
                plots["Failure"] = status_graph
            cell_to_plot[model] = plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_to_plot,
        )

        # Structure viewer for every plot. Keyed on each point's material id
        # (customdata), not its index: the violin/status plots filter and reorder
        # points, so index-based matching would show the wrong crystal.
        struct_template = "/assets/bulk_crystal/thermal_conductivity/{id}.xyz"
        struct_placeholder = f"{BENCHMARK_NAME}-struct-placeholder"
        scatter_ids = [f"{BENCHMARK_NAME}-figure", f"{BENCHMARK_NAME}-fast-figure"]
        if violin_data:
            scatter_ids.append(violin_id)
        if fast_violin_data:
            scatter_ids.append(fast_violin_id)
        if status_data:
            scatter_ids.append(status_id)
        for scatter_id in scatter_ids:
            struct_from_scatter(
                scatter_id, struct_placeholder, struct_template=struct_template
            )


def get_app() -> ThermalConductivityApp:
    """
    Get Thermal Conductivity benchmark app layout and callback registration.

    Returns
    -------
    ThermalConductivityApp
        Benchmark layout and callback registration.
    """
    return ThermalConductivityApp(
        name=BENCHMARK_NAME,
        description="Thermal conductivity for 103 binary crystals.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "thermal_conductivity.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
