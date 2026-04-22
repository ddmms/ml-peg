"""Run graphene oxide benchmark app."""

from __future__ import annotations

import json

from dash import dcc
from dash.html import Div
import plotly.graph_objects as go

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "graphene oxide"
DATA_PATH = APP_ROOT / "data" / "carbon" / "graphene_oxide"
METRIC_KEY = "Energy MAE"


class GrapheneOxideApp(BaseApp):
    """Graphene oxide benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_path = DATA_PATH / "graphene_oxide_scatter.json"
        cell_plots: dict[str, dict] = {model: {} for model in MODELS}

        if scatter_path.exists():
            with scatter_path.open(encoding="utf8") as f:
                scatter_data = json.load(f)
            models_data = scatter_data.get("models", {})
            for model in MODELS:
                fig_dict = models_data.get(model, {}).get("figures", {}).get(METRIC_KEY)
                if fig_dict:
                    cell_plots[model][METRIC_KEY] = dcc.Graph(
                        figure=go.Figure(fig_dict),
                        id=f"{BENCHMARK_NAME}-{model}-scatter",
                    )

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_plots,
        )

        for model in MODELS:
            model_dir = DATA_PATH / model
            if model_dir.exists():
                structs = sorted(model_dir.glob("*.xyz"), key=lambda p: int(p.stem))
                asset_paths = [
                    f"/assets/carbon/graphene_oxide/{model}/{p.name}" for p in structs
                ]
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-scatter",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=asset_paths,
                )


def get_app() -> GrapheneOxideApp:
    """
    Get graphene oxide benchmark app layout and callback registration.

    Returns
    -------
    GrapheneOxideApp
        Benchmark layout and callback registration.
    """
    return GrapheneOxideApp(
        name="Graphene Oxide",
        description=(
            "Graphene oxide benchmark covering 3813 structures with varying oxidation "
            "coverage (O/C ratio 0.1–0.5), hydroxyl/epoxide ratio (OH/O 0–1), and "
            "edge functionalisation. Energies relative to isolated C, H, O atoms "
            "(DFT/PBE)."
        ),
        table_path=DATA_PATH / "graphene_oxide_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )
