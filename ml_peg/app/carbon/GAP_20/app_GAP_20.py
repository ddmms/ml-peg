"""Run GAP-20 carbon benchmark app."""

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
BENCHMARK_NAME = "GAP-20 carbon"
DATA_PATH = APP_ROOT / "data" / "carbon" / "GAP_20"
METRIC_KEY = "Energy MAE"


class GAP20App(BaseApp):
    """GAP-20 carbon benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_path = DATA_PATH / "GAP_20_scatter.json"
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
                    f"/assets/carbon/GAP_20/{model}/{p.name}" for p in structs
                ]
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-scatter",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=asset_paths,
                )


def get_app() -> GAP20App:
    """
    Get GAP-20 carbon benchmark app layout and callback registration.

    Returns
    -------
    GAP20App
        Benchmark layout and callback registration.
    """
    return GAP20App(
        name="GAP-20 Carbon",
        description=(
            "Diverse carbon benchmark from the GAP-20 training dataset: sp2 (graphene, "
            "graphite, fullerenes, nanotubes), sp³ (diamond, high-pressure phases, "
            "crystalline allotropes), amorphous/liquid, and surface/defect structures. "
            "Energies relative to isolated carbon atom (DFT/optB88vdW)."
        ),
        table_path=DATA_PATH / "GAP_20_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )
