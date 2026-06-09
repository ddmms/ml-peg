"""Run diverse carbon structures benchmark app."""

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
BENCHMARK_NAME = "diverse carbon structures"
DATA_PATH = APP_ROOT / "data" / "carbon" / "diverse_carbon_structures"
ASSETS_PREFIX = "assets/carbon/diverse_carbon_structures"

# (category_folder, metric_name matching table column and scatter JSON key)
ENERGY_CONFIG = [
    ("sp2", "sp2 Bonded MAE"),
    ("sp3", "sp3 Bonded MAE"),
    ("amorphous", "Amorphous/Liquid MAE"),
    ("general_bulk", "General Bulk MAE"),
    ("general_clusters", "General Clusters MAE"),
]


class DiverseCarbonStructuresApp(BaseApp):
    """Diverse carbon structures benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_path = DATA_PATH / "diverse_carbon_structures_scatter.json"
        cell_plots: dict[str, dict] = {model: {} for model in MODELS}

        if scatter_path.exists():
            with scatter_path.open(encoding="utf8") as f:
                scatter_data = json.load(f)
            models_data = scatter_data.get("models", {})
            for cat_folder, metric_name in ENERGY_CONFIG:
                for model in MODELS:
                    fig_dict = (
                        models_data.get(model, {}).get("figures", {}).get(metric_name)
                    )
                    if fig_dict:
                        cell_plots[model][metric_name] = dcc.Graph(
                            figure=go.Figure(fig_dict),
                            id=f"{BENCHMARK_NAME}-{model}-{cat_folder}",
                        )

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_plots,
        )

        for cat_folder, _ in ENERGY_CONFIG:
            for model in MODELS:
                model_dir = DATA_PATH / model / cat_folder
                if model_dir.exists():
                    structs = sorted(model_dir.glob("*.xyz"), key=lambda p: int(p.stem))
                    asset_paths = [
                        f"/{ASSETS_PREFIX}/{model}/{cat_folder}/{p.name}"
                        for p in structs
                    ]
                    struct_from_scatter(
                        scatter_id=f"{BENCHMARK_NAME}-{model}-{cat_folder}",
                        struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                        structs=asset_paths,
                    )


def get_app() -> DiverseCarbonStructuresApp:
    """
    Get diverse carbon structures benchmark app layout and callback registration.

    Returns
    -------
    DiverseCarbonStructuresApp
        Benchmark layout and callback registration.
    """
    return DiverseCarbonStructuresApp(
        name="Diverse Carbon Structures",
        description=(
            "Diverse benchmark across many carbon bonding environments: sp² (graphene, "
            "graphite, fullerenes, nanotubes), sp³ (diamond, high-pressure phases), "
            "amorphous/liquid, general bulk crystals, and small clusters. "
        ),
        table_path=DATA_PATH / "diverse_carbon_structures_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )
