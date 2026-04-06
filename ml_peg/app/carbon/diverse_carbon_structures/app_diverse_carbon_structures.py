"""Run diverse carbon structures benchmark app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.app.utils.load import collect_traj_assets, read_density_plot_for_model
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "diverse carbon structures"
DATA_PATH = APP_ROOT / "data" / "carbon" / "diverse_carbon_structures"
ASSETS_PREFIX = "assets/carbon/diverse_carbon_structures"

# (category_folder, metric_name, figure_filename)
ENERGY_CONFIG = [
    ("sp2", "sp2 bonded MAE", "figure_sp2_density.json"),
    ("sp3", "sp3 bonded MAE", "figure_sp3_density.json"),
    ("amorphous", "amorphous/liquid MAE", "figure_amorphous_density.json"),
    ("general_bulk", "general bulk MAE", "figure_general_bulk_density.json"),
    (
        "general_clusters",
        "general clusters MAE",
        "figure_general_clusters_density.json",
    ),
]


class DiverseCarbonStructuresApp(BaseApp):
    """Diverse carbon structures benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Build {model: {metric_name: density_graph}} for plot_from_table_cell
        cell_plots: dict[str, dict] = {model: {} for model in MODELS}
        for cat_folder, metric_name, figure_file in ENERGY_CONFIG:
            for model in MODELS:
                graph = read_density_plot_for_model(
                    filename=DATA_PATH / figure_file,
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-{cat_folder}",
                )
                if graph is not None:
                    cell_plots[model][metric_name] = graph

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_plots,
        )

        # Wire each energy density scatter to the structure viewer
        for cat_folder, _, _ in ENERGY_CONFIG:
            struct_trajs = collect_traj_assets(
                data_path=DATA_PATH,
                assets_prefix=ASSETS_PREFIX,
                models=MODELS,
                traj_dirname=f"density_traj_{cat_folder}",
            )
            for model, asset_paths in struct_trajs.items():
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{cat_folder}",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=asset_paths,
                    mode="traj",
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
