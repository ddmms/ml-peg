"""Low-dimensional (2D/1D) crystal relaxation benchmark app."""

from __future__ import annotations

import json

from dash import Dash
from dash.dcc import Graph
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.app.utils.load import collect_traj_assets, read_density_plot_for_model
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Low-Dimensional Relaxation"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html"
    "#low-dimensional-relaxation"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "low_dimensional_relaxation"
ASSETS_PREFIX = "/assets/bulk_crystal/low_dimensional_relaxation"

# (plot_json_filename, metric_cell_name, plot_id_suffix, traj_dirname)
PLOT_CONFIGS = [
    ("figure_area_2d.json", "Area MAE (2D)", "area-2d", "density_traj_area_2d"),
    (
        "figure_energy_2d.json",
        "Energy MAE (2D)",
        "energy-2d",
        "density_traj_energy_2d",
    ),
    (
        "figure_length_1d.json",
        "Length MAE (1D)",
        "length-1d",
        "density_traj_length_1d",
    ),
    (
        "figure_energy_1d.json",
        "Energy MAE (1D)",
        "energy-1d",
        "density_traj_energy_1d",
    ),
]

# (figure_json_filename, metric_cell_name, plot_id_suffix, mirror_dirname)
# The figure JSON is a dict of {model_name: plotly_figure_dict} so each cell
# can render its own model's violin without filtering.
VIOLIN_CONFIGS = [
    (
        "figure_force_violin_2d.json",
        "Convergence (2D)",
        "force-violin-2d",
        "force_violin_2d",
    ),
    (
        "figure_force_violin_1d.json",
        "Convergence (1D)",
        "force-violin-1d",
        "force_violin_1d",
    ),
]


class LowDimensionalRelaxationApp(BaseApp):
    """Low-dimensional relaxation benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        density_plots: dict = {}
        for model in MODELS:
            plots = {}
            for filename, metric_name, plot_id_suffix, _ in PLOT_CONFIGS:
                plot_path = DATA_PATH / filename
                if plot_path.exists():
                    graph = read_density_plot_for_model(
                        filename=plot_path,
                        model=model,
                        id=f"{BENCHMARK_NAME}-{model}-{plot_id_suffix}-figure",
                    )
                    if graph is not None:
                        plots[metric_name] = graph
            if plots:
                density_plots[model] = plots

        # Convergence violin plots: one figure JSON per dim, keyed by model.
        for filename, metric_name, plot_id_suffix, _ in VIOLIN_CONFIGS:
            plot_path = DATA_PATH / filename
            if not plot_path.exists():
                continue
            with open(plot_path) as f:
                figures_per_model = json.load(f)
            for model, fig_dict in figures_per_model.items():
                if model not in MODELS:
                    continue
                graph = Graph(
                    id=f"{BENCHMARK_NAME}-{model}-{plot_id_suffix}-figure",
                    figure=fig_dict,
                )
                density_plots.setdefault(model, {})[metric_name] = graph

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=density_plots,
        )

        for _, _, plot_id_suffix, traj_dirname in PLOT_CONFIGS:
            trajs = collect_traj_assets(
                data_path=DATA_PATH,
                assets_prefix=ASSETS_PREFIX,
                models=MODELS,
                traj_dirname=traj_dirname,
            )
            for model, paths in trajs.items():
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{plot_id_suffix}-figure",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=paths,
                    mode="traj",
                )

        # Violin click-to-structure: each violin point corresponds to one
        # mirrored xyz (numbered 0.xyz, 1.xyz, ...) in the same order as the
        # violin's samples. A click on point N renders structs[N].
        for _, _, plot_id_suffix, mirror_dirname in VIOLIN_CONFIGS:
            mirrors = collect_traj_assets(
                data_path=DATA_PATH,
                assets_prefix=ASSETS_PREFIX,
                models=MODELS,
                traj_dirname=mirror_dirname,
                suffix=".xyz",
            )
            for model, paths in mirrors.items():
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{plot_id_suffix}-figure",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=paths,
                    mode="struct",
                )


def get_app() -> LowDimensionalRelaxationApp:
    """
    Get low-dimensional relaxation benchmark app.

    Returns
    -------
    LowDimensionalRelaxationApp
        Benchmark layout and callback registration.
    """
    return LowDimensionalRelaxationApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in relaxing low-dimensional crystal structures. "
            "2D structures are evaluated on area per atom, and 1D structures "
            "on length per atom. Structures from the Alexandria database are "
            "relaxed with cell masks to constrain relaxation to the appropriate "
            "dimensions and compared to PBE reference calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "low_dimensional_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    ld_app = get_app()
    full_app.layout = ld_app.layout
    ld_app.register_callbacks()
    full_app.run(port=8064, debug=True)
