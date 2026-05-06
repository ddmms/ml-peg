"""Run elasticity benchmark app."""

from __future__ import annotations

import json

from dash import Dash
from dash.dcc import Graph
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.app.utils.load import read_density_plot_for_model
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Elasticity"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#elasticity"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "elasticity"


class ElasticityApp(BaseApp):
    """Elasticity benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Load pre-generated per-model violin figure for elastic tensor only
        tensor_violin_data: dict[str, dict] = {}
        tensor_violin_path = DATA_PATH / "figure_elastic_tensor_violin.json"
        if tensor_violin_path.exists():
            with open(tensor_violin_path) as f:
                tensor_violin_data = json.load(f)

        # Build plots: density scatter for bulk/shear, violin for elastic tensor
        density_plots = {}
        for model in MODELS:
            plots = {
                "Bulk modulus MAE": read_density_plot_for_model(
                    filename=DATA_PATH / "figure_bulk_density.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-bulk-figure",
                ),
                "Shear modulus MAE": read_density_plot_for_model(
                    filename=DATA_PATH / "figure_shear_density.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-shear-figure",
                ),
            }
            fig_dict = tensor_violin_data.get(model)
            if fig_dict is not None:
                plots["Elasticity tensor MAE"] = Graph(
                    id=f"{BENCHMARK_NAME}-{model}-tensor-violin",
                    figure=fig_dict,
                )
            model_plots = {k: v for k, v in plots.items() if v is not None}
            if model_plots:
                density_plots[model] = model_plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=density_plots,
        )

        # Structure visualization: density scatter (traj mode) and violin (struct mode)
        for model in MODELS:
            for prop, traj_subdir in [
                ("bulk", "density_bulk"),
                ("shear", "density_shear"),
            ]:
                traj_dir = DATA_PATH / model / traj_subdir
                if not traj_dir.exists():
                    continue
                traj_files = sorted(
                    traj_dir.glob("*.extxyz"), key=lambda p: int(p.stem)
                )
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{prop}-figure",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=[
                        f"/assets/bulk_crystal/elasticity/{model}/{traj_subdir}/{p.name}"
                        for p in traj_files
                    ],
                    mode="traj",
                )

            violin_struct_dir = DATA_PATH / model
            xyz_files = (
                sorted(violin_struct_dir.glob("*.xyz"), key=lambda p: int(p.stem))
                if violin_struct_dir.exists()
                else []
            )
            if xyz_files:
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-tensor-violin",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=[
                        f"/assets/bulk_crystal/elasticity/{model}/{p.name}"
                        for p in xyz_files
                    ],
                    mode="struct",
                )


def get_app() -> ElasticityApp:
    """
    Get elasticity benchmark app layout and callback registration.

    Returns
    -------
    ElasticityApp
        Benchmark layout and callback registration.
    """
    return ElasticityApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance when predicting VRH bulk and shear moduli for crystalline "
            "materials compared against Materials Project reference data."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "elasticity_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    elasticity_app = get_app()
    full_app.layout = elasticity_app.layout
    elasticity_app.register_callbacks()
    full_app.run(port=8054, debug=True)
