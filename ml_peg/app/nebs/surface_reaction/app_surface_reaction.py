"""Run Li diffusion app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Surface reaction"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nebs.html#surface-reactionn"
DATA_PATH = APP_ROOT / "data" / "nebs" / "surface_reaction"


class SurfaceReactionApp(BaseApp):
    """Surface reaction benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_plots = {
            model: {
                "Forward Barrier Error": read_plot(
                    DATA_PATH / f"figure_{model}_neb_transfer_id_601_1482_1_211-5.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-transfer_id_601_1482_1_211-5",
                ),
#                "Path C error": read_plot(
#                    DATA_PATH / f"figure_{model}_neb_c.json",
#                    id=f"{BENCHMARK_NAME}-{model}-figure-c",
#                ),
            }
            for model in MODELS
        }

        # Assets dir will be parent directory
        assets_dir = "assets/nebs/surface_reaction"
        structs = {
            model: {
                "Forward Barrier Error": f"{assets_dir}/{model}/{model}-transfer_id_601_1482_1_211-5.xyz",
#                "Path C error": f"{assets_dir}/{model}/{model}-c-neb-band.extxyz",
            }
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=scatter_plots,
        )

        for model in MODELS:
            for path in ("transfer_id_601_1482_1_211-5", ):
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-figure-{path}",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=structs[model][f"Forward Barrier Error"],
                    mode="traj",
                )


def get_app() -> SurfaceReactionApp:
    """
    Get Li diffusion benchmark app layout and callback registration.

    Returns
    -------
    LiDiffusionApp
        Benchmark layout and callback registration.
    """
    return SurfaceReactionApp(
        name=BENCHMARK_NAME,
        description=("Performance in predicting energy barriers for Surface reaction."),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "surface_reaction_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    surface_reaction_app = get_app()
    full_app.layout = surface_reaction_app.layout
    surface_reaction_app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
