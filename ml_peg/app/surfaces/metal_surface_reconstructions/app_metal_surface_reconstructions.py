

"""Run Metal surface reconstructions app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Metal Surfaces"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#metal_surfaces"
)
DATA_PATH = APP_ROOT / "data" / "surfaces" / "metal_surfaces"



class Metal_surface_reconstructions(BaseApp):
    """Metal surface reconstructions benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "slab_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Assets dir will be parent directory - individual files for each system
        structs_dir = DATA_PATH / MODELS[0]
        structs = [
            f"assets/surfaces/metal_surfaces/{MODELS[0]}/{struct_file.stem}.xyz"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter, "Displacement": scatter},
        )
        

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )



def get_app() -> Metal_surface_reconstructions:
    """
    Get Metal surface reconstructions benchmark app layout and callback registration.

    Returns
    -------
    Metal surface reconstructions App
        Benchmark layout and callback registration.
    """
    return Metal_surface_reconstructions(
        name=BENCHMARK_NAME,
        description="Energies for two surface reconstuctions.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "metal_surfaces_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )

