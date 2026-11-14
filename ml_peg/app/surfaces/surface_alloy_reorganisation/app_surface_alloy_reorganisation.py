"""Run surface alloy reorganisation app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Surface Alloy Reorganisation"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#surface-alloy-reorganisation"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "surface_alloy_reorganisation"
INFO_PATH = DATA_PATH / "info.json"


class SurfaceAlloyReorganisationApp(BaseApp):
    """Surface Alloy Reorganisation benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_surface_alloy_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Assets dir will be parent directory - individual files for each system
        structs_dir = DATA_PATH / MODELS[0]
        structs = [
            f"/assets/surfaces/surface_alloy_reorganisation/{MODELS[0]}/{struct_file.stem}.xyz"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-energies-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-energies-struct-placeholder",
            structs=structs,
            mode="traj",
        )


def get_app() -> SurfaceAlloyReorganisationApp:
    """
    Get Surface alloy reorganisation benchmark app layout and callback registration.

    Returns
    -------
    Surface_Alloy_Reorganisation_App
        Benchmark layout and callback registration.
    """
    return SurfaceAlloyReorganisationApp(
        name=BENCHMARK_NAME,
        description="Reaction energies for reactions of surface alloys",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "surface_alloy_reorganisation_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-energies-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-energies-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
