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

BENCHMARK_NAME = "Surface Alloy Reorganisation"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#surface-alloy-reorganisation"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "surface_alloy_reorganisation"
INFO_PATH = DATA_PATH / "info.json"

# Input geometries are shared by all models, so trajectories (initial, transition,
# final) are written once from the mock outputs, named by reaction identifier.
STRUCT_TEMPLATE = "/assets/surfaces/surface_alloy_reorganisation/mock/{id}.xyz"


class SurfaceAlloyReorganisationApp(BaseApp):
    """Surface Alloy Reorganisation benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        energies_id = f"{BENCHMARK_NAME}-energies-figure"
        barriers_id = f"{BENCHMARK_NAME}-barriers-figure"

        energies_scatter = read_plot(
            DATA_PATH / "figure_surface_alloy_energies.json", id=energies_id
        )
        barriers_scatter = read_plot(
            DATA_PATH / "figure_surface_alloy_barriers.json", id=barriers_id
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "MAE_energies": energies_scatter,
                "MAE_barriers": barriers_scatter,
            },
        )

        # Match structures by the reaction identifier in each point's customdata,
        # as the energy plot excludes reactions with zero reference energy
        for scatter_id in (energies_id, barriers_id):
            struct_from_scatter(
                scatter_id=scatter_id,
                struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                mode="traj",
                struct_template=STRUCT_TEMPLATE,
            )


def get_app() -> SurfaceAlloyReorganisationApp:
    """
    Get Surface alloy reorganisation benchmark app layout and callback registration.

    Returns
    -------
    SurfaceAlloyReorganisationApp
        Benchmark layout and callback registration.
    """
    return SurfaceAlloyReorganisationApp(
        name=BENCHMARK_NAME,
        description=(
            "Reaction energies and barriers for reorganisation reactions of "
            "surface alloys"
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "surface_alloy_reorganisation_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
