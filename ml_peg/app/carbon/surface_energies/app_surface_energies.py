"""Run surface energies app."""

from __future__ import annotations

from dash.dcc import Graph
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Surface Energies"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/carbon.html#surface-energies"
)
DATA_PATH = APP_ROOT / "data" / "carbon" / "surface_energies"
INFO_PATH = DATA_PATH / "info.json"

# Table column, figure JSON suffix
PLOTS = {
    "As-cut surface energy MAE": "as_cut",
    "Relaxed surface energy MAE": "relaxed",
}


def get_mock_structs(scatter: Graph) -> list[str]:
    """
    Get mock structure paths in the same order as a scatter plot's points.

    Parameters
    ----------
    scatter
        Scatter plot whose first hoverdata field is the system name.

    Returns
    -------
    list[str]
        Asset paths to the mock structure for each point.
    """
    traces = scatter.figure.data if scatter.figure else ()
    if not traces or traces[0].customdata is None:
        return []
    return [
        f"/assets/carbon/surface_energies/mock/{point[0]}.extxyz"
        for point in traces[0].customdata
    ]


class SurfaceEnergiesApp(BaseApp):
    """Surface energies benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatters = {
            column: read_plot(
                DATA_PATH / f"figure_surface_energies_{suffix}.json",
                id=f"{BENCHMARK_NAME}-{suffix}-figure",
            )
            for column, suffix in PLOTS.items()
        }

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot=scatters,
        )

        for scatter in scatters.values():
            struct_from_scatter(
                scatter_id=scatter.id,
                struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                structs=get_mock_structs(scatter),
                mode="struct",
            )


def get_app() -> SurfaceEnergiesApp:
    """
    Get surface energies benchmark app layout and callback registration.

    Returns
    -------
    SurfaceEnergiesApp
        Benchmark layout and callback registration.
    """
    return SurfaceEnergiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Single-point as-cut and relaxed surface energies for diamond {100} and"
            " graphite (0001), plus as-cut only for amorphous carbon, which has no"
            " relaxed reference. Reference data is optB88-vdW."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "surface_energies_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
        framework_ids="gap-20",
    )
