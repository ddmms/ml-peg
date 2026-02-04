"""Low-dimensional (2D/1D) crystal relaxation benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Low-Dimensional Relaxation"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html"
    "#low-dimensional-relaxation"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "low_dimensional_relaxation"


class LowDimensionalRelaxationApp(BaseApp):
    """Low-dimensional relaxation benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Define plot paths and their metric mappings
        plot_configs = [
            ("figure_area_2d.json", "Area MAE (2D)", "area-2d"),
            ("figure_energy_2d.json", "Energy MAE (2D)", "energy-2d"),
            ("figure_length_1d.json", "Length MAE (1D)", "length-1d"),
            ("figure_energy_1d.json", "Energy MAE (1D)", "energy-1d"),
        ]

        # Build column-to-plot mapping
        column_to_plot = {}
        for filename, metric_name, plot_id_suffix in plot_configs:
            plot_path = DATA_PATH / filename
            if plot_path.exists():
                scatter = read_plot(
                    plot_path,
                    id=f"{BENCHMARK_NAME}-figure-{plot_id_suffix}",
                )
                column_to_plot[metric_name] = scatter

        if column_to_plot:
            plot_from_table_column(
                table_id=self.table_id,
                plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
                column_to_plot=column_to_plot,
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
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    ld_app = get_app()
    full_app.layout = ld_app.layout
    ld_app.register_callbacks()
    full_app.run(port=8064, debug=True)
