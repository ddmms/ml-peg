"""Quasiharmonic benchmark app layout and callbacks."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Quasiharmonic"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html"
    "#quasiharmonic"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "quasiharmonic"


class QuasiharmonicApp(BaseApp):
    """Quasiharmonic benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Load pre-generated scatter plots
        scatter_lattice = None
        scatter_volume = None
        scatter_thermal_expansion = None
        scatter_bulk_modulus = None

        lattice_path = DATA_PATH / "figure_qha_lattice_constants.json"
        volume_path = DATA_PATH / "figure_qha_volume.json"
        thermal_expansion_path = DATA_PATH / "figure_qha_thermal_expansion.json"
        bulk_modulus_path = DATA_PATH / "figure_qha_bulk_modulus.json"

        if lattice_path.exists():
            scatter_lattice = read_plot(
                lattice_path,
                id=f"{BENCHMARK_NAME}-figure-lattice",
            )

        if volume_path.exists():
            scatter_volume = read_plot(
                volume_path,
                id=f"{BENCHMARK_NAME}-figure-volume",
            )

        if thermal_expansion_path.exists():
            scatter_thermal_expansion = read_plot(
                thermal_expansion_path,
                id=f"{BENCHMARK_NAME}-figure-thermal-expansion",
            )

        if bulk_modulus_path.exists():
            scatter_bulk_modulus = read_plot(
                bulk_modulus_path,
                id=f"{BENCHMARK_NAME}-figure-bulk-modulus",
            )

        # Build column-to-plot mapping
        column_to_plot = {}
        if scatter_lattice is not None:
            column_to_plot["Lattice constant MAE"] = scatter_lattice
        if scatter_volume is not None:
            column_to_plot["Volume MAE"] = scatter_volume
        if scatter_thermal_expansion is not None:
            column_to_plot["Thermal expansion MAE"] = scatter_thermal_expansion
        if scatter_bulk_modulus is not None:
            column_to_plot["Bulk modulus MAE"] = scatter_bulk_modulus

        if column_to_plot:
            plot_from_table_column(
                table_id=self.table_id,
                plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
                column_to_plot=column_to_plot,
            )


def get_app() -> QuasiharmonicApp:
    """
    Get quasiharmonic benchmark app.

    Returns
    -------
    QuasiharmonicApp
        Benchmark layout and callback registration.
    """
    return QuasiharmonicApp(
        name=BENCHMARK_NAME,
        description=(
            "Temperature-dependent thermodynamic properties using the "
            "quasiharmonic approximation (QHA). Evaluates MLIP predictions of "
            "lattice constants, volume, thermal expansion, bulk modulus, "
            "heat capacity, entropy, and Gruneisen parameter as a function "
            "of temperature and pressure."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "quasiharmonic_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    qha_app = get_app()
    full_app.layout = qha_app.layout
    qha_app.register_callbacks()
    full_app.run(port=8063, debug=True)
