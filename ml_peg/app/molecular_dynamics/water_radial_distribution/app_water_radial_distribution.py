"""Run water radial distribution benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "WaterRDF"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#water-radial-distribution"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "water_radial_distribution"


class WaterRDFApp(BaseApp):
    """Water radial distribution benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_rdf.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"RDF RMSE": scatter},
        )


def get_app() -> WaterRDFApp:
    """
    Get water radial distribution benchmark app layout and callback registration.

    Returns
    -------
    WaterRDFApp
        Benchmark layout and callback registration.
    """
    return WaterRDFApp(
        name="Water RDF",
        framework_ids="mlip_audit",
        description=(
            "Performance in reproducing the oxygen-oxygen radial distribution "
            "function of liquid water from a short NVT molecular dynamics "
            "simulation. Reference data from experiment."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "water_radial_distribution_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8070, debug=True)
