"""Run BMIMCl RDF benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "BMIMCl Cl-C RDF"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular.html#bmimcl-rdf"
)
DATA_PATH = APP_ROOT / "data" / "molecular" / "BMIMCl_RDF"


class BMIMClRDFApp(BaseApp):
    """BMIMCl RDF benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        rdf_plot = read_plot(
            DATA_PATH / "figure_rdf.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Cl-C Bonds Formed": rdf_plot},
        )


def get_app() -> BMIMClRDFApp:
    """
    Get BMIMCl RDF benchmark app layout and callback registration.

    Returns
    -------
    BMIMClRDFApp
        Benchmark layout and callback registration.
    """
    return BMIMClRDFApp(
        name=BENCHMARK_NAME,
        description=(
            "Tests whether MLIPs incorrectly predict Cl-C bond formation "
            "in BMIMCl ionic liquid. Bonds should NOT form."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "bmimcl_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    bmimcl_app = get_app()
    full_app.layout = bmimcl_app.layout
    bmimcl_app.register_callbacks()

    # Run app
    full_app.run(port=8054, debug=True)
