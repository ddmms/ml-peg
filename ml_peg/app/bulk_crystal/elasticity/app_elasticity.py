"""Run elasticity benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Elasticity"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk.html#elasticity"
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "elasticity"


class ElasticityApp(BaseApp):
    """Elasticity benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        bulk_plot = read_plot(
            DATA_PATH / "figure_bulk_density.json",
            id=f"{BENCHMARK_NAME}-bulk-figure",
        )
        shear_plot = read_plot(
            DATA_PATH / "figure_shear_density.json",
            id=f"{BENCHMARK_NAME}-shear-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Bulk modulus MAE": bulk_plot,
                "Shear modulus MAE": shear_plot,
            },
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
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    elasticity_app = get_app()
    full_app.layout = elasticity_app.layout
    elasticity_app.register_callbacks()
    full_app.run(port=8054, debug=True)
