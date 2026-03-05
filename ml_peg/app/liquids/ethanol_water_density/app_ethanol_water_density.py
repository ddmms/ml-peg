# TODO: This does not work. Fix this

"""Run ethanol–water density (decomposition curves) app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CATEGORY = "liquids"
BENCHMARK_NAME = "ethanol_water_density"

DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/liquids.html#ethanol-water-density-curves"

DATA_PATH = APP_ROOT / "data" / CATEGORY / BENCHMARK_NAME


class EthanolWaterDecompositionCurvesApp(BaseApp):
    """Ethanol–water density benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        parity = read_plot(
            DATA_PATH / "density_parity.json", id=f"{BENCHMARK_NAME}-figure"
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "density": parity,
            },
        )


def get_app() -> EthanolWaterDecompositionCurvesApp:
    """
    Get ethanol–water benchmark app layout and callback registration.

    Returns
    -------
    EthanolWaterDecompositionCurvesApp
        Benchmark layout and callback registration.
    """
    return EthanolWaterDecompositionCurvesApp(
        name=BENCHMARK_NAME,
        description=(
            "Ethanol–water mixture density at 293.15 K. Metrics include density RMSE, "
            "excess-volume RMSE, and error in the mole-fraction"
            "location of the maximum excess volume."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "density_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    # assets_folder should be the parent of the "assets/<category>/<benchmark>/..." tree
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    app = get_app()
    full_app.layout = app.layout
    app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
