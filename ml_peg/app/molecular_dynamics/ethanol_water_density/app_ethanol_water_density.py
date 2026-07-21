"""Run ethanol–water density (decomposition curves) app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

CATEGORY = "molecular_dynamics"
BENCHMARK_NAME = "Ethanol-water densities"

# Get all models
MODELS = get_model_names(current_models)

DOCS_URL = f"https://ddmms.github.io/ml-peg/user_guide/benchmarks/{CATEGORY}.html#water-ethanol-density-curves"

DATA_PATH = APP_ROOT / "data" / CATEGORY / "ethanol_water_density"
INFO_PATH = DATA_PATH / "info.json"


class EthanolWaterDecompositionCurvesApp(BaseApp):
    """Ethanol–water density benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_plots = {
            model: {
                "RMSE density": read_plot(
                    DATA_PATH / model / "figure_density.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-density",
                ),
                "RMSE excess volume": read_plot(
                    DATA_PATH / model / "figure_excess_volume.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-excess-volume",
                ),
                "Peak x error": read_plot(
                    DATA_PATH / model / "figure_excess_volume_minimum.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-excess-volume-minimum",
                ),
            }
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=scatter_plots,
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
            "excess-volume RMSE, and error in the mole-fraction "
            "location of the maximum excess volume."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "density_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
        info_path=INFO_PATH,
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
