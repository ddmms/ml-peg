"""Run water density app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Water Density"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#water-density"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "water_density"


class WaterDensityApp(BaseApp):
    """Water density benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_water_density.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )


def get_app() -> WaterDensityApp:
    """
    Get water density benchmark app layout and callback registration.

    Returns
    -------
    WaterDensityApp
        Benchmark layout and callback registration.
    """
    return WaterDensityApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting water density at different temperatures."
            " Reference data is experimental densities."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "water_density_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8063, debug=True)
