"""Run battery electrolyte benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)

BENCHMARK_NAME = "Battery electrolyte"
DATA_PATH = APP_ROOT / "data" / "bulk_liquids" / "battery_electrolyte"


class BatteryElectrolyteApp(BaseApp):
    """Battery electrolyte benchmark app layout and callbacks."""

    # def register_callbacks(self) -> None:
    #     """Register callbacks to app."""
    #     # density_plots = {
    #     #     model: {
    #     #         "Bulk modulus MAE": read_density_plot_for_model(
    #     #             filename=DATA_PATH / "figure_bulk_density.json",
    #     #             model=model,
    #     #             id=f"{BENCHMARK_NAME}-{model}-bulk-figure",
    #     #         ),
    #     #         "Shear modulus MAE": read_density_plot_for_model(
    #     #             filename=DATA_PATH / "figure_shear_density.json",
    #     #             model=model,
    #     #             id=f"{BENCHMARK_NAME}-{model}-shear-figure",
    #     #         ),
    #     #     }
    #     #     for model in MODELS
    #     # }

    #     plot_from_table_cell(
    #         table_id=self.table_id,
    #         plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
    #         cell_to_plot=density_plots,
    #     )


def get_app() -> BatteryElectrolyteApp:
    """
    Get elasticity benchmark app layout and callback registration.

    Returns
    -------
    ElasticityApp
        Benchmark layout and callback registration.
    """
    return BatteryElectrolyteApp(
        name=BENCHMARK_NAME,
        description=(
            "Evaluate model inter/intra property prediction"
            "for different densities of LIB full electrolyte"
            "and neat solvent configs"
        ),
        # docs_url=DOCS_URL,
        table_path=DATA_PATH / "Inter_intra_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    BatteryElectrolyte_app = get_app()
    full_app.layout = BatteryElectrolyte_app.layout
    # BatteryElectrolyte_app.register_callbacks()
    full_app.run(port=8054, debug=True)
