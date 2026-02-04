"""Run water slab dipoles app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
    #    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Dipoles of Water Slabs"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#water_slab_dipoles"
DATA_PATH = APP_ROOT / "data" / "physicality" / "water_slab_dipoles"


class WaterSlabDipolesApp(BaseApp):
    """Water slab dipole benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        print(" Registering callbacks to app")
        hists = {
            model: {
                "sigma": read_plot(
                    DATA_PATH / f"figure_{model}_dipoledistr.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                ),
                "Fraction Breakdown Candidates": read_plot(
                    DATA_PATH / f"figure_{model}_dipoledistr.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                ),
            }
            for model in MODELS
        }
        for model in MODELS:
            print("Read plot from ", DATA_PATH / f"figure_{model}_dipoledistr.json")
        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=hists,
        )


def get_app() -> WaterSlabDipolesApp:
    """
    Get water slab dipoles benchmark app layout and callback registration.

    Returns
    -------
    WaterSlabDipolesApp
        Benchmark layout and callback registration.
    """
    print("Id of extra components: ", f"{BENCHMARK_NAME}-figure-placeholder")
    return WaterSlabDipolesApp(
        name=BENCHMARK_NAME,
        description="Dipole distribution of a 38 A water slab",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "water_slab_dipoles_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    dipole_app = get_app()
    full_app.layout = dipole_app.layout
    dipole_app.register_callbacks()

    # Run app
    full_app.run(port=8055, debug=True)
