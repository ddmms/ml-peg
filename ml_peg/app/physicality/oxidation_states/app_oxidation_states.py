"""Run oxidation states app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.calcs import CALCS_ROOT

# MODELS = get_model_names(current_models)
MODELS = ["mace-mp-0b3", "omol", "0totchrg_2L_fl32_new"]

BENCHMARK_NAME = "Iron Oxidation States"
DATA_PATH = APP_ROOT / "data" / "physicality" / "oxidation_states"
REF_PATH = CALCS_ROOT / "physicality" / "oxidation_states" / "data"


class FeOxidationStatesApp(BaseApp):
    """Fe Oxidation States benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_plots = {
            model: {
                "Fe-O RDF Peak Split": read_plot(
                    DATA_PATH / f"Fe-O_{model}_RDF_scatter.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-Fe-O-RDF",
                ),
                "Peak Within DFT Ref": read_plot(
                    DATA_PATH / f"Fe-O_{model}_RDF_scatter.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-Fe-O-RDF",
                ),
            }
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=scatter_plots,
        )


def get_app() -> FeOxidationStatesApp:
    """
    Get Fe Oxidation States benchmark app layout and callback registration.

    Returns
    -------
    FeOxidationStatesApp
        Benchmark layout and callback registration.
    """
    return FeOxidationStatesApp(
        name=BENCHMARK_NAME,
        description=(
            "Evaluate model ability to capture different oxidation states of Fe"
            "from aqueous Fe 2Cl and Fe 3Cl MD RDFs"
        ),
        # docs_url=DOCS_URL,
        table_path=DATA_PATH / "oxidation_states_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(
        __name__,
        assets_folder=DATA_PATH.parent.parent,
        suppress_callback_exceptions=True,
    )

    # Construct layout and register callbacks
    FeOxidationStatesApp = get_app()
    full_app.layout = FeOxidationStatesApp.layout
    FeOxidationStatesApp.register_callbacks()

    # Run app
    full_app.run(port=8054, debug=True)
