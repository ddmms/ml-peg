"""Run HPHT_CH4_H2O benchmark app."""

from __future__ import annotations

from dash import Dash, Input, Output, callback, dcc
from dash.html import Div
import plotly.io as pio

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "HPHT_CH4_H2O"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_reactions.html#hpht-ch4-h2o"
DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / BENCHMARK_NAME
CALCS_ROOT = APP_ROOT.parent / "calcs"


class HPHTCH4H2OApp(BaseApp):
    """HPHT_CH4_H2O benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_reaction_free_energy.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        scatter_barrier = read_plot(
            DATA_PATH / "figure_barrier_free_energy.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "FEP_MAE": scatter,
                "DF_MAE": scatter,
                "DF#_MAE": scatter_barrier,
            },
        )

        @callback(
            Output(f"{BENCHMARK_NAME}-fes-plot", "children"),
            Input(f"{BENCHMARK_NAME}-figure", "clickdata"),
        )
        def update_fes_plot(clickdata):
            """
            Display all the free energy profiles of the selected structure.

            Parameters
            ----------
            clickdata : dict or None
                Dash click event data containing the selected structure.

            Returns
            -------
            str or dash.dcc.Graph
                Free energy profile graph or an error message.
            """
            if clickdata is None:
                return "Click on a point to show free energy profile"
            try:
                point = clickdata["points"][0]
                structure = point["customdata"][0]

                figure_file = DATA_PATH / "fes_plots" / f"{structure}.json"

                if not figure_file.exists():
                    return f"No free energy profile available for {structure}"
                fig = pio.read_json(figure_file)
                return dcc.Graph(figure=fig)
            except Exception as e:
                return f"Error loading free energy profile: {e}"


def get_app() -> HPHTCH4H2OApp:
    """
    Get the configured HPHT_CH4_H2O benchmark application.

    Returns
    -------
    HPHTCH4H2OApp
        Configured benchmark application with registered callbacks.
    """
    return HPHTCH4H2OApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting free energy profiles of proton hopping"
            "in CH4/H2O mixtures under high pressure (HP) and high temperature (HT)"
            "Reference data from DFT-MD (PBE) simulations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "fes_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-fes-plot"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    HPHT_CH4_H2O_app = get_app()
    HPHT_CH4_H2O_app.app = full_app
    full_app.layout = HPHT_CH4_H2O_app.layout
    HPHT_CH4_H2O_app.register_callbacks()

    full_app.run(port=8055, debug=True)
