"""Run HPHT_CH4_H2O benchmark app."""

from __future__ import annotations

from dash import callback, Dash, Input, Output, dcc
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models
import plotly.graph_objects as go
import numpy as np

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "HPHT_CH4_H2O"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular.html#proton"
DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / BENCHMARK_NAME
CALCS_ROOT = APP_ROOT.parent / "calcs"

class HPHT_CH4_H2OApp(BaseApp):
    """PROTON benchmark app layout and callbacks."""

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
            Input(f"{BENCHMARK_NAME}-figure", "clickData"),
        )
        def update_fes_plot(clickData):

            if clickData is None:
                return "Click on a point to show free energy profile"
            try:
                point = clickData['points'][0]
                structure = point['customdata'][0]
                model = MODELS[0]
                ref_file = DATA_PATH / f"{structure}.data"
                model_file = DATA_PATH / model / f"{structure}.data"
                ref = np.loadtxt(ref_file)
                model_data = np.loadtxt(model_file)

                bins = ref[:, 0]
                F_ref = ref[:, 1]
                F_model = model_data[:, 1]

                fig = go.Figure()
                
                fig.add_trace(
                    go.Scatter(
                        x=bins,
                        y=F_ref,
                        mode="lines",
                        name="Reference",
                    )
                )
                for model in MODELS:
                    model_file = DATA_PATH / model / f"{structure}.data"
                    if not model_file.exists():
                        continue
                    model_data = np.loadtxt(model_file)
                    F_model = model_data[:, 1]

                    fig.add_trace(
                        go.Scatter(
                            x=bins,
                            y=F_model,
                            mode="lines",
                            name=model,
                        )
                    )
                fig.update_layout(
                    title=f"Free Energy Profile - {structure}",
                    xaxis_title="Reaction coordinate",
                    yaxis_title="Free energy (kJ/mol)",
                    template="plotly_white",
                )
                return dcc.Graph(figure=fig)
            except Exception as e:
                return f"Erreur : {e}"

def get_app() -> HPHT_CH4_H2OApp:
    """
    Get PROTON benchmark app layout and callback registration.

    Returns
    -------
    PROTONApp
        Benchmark layout and callback registration.
    """
    return HPHT_CH4_H2OApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting free energy profiles for the HPHT_CH4_H2O benchmark."
        ),
        docs_url=DOCS_URL,
    table_path= DATA_PATH / "fes_metrics_table.json",
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

