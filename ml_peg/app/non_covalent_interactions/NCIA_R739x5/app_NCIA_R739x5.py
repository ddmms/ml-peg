"""Run NCIA_R739x5 app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_density_plot_for_model
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "NCIA R739x5"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "non_covalent_interactions.html#ncia-r739x5"
)
DATA_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "NCIA_R739x5"


class NCIAR739x5App(BaseApp):
    """NCIA_R739x5 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        density_plots: dict[str, dict] = {}
        for model in MODELS:
            density_graph = read_density_plot_for_model(
                filename=DATA_PATH / "figure_ncia_r739x5_density.json",
                model=model,
                id=f"{BENCHMARK_NAME}-{model}-density",
            )
            if density_graph is not None:
                density_plots[model] = {"MAE": density_graph}

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=density_plots,
        )


def get_app() -> NCIAR739x5App:
    """
    Get NCIA_R739x5 benchmark app layout and callback registration.

    Returns
    -------
    NCIAR739x5App
        Benchmark layout and callback registration.
    """
    return NCIAR739x5App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting radical interaction energies "
            "for the NCIA R739x5 dataset (radical complexes). "
            "Reference data from CCSD(T) calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "ncia_r739x5_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8060, debug=True)
