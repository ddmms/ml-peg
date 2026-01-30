"""Run RDB7 app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
)
from ml_peg.app.utils.load import read_density_plot_for_model
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "RDB7"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_reactions.html#rdb7"
)
DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / "RDB7"


class RDB7App(BaseApp):
    """RDB7 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Build plots for models with data (read_density_plot_for_model
        # returns None for models without data)
        density_plots: dict[str, dict] = {}
        for model in MODELS:
            plots = {
                "MAE": read_density_plot_for_model(
                    filename=DATA_PATH / "figure_barrier_density.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-barrier-figure",
                ),
            }
            # Filter out None values (models without data for that metric)
            model_plots = {k: v for k, v in plots.items() if v is not None}
            if model_plots:
                density_plots[model] = model_plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=density_plots,
        )


def get_app() -> RDB7App:
    """
    Get RDB7 benchmark app layout and callback registration.

    Returns
    -------
    RDB7App
        Benchmark layout and callback registration.
    """
    return RDB7App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting barrier heights for the "
            "RDB7 pericyclic reactions benchmark. "
            "Reference data from CCSD(T)-F12/cc-pVDZ-F12 calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "rdb7_barriers_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            # Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8068, debug=True)
