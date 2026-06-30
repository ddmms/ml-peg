"""Run HF/SbF5 density app."""

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

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "HF/SbF5 Mixture Densities"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "superacids.html#hf-sbf5-mixture-densities"
)
DATA_PATH = APP_ROOT / "data" / "superacids" / "HF_SbF5_density"


class HFSbF5DensityApp(BaseApp):
    """HF/SbF5 density benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_density.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAPE": scatter},
        )


def get_app() -> HFSbF5DensityApp:
    """
    Get HF/SbF5 density benchmark app layout and callback registration.

    Returns
    -------
    HFSbF5DensityApp
        Benchmark layout and callback registration.
    """
    return HFSbF5DensityApp(
        name=BENCHMARK_NAME,
        description=("Liquid densities of HF/SbF5 mixtures at varying compositions."),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "hf_sbf5_density_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    app = get_app()
    full_app.layout = app.layout
    app.register_callbacks()

    # Run app
    full_app.run(port=8056, debug=True)
