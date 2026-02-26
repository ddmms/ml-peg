"""Run SSE-MD app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "SSE-MD Scores"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/electrolytes.html#sse-md"
)
DATA_PATH = APP_ROOT / "electrolytes" / "SSEMD"


class SSEMDApp(BaseApp):
    """SSE-MD benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_ssemd_scores.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"RDF Score": scatter},
        )


def get_app() -> SSEMDApp:
    """
    Get SSE-MD benchmark app layout and callback registration.

    Returns
    -------
    SSEMDApp
        Benchmark layout and callback registration.
    """
    return SSEMDApp(
        name=BENCHMARK_NAME,
        description=(
            "RDF similarity scores for solid-state electrolyte systems, "
            "comparing MLIP MD trajectories against AIMD reference data."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "ssemd_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    ssemd_app = get_app()
    full_app.layout = ssemd_app.layout
    ssemd_app.register_callbacks()

    # Run app
    full_app.run(port=8056, debug=True)
