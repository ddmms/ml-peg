"""Run S30L app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)

BENCHMARK_NAME = "S30L"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/supramolecular.html#s30l"
)
DATA_PATH = APP_ROOT / "data" / "supramolecular" / "S30L"


class S30LApp(BaseApp):
    """S30L benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_interaction_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Assets dir will be parent directory - individual files for each system
        structs = [
            f"assets/supramolecular/S30L/{MODELS[0]}/{i}.xyz"
            for i in range(30)  # S30L has 30 systems
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Neutral MAE": scatter,
                "Charged MAE": scatter,
                "Overall MAE": scatter,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> S30LApp:
    """
    Get S30L benchmark app layout and callback registration.

    Returns
    -------
    S30LApp
        Benchmark layout and callback registration.
    """
    return S30LApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting interaction energies for 30 "
            "host-guest supramolecular complexes."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "s30l_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    s30l_app = get_app()
    full_app.layout = s30l_app.layout
    s30l_app.register_callbacks()

    # Run app
    full_app.run(port=8054, debug=True)
