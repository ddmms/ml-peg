"""Run LNCI16 app."""

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
BENCHMARK_NAME = "LNCI16"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/supramolecular.html#lnci16"
)
DATA_PATH = APP_ROOT / "data" / "supramolecular" / "LNCI16"


class LNCI16App(BaseApp):
    """LNCI16 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_interaction_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Assets dir will be parent directory - individual files for each system
        structs = [
            f"assets/supramolecular/LNCI16/{MODELS[0]}/{i}.xyz"
            for i in range(16)  # LNCI16 has 16 systems
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> LNCI16App:
    """
    Get LNCI16 benchmark app layout and callback registration.

    Returns
    -------
    LNCI16App
        Benchmark layout and callback registration.
    """
    return LNCI16App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting interaction energies for 16 "
            "large non-covalent complexes including proteins, DNA, "
            "and supramolecular assemblies."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "lnci16_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    lnci16_app = get_app()
    full_app.layout = lnci16_app.layout
    lnci16_app.register_callbacks()

    # Run app
    full_app.run(port=8053, debug=True)
