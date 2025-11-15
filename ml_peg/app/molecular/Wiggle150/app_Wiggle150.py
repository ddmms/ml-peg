"""Run Wiggle150 app."""

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
BENCHMARK_NAME = "Wiggle150"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular.html#wiggle150"
)
DATA_PATH = APP_ROOT / "data" / "molecular" / "Wiggle150"


class Wiggle150App(BaseApp):
    """Wiggle150 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_relative_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        structs_dir = DATA_PATH / MODELS[0]
        structure_files = sorted(
            structs_dir.glob("*.xyz"), key=lambda path: int(path.stem)
        )

        structs = [
            f"assets/molecular/{BENCHMARK_NAME}/{MODELS[0]}/{path.name}"
            for path in structure_files
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


def get_app() -> Wiggle150App:
    """
    Get Wiggle150 benchmark app layout and callback registration.

    Returns
    -------
    Wiggle150App
        Benchmark layout and callback registration.
    """
    return Wiggle150App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting relative conformer energies for the "
            "150-structure Wiggle dataset (ado, bpn, efa)."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "wiggle150_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    wiggle_app = get_app()
    full_app.layout = wiggle_app.layout
    wiggle_app.register_callbacks()

    # Run app
    full_app.run(port=8054, debug=True)
