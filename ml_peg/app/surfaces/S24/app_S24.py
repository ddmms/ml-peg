"""Run S24 app."""

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
BENCHMARK_NAME = "S24"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#s24"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "S24"


class S24App(BaseApp):
    """S24 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_adsorption_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        structs_dir = DATA_PATH / MODELS[0]

        # Assets dir will be parent directory
        structs = [
            f"assets/surfaces/S24/{MODELS[0]}/{struct_file.stem}.xyz"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
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
            mode="traj",
        )


def get_app() -> S24App:
    """
    Get S24 benchmark app layout and callback registration.

    Returns
    -------
    S24App
        Benchmark layout and callback registration.
    """
    return S24App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting adsorption energies for 24 "
            "molecule-surface combinations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "s24_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    s24_app = get_app()
    full_app.layout = s24_app.layout
    s24_app.register_callbacks()

    # Run app
    full_app.run(port=8052, debug=True)
