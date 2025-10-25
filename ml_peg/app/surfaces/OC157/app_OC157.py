"""Run OC157 app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div
import numpy as np

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
BENCHMARK_NAME = "OC157"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#oc157"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "OC157"


class OC157App(BaseApp):
    """OC157 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_rel_energies.json", id=f"{BENCHMARK_NAME}-figure"
        )

        structs_dir = DATA_PATH / MODELS[0]

        # Assets dir will be parent directory
        structs = list(
            np.repeat(
                [
                    f"assets/surfaces/OC157/{MODELS[0]}/{i}.xyz"
                    for i in range(len(list(structs_dir.glob("*.xyz"))))
                ],
                3,
            )
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter, "Ranking Error": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="traj",
        )


def get_app() -> OC157App:
    """
    Get OC157 benchmark app layout and callback registration.

    Returns
    -------
    OC157App
        Benchmark layout and callback registration.
    """
    return OC157App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting relative energies between 3 structures for 157 "
            "molecule-surface combinations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "oc157_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    oc157_app = get_app()
    full_app.layout = oc157_app.layout
    oc157_app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
