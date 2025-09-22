"""Run PLA15 app."""

from __future__ import annotations

from pathlib import Path

from dash import Dash
from dash.html import Div

from mlip_testing.app import APP_ROOT
from mlip_testing.app.base_app import BaseApp
from mlip_testing.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from mlip_testing.app.utils.load import read_plot
from mlip_testing.calcs.models.models import MODELS

BENCHMARK_NAME = Path(__file__).name.removeprefix("app_").removesuffix(".py")
DATA_PATH = APP_ROOT / "data" / "supramolecular" / "PLA15"


class PLA15App(BaseApp):
    """PLA15 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_interaction_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        structs_dir = DATA_PATH / list(MODELS.keys())[0]
        # Assets dir will be parent directory - individual files for each system
        structs = [
            f"assets/supramolecular/PLA15/{list(MODELS.keys())[0]}/{i}.xyz"
            for i in range(len(list(structs_dir.glob("*.xyz"))))
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


def get_app() -> PLA15App:
    """
    Get PLA15 benchmark app layout and callback registration.

    Returns
    -------
    PLA15App
        Benchmark layout and callback registration.
    """
    return PLA15App(
        name=BENCHMARK_NAME,
        title="PLA15",
        description=(
            "Performance in predicting protein-ligand interaction energies for 15 "
            "complete active site complexes."
        ),
        table_path=DATA_PATH / "pla15_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    pla15_app = get_app()
    full_app.layout = pla15_app.layout
    pla15_app.register_callbacks()

    # Run app
    full_app.run(port=8055, debug=True)
