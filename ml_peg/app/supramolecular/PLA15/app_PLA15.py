"""Run PLA15 app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "PLA15"
DATA_PATH = APP_ROOT / "data" / "supramolecular" / "PLA15"
INFO_PATH = DATA_PATH / "info.json"


class PLA15App(BaseApp):
    """PLA15 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_interaction_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        structs_dir = DATA_PATH / "mock"
        structs = [
            f"/assets/supramolecular/PLA15/mock/{path.name}"
            for path in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "MAE": scatter,
                "Ion-Ion MAE": scatter,
                "Ion-Neutral MAE": scatter,
                "R²": scatter,
            },
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
        description=(
            "Performance in predicting protein-ligand interaction energies for 15 "
            "complete active site complexes."
        ),
        table_path=DATA_PATH / "pla15_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
        framework_ids="mace-polar-1",
    )
