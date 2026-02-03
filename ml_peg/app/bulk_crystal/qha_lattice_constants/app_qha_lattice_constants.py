"""Run QHA lattice constants benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "QHA lattice constants"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html"
    "#qha-lattice-constants"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "qha_lattice_constants"


class QHALatticeConstantsApp(BaseApp):
    """QHA lattice constants benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_qha_lattice_constants.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Lattice constant MAE (QHA)": scatter,
            },
        )


def get_app() -> QHALatticeConstantsApp:
    """
    Get QHA lattice constants benchmark app.

    Returns
    -------
    QHALatticeConstantsApp
        Benchmark layout and callback registration.
    """
    return QHALatticeConstantsApp(
        name=BENCHMARK_NAME,
        description=(
            "Temperature-dependent lattice constants obtained from a "
            "quasi-harmonic workflow."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "qha_lattice_constants_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    qha_app = get_app()
    full_app.layout = qha_app.layout
    qha_app.register_callbacks()
    full_app.run(port=8062, debug=True)
