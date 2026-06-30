"""Run Tautobase tautomer benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Tautomers"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_reactions.html#tautomers"
DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / "tautomers"


class TautomersApp(BaseApp):
    """Tautomers benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_tautomers.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )


def get_app() -> TautomersApp:
    """
    Get tautomers benchmark app layout and callback registration.

    Returns
    -------
    TautomersApp
        Benchmark layout and callback registration.
    """
    return TautomersApp(
        name=BENCHMARK_NAME,
        framework_id="mlip_audit",
        description=(
            "Performance in predicting relative energies of tautomer pairs. "
            "Reference data from the Tautobase dataset."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "tautomers_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8068, debug=True)
