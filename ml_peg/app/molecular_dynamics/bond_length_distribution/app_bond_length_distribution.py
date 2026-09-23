"""Run bond length distribution benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "BondLength"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#bond-length-distribution"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "bond_length_distribution"


class BondLengthApp(BaseApp):
    """Bond length distribution benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        histogram = read_plot(
            DATA_PATH / "figure_bond_length_hist.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Bond Length Deviation": histogram},
        )


def get_app() -> BondLengthApp:
    """
    Get bond length distribution benchmark app layout and callback registration.

    Returns
    -------
    BondLengthApp
        Benchmark layout and callback registration.
    """
    return BondLengthApp(
        name="Bond Length Distribution",
        framework_ids="mlip_audit",
        description=(
            "Performance in maintaining physically reasonable covalent bond "
            "lengths during molecular dynamics of small organic molecules. "
            "Reference bond lengths are taken from QM-optimised geometries."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "bond_length_distribution_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8070, debug=True)
