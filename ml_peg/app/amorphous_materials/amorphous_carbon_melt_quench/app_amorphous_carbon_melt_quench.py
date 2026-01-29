"""Run amorphous carbon melt-quench app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Melt-quench carbon"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "amorphous_materials.html#amorphous-carbon-melt-quench"
)
DATA_PATH = APP_ROOT / "data" / "amorphous_materials" / "amorphous_carbon_melt_quench"


class AmorphousCarbonMeltQuenchApp(BaseApp):
    """Amorphous carbon melt-quench benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_sp3_vs_density.json",
            id="amorphous-carbon-melt-quench-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id="amorphous-carbon-melt-quench-figure-placeholder",
            column_to_plot={
                "MAE vs DFT": scatter,
                "MAE vs Expt": scatter,
            },
        )


def get_app() -> AmorphousCarbonMeltQuenchApp:
    """
    Get amorphous carbon melt-quench benchmark app layout and callbacks.

    Returns
    -------
    AmorphousCarbonMeltQuenchApp
        Benchmark layout and callback registration.
    """
    return AmorphousCarbonMeltQuenchApp(
        name=BENCHMARK_NAME,
        description=(
            "Melt-quench simulations of amorphous carbon; compare sp3 fraction versus "
            "density to DFT and experimental references."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "amorphous_carbon_melt_quench_metrics_table.json",
        extra_components=[Div(id="amorphous-carbon-melt-quench-figure-placeholder")],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()

    full_app.run(port=8052, debug=True)
