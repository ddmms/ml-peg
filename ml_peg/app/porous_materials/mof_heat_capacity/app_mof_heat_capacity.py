"""Run MOF heat capacity benchmark app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "MOF heat capacity"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "porous_materials.html#mof-heat-capacity"
)
DATA_PATH = APP_ROOT / "data" / "porous_materials" / "mof_heat_capacity"


class MOFHeatCapacityApp(BaseApp):
    """MOF heat capacity benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        parity = read_plot(
            DATA_PATH / "figure_heat_capacity.json", id=f"{BENCHMARK_NAME}-figure"
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Cv MAE": parity},
        )


def get_app() -> MOFHeatCapacityApp:
    """
    Get MOF heat capacity benchmark app layout and callback registration.

    Returns
    -------
    MOFHeatCapacityApp
        Benchmark layout and callback registration.
    """
    return MOFHeatCapacityApp(
        name=BENCHMARK_NAME,
        description=(
            "Isochoric heat capacity of metal-organic frameworks at 300 K, from "
            "the phonon density of states, compared against experimental "
            "calorimetry. The percentage of imaginary modes from the same phonon "
            "calculation is reported alongside it."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "mof_heat_capacity_metrics_table.json",
        extra_components=[Div(id=f"{BENCHMARK_NAME}-figure-placeholder")],
    )
