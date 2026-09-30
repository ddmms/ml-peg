"""Run MOF bulk modulus benchmark app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "MOF bulk modulus"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "porous_materials.html#mof-bulk-modulus"
)
DATA_PATH = APP_ROOT / "data" / "porous_materials" / "mof_bulk_modulus"


class MOFBulkModulusApp(BaseApp):
    """MOF bulk modulus benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        parity = read_plot(
            DATA_PATH / "figure_bulk_modulus.json", id=f"{BENCHMARK_NAME}-figure"
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Bulk modulus MAE": parity},
        )


def get_app() -> MOFBulkModulusApp:
    """
    Get MOF bulk modulus benchmark app layout and callback registration.

    Returns
    -------
    MOFBulkModulusApp
        Benchmark layout and callback registration.
    """
    return MOFBulkModulusApp(
        name=BENCHMARK_NAME,
        description=(
            "Bulk modulus of metal-organic frameworks from a Birch-Murnaghan fit "
            "to an isotropic energy-volume scan, compared against experimental "
            "values."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "mof_bulk_modulus_metrics_table.json",
        extra_components=[Div(id=f"{BENCHMARK_NAME}-figure-placeholder")],
    )
