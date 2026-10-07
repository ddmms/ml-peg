"""Run HF structure factor app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "HF Structure Factor"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "superacids.html#hf-structure-factor"
)
DATA_PATH = APP_ROOT / "data" / "superacids" / "HF_structure"


class HFStructureApp(BaseApp):
    """HF structure factor benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_sq.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "S(q) R-factor": scatter,
                "First Peak Position Error": scatter,
            },
        )


def get_app() -> HFStructureApp:
    """
    Get HF structure factor benchmark app layout and callback registration.

    Returns
    -------
    HFStructureApp
        Benchmark layout and callback registration.
    """
    return HFStructureApp(
        name=BENCHMARK_NAME,
        description=(
            "Total neutron structure factor of liquid HF, from NPT molecular "
            "dynamics with all H transmuted to D."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "hf_structure_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )
