"""Run phonon dispersion app."""

from __future__ import annotations

import json

from dash import Dash, dcc, html

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import register_phonon_callbacks
from ml_peg.calcs import CALCS_ROOT

DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "phonons"
TABLE_PATH = DATA_PATH / "phonon_metrics_table.json"
SCATTER_PATH = DATA_PATH / "phonon_interactive.json"
BENCHMARK_NAME = "Phonons"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#phonons"
)
CALC_BASE = CALCS_ROOT / "bulk_crystal" / "phonons"

with SCATTER_PATH.open(encoding="utf8") as handle:
    INTERACTIVE_DATA = json.load(handle)

PLOT_CONTAINER_ID = f"{BENCHMARK_NAME}-plot-container"
DISPERSION_CONTAINER_ID = f"{BENCHMARK_NAME}-dispersion-container"
LAST_CELL_STORE_ID = f"{BENCHMARK_NAME}-last-cell"
SCATTER_META_STORE_ID = f"{BENCHMARK_NAME}-scatter-meta"
SCATTER_GRAPH_ID = f"{BENCHMARK_NAME}-scatter"


class PhononApp(BaseApp):
    """Phonon benchmark app wiring callbacks and layout."""

    def register_callbacks(self) -> None:
        """Register scatter/dispersion callbacks via shared helper."""
        register_phonon_callbacks(
            table_id=self.table_id,
            table_data=self.table.data,
            plot_container_id=PLOT_CONTAINER_ID,
            dispersion_container_id=DISPERSION_CONTAINER_ID,
            scatter_meta_store_id=SCATTER_META_STORE_ID,
            last_cell_store_id=LAST_CELL_STORE_ID,
            scatter_graph_id=SCATTER_GRAPH_ID,
            interactive_data=INTERACTIVE_DATA,
            calc_root=CALC_BASE,
        )


def get_app() -> PhononApp:
    """
    Construct the PhononApp instance.

    Returns
    -------
    PhononApp
        Configured application with table + scatter/dispersion panels.
    """
    return PhononApp(
        name=BENCHMARK_NAME,
        description=(
            "Accuracy of MLIPs in predicting phonon dispersions and vibrational "
            "thermodynamics for bulk crystals."
        ),
        docs_url=DOCS_URL,
        table_path=TABLE_PATH,
        extra_components=[
            dcc.Store(id=LAST_CELL_STORE_ID),
            dcc.Store(id=SCATTER_META_STORE_ID),
            html.Div(
                [
                    html.Div(
                        "Click on a metric to view scatter plots.",
                        id=PLOT_CONTAINER_ID,
                        style={"flex": "1", "minWidth": 0},
                    ),
                    html.Div(
                        "Click on a scatter point to view the dispersion plot.",
                        id=DISPERSION_CONTAINER_ID,
                        style={"flex": "1", "minWidth": 0},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "24px",
                    "alignItems": "stretch",
                    "flexWrap": "wrap",
                },
            ),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    phonon_app = get_app()
    full_app.layout = phonon_app.layout
    phonon_app.register_callbacks()
    full_app.run(port=8060, debug=True)
