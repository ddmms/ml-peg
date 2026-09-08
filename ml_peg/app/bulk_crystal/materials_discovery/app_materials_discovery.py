"""Run the materials-discovery benchmark app."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp

BENCHMARK_NAME = "Materials discovery"
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "materials_discovery"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "bulk_crystal.html#materials-discovery-evaluation"
)


class MaterialsDiscoveryApp(BaseApp):
    """Table-only Matbench Discovery benchmark app."""

    def register_callbacks(self) -> None:
        """Register benchmark-specific callbacks."""


def get_app() -> MaterialsDiscoveryApp:
    """
    Return the materials-discovery benchmark app.

    Returns
    -------
    MaterialsDiscoveryApp
        Configured materials-discovery application.
    """
    return MaterialsDiscoveryApp(
        name=BENCHMARK_NAME,
        description=(
            "Classification and regression performance on the WBM materials "
            "discovery test set."
        ),
        docs_url=DOCS_URL,
        framework_ids="matbench-discovery",
        table_path=DATA_PATH / "materials_discovery_metrics_table.json",
        extra_components=[],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8051, debug=True)
