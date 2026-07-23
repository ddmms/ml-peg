"""Run reference geometry stability benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "ReferenceGeometryStability"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular.html#reference-geometry-stability"
DATA_PATH = APP_ROOT / "data" / "molecular" / "reference_geometry_stability"


class ReferenceGeometryStabilityApp(BaseApp):
    """Reference geometry stability benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        histogram = read_plot(
            DATA_PATH / "figure_rmsd_histogram.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Avg RMSD": histogram},
        )


def get_app() -> ReferenceGeometryStabilityApp:
    """
    Get reference geometry stability benchmark app layout and callbacks.

    Returns
    -------
    ReferenceGeometryStabilityApp
        Benchmark layout and callback registration.
    """
    return ReferenceGeometryStabilityApp(
        name="Reference Geometry Stability",
        framework_ids="mlip_audit",
        description=(
            "Ability to preserve the ground-state geometry of small organic "
            "molecules during energy minimization, measured as the heavy-atom "
            "RMSD from the reference structure. Geometries from the OpenFF "
            "industry dataset."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "reference_geometry_stability_metrics_table.json",
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
    full_app.run(port=8073, debug=True)
