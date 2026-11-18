"""Run reference geometry stability benchmark app."""

from __future__ import annotations

from dash import Input, Output, clientside_callback
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
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

        molecules = read_plot(
            DATA_PATH / "figure_molecule_rmsd.json", id=f"{BENCHMARK_NAME}-molecules"
        )
        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Avg RMSD": molecules,
                "Avg RMSD (stable only)": molecules,
                "Geometry Score": histogram,
            },
        )
        # A structure selection belongs to the plot that produced it.
        clientside_callback(
            "() => null",
            Output(f"{BENCHMARK_NAME}-structure", "children", allow_duplicate=True),
            Input(f"{BENCHMARK_NAME}-figure-placeholder", "children"),
            prevent_initial_call=True,
        )
        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-molecules",
            struct_id=f"{BENCHMARK_NAME}-structure",
            struct_template="/assets/molecular/reference_geometry_stability/structures/{id}.xyz",
            mode="traj",
            follow_frames=False,
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
            Div(id=f"{BENCHMARK_NAME}-structure"),
        ],
    )
