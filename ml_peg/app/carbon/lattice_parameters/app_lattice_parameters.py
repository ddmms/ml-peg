"""Run lattice parameters app."""

from __future__ import annotations

from dash.dcc import Graph
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Lattice Parameters"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/carbon.html#lattice-parameters"
DATA_PATH = APP_ROOT / "data" / "carbon" / "lattice_parameters"
INFO_PATH = DATA_PATH / "info.json"

# Table column, figure JSON suffix
PLOTS = {
    "Lattice parameter MAPE": "lattice_parameter",
    "Bond length MAPE": "bond_length",
    "Energy above graphite MAE": "energy_above_graphite",
    "Convergence": "convergence",
}


def get_mock_structs(scatter: Graph) -> list[str]:
    """
    Get mock structure paths in the same order as a scatter plot's points.

    Parameters
    ----------
    scatter
        Scatter plot whose first hoverdata field is the system name.

    Returns
    -------
    list[str]
        Asset paths to the mock structure for each point.
    """
    traces = scatter.figure.data if scatter.figure else ()
    if not traces or traces[0].customdata is None:
        return []
    return [
        f"/assets/carbon/lattice_parameters/mock/{point[0]}.extxyz"
        for point in traces[0].customdata
    ]


class LatticeParametersApp(BaseApp):
    """Lattice parameters benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatters = {
            column: read_plot(
                DATA_PATH / f"figure_lattice_parameters_{suffix}.json",
                id=f"{BENCHMARK_NAME}-{suffix}-figure",
            )
            for column, suffix in PLOTS.items()
        }

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot=scatters,
        )

        for scatter in scatters.values():
            struct_from_scatter(
                scatter_id=scatter.id,
                struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                structs=get_mock_structs(scatter),
                mode="struct",
            )


def get_app() -> LatticeParametersApp:
    """
    Get lattice parameters benchmark app layout and callback registration.

    Returns
    -------
    LatticeParametersApp
        Benchmark layout and callback registration.
    """
    return LatticeParametersApp(
        name=BENCHMARK_NAME,
        description=(
            "Lattice parameters, neighbour bond lengths, and energy above"
            " graphite for carbon allotropes. Reference data is optB88-vdW."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "lattice_parameters_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
        framework_ids="gap-20",
    )
