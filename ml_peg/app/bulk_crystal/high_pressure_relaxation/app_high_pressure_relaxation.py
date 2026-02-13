"""Run high-pressure crystal relaxation benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "High-pressure relaxation"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#high-pressure-relaxation"
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "high_pressure_relaxation"


PRESSURES = [0, 25, 50, 75, 100, 125, 150]


class HighPressureRelaxationApp(BaseApp):
    """High-pressure crystal relaxation benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_volume = read_plot(
            DATA_PATH / "figure_volume.json", id=f"{BENCHMARK_NAME}-figure"
        )
        scatter_energy = read_plot(
            DATA_PATH / "figure_energy.json", id=f"{BENCHMARK_NAME}-figure"
        )

        # Build column_to_plot mapping for all pressures
        column_to_plot = {}
        for pressure in PRESSURES:
            column_to_plot[f"Volume MAE ({pressure} GPa)"] = scatter_volume
            column_to_plot[f"Energy MAE ({pressure} GPa)"] = scatter_energy

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot=column_to_plot,
        )


def get_app() -> HighPressureRelaxationApp:
    """
    Get high-pressure relaxation benchmark app layout and callback registration.

    Returns
    -------
    HighPressureRelaxationApp
        Benchmark layout and callback registration.
    """
    return HighPressureRelaxationApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance when relaxing crystal structures under high pressure "
            "(0-150 GPa). Evaluates volume, energy, and convergence percentage "
            "against PBE reference calculations from the Alexandria database. "
            "Please also reference Loew et al 2026 J. Phys. Mater. 9 015010"
            " (https://iopscience.iop.org/article/10.1088/2515-7639/ae2ba8) "
            "when using this benchmark."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "high_pressure_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    hp_app = get_app()
    full_app.layout = hp_app.layout
    hp_app.register_callbacks()
    full_app.run(port=8055, debug=True)
