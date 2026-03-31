"""Run equation of state benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Equation of State"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#equation-of-state"
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "equation_of_state"


class EquationOfStateApp(BaseApp):
    """Equation of State benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        _metrics = [
            ("Δ", "delta_periodic_table"),
            ("Phase energy", "phase_energy_periodic_table"),
            ("Phase stability", "phase_stability_periodic_table"),
        ]
        cell_to_plot = {}
        for model in MODELS:
            plots = {}
            for column_id, file_suffix in _metrics:
                path = DATA_PATH / model / f"{file_suffix}.json"
                if path.exists():
                    plots[column_id] = read_plot(
                        filename=path,
                        id=f"{BENCHMARK_NAME}-{model}-{file_suffix}",
                    )
            if plots:
                cell_to_plot[model] = plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_to_plot,
        )


def get_app() -> EquationOfStateApp:
    """
    Get equation of state benchmark app layout and callback registration.

    Returns
    -------
    EquationOfStateApp
        Benchmark layout and callback registration.
    """
    return EquationOfStateApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance when calculating the equation of state for different "
            "bulk crystal (W, Mo, Nb) structures "
            "scomapred to PBE data from literature."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "eos_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    equation_of_state_app = get_app()
    full_app.layout = equation_of_state_app.layout
    equation_of_state_app.register_callbacks()
    full_app.run(port=8054, debug=True)
