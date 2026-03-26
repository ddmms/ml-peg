"""Run equation of state benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
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
        # Build plots for models with data (read_density_plot_for_model
        # returns None for models without data)
        return


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
