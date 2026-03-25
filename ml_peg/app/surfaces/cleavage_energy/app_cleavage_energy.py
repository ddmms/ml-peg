"""Run cleavage energy app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.app.utils.load import collect_traj_assets, read_density_plot_for_model
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Cleavage Energy"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#cleavage-energy"
)
DATA_PATH = APP_ROOT / "data" / "surfaces" / "cleavage_energy"


class CleavageEnergyApp(BaseApp):
    """Cleavage energy benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """No interactive callbacks needed."""


def get_app() -> CleavageEnergyApp:
    """
    Get cleavage energy benchmark app layout and callback registration.

    Returns
    -------
    CleavageEnergyApp
        Benchmark layout and callback registration.
    """
    scatter = read_plot(
        DATA_PATH / "figure_cleavage_energies.json",
        id=f"{BENCHMARK_NAME}-figure",
    )

    return CleavageEnergyApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting cleavage energies for "
            "36,718 surface configurations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "cleavage_energy_metrics_table.json",
        extra_components=[
            Div(scatter, style={"marginTop": "20px"}),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(
        __name__,
        assets_folder=DATA_PATH.parent.parent,
    )

    cleavage_energy_app = get_app()
    full_app.layout = cleavage_energy_app.layout
    cleavage_energy_app.register_callbacks()

    full_app.run(port=8056, debug=True)
