"""Run energy_response app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "energy_response Total Energies"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/electric_field.html#energy_response"
)
DATA_PATH = APP_ROOT / "data" / "electric_field" / "energy_response"


class EnergyResponseApp(BaseApp):
    """energy_response benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_total_energy_response.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Assets dir will be parent directory - individual files for each system
        structs_dir = DATA_PATH / MODELS[0]
        structs = [
            f"assets/electric_field/energy_response/{MODELS[0]}/{struct_file.stem}.xyz"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> energy_responseApp:
    """
    Get energy_response benchmark app layout and callback registration.

    Returns
    -------
    energy_responseApp
        Benchmark layout and callback registration.
    """
    return EnergyResponseApp(
        name=BENCHMARK_NAME,
        description="Total energies for linear organic molecules.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "energy_response_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    energy_response_app = get_app()
    full_app.layout = energy_response_app.layout
    energy_response_app.register_callbacks()

    # Run app
    full_app.run(port=8053, debug=True)