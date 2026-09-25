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
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Energy response"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/electric_field.html#energy_response"
DATA_PATH = APP_ROOT / "data" / "electric_field" / "energy_response"


class EnergyResponseApp(BaseApp):
    """energy_response benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_total = read_plot(
            DATA_PATH / "figure_energy_responses.json",
            id=f"{BENCHMARK_NAME}-figure-total",
        )
        scatter_alkane = read_plot(
            DATA_PATH / "figure_alkane_energy_responses.json",
            id=f"{BENCHMARK_NAME}-figure-alkane",
        )
        scatter_cumulene = read_plot(
            DATA_PATH / "figure_cumulene_energy_responses.json",
            id=f"{BENCHMARK_NAME}-figure-cumulene",
        )

        structs_dir = DATA_PATH / MODELS[0]
        assets_prefix = f"assets/electric_field/energy_response/{MODELS[0]}"

        alkane_structs = [
            f"{assets_prefix}/{p.name}"
            for p in sorted(structs_dir.glob("ALKANES_[0-9]*.xyz"))
        ]
        cumulene_structs = [
            f"{assets_prefix}/{p.name}"
            for p in sorted(structs_dir.glob("CUMULENES_[0-9]*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Total MAE": scatter_total,
                "Alkanes MAE": scatter_alkane,
                "Cumulenes MAE": scatter_cumulene,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure-total",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=alkane_structs + cumulene_structs,
            mode="struct",
        )
        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure-alkane",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=alkane_structs,
            mode="struct",
        )
        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure-cumulene",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=cumulene_structs,
            mode="struct",
        )


def get_app() -> EnergyResponseApp:
    """
    Get energy_response benchmark app layout and callback registration.

    Returns
    -------
    energy_responseApp
        Benchmark layout and callback registration.
    """
    return EnergyResponseApp(
        name=BENCHMARK_NAME,
        description="Energy responses of linear organic molecules to external electric fields.",
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
