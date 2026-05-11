"""Run polymer densities app."""

from __future__ import annotations

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

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Polymer Densities"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#polymer-densities"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "polymers"


class PolymerDensitiesApp(BaseApp):
    """Polymer densities benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_polymers.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        model_dir = DATA_PATH / MODELS[0]
        if model_dir.exists():
            labels = sorted([f.stem for f in model_dir.glob("*.xyz")])
            structs = [
                f"/assets/molecular_dynamics/polymers/{MODELS[0]}/{label}.xyz"
                for label in labels
            ]
        else:
            structs = []

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


def get_app() -> PolymerDensitiesApp:
    """
    Get polymer densities benchmark app layout and callback registration.

    Returns
    -------
    PolymerDensitiesApp
        Benchmark layout and callback registration.
    """
    return PolymerDensitiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting amorphous polymer densities. "
            "Reference data is experimental densities."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "polymers_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )
