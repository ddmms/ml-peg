"""Run lanthanide isomer complex benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column, struct_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Lanthanide Isomer Complexes"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/lanthanides.html"
    "#isomer-complexes"
)
DATA_PATH = APP_ROOT / "data" / "lanthanides" / "isomer_complexes"


class IsomerComplexesApp(BaseApp):
    """Lanthanide isomer complex benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_isomer_complexes.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )

        # Use first model's structures for visualization
        if MODELS:
            structs_dir = DATA_PATH / MODELS[0]
            structs = [
                f"assets/lanthanides/isomer_complexes/{MODELS[0]}/{struct_file.stem}.xyz"
                for struct_file in sorted(structs_dir.glob("*.xyz"))
            ]

            struct_from_scatter(
                scatter_id=f"{BENCHMARK_NAME}-figure",
                struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                structs=structs,
                mode="struct",
            )


def get_app() -> IsomerComplexesApp:
    """
    Get lanthanide isomer complex benchmark app layout and callback registration.

    Returns
    -------
    IsomerComplexesApp
        Benchmark layout and callback registration.
    """
    return IsomerComplexesApp(
        name=BENCHMARK_NAME,
        description=(
            "Relative energies of lanthanide isomer complexes compared to r2SCAN-3c."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "isomer_complexes_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    app_instance = get_app()
    full_app.layout = app_instance.layout
    app_instance.register_callbacks()

    # Run app
    full_app.run(port=8061, debug=True)
