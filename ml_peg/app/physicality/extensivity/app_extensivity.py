"""Run extensivity app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import struct_from_table
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Extensivity"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#extensivity"
)
DATA_PATH = APP_ROOT / "data" / "physicality" / "extensivity"


class ExtensivityApp(BaseApp):
    """Extensivity benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Assets dir will be parent directory - individual files for each system
        assets_dir = f"assets/physicality/extensivity/{MODELS[0]}"
        structs = {
            "ΔE": f"{assets_dir}/slabs.xyz",
        }

        struct_from_table(
            table_id=self.table_id,
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            column_to_struct=structs,
            mode="struct",
        )


def get_app() -> ExtensivityApp:
    """
    Get extensivity benchmark app layout and callback registration.

    Returns
    -------
    ExtensivityApp
        Benchmark layout and callback registration.
    """
    return ExtensivityApp(
        name=BENCHMARK_NAME,
        description="Extensivity of slab energies, comparing two isolated slabs to the "
        "two slabs with a large separation.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "extensivity_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    extensivity_app = get_app()
    full_app.layout = extensivity_app.layout
    extensivity_app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
