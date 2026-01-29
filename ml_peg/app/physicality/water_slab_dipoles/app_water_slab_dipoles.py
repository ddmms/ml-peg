"""Run water slab dipoles app."""

from __future__ import annotations

# from dash import Dash
# from dash.html import Div
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp

# from ml_peg.app.utils.build_callbacks import (
#    plot_from_table_column,
#    struct_from_scatter,
# )
# from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Dipoles of Water Slabs"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#water_slab_dipoles"
DATA_PATH = APP_ROOT / "data" / "molecular_crystal" / "water_slab_dipoles"


class WaterSlabDipolesApp(BaseApp):
    """Water slab dipole benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        return


def get_app() -> WaterSlabDipolesApp:
    """
    Get water slab dipoles benchmark app layout and callback registration.

    Returns
    -------
    WaterSlabDipolesApp
        Benchmark layout and callback registration.
    """
    return WaterSlabDipolesApp(
        name=BENCHMARK_NAME,
        description="Dipole distribution of a 38 A water slab",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "water_slab_dipoles_metrics_table.json",
        extra_components=[],
    )
