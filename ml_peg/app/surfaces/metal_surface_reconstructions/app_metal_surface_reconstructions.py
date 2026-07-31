"""Run Metal surface reconstructions app."""

from __future__ import annotations

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
BENCHMARK_NAME = "Metal Surfaces"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#metal-surface-reconstructions"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "metal_surfaces"
INFO_PATH = DATA_PATH / "info.json"


class MetalSurfaceApp(BaseApp):
    """Metal surface reconstructions benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "slab_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Systems from info match the order of the scatter data
        if self.info:
            systems = self.info["system"]
        else:
            structs_dir = DATA_PATH / MODELS[0]
            systems = [
                struct_file.stem for struct_file in sorted(structs_dir.glob("*.xyz"))
            ]

        # Assets dir will be parent directory - individual files for each system
        structs = [
            f"assets/surfaces/metal_surfaces/{MODELS[0]}/{system}.xyz"
            for system in systems
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter, "Displacement": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> MetalSurfaceApp:
    """
    Get Metal surface reconstructions benchmark app layout and callback registration.

    Returns
    -------
    MetalSurfaceApp
        Benchmark layout and callback registration.
    """
    return MetalSurfaceApp(
        name=BENCHMARK_NAME,
        description="Energies for two surface reconstuctions.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "metal_surfaces_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
