"""Run surface energies app."""

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

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Surface Energies"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/carbon.html#surface-energies"
)
DATA_PATH = APP_ROOT / "data" / "carbon" / "surface_energies"
INFO_PATH = DATA_PATH / "info.json"


class SurfaceEnergiesApp(BaseApp):
    """Surface energies benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_surface_energies_relaxed.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Relaxed surface energy MAE (D3)": scatter},
        )

        model_dir = DATA_PATH / MODELS[0]
        if model_dir.exists():
            labels = sorted(f.stem for f in model_dir.glob("*.xyz"))
            structs = [
                f"/assets/carbon/surface_energies/{MODELS[0]}/{label}.xyz"
                for label in labels
            ]
        else:
            structs = []

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> SurfaceEnergiesApp:
    """
    Get surface energies benchmark app layout and callback registration.

    Returns
    -------
    SurfaceEnergiesApp
        Benchmark layout and callback registration.
    """
    return SurfaceEnergiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Single-point as-cut and relaxed surface energies for diamond {100} and"
            " graphite (0001), plus as-cut only for amorphous carbon, which has no"
            " relaxed reference. Reference data is optB88-vdW."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "surface_energies_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
        framework_ids="gap-20",
    )
