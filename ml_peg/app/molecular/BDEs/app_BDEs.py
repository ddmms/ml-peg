"""Run BDEs app."""

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

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Bond Dissociation Energies"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular.html#bdes"
DATA_PATH = APP_ROOT / "data" / "molecular" / "BDEs"


class BDEsApp(BaseApp):
    """BDEs benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_bdes = read_plot(
            DATA_PATH / "figure.CYP3A4.dft_opt_geometry.BDEs.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        scatter_ranks = read_plot(
            DATA_PATH / "figure.CYP3A4.dft_opt_geometry.BDE_ranks.json",
            id=f"{BENCHMARK_NAME}-ranks-figure",
        )

        # Assets dir will be parent directory - individual files for each system
        structs_dir = DATA_PATH / MODELS[0]
        structs = [
            f"assets/molecular/BDEs/{MODELS[0]}/{struct_file.stem}.xyz"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Direct BDE": scatter_bdes, "BDE rank": scatter_ranks},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> BDEsApp:
    """
    Get bond dissociation energy benchmark app.

    Returns
    -------
    BDEsApp
        Benchmark layout and callback registration.
    """
    return BDEsApp(
        name=BENCHMARK_NAME,
        description="Bond Dissociation Energies",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "metrics_table.CYP3A4.dft_opt_geometry.BDEs.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )
