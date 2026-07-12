"""Run Thermal thermal_Conductivity app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Thermal Conductivity"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/"
    "benchmarks/bulk_crystal.html#thermal-conductivity"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "thermal_conductivity"
INFO_PATH = DATA_PATH / "info.json"


class ThermalConductivityApp(BaseApp):
    """Thermal Conductivity benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_thermal_conductivity.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        fast_scatter = read_plot(
            DATA_PATH / "figure_fast_thermal_conductivity.json",
            id=f"{BENCHMARK_NAME}-fast-figure",
        )

        # Assets dir will be parent directory - individual files for each system
        # structs_dir = DATA_PATH / MODELS[0]
        # structs = [
        #     f"assets/bulk_crystal/thermal_conductivity/"
        #     f"{MODELS[0]}/{struct_file.stem}.xyz"
        #     for struct_file in sorted(structs_dir.glob("*.xyz"))
        # ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "kSRE": scatter,
                "kSRME": scatter,
                "Fast kSRE": fast_scatter,
                "Fast kSRME": fast_scatter,
            },
        )

        # struct_from_scatter(
        #     scatter_id=f"{BENCHMARK_NAME}-figure",
        #     struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
        #     structs=structs,
        #     mode="struct",
        # )


def get_app() -> ThermalConductivityApp:
    """
    Get Thermal Conductivity benchmark app layout and callback registration.

    Returns
    -------
    ThermalConductivityApp
        Benchmark layout and callback registration.
    """
    return ThermalConductivityApp(
        name=BENCHMARK_NAME,
        description="Thermal conductivity for 103 binary crystals.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "thermal_conductivity.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            # Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
