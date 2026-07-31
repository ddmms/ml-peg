"""Run battery electrolyte densities app."""

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

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Battery Electrolyte Densities"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html"
    "#battery-electrolyte-densities"
)
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "battery_electrolyte_densities"
INFO_PATH = DATA_PATH / "info.json"


class BatteryElectrolyteDensitiesApp(BaseApp):
    """Battery electrolyte densities benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_battery_electrolyte_densities.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        model_dir = DATA_PATH / MODELS[0]
        if model_dir.exists():
            labels = sorted([f.stem for f in model_dir.glob("*.xyz")])
            structs = [
                "/assets/molecular_dynamics/battery_electrolyte_densities/"
                f"{MODELS[0]}/{label}.xyz"
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


def get_app() -> BatteryElectrolyteDensitiesApp:
    """
    Get battery electrolyte densities benchmark app layout and callbacks.

    Returns
    -------
    BatteryElectrolyteDensitiesApp
        Benchmark layout and callback registration.
    """
    return BatteryElectrolyteDensitiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting densities of Na-ion battery electrolytes "
            "in glyme and carbonate solvents, and of the corresponding neat "
            "solvents. Reference data are experimental densities at 298.2 K."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "battery_electrolyte_densities_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
        framework_ids="omol25-electrolytes",
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8064, debug=True)
