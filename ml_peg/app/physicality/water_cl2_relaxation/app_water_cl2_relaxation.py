"""Run liquid densities app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Water-Cl2 relaxation"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#water-cl2-relaxation"
DATA_PATH = APP_ROOT / "data" / "physicality" / "water_cl2_relaxation"
INFO_PATH = DATA_PATH / "info.json"


class WaterCl2RelaxationApp(BaseApp):
    """Water-Cl2 relaxation benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_plots = {
            model: {
                "Cl2_stability": read_plot(
                    DATA_PATH / model / "figure_water_cl2.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                )
            }
            for model in MODELS
        }

        structs = {}
        for model in MODELS:
            model_dir = DATA_PATH / model
            if model_dir.exists():
                structs[model] = (
                    f"/assets/physicality/water_cl2_relaxation/{model}/relaxation.xyz"
                )
            else:
                print(f"Model directory does not exist: {model_dir}")
                structs[model] = ""

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=scatter_plots,
        )

        for model in MODELS:
            struct_from_scatter(
                scatter_id=f"{BENCHMARK_NAME}-{model}-figure",
                struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                structs=structs[model],
                mode="traj",
            )


def get_app() -> WaterCl2RelaxationApp:
    """
    Get water-Cl2 relaxation benchmark app layout and callback registration.

    Returns
    -------
    WaterCl2RelaxationApp
        Benchmark layout and callback registration.
    """
    return WaterCl2RelaxationApp(
        name=BENCHMARK_NAME,
        framework_ids="mace-polar-1",
        description=("Performance in predicting water-Cl2 relaxation."),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "water_cl2_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8063, debug=True)
