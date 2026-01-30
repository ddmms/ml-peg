"""Run O diffusion on 2D TMDs app."""

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
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "O diffusion on 2D TMDs"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nebs.html#oxygen-adatom-diffusion-on-tmds"
DATA_PATH = APP_ROOT / "data" / "nebs" / "O_diffusion_2D_TMDs"
COMPOUNDS = ["MoS2", "MoSe2", "MoTe2", "WS2", "WSe2", "WTe2"]


class ODiffusionTMDsApp(BaseApp):
    """O diffusion on 2D TMDs benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_plots = {
            model: {
                f"{compound} barrier error": read_plot(
                    DATA_PATH / f"figure_{model}_O_diffusion_{compound}.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-{compound}",
                )
                for compound in COMPOUNDS
            }
            for model in MODELS
        }

        # Assets dir will be parent directory
        assets_dir = "assets/nebs/O_diffusion_2D_TMDs"
        structs = {
            model: {
                f"{compound} barrier error": (
                    f"{assets_dir}/{model}/{model}-{compound}-neb-band.extxyz"
                )
                for compound in COMPOUNDS
            }
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=scatter_plots,
        )

        for model in MODELS:
            for compound in COMPOUNDS:
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-figure-{compound}",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=structs[model][f"{compound} barrier error"],
                    mode="traj",
                )


def get_app() -> ODiffusionTMDsApp:
    """
    Get O diffusion on 2D TMDs benchmark app layout and callback registration.

    Returns
    -------
    ODiffusionTMDsApp
        Benchmark layout and callback registration.
    """
    return ODiffusionTMDsApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting oxygen adatom diffusion barriers on 2D TMDs."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "O_diffusion_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    o_diffusion_app = get_app()
    full_app.layout = o_diffusion_app.layout
    o_diffusion_app.register_callbacks()

    # Run app
    full_app.run(port=8056, debug=True)
