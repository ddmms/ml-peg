"""Run Li diffusion app."""

from __future__ import annotations

from pathlib import Path

from dash import Dash
from dash.html import Div

from mlip_testing.app import APP_ROOT
from mlip_testing.app.base_app import BaseApp
from mlip_testing.app.utils.build_callbacks import (
    plot_from_table_cell,
    struct_from_scatter,
)
from mlip_testing.app.utils.load import read_plot
from mlip_testing.calcs.models.models import MODELS

BENCHMARK_NAME = Path(__file__).name.removeprefix("app_").removesuffix(".py")
DATA_PATH = APP_ROOT / "data" / "nebs" / "li_diffusion"


class LiDiffusionApp(BaseApp):
    """Li diffusion benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_plots = {
            model: {
                "Path B error": read_plot(
                    DATA_PATH / f"figure_{model}_neb_b.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-b",
                ),
                "Path C error": read_plot(
                    DATA_PATH / f"figure_{model}_neb_c.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-c",
                ),
            }
            for model in MODELS
        }

        # Assets dir will be parent directory
        assets_dir = "assets/nebs/li_diffusion"
        structs = {
            model: {
                "Path B error": f"{assets_dir}/{model}/{model}-b-neb-band.extxyz",
                "Path C error": f"{assets_dir}/{model}/{model}-c-neb-band.extxyz",
            }
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=scatter_plots,
        )

        for model in MODELS:
            for path in ("b", "c"):
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-figure-{path}",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=structs[model][f"Path {path.upper()} error"],
                    mode="traj",
                )


def get_app() -> LiDiffusionApp:
    """
    Get Li diffusion benchmark app layout and callback registration.

    Returns
    -------
    LiDiffusionApp
        Benchmark layout and callback registration.
    """
    return LiDiffusionApp(
        name="NEBs",
        title="NEBs",
        description=("Performance in predicting energy barriers for Li diffision."),
        table_path=DATA_PATH / "li_diffusion_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    li_diffusion_app = get_app()
    full_app.layout = li_diffusion_app.layout
    li_diffusion_app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
