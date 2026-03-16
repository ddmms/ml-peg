"""Run PuO2 force benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.app.utils.load import collect_traj_assets, read_density_plot_for_model
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Plutonium Dioxide"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/actinides.html#plutonium_dioxide"
DATA_PATH = APP_ROOT / "data" / "actinides" / "plutonium_dioxide"


class PuO2App(BaseApp):
    """PuO2 benchmark app."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        density_plots = {}

        for model in MODELS:
            # We explicitly map the exact Column Header strings from metrics.yml
            # to their respective JSON figure files.
            plots = {
                "Energy MAE": read_density_plot_for_model(
                    filename=DATA_PATH / "figure_energy_density.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-energy-figure",
                ),
                "Force MAE": read_density_plot_for_model(
                    filename=DATA_PATH / "figure_force_density.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-force-figure",
                ),
                "Stress MAE": read_density_plot_for_model(
                    filename=DATA_PATH / "figure_stress_density.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-stress-figure",
                ),
            }

            # This comprehension is key: it prevents the app from trying to
            # link to a plot if the data is missing for that specific model.
            model_plots = {k: v for k, v in plots.items() if v is not None}

            if model_plots:
                density_plots[model] = model_plots

        # This links the sanitized ID to the compiled dictionary of figures
        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=density_plots,
        )

        assets_prefix = "assets/actinides/plutonium_dioxide"

        for prop, scatter_suffix in [
            ("density_traj_energy", "energy-figure"),
            ("density_traj_force", "force-figure"),
            ("density_traj_stress", "stress-figure"),
        ]:
            struct_trajs = collect_traj_assets(
                data_path=DATA_PATH,
                assets_prefix=assets_prefix,
                models=MODELS,
                traj_dirname=prop,
            )
            for model in struct_trajs:
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{scatter_suffix}",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=struct_trajs[model],
                    mode="traj",
                )


def get_app() -> PuO2App:
    """
    Get PuO2 benchmark app instance with layout and callbacks.

    Returns
    -------
    PuO2App
        Benchmark layout and callback registration.
    """
    return PuO2App(
        name=BENCHMARK_NAME,
        description=("Basic performance metrics on plutonium (IV) dioxide"),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "puo2_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__)
    puo2_app = get_app()
    full_app.layout = puo2_app.layout
    puo2_app.register_callbacks()
    full_app.run(port=8050, debug=True)
