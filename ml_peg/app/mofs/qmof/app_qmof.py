"""Run QMOF app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.app.utils.load import collect_traj_assets, read_density_plot_for_model
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "QMOF"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/mofs.html#qmof"
DATA_PATH = APP_ROOT / "data" / "mofs" / "qmof"
INFO_PATH = DATA_PATH / "info.json"


class QMOFApp(BaseApp):
    """QMOF benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        density_plots: dict[str, dict] = {}
        for model in MODELS:
            density_graph = read_density_plot_for_model(
                filename=DATA_PATH / "figure_qmof_density.json",
                model=model,
                id=f"{BENCHMARK_NAME}-{model}-density",
            )
            if density_graph is not None:
                density_plots[model] = {"MAE": density_graph}

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=density_plots,
        )

        struct_trajs = collect_traj_assets(
            data_path=DATA_PATH,
            assets_prefix="/assets/mofs/qmof",
            models=MODELS,
            traj_dirname="density_traj",
            suffix=".extxyz",
        )
        for model in struct_trajs:
            struct_from_scatter(
                scatter_id=f"{BENCHMARK_NAME}-{model}-density",
                struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                structs=struct_trajs[model],
                mode="traj",
            )


def get_app() -> QMOFApp:
    """
    Get QMOF benchmark app layout and callback registration.

    Returns
    -------
    QMOFApp
        Benchmark layout and callback registration.
    """
    return QMOFApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting energies of metal organic frameworks from the "
            "QMOF database, compared against PBE+D3 reference data."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "qmof_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
