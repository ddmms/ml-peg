"""MC500 molecular-crystal relaxation benchmark app."""

from __future__ import annotations

import json

from dash.dcc import Graph
from dash.html import Div, P

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "MC500 Molecular Crystal Relaxation"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_crystal.html#mc500"
)
DATA_PATH = APP_ROOT / "data" / "molecular_crystal" / "MC500"
INFO_PATH = DATA_PATH / "info.json"
ASSETS_PREFIX = "/assets/molecular_crystal/MC500"

METRICS = (
    "Mean RMSCD excluding H",
    "Mean RMSCD all atoms",
    "Structures within threshold",
    "Convergence",
)


class MC500App(BaseApp):
    """MC500 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register table, plot, and structure callbacks."""
        figure_path = DATA_PATH / "figure_rmscd.json"
        figures = {}
        if figure_path.exists():
            with figure_path.open(encoding="utf-8") as f:
                figures = json.load(f)

        plots: dict[str, dict[str, Graph]] = {}
        for model_name in MODELS:
            figure = figures.get(model_name)
            if figure is None:
                continue
            graph = Graph(
                id=f"{BENCHMARK_NAME}-{model_name}-figure",
                figure=figure,
            )
            plots[model_name] = dict.fromkeys(METRICS, graph)

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=plots,
        )

        for model_name in MODELS:
            struct_dir = DATA_PATH / model_name / "structures"
            trajectories = [
                f"{ASSETS_PREFIX}/{model_name}/structures/{path.name}"
                for path in sorted(struct_dir.glob("*.xyz"))
            ]
            if not trajectories:
                continue
            struct_from_scatter(
                scatter_id=f"{BENCHMARK_NAME}-{model_name}-figure",
                struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                structs=trajectories,
                mode="traj",
                follow_frames=False,
            )


def get_app() -> MC500App:
    """
    Get the MC500 app.

    Returns
    -------
    MC500App
        MC500 benchmark app.
    """
    return MC500App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in relaxing 500 molecular-crystal structures relative to "
            "r2SCAN+MBD reference geometries."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "mc500_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            P("Click a point to compare the reference and relaxed structures."),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
