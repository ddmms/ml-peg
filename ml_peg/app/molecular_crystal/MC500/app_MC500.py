"""MC500 molecular-crystal relaxation benchmark app."""

from __future__ import annotations

import json

from dash import Input, Output, callback
from dash.dcc import Graph
from dash.html import B, Div, Iframe, P

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    _register_point_highlight,
    plot_from_table_cell,
)
from ml_peg.app.utils.weas import generate_weas_html
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

IFRAME_STYLE = {
    "height": "550px",
    "width": "100%",
    "border": "1px solid #ddd",
    "borderRadius": "5px",
}
GRID_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(2, minmax(0, 1fr))",
    "gap": "8px",
}
CAPTION_STYLE = {
    "fontSize": "1.15rem",
    "borderLeft": "4px solid #636efa",
    "paddingLeft": "10px",
    "margin": "8px 0 14px 0",
}


def _structure_panel(trajectory: str, title: str, frame: int) -> Div:
    """
    Build a labelled WEAS viewer opened at one frame of an MC500 trajectory.

    Parameters
    ----------
    trajectory
        URL of the two-frame reference and relaxed trajectory.
    title
        Label displayed above the viewer.
    frame
        Frame at which to open the trajectory.

    Returns
    -------
    Div
        Labelled structure viewer.
    """
    return Div(
        [
            P(B(title)),
            Iframe(
                srcDoc=generate_weas_html(trajectory, mode="traj", index=frame),
                style=IFRAME_STYLE,
            ),
        ]
    )


def struct_pair_from_scatter(
    scatter_id: str,
    struct_id: str,
    trajectories: list[str],
    model_name: str,
) -> None:
    """
    Show reference and MLIP-relaxed structures beside each other on point click.

    Parameters
    ----------
    scatter_id
        ID of the clickable RMSCD graph.
    struct_id
        ID of the structure placeholder.
    trajectories
        Two-frame trajectories in the same order as the scatter points.
    model_name
        Name of the model that produced the relaxed structures.
    """
    _register_point_highlight(scatter_id, follow_frames=False)

    @callback(
        Output(struct_id, "children", allow_duplicate=True),
        Input(scatter_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_structures(click_data):
        """
        Build the side-by-side structure comparison for a clicked point.

        Parameters
        ----------
        click_data
            Plotly data for the clicked RMSCD point.

        Returns
        -------
        Div
            Reference and relaxed structure viewers.
        """
        if not click_data:
            return Div("Click on a point to view structures.")

        point = click_data["points"][0]
        index = point["pointNumber"]
        if index >= len(trajectories):
            return Div("Structures unavailable for this point.")

        custom_data = point.get("customdata") or []
        refcode = custom_data[0] if custom_data else f"structure {index + 1}"
        trajectory = trajectories[index]
        return Div(
            [
                P(B(f"CSD refcode: {refcode}"), style=CAPTION_STYLE),
                Div(
                    [
                        _structure_panel(
                            trajectory,
                            "r2SCAN+MBD reference",
                            frame=0,
                        ),
                        _structure_panel(
                            trajectory,
                            f"MLIP relaxed ({model_name})",
                            frame=1,
                        ),
                    ],
                    style=GRID_STYLE,
                ),
            ]
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
            struct_pair_from_scatter(
                scatter_id=f"{BENCHMARK_NAME}-{model_name}-figure",
                struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                trajectories=trajectories,
                model_name=model_name,
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
