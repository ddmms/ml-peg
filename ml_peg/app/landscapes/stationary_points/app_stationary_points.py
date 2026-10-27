"""Run Landscape17 stationary points app."""

from __future__ import annotations

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
BENCHMARK_NAME = "Landscape17-stationary-points"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/landscapes.html"
    "#stationary-points"
)
DATA_PATH = APP_ROOT / "data" / "landscapes" / "stationary_points"
INFO_PATH = DATA_PATH / "info.json"

# Which figure to show when a metric column is selected. Several metrics share
# a figure: the median and the upper percentile of one distribution are two
# readings of the same violin, and showing it for either is what makes the
# pair interpretable -- a median and a 90th percentile far apart mean a tail,
# which is visible in the plot and not in the two numbers.
FIGURES = {
    "Energy RMSE": "figure_stationary_energies.json",
    "Residual force (median)": "figure_residual_forces.json",
    "Residual force (90th pct)": "figure_residual_forces.json",
    "Path force RMSE": "figure_path_forces.json",
    "Minima drift (median)": "figure_minima_drift.json",
    "Minima drift (mean)": "figure_minima_drift.json",
    "Saddle drift (median)": "figure_saddle_drift.json",
    "Saddle drift (mean)": "figure_saddle_drift.json",
}


# Figures whose points are individual structures, and the directory the
# analysis staged those structures into. The files are named by position, so
# sorting them reproduces the order the points appear in.
STRUCTURES = {
    "figure_stationary_energies.json": "points",
    "figure_minima_drift.json": "minima",
    "figure_saddle_drift.json": "saddles",
}


class StationaryPointsApp(BaseApp):
    """Landscape17 stationary points benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        for filename, subdirectory in STRUCTURES.items():
            staged = DATA_PATH / MODELS[0] / subdirectory
            if not staged.exists():
                continue
            struct_from_scatter(
                scatter_id=f"{BENCHMARK_NAME}-{filename.removesuffix('.json')}",
                struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                structs=[
                    f"/assets/landscapes/stationary_points/{MODELS[0]}"
                    f"/{subdirectory}/{path.name}"
                    for path in sorted(staged.glob("*.xyz"))
                ],
                mode="struct",
            )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                metric: read_plot(
                    DATA_PATH / filename,
                    id=f"{BENCHMARK_NAME}-{filename.removesuffix('.json')}",
                )
                for metric, filename in FIGURES.items()
                if (DATA_PATH / filename).exists()
            },
        )


def get_app() -> StationaryPointsApp:
    """
    Get Landscape17 stationary points benchmark app layout and callbacks.

    Returns
    -------
    StationaryPointsApp
        Benchmark layout and callback registration.
    """
    return StationaryPointsApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance on the DFT stationary points of Landscape17: six "
            "molecules mapped as networks of minima joined by transition "
            "states, 28 minima and 67 transition states at wB97X/6-31G(d). "
            "Energies and forces are evaluated at those geometries and along "
            "the 11,173 steepest-descent frames joining them, and each "
            "structure is then reconverged under the model to measure how far "
            "it travels and whether its transition states still connect the "
            "same minima."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "stationary_points_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
