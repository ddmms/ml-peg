"""Run Landscape17 landscape search app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Landscape17-landscape-search"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/landscapes.html"
    "#landscape-search"
)
DATA_PATH = APP_ROOT / "data" / "landscapes" / "landscape_search"

# Which figure to show when a metric column is clicked. The median and the
# 90th percentile of one distribution share a violin, which is what makes the
# pair readable: a 90th percentile far above the median means a tail of badly
# placed structures rather than a uniform offset, and that is visible in the
# plot and not in the two numbers. The four fractions summarise counts rather
# than a distribution, so they have no figure and clicking them does nothing.
#
# Every key must match a metric name in metrics.yml exactly. A stale key is
# silent: the callback raises PreventUpdate and the column stops responding.
FIGURES = {
    "Minima RMSD (median)": "figure_minima_rmsd.json",
    "Minima RMSD (90th pct)": "figure_minima_rmsd.json",
    "Saddle RMSD (median)": "figure_saddle_rmsd.json",
    "Saddle RMSD (90th pct)": "figure_saddle_rmsd.json",
}


class LandscapeSearchApp(BaseApp):
    """Landscape17 landscape search benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
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


def get_app() -> LandscapeSearchApp:
    """
    Get Landscape17 landscape search benchmark app layout and callbacks.

    Returns
    -------
    LandscapeSearchApp
        Benchmark layout and callback registration.
    """
    return LandscapeSearchApp(
        name=BENCHMARK_NAME,
        description=(
            "Whether a model finds the Landscape17 landscape for itself. "
            "Each of six molecules is searched from five rMD17 starting "
            "conformers with basin hopping and eigenvector following, with "
            "no reference geometry supplied, and what the search finds is "
            "matched against the 28 DFT minima and 67 transition states. "
            "This asks a harder question than reconverging a known "
            "structure: a model can hold every DFT stationary point it is "
            "handed and still fail to locate them unaided, and it can also "
            "invent minima the reference does not contain."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "landscape_search_metrics_table.json",
        extra_components=[Div(id=f"{BENCHMARK_NAME}-figure-placeholder")],
    )
