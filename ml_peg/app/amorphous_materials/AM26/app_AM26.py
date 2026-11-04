"""Display the AM26 scored table and precomputed static diagnostic heatmaps."""

from __future__ import annotations

from dash.html import P

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "AM26"
DATA_PATH = APP_ROOT / "data" / "amorphous_materials" / "AM26"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/amorphous_materials.html#am26"
)


class AM26App(BaseApp):
    """Use the standard ML-PEG table with two static all-model heatmaps."""

    def register_callbacks(self) -> None:
        """Keep standard table behaviour; static figures require no callbacks."""


def get_app() -> AM26App:
    """
    Load the standard table, element metadata, and both saved Plotly figures.

    Returns
    -------
    AM26App
        Benchmark component discovered under the existing amorphous category.
        Missing figure files follow read_plot's normal empty-graph behaviour.
    """
    return AM26App(
        name=BENCHMARK_NAME,
        description=(
            "Raw zero-shot energy and force errors across five AM26 "
            "amorphous-material systems."
        ),
        table_path=DATA_PATH / "am26_metrics_table.json",
        docs_url=DOCS_URL,
        info_path=DATA_PATH / "info.json",
        extra_components=[
            P(
                "Diagnostic heatmaps show all recorded models and do not "
                "follow table filters."
            ),
            read_plot(
                DATA_PATH / "figure_energy_relrmse_heatmap.json",
                id="AM26-energy-relrmse-heatmap",
            ),
            read_plot(
                DATA_PATH / "figure_force_relrmse_heatmap.json",
                id="AM26-force-relrmse-heatmap",
            ),
        ],
    )
