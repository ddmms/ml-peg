"""Run protein sampling benchmark app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "ProteinSampling"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/biomolecules.html#protein-sampling"
DATA_PATH = APP_ROOT / "data" / "biomolecules" / "protein_sampling"


class ProteinSamplingApp(BaseApp):
    """Protein sampling benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Show the residue breakdown for the selected metric."""
        plots = {}
        for metric in self.metrics:
            slug = metric.lower().replace(" ", "_")
            plots[metric] = read_plot(
                DATA_PATH / f"figure_{slug}.json", id=f"{BENCHMARK_NAME}-plot"
            )
        plot_from_table_column(self.table_id, f"{BENCHMARK_NAME}-figure", plots)


def get_app() -> ProteinSamplingApp:
    """
    Get protein sampling benchmark app layout and callback registration.

    Returns
    -------
    ProteinSamplingApp
        Benchmark layout and callback registration.
    """
    return ProteinSamplingApp(
        name="Protein Sampling",
        framework_ids="mlip_audit",
        description=(
            "Performance in exploring protein conformational space during molecular "
            "dynamics. Sampled backbone dihedral distributions are compared against "
            "reference distributions via distribution RMSD, Hellinger distance, and "
            "the ratio of outlying conformations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "protein_sampling_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure"),
        ],
    )
