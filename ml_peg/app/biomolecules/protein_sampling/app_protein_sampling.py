"""Run protein sampling benchmark app."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp

BENCHMARK_NAME = "ProteinSampling"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/biomolecules.html#protein-sampling"
DATA_PATH = APP_ROOT / "data" / "biomolecules" / "protein_sampling"


class ProteinSamplingApp(BaseApp):
    """Protein sampling benchmark app layout and callbacks."""


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
        extra_components=[],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8071, debug=True)
