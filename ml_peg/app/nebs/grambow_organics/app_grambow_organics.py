"""Run Grambow organics NEB convergence benchmark app."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp

BENCHMARK_NAME = "GrambowOrganics"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nebs.html#grambow-organics"
)
DATA_PATH = APP_ROOT / "data" / "nebs" / "grambow_organics"


class GrambowOrganicsApp(BaseApp):
    """Grambow organics NEB benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""


def get_app() -> GrambowOrganicsApp:
    """
    Get Grambow organics benchmark app layout and callback registration.

    Returns
    -------
    GrambowOrganicsApp
        Benchmark layout and callback registration.
    """
    return GrambowOrganicsApp(
        name="Grambow Organics",
        framework_ids="mlip_audit",
        description=(
            "Fraction of nudged elastic band simulations that converge for 100 "
            "elementary organic reactions sampled from the Grambow dataset. "
            "Assesses whether the model can produce a stable reaction path."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "grambow_organics_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8072, debug=True)
