"""Run molecular dynamics stability app."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Stability"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#stability"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "stability"
INFO_PATH = DATA_PATH / "info.json"


class StabilityApp(BaseApp):
    """Stability benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""


def get_app() -> StabilityApp:
    """
    Get stability benchmark app layout and callback registration.

    Returns
    -------
    StabilityApp
        Benchmark layout and callback registration.
    """
    return StabilityApp(
        name=BENCHMARK_NAME,
        framework_ids="mlip_audit",
        description=(
            "Fraction of short molecular dynamics simulations that complete "
            "without error, across small molecules, peptides and proteins in "
            "vacuum and solvent."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "stability_metrics_table.json",
        extra_components=[
            read_plot(
                filename=DATA_PATH / "stability_progress.json",
                id=f"{BENCHMARK_NAME}-progress-figure",
            )
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8067, debug=True)
