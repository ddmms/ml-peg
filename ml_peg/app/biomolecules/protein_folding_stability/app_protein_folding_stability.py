"""Run protein folding stability benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "ProteinFoldingStability"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/biomolecules.html#protein-folding-stability"
DATA_PATH = APP_ROOT / "data" / "biomolecules" / "protein_folding_stability"


class ProteinFoldingStabilityApp(BaseApp):
    """Protein folding stability benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_rmsd_trajectory.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"RMSD": scatter},
        )


def get_app() -> ProteinFoldingStabilityApp:
    """
    Get protein folding stability benchmark app layout and callback registration.

    Returns
    -------
    ProteinFoldingStabilityApp
        Benchmark layout and callback registration.
    """
    return ProteinFoldingStabilityApp(
        name="Protein Folding Stability",
        framework_ids="mlip_audit",
        description=(
            "Performance in keeping small proteins folded during molecular "
            "dynamics started from their native conformation. The RMSD, TM "
            "score, and radius of gyration relative to experimental reference "
            "structures are tracked along the trajectory."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "protein_folding_stability_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8070, debug=True)
