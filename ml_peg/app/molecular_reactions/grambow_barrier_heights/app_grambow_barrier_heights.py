"""Run Grambow barrier heights benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_density_plot_for_model
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "GrambowBarrierHeights"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_reactions.html#grambow-barrier-heights"
DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / "grambow_barrier_heights"


class GrambowBarrierHeightsApp(BaseApp):
    """Grambow barrier heights benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        density_plots: dict[str, dict] = {}
        for model in MODELS:
            graph = read_density_plot_for_model(
                filename=DATA_PATH / "figure_barrier_density.json",
                model=model,
                id=f"{BENCHMARK_NAME}-{model}-barrier-figure",
            )
            if graph is not None:
                density_plots[model] = {"Barrier Height MAE": graph}

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=density_plots,
        )


def get_app() -> GrambowBarrierHeightsApp:
    """
    Get Grambow barrier heights benchmark app layout and callback registration.

    Returns
    -------
    GrambowBarrierHeightsApp
        Benchmark layout and callback registration.
    """
    return GrambowBarrierHeightsApp(
        name="Grambow Barrier Heights",
        framework_ids="mlip_audit",
        description=(
            "Performance in predicting reaction barrier heights (and reaction "
            "energies) for elementary organic reactions from the Grambow "
            "dataset. Reference data from wB97X-D3/def2-TZVP calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "grambow_barrier_heights_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(
        __name__,
        assets_folder=DATA_PATH.parent.parent,
        suppress_callback_exceptions=True,
    )
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8071, debug=True)
