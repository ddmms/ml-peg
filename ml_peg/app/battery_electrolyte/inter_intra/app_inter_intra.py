"""Run inter intra benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_plot
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)

BENCHMARK_NAME = "Inter-Intra"
DATA_PATH = APP_ROOT / "data" / "battery_electrolyte" / "inter_intra"
REF_PATH = CALCS_ROOT / "battery_electrolyte" / "inter_intra" / "data"


class InterIntraApp(BaseApp):
    """Inter intra benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        parity_plots = {
            model: {
                "Intra-Forces": read_plot(
                    DATA_PATH / f"intra-forces_parity_{model}.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                ),
                "Inter-Forces": read_plot(
                    DATA_PATH / f"inter-forces_parity_{model}.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                ),
                "Inter-Energy": read_plot(
                    DATA_PATH / f"inter-energy_parity_{model}.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                ),
                "Intra-Virial": read_plot(
                    DATA_PATH / f"intra-virial_parity_{model}.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                ),
                "Inter-Virial": read_plot(
                    DATA_PATH / f"inter-virial_parity_{model}.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                ),
            }
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=parity_plots,
        )


def get_app() -> InterIntraApp:
    """
    Get inter intra benchmark app layout and callback registration.

    Returns
    -------
    InterIntraApp
        Benchmark layout and callback registration.
    """
    return InterIntraApp(
        name=BENCHMARK_NAME,
        description=(
            "Evaluate model inter/intra property prediction "
            "for different densities of LIB full electrolyte"
            " and neat solvent configs"
        ),
        # docs_url=DOCS_URL,
        table_path=DATA_PATH / "inter_intra_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    InterIntra_app = get_app()
    full_app.layout = InterIntra_app.layout
    InterIntra_app.register_callbacks()
    full_app.run(port=8054, debug=True)
