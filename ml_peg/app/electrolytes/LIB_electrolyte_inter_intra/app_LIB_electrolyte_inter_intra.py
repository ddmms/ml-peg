"""Run LIB electrolyte inter intra benchmark app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_density_plot_for_model, read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

BENCHMARK_NAME = "LIB electrolyte Inter-Intra Properties"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/electrolytes.html#lib-electrolyte-inter-intra-properties"
DATA_PATH = APP_ROOT / "data" / "electrolytes" / "LIB_electrolyte_inter_intra"
INFO_PATH = (
    APP_ROOT / "data" / "electrolytes" / "LIB_electrolyte_inter_intra" / "info.json"
)


class LIBelectrolyteInterIntraApp(BaseApp):
    """LIB electrolyte inter intra benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        plots = {
            model: {
                "Intra-Forces": read_density_plot_for_model(
                    DATA_PATH / "intra-forces_density_parity.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                ),
                "Inter-Forces": read_density_plot_for_model(
                    DATA_PATH / "inter-forces_density_parity.json",
                    model=model,
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
            cell_to_plot=plots,
        )


def get_app() -> LIBelectrolyteInterIntraApp:
    """
    Get LIB electrolyte inter intra benchmark app layout and callback registration.

    Returns
    -------
    LIBelectrolyteInterIntraApp
        Benchmark layout and callback registration.
    """
    return LIBelectrolyteInterIntraApp(
        name=BENCHMARK_NAME,
        description=(
            "Evaluate model inter/intra property prediction "
            "for different densities of LIB electrolyte"
            " and neat solvent configs"
        ),
        # docs_url=DOCS_URL,
        table_path=DATA_PATH / "inter_intra_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
        info_path=INFO_PATH,
    )
