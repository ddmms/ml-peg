"""Run 3dTMV barriers app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "BH9"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "molecular.html#3dTMV-tm-complexes"
)
DATA_PATH = APP_ROOT / "data" / "tm_complexes" / "3dTMV"


class Benchmark3dTMVApp(BaseApp):
    """3dTMV benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_3dtmv.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        model_dir = DATA_PATH / MODELS[0]
        if model_dir.exists():
            # Note: sorting different to rxn_count order in calc
            ts_files = sorted(model_dir.glob("*.xyz"))
            structs = [
                f"assets/tm_complexes/3dTMV/{MODELS[0]}/{ts_file.name}"
                for ts_file in ts_files
            ]
        else:
            structs = []

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> Benchmark3dTMVApp:
    """
    Get 3dTMV benchmark app layout and callback registration.

    Returns
    -------
    Benchmark3dTMVApp
        Benchmark layout and callback registration.
    """
    return Benchmark3dTMVApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting vertical ionization energies for the "
            "3dTMV dataset of 28 transition metal complexes."
            "Reference data from ph-AFQMC calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "3dtmv_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8071, debug=True)
