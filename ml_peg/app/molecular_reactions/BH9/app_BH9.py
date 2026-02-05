"""Run BH9 barriers app."""

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
    "molecular.html#bh9-reaction-barriers"
)
DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / "BH9"


class BH9App(BaseApp):
    """BH9 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_bh9_barriers.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        model_dir = DATA_PATH / MODELS[0]
        if model_dir.exists():
            # Note: sorting different to rxn_count order in calc
            ts_files = sorted(model_dir.glob("*.xyz"))
            structs = [
                f"assets/molecular_reactions/BH9/{MODELS[0]}/{ts_file.name}"
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


def get_app() -> BH9App:
    """
    Get BH9 benchmark app layout and callback registration.

    Returns
    -------
    BH9App
        Benchmark layout and callback registration.
    """
    return BH9App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting hydrolysis reaction barriers for the "
            "BH9 dataset of nine aqueous reactions spanning multiple functional "
            "groups. Reference data from CCSD(T) calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "bh9_barriers_metrics_table.json",
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
