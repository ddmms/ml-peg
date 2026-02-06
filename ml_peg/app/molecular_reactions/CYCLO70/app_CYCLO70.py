"""Run CYCLO70 app."""

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
BENCHMARK_NAME = "CYCLO70"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_reactions.html#cyclo70"
DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / "CYCLO70"


class CYCLO70App(BaseApp):
    """CYCLO70 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_cyclo70_barriers.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        model_dir = DATA_PATH / MODELS[0]
        if model_dir.exists():
            labels = sorted([f.stem for f in model_dir.glob("*_forward.xyz")])
            structs = [
                f"assets/molecular_reactions/CYCLO70/{MODELS[0]}/{label}_{dir}.xyz"
                for label in labels
                for dir in ("forward", "reverse")
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


def get_app() -> CYCLO70App:
    """
    Get CYCLO70 benchmark app layout and callback registration.

    Returns
    -------
    CYCLO70App
        Benchmark layout and callback registration.
    """
    return CYCLO70App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting barrier heights for the "
            "CYCLO70 pericyclic reactions benchmark. "
            "Reference data from CCSD(T) calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "cyclo70_barriers_metrics_table.json",
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
    full_app.run(port=8068, debug=True)
