"""Run NCIA_D442x10 app."""

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
BENCHMARK_NAME = "NCIA_D442x10"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/non_covalent_interactions.html#ncia-d442x10"
DATA_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "ncia_d442x10"


class NCIAD442x10App(BaseApp):
    """NCIA_D442x10 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_ncia_d442x10.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Get list of structure labels from first model directory
        model_dir = DATA_PATH / MODELS[0]
        if model_dir.exists():
            labels = sorted([f.stem for f in model_dir.glob("*.xyz")])
            structs = [
                f"assets/non_covalent_interactions/ncia_d442x10/{MODELS[0]}/{label}.xyz"
                for label in labels
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


def get_app() -> NCIAD442x10App:
    """
    Get NCIA_D442x10 benchmark app layout and callback registration.

    Returns
    -------
    NCIAD442x10App
        Benchmark layout and callback registration.
    """
    return NCIAD442x10App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting London dispersion interaction energies "
            "for the NCIA D442x10 dataset (442 systems, 10 points per curve). "
            "Reference data from CCSD(T) calculations. Noble gases are excluded."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "ncia_d442x10_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    ncia_d442x10_app = get_app()
    full_app.layout = ncia_d442x10_app.layout
    ncia_d442x10_app.register_callbacks()

    # Run app
    full_app.run(port=8055, debug=True)
