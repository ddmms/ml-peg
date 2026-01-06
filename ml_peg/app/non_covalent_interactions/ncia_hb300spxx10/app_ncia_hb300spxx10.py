"""Run NCIA_HB300SPXx10 app."""

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
BENCHMARK_NAME = "NCIA_HB300SPXx10"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "non_covalent_interactions.html#ncia-hb300spxx10"
)
DATA_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "ncia_hb300spxx10"


class NCIANHB300SPXx10App(BaseApp):
    """NCIA_HB300SPXx10 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_ncia_hb300spxx10.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        model_dir = DATA_PATH / MODELS[0]
        if model_dir.exists():
            labels = sorted([f.stem for f in model_dir.glob("*.xyz")])
            structs = [
                (
                    "assets/non_covalent_interactions/ncia_hb300spxx10/"
                    f"{MODELS[0]}/{label}.xyz"
                )
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


def get_app() -> NCIANHB300SPXx10App:
    """
    Get NCIA_HB300SPXx10 benchmark app layout and callback registration.

    Returns
    -------
    NCIANHB300SPXx10App
        Benchmark layout and callback registration.
    """
    return NCIANHB300SPXx10App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting ionic hydrogen bond interaction energies "
            "for the NCIA HB300SPXx10 dataset (ionic dimers from HB300). "
            "Reference data from CCSD(T) calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "ncia_hb300spxx10_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()

    # Run app
    full_app.run(port=8057, debug=True)
