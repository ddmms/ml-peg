"""Run IONPI19 app."""

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
BENCHMARK_NAME = "IONPI19"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/non_covalent_interactions.html#ionpi19"
DATA_PATH = APP_ROOT / "data" / "non_covalent_interactions" / "ionpi19"


class IONPI19App(BaseApp):
    """IONPI19 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_ionpi19.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Build list of structure files to display
        # Systems 1-17 have AB complexes, systems 18-19 only have fragments
        structs = []
        for i in range(1, 18):
            # Systems 1-17: show the complex (AB)
            structs.append(
                f"assets/non_covalent_interactions/ionpi19/{MODELS[0]}/{i}_AB.xyz"
            )
        # Systems 18-19: show fragment A (no complex available)
        structs.append(f"assets/non_covalent_interactions/ionpi19/{MODELS[0]}/18_A.xyz")
        structs.append(f"assets/non_covalent_interactions/ionpi19/{MODELS[0]}/19_A.xyz")

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter, "RMSE": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> IONPI19App:
    """
    Get IONPI19 benchmark app layout and callback registration.

    Returns
    -------
    IONPI19App
        Benchmark layout and callback registration.
    """
    return IONPI19App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting ion-pi interaction energies for 19 systems. "
            "Reference data from CCSD(T) calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "ionpi19_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    ionpi19_app = get_app()
    full_app.layout = ionpi19_app.layout
    ionpi19_app.register_callbacks()

    full_app.run(port=8054, debug=True)
