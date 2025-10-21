"""Run GMTKN55 app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "GMTKN55"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular.html#gmtkn55"
DATA_PATH = APP_ROOT / "data" / "molecular" / "GMTKN55"


class GMTKN55App(BaseApp):
    """GMTKN55 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_rel_energies.json", id=f"{BENCHMARK_NAME}-figure"
        )

        # structs_dir = DATA_PATH / MODELS[0]

        # # Assets dir will be parent directory
        # structs = list(
        #     np.repeat(
        #         [
        #             f"assets/surfaces/GMTKN55/{MODELS[0]}/{i}.xyz"
        #             for i in range(len(list(structs_dir.glob("*.xyz"))))
        #         ],
        #         3,
        #     )
        # )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Relative Energy Error": scatter, "Ranking Error": scatter},
        )

        # struct_from_scatter(
        #     scatter_id=f"{BENCHMARK_NAME}-figure",
        #     struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
        #     structs=structs,
        #     mode="traj",
        # )


def get_app() -> GMTKN55App:
    """
    Get GMTNK55 benchmark app layout and callback registration.

    Returns
    -------
    GMTNK55App
        Benchmark layout and callback registration.
    """
    return GMTKN55App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting relative energies for 55 subsets of molecules, "
            "inclding intramolecular non-covalent interactions (NCIs), intermolecular "
            "NCIs, small systems, large systems and barrier heights."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "gmtkn55_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            # Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    gmtkn55_app = get_app()
    full_app.layout = gmtkn55_app.layout
    gmtkn55_app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
