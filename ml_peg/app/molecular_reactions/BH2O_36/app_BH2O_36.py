"""Run BH2O-36 barriers app."""

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
BENCHMARK_NAME = "BH2O-36"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular.html#bh2o-36"
DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / "BH2O_36"


class BH2O36App(BaseApp):
    """BH2O-36 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_bh2o_36_barriers.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        model_dir = DATA_PATH / MODELS[0]
        if model_dir.exists():
            # Get system names from trajectory files
            labels = sorted(
                {
                    f.stem.replace("_rct_to_ts", "")
                    for f in model_dir.glob("*_rct_to_ts.xyz")
                }
            )
            asset_prefix = f"assets/molecular_reactions/BH2O_36/{MODELS[0]}/"
            # Each system has 2 data points:
            # TS-Reactants (rct->TS), TS-Products (pro->TS)
            structs = [
                path
                for label in labels
                for path in [
                    f"{asset_prefix}{label}_rct_to_ts.xyz",
                    f"{asset_prefix}{label}_pro_to_ts.xyz",
                ]
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
            mode="traj",  # Use trajectory mode to cycle between relevant structures
        )


def get_app() -> BH2O36App:
    """
    Get BH2O-36 benchmark app layout and callback registration.

    Returns
    -------
    BH2O36App
        Benchmark layout and callback registration.
    """
    return BH2O36App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting hydrolysis reaction barriers for the "
            "BH2O-36 dataset (36 aqueous bimolecular reactions). Reference data "
            "from CCSD(T) calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "bh2o_36_barriers_metrics_table.json",
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
    full_app.run(port=8070, debug=True)
