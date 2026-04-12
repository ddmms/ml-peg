"""Run CMRAds200 app."""

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

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "CMRAds200"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#cmrads200"
)
DATA_PATH = APP_ROOT / "data" / "surfaces" / "CMRAds200"


class CMRAds200App(BaseApp):
    """CMRAds200 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_adsorption_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        structs_dir = DATA_PATH / MODELS[0]

        # Assets dir will be parent directory
        structs = [
            f"/assets/surfaces/CMRAds200/{MODELS[0]}/{struct_file.stem}.xyz"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="traj",
        )


def get_app() -> CMRAds200App:
    """
    Get CMRAds200 benchmark app layout and callback registration.

    Returns
    -------
    CMRAds200App
        Benchmark layout and callback registration.
    """
    return CMRAds200App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting adsorption energies for 200 "
            "adsorbate-surface systems."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "cmrads_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    cmrads200_app = get_app()
    full_app.layout = cmrads200_app.layout
    cmrads200_app.register_callbacks()

    # Run app
    full_app.run(port=8056, debug=True)
