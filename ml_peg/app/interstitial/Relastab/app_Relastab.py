"""Run Relastab app."""

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
BENCHMARK_NAME = "Relastab Relative Stability"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/interstitial.html#relastab"
)
DATA_PATH = APP_ROOT / "data" / "interstitial" / "Relastab"


class RelastabApp(BaseApp):
    """Relastab benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_energy.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Assuming first model has structure files
        structs_dir = DATA_PATH / MODELS[0]
        # Sort glob to match analysis order
        structs = [
            f"assets/interstitial/Relastab/{MODELS[0]}/{struct_file.name}"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"KendallTau": scatter, "Spearman": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> RelastabApp:
    """
    Get Relastab benchmark app.

    Returns
    -------
    RelastabApp
        Benchmark layout and callback registration.
    """
    return RelastabApp(
        name=BENCHMARK_NAME,
        description="Relative stability ranking of interstitial Fe configurations.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "relastab_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    rel_app = get_app()
    full_app.layout = rel_app.layout
    rel_app.register_callbacks()

    # Run app
    full_app.run(port=8055, debug=True)
