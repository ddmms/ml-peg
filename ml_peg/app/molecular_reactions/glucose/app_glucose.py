"""Run glucose app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Glucose"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_reactions.html"
)
DATA_PATH = APP_ROOT / "data" / "molecular_reactions" / "glucose"
INFO_PATH = DATA_PATH / "info.json"


class GlucoseApp(BaseApp):
    """Glucose benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_glucose_barriers.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Find structures from the first available model.
        structs = []

        for model_name in MODELS:
            model_dir = DATA_PATH / model_name

            if model_dir.exists():
                base_labels = sorted(
                    f.stem.rsplit("_", 1)[0] for f in model_dir.glob("*_rct.xyz")
                )

                structs = [
                    f"/assets/molecular_reactions/glucose/{model_name}/{label}_rct.xyz"
                    for label in base_labels
                ]
                break

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


def get_app() -> GlucoseApp:
    """
    Get glucose benchmark app layout and callback registration.

    Returns
    -------
    GlucoseApp
        Benchmark layout and callback registration.
    """
    return GlucoseApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting reaction barriers "
            "for the glucose reaction benchmark."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "glucose_barriers_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
        framework_ids="mace-polar-1",
    )
