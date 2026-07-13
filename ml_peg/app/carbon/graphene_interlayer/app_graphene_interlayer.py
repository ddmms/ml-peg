"""Run bilayer-graphene interlayer app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.analysis.carbon.curve_metrics import SHAPE_METRICS
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

CATEGORY = "carbon"
BENCHMARK_NAME = "Graphene interlayer"

MODELS = get_model_names(current_models)

METRIC_COLUMNS = list(SHAPE_METRICS) + ["Min distance error", "Min energy error"]

DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/carbon.html"
    "#graphene-interlayer"
)

DATA_PATH = APP_ROOT / "data" / CATEGORY / "graphene_interlayer"
INFO_PATH = DATA_PATH / "info.json"


class GrapheneInterlayerApp(BaseApp):
    """Bilayer-graphene interlayer benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks for interlayer curve and WEAS structure viewer."""
        cell_to_plot = {
            model: {
                column: read_plot(
                    DATA_PATH / model / "figure_interlayer.json",
                    id=f"{BENCHMARK_NAME}-{model}-{column}-figure",
                )
                for column in METRIC_COLUMNS
            }
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_to_plot,
        )

        for model in MODELS:
            traj_path = f"/assets/carbon/graphene_interlayer/{model}/interlayer.extxyz"
            for column in METRIC_COLUMNS:
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{column}-figure",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=traj_path,
                    mode="traj",
                )


def get_app() -> GrapheneInterlayerApp:
    """
    Get bilayer-graphene interlayer benchmark app layout and callback registration.

    Returns
    -------
    GrapheneInterlayerApp
        Benchmark layout and callback registration.
    """
    return GrapheneInterlayerApp(
        name=BENCHMARK_NAME,
        description=(
            "Energy of bilayer graphene as a function of interlayer separation, "
            "compared against a PBE+D2 reference curve. Tests how well a model "
            "captures long-range dispersion interactions."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "graphene_interlayer_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)
    app = get_app()
    full_app.layout = app.layout
    app.register_callbacks()
    full_app.run(port=8053, debug=True)
