"""Run carbon binding-curves app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.analysis.carbon.curve_metrics import SHAPE_METRICS
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
    struct_from_multi_scatters,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

CATEGORY = "carbon"
BENCHMARK_NAME = "Carbon binding curves"

MODELS = get_model_names(current_models)

METRIC_COLUMNS = list(SHAPE_METRICS) + ["Min distance error", "Min energy error"]

STRUCTURE_NAMES = ("dimer", "graphene", "diamond", "sc", "bcc", "fcc")

DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/carbon.html#binding-curves"
)

DATA_PATH = APP_ROOT / "data" / CATEGORY / "binding_curves"
INFO_PATH = DATA_PATH / "info.json"


class BindingCurvesApp(BaseApp):
    """Carbon binding-curves benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks for binding-curve figures and WEAS structure viewer."""
        cell_to_plot = {
            model: {
                column: read_plot(
                    DATA_PATH / model / "figure_binding_curves.json",
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

        assets_base = "/assets/carbon/binding_curves"
        for model in MODELS:
            # Traces 0-5: model curves; traces 6-11: reference curves (no structs).
            structs = [f"{assets_base}/{model}/{s}.extxyz" for s in STRUCTURE_NAMES] + [
                None
            ] * len(STRUCTURE_NAMES)

            for column in METRIC_COLUMNS:
                struct_from_multi_scatters(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{column}-figure",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=structs,
                    mode="traj",
                )


def get_app() -> BindingCurvesApp:
    """
    Get carbon binding-curves benchmark app layout and callback registration.

    Returns
    -------
    BindingCurvesApp
        Benchmark layout and callback registration.
    """
    return BindingCurvesApp(
        name=BENCHMARK_NAME,
        description=(
            "Energy vs nearest-neighbour distance for six carbon structures "
            "(dimer, graphene, diamond, simple cubic, BCC, FCC), compared against "
            "PBE+D2 reference curves. Tests whether a model predicts the correct "
            "equilibrium bond length and binding energy across a range of bonding "
            "environments."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "binding_curves_metrics_table.json",
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
    full_app.run(port=8052, debug=True)
