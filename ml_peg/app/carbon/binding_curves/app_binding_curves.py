"""Run carbon binding-curves app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.analysis.carbon.curve_metrics import SHAPE_METRICS
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

CATEGORY = "carbon"
BENCHMARK_NAME = "Carbon binding curves"

MODELS = get_model_names(current_models)

METRIC_COLUMNS = list(SHAPE_METRICS) + ["Min distance error", "Min energy error"]

DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/carbon.html#binding-curves"
)

DATA_PATH = APP_ROOT / "data" / CATEGORY / "binding_curves"
INFO_PATH = DATA_PATH / "info.json"


class BindingCurvesApp(BaseApp):
    """Carbon binding-curves benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callback to show a model's binding curves on cell click."""
        # Every metric column for a model shows that model's overlaid
        # binding-curve figure (model curves plus the PBE+D2 reference).
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
            "Energy vs C–C nearest-neighbour distance for six carbon structures "
            "(dimer, graphene, diamond, simple cubic, BCC, FCC). Metrics cover "
            "curve-shape physicality (force flips, energy minima and inflections, "
            "repulsive/attractive monotonicity) and the error in the location and "
            "depth of the energy minimum vs a PBE+D2 reference."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "binding_curves_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)
    app = get_app()
    full_app.layout = app.layout
    app.register_callbacks()
    full_app.run(port=8052, debug=True)
