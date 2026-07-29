"""Run Jacobian symmetry app."""

from __future__ import annotations

from dash import Dash, Input, Output, callback
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
    plot_with_download_controls,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Jacobian Symmetry"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#jacobian-symmetry"
DATA_PATH = APP_ROOT / "data" / "physicality" / "jacobian_symmetry"
INFO_PATH = DATA_PATH / "info.json"


# TODO: if another benchmark needs this same click-a-bar-to-show-a-figure
# pattern, move this into ml_peg/app/utils/build_callbacks.py alongside
# plot_from_scatter / struct_from_scatter instead of duplicating it here.
def _register_bar_heatmap(bar_id: str, model: str, plot_id: str) -> None:
    """
    Show a structure's antisymmetric-Jacobian heatmap when its bar is clicked.

    The heatmap JSON is read from disk on demand, keyed by the clicked bar's
    structure name, rather than pre-loading all ten heatmaps per model up front.

    Parameters
    ----------
    bar_id
        ID for the model's Dash bar chart being clicked.
    model
        Name of the model whose bar chart this is.
    plot_id
        ID for the Dash placeholder Div where the heatmap is rendered.
    """

    @callback(
        Output(plot_id, "children", allow_duplicate=True),
        Input(bar_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_heatmap(click_data) -> Div:
        """
        Show the heatmap for the clicked bar's structure.

        Parameters
        ----------
        click_data
            Clicked bar in the bar chart.

        Returns
        -------
        Div
            Heatmap on bar click, or an empty div.
        """
        if not click_data:
            return Div()
        struct = click_data["points"][0]["x"]
        return plot_with_download_controls(
            read_plot(
                DATA_PATH / f"figure_{model}_{struct}_janti.json",
                id=f"{BENCHMARK_NAME}-{model}-{struct}-heatmap",
            )
        )


def _register_clear_heatmap(table_id: str, plot_id: str) -> None:
    """
    Clear the heatmap when a different model's table cell is selected.

    Stops a stale heatmap from a previous model lingering after switching.

    Parameters
    ----------
    table_id
        ID for the Dash table being clicked.
    plot_id
        ID for the heatmap placeholder Div to clear.
    """

    @callback(
        Output(plot_id, "children", allow_duplicate=True),
        Input(table_id, "active_cell"),
        prevent_initial_call=True,
    )
    def clear_heatmap(active_cell) -> Div:
        """
        Clear the heatmap placeholder on table cell selection.

        Parameters
        ----------
        active_cell
            Clicked cell in the Dash table.

        Returns
        -------
        Div
            Empty div, clearing the placeholder.
        """
        return Div()


class JacobianSymmetryApp(BaseApp):
    """Jacobian symmetry benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Clicking either metric cell for a model shows that model's bar chart
        # of lambda for each of the ten structures.
        bars = {
            model: {
                "mean lambda": read_plot(
                    DATA_PATH / f"figure_{model}_lambda_by_structure.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                ),
                "max lambda": read_plot(
                    DATA_PATH / f"figure_{model}_lambda_by_structure.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure",
                ),
            }
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=bars,
        )

        # Clicking a structure's bar shows that structure's antisymmetric-
        # Jacobian heatmap; selecting a new model's cell clears the old one.
        for model in MODELS:
            _register_bar_heatmap(
                bar_id=f"{BENCHMARK_NAME}-{model}-figure",
                model=model,
                plot_id=f"{BENCHMARK_NAME}-heatmap-placeholder",
            )
        _register_clear_heatmap(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-heatmap-placeholder",
        )


def get_app() -> JacobianSymmetryApp:
    """
    Get Jacobian symmetry benchmark app layout and callback registration.

    Returns
    -------
    JacobianSymmetryApp
        Benchmark layout and callback registration.
    """
    return JacobianSymmetryApp(
        name=BENCHMARK_NAME,
        description=(
            "Fraction of the force Jacobian's antisymmetric component, "
            "across 10 diverse structures."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "jacobian_symmetry_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-heatmap-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(
        __name__, assets_folder=DATA_PATH.parent, suppress_callback_exceptions=True
    )

    jacobian_symmetry_app = get_app()
    full_app.layout = jacobian_symmetry_app.layout
    jacobian_symmetry_app.register_callbacks()

    full_app.run(port=8073, debug=True)
