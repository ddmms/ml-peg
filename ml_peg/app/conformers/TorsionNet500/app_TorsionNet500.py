"""Run TorsionNet500 dihedral scan benchmark app."""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, State, callback
from dash.dcc import Store
from dash.html import Div, Iframe

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
    plot_with_download_controls,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.weas import generate_weas_html
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "TorsionNet500"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/conformers.html#torsionnet500"
)
DATA_PATH = APP_ROOT / "data" / "conformers" / "TorsionNet500"
INFO_PATH = DATA_PATH / "info.json"
ASSETS_DIR = "/assets/conformers/TorsionNet500"


def _register_curve_callback(
    scatter_id: str,
    plot_id: str,
    curve_dir: Path,
    labels: list[str],
    struct_plot_id: str,
) -> None:
    """
    Attach callbacks that show a torsion curve and structure on point clicks.

    Unlike `plot_from_scatter`, this reads the clicked fragment's curve JSON from
    disk on demand instead of requiring every curve to be pre-loaded into a
    `plots_list` up front - with up to 500 fragments per model, pre-loading all of
    them for every model would be needlessly expensive when only one is ever shown
    at a time.

    A second callback shows the 3D structure at the torsion angle of whichever
    point on the curve is clicked, using WEAS trajectory mode. The curve's
    fragment label is tracked in a `Store` so the structure callback (triggered by
    clicks on the curve, whose id is fixed and reused across fragments) knows which
    fragment's trajectory file to load.

    Parameters
    ----------
    scatter_id
        ID of the per-model fragment-barrier scatter plot.
    plot_id
        ID of the shared placeholder Div where curves are rendered.
    curve_dir
        Directory containing this model's per-fragment torsion curve JSON files.
    labels
        Fragment labels, in the same order as the scatter's points.
    struct_plot_id
        ID of the shared placeholder Div where structures are rendered.
    """
    label_store_id = f"{scatter_id}-curve-label"

    @callback(
        Output(plot_id, "children", allow_duplicate=True),
        Output(label_store_id, "data", allow_duplicate=True),
        Output(struct_plot_id, "children", allow_duplicate=True),
        Input(scatter_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_curve(click_data):
        """
        Register callback to show a torsion curve when a scatter point is clicked.

        Also clears any structure shown from a previously displayed curve, so a
        stale structure from a different fragment/model doesn't linger on screen
        until the new curve is itself clicked.

        Parameters
        ----------
        click_data
            Clicked data point in the fragment-barrier scatter plot.

        Returns
        -------
        tuple[Div, str | None, Div]
            Torsion curve plot on scatter click, the clicked fragment's label, and
            an empty Div to clear any previously displayed structure.
        """
        if not click_data:
            return Div(), None, Div()
        point = click_data["points"][0]
        # The y = x reference line trace carries no customdata; ignore clicks on
        # it so they don't map to an unrelated fragment.
        if point.get("customdata") is None:
            return Div(), None, Div()
        idx = point["pointNumber"]
        if idx < 0 or idx >= len(labels):
            return Div(), None, Div()
        label = labels[idx]
        curve_path = curve_dir / f"{label}.json"
        graph = read_plot(curve_path, id=f"{scatter_id}-curve")
        return plot_with_download_controls(graph), label, Div()

    @callback(
        Output(struct_plot_id, "children", allow_duplicate=True),
        Input(f"{scatter_id}-curve", "clickData"),
        State(label_store_id, "data"),
        prevent_initial_call="initial_duplicate",
    )
    def show_struct(click_data, label):
        """
        Register callback to show a structure when a point on the curve is clicked.

        Parameters
        ----------
        click_data
            Clicked data point in the torsion curve plot.
        label
            Fragment label of the currently displayed curve.

        Returns
        -------
        Div
            Structure at the clicked dihedral angle, in WEAS trajectory mode.
        """
        if not click_data or not label:
            return Div()
        idx = click_data["points"][0]["pointNumber"]
        traj_path = f"{ASSETS_DIR}/torsion_trajectories/{label}.xyz"
        return Div(
            Iframe(
                srcDoc=generate_weas_html(traj_path, mode="traj", index=idx),
                style={
                    "height": "550px",
                    "width": "100%",
                    "border": "1px solid #ddd",
                    "borderRadius": "5px",
                },
            )
        )


class TorsionNet500App(BaseApp):
    """TorsionNet500 dihedral scan benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Build a per-fragment barrier-error scatter per model (fragment index vs
        # that fragment's barrier height error), and record fragment labels in
        # scatter-point order so a later click on a point can be mapped back to
        # its curve file.
        scatter_cell_to_plot: dict[str, dict[str, Div]] = {}
        fragment_labels: dict[str, list[str]] = {}
        for model_name in MODELS:
            barrier_parity_path = (
                DATA_PATH / model_name / "fragment_barrier_parity.json"
            )
            curve_dir = DATA_PATH / model_name / "torsion_curves"
            if not barrier_parity_path.exists() or not curve_dir.exists():
                continue
            scatter_cell_to_plot[model_name] = {
                "Barrier Height MAE": read_plot(
                    barrier_parity_path,
                    id=f"{BENCHMARK_NAME}-{model_name}-barrier-figure",
                ),
            }
            fragment_labels[model_name] = [
                curve_file.stem for curve_file in sorted(curve_dir.glob("*.json"))
            ]

        # Clicking a model's Barrier Height MAE cell shows that model's
        # per-fragment barrier-error scatter.
        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-scatter-placeholder",
            cell_to_plot=scatter_cell_to_plot,
        )

        # Clicking a point on the scatter shows that fragment's torsion curve.
        for model_name, labels in fragment_labels.items():
            curve_dir = DATA_PATH / model_name / "torsion_curves"
            _register_curve_callback(
                scatter_id=f"{BENCHMARK_NAME}-{model_name}-barrier-figure",
                plot_id=f"{BENCHMARK_NAME}-curve-placeholder",
                curve_dir=curve_dir,
                labels=labels,
                struct_plot_id=f"{BENCHMARK_NAME}-struct-placeholder",
            )


def get_app() -> TorsionNet500App:
    """
    Get TorsionNet500 benchmark app layout and callback registration.

    Returns
    -------
    TorsionNet500App
        Benchmark layout and callback registration.
    """
    return TorsionNet500App(
        name=BENCHMARK_NAME,
        framework_ids="mlip_audit",
        description=(
            "Performance in predicting torsion energy barriers for drug-like "
            "molecules from systematic dihedral scans. Reference data from "
            "wB97M-D3(BJ)/def2-TZVPPD calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "torsionnet500_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-scatter-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-curve-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
            # One Store per model, remembering which fragment's curve is currently
            # displayed, so a click on the curve (whose id is fixed and reused
            # across fragments) can be mapped back to the right trajectory file.
            *[
                Store(id=f"{BENCHMARK_NAME}-{model_name}-barrier-figure-curve-label")
                for model_name in MODELS
            ],
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8069, debug=True)
