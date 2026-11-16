"""Run TorsionNet500 app."""

from __future__ import annotations

from pathlib import Path

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    clear_from_table_cell,
    plot_from_scatter_files,
    plot_from_table_cell,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
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


def _trajectory_paths() -> dict[str, str]:
    """
    Get the torsion scan trajectory asset path for each fragment.

    Returns
    -------
    dict[str, str]
        Asset path of each fragment's trajectory, keyed by fragment label.
        Geometries are identical across models, so these are shared.
    """
    traj_dir = DATA_PATH / "torsion_trajectories"
    return {
        path.stem: f"{ASSETS_DIR}/torsion_trajectories/{path.name}"
        for path in sorted(traj_dir.glob("*.xyz"))
    }


class TorsionNet500App(BaseApp):
    """TorsionNet500 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Build an RMSE and an MAE scatter plot per model (fragment index vs
        # that fragment's metric value).
        scatter_cell_to_plot: dict[str, dict[str, Div]] = {}
        curve_dirs: dict[str, Path] = {}

        for model_name in MODELS:
            rmse_scatter_path = DATA_PATH / model_name / "fragment_rmse_scatter.json"
            mae_scatter_path = DATA_PATH / model_name / "fragment_mae_scatter.json"
            curve_dir = DATA_PATH / model_name / "torsion_curves"

            if (
                not rmse_scatter_path.exists()
                or not mae_scatter_path.exists()
                or not curve_dir.exists()
            ):
                continue

            scatter_cell_to_plot[model_name] = {
                "RMSE": read_plot(
                    rmse_scatter_path, id=f"{BENCHMARK_NAME}-{model_name}-rmse-figure"
                ),
                "MAE": read_plot(
                    mae_scatter_path, id=f"{BENCHMARK_NAME}-{model_name}-mae-figure"
                ),
            }
            curve_dirs[model_name] = curve_dir

        curve_placeholder = f"{BENCHMARK_NAME}-curve-placeholder"
        struct_placeholder = f"{BENCHMARK_NAME}-struct-placeholder"

        # Clicking a model's RMSE or MAE cell shows that model's corresponding
        # fragment scatter, and clears the previous model's curve and structure.
        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-scatter-placeholder",
            cell_to_plot=scatter_cell_to_plot,
        )
        clear_from_table_cell(
            table_id=self.table_id,
            plot_ids=[curve_placeholder, struct_placeholder],
        )

        # Clicking a point on either scatter shows that fragment's curve. The
        # curve doesn't depend on which metric was clicked through, so both
        # scatters for a model share the same curve_dir.
        trajectories = _trajectory_paths()

        for model_name, curve_dir in curve_dirs.items():
            for metric in ("rmse", "mae"):
                # Showing a new curve clears the previous curve's structure.
                curve_id = plot_from_scatter_files(
                    scatter_id=f"{BENCHMARK_NAME}-{model_name}-{metric}-figure",
                    plot_id=curve_placeholder,
                    plot_dir=curve_dir,
                    clear_plot_ids=[struct_placeholder],
                )

                # Curve points and trajectory frames are both ordered by angle,
                # so clicking a point shows the structure at that angle.
                struct_from_scatter(
                    scatter_id=curve_id,
                    struct_id=struct_placeholder,
                    structs=trajectories,
                    mode="traj",
                    follow_frames=True,
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
        description=(
            "Accuracy of predicted torsional energy profiles for 500 molecular "
            "fragments, benchmarked against CCSD(T)/CBS reference scans."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "torsionnet500_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-scatter-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-curve-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    torsionnet500_app = get_app()
    full_app.layout = torsionnet500_app.layout
    torsionnet500_app.register_callbacks()

    # Run app
    full_app.run(port=8072, debug=True)
