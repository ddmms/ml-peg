"""Run Sellers torsion scan app."""

from __future__ import annotations

from pathlib import Path

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
BENCHMARK_NAME = "Sellers_Tors"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/conformers.html#sellers-tors"
)
DATA_PATH = APP_ROOT / "data" / "conformers" / "Sellers_Tors"
INFO_PATH = DATA_PATH / "info.json"
ASSETS_DIR = "/assets/conformers/Sellers_Tors"

# Table column, and the file key of its per-fragment scatter
METRIC_SCATTERS = {
    "MAE": "mae",
    "RMSE": "rmse",
    "Barrier MAE": "barrier_error",
}


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


class SellersTorsApp(BaseApp):
    """Sellers torsion scan benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Build an MAE, RMSE and barrier error scatter plot per model (fragment
        # index vs that fragment's metric value).
        scatter_cell_to_plot: dict[str, dict[str, Div]] = {}
        curve_dirs: dict[str, Path] = {}

        for model_name in MODELS:
            scatter_paths = {
                column: DATA_PATH / model_name / f"fragment_{key}_scatter.json"
                for column, key in METRIC_SCATTERS.items()
            }
            curve_dir = DATA_PATH / model_name / "torsion_curves"

            if (
                not all(path.exists() for path in scatter_paths.values())
                or not curve_dir.exists()
            ):
                continue

            scatter_cell_to_plot[model_name] = {
                column: read_plot(
                    path,
                    id=f"{BENCHMARK_NAME}-{model_name}-{METRIC_SCATTERS[column]}-figure",
                )
                for column, path in scatter_paths.items()
            }
            curve_dirs[model_name] = curve_dir

        curve_placeholder = f"{BENCHMARK_NAME}-curve-placeholder"
        struct_placeholder = f"{BENCHMARK_NAME}-struct-placeholder"

        # Clicking a model's metric cell shows that model's corresponding
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

        # Clicking a point on any scatter shows that fragment's curve. The curve
        # doesn't depend on which metric was clicked through, so all scatters for
        # a model share the same curve_dir.
        trajectories = _trajectory_paths()

        for model_name, curve_dir in curve_dirs.items():
            for metric in METRIC_SCATTERS.values():
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


def get_app() -> SellersTorsApp:
    """
    Get Sellers torsion scan benchmark app layout and callback registration.

    Returns
    -------
    SellersTorsApp
        Benchmark layout and callback registration.
    """
    return SellersTorsApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting torsional energy profiles of 62 neutral "
            "drug-like fragments (alkyl, aryl, biaryl and aryl amide torsions) "
            "with 36 dihedral angles each. "
            "Reference data from CCSD(T)/CBS calculations on MP2/6-311+G**-optimised "
            "constrained geometries."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "sellers_tors_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-scatter-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-curve-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
