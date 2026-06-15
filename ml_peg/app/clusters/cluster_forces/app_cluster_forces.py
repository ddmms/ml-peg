"""Run app for the cluster-force benchmark."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.app.utils.load import read_density_plot_for_model
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

BENCHMARK_NAME = "Cluster Forces"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/clusters.html#cluster-forces"
DATA_PATH = APP_ROOT / "data" / "clusters" / "cluster_forces"
CLUSTER_SIZES = (3, 4, 5, 6, 7, 8)
REFERENCES = (
    ("mad2", "MAD2"),
    ("omol25", "OMOL25"),
)


def _metric_name(reference_label: str, cluster_size: int) -> str:
    """
    Return the metric name for a reference and cluster size.

    Parameters
    ----------
    reference_label
        Reference force-set label.
    cluster_size
        Number of atoms in the cluster.

    Returns
    -------
    str
        Metric label.
    """
    return f"{reference_label} Force MAE ({cluster_size} atoms)"


class ClusterForcesApp(BaseApp):
    """Cluster-force benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register force parity and structure callbacks."""
        density_plots: dict[str, dict] = {}
        for model in MODELS:
            model_plots = {}
            for reference_key, reference_label in REFERENCES:
                for cluster_size in CLUSTER_SIZES:
                    metric_name = _metric_name(reference_label, cluster_size)
                    figure_path = (
                        DATA_PATH
                        / (
                            f"figure_force_parity_{reference_key}_"
                            f"{cluster_size}atoms.json"
                        )
                    )
                    if not figure_path.exists():
                        continue
                    density_graph = read_density_plot_for_model(
                        filename=figure_path,
                        model=model,
                        id=(
                            f"{BENCHMARK_NAME}-{model}-{reference_key}-"
                            f"{cluster_size}atoms-density"
                        ),
                    )
                    if density_graph is not None:
                        model_plots[metric_name] = density_graph
            if model_plots:
                density_plots[model] = model_plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=density_plots,
        )

        for model in MODELS:
            for reference_key, _ in REFERENCES:
                for cluster_size in CLUSTER_SIZES:
                    traj_dir = (
                        DATA_PATH
                        / model
                        / "density_traj"
                        / f"{reference_key}_{cluster_size}atoms"
                    )
                    if not traj_dir.exists():
                        continue
                    traj_files = sorted(
                        traj_dir.glob("*.extxyz"), key=lambda path: int(path.stem)
                    )
                    struct_from_scatter(
                        scatter_id=(
                            f"{BENCHMARK_NAME}-{model}-{reference_key}-"
                            f"{cluster_size}atoms-density"
                        ),
                        struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                        structs=[
                            (
                                f"/assets/clusters/cluster_forces/{model}/"
                                f"density_traj/{reference_key}_{cluster_size}atoms/"
                                f"{path.name}"
                            )
                            for path in traj_files
                        ],
                        mode="traj",
                    )


def get_app() -> ClusterForcesApp:
    """
    Get cluster-force benchmark app.

    Returns
    -------
    ClusterForcesApp
        App instance.
    """
    return ClusterForcesApp(
        name=BENCHMARK_NAME,
        description=(
            "Component-wise force MAE for randomly generated neutral 3- to 8-atom "
            "clusters, split by reference force set and cluster size."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "cluster_forces_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8061, debug=True)
