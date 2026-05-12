"""Run app for the cluster-force benchmark."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Cluster Forces"
DATA_PATH = APP_ROOT / "data" / "clusters" / "cluster_forces"
CLUSTER_SIZES = (3, 4, 5, 6, 7, 8)


def _metric_name(cluster_size: int) -> str:
    """
    Return the metric name for a cluster size.

    Parameters
    ----------
    cluster_size
        Number of atoms in the cluster.

    Returns
    -------
    str
        Metric label.
    """
    return f"Force MAE ({cluster_size} atoms)"


class ClusterForcesApp(BaseApp):
    """Cluster-force benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register force parity callbacks."""
        column_to_plot = {
            _metric_name(cluster_size): read_plot(
                DATA_PATH / f"figure_force_parity_{cluster_size}mer.json",
                id=f"{BENCHMARK_NAME}-{cluster_size}mer-force-parity-figure",
            )
            for cluster_size in CLUSTER_SIZES
        }
        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot=column_to_plot,
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
            "Component-wise force MAE for neutral small clusters, split by cluster "
            "size. Models are routed to MAD2 or OMOL25 reference forces according "
            "to their training domain."
        ),
        table_path=DATA_PATH / "cluster_forces_metrics_table.json",
        extra_components=[Div(id=f"{BENCHMARK_NAME}-figure-placeholder")],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8061, debug=True)
