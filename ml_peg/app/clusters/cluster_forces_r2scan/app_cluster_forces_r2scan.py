"""Run app for the r2SCAN cluster force benchmark."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.utils.cluster_forces import ClusterForcesApp

BENCHMARK_NAME = "Cluster Forces r2SCAN"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/clusters.html#cluster-forces-r2scan"
DATA_PATH = APP_ROOT / "data" / "clusters" / "cluster_forces_r2scan"


def get_app() -> ClusterForcesApp:
    """
    Get the r2SCAN cluster force benchmark app.

    Returns
    -------
    ClusterForcesApp
        Benchmark layout and callback registration.
    """
    return ClusterForcesApp(
        name=BENCHMARK_NAME,
        description=(
            "Component-wise force MAE for randomly generated neutral 3- to 8-atom "
            "clusters, compared to r2SCAN reference forces. Models are evaluated "
            "without added dispersion corrections."
        ),
        docs_url=DOCS_URL,
        data_path=DATA_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8061, debug=True)
