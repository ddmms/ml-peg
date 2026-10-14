"""Run optimised inference speed app."""

from __future__ import annotations

from dash import Dash

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Optimised Inference Speed"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#optimised-inference-speed"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "optimized_inference_speed"


class OptimizedInferenceSpeedApp(BaseApp):
    """Optimised inference speed benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""


def get_app() -> OptimizedInferenceSpeedApp:
    """
    Get optimised inference speed benchmark app layout and callback registration.

    Returns
    -------
    OptimizedInferenceSpeedApp
        Benchmark layout and callback registration.
    """
    return OptimizedInferenceSpeedApp(
        name=BENCHMARK_NAME,
        description=(
            "Speed and memory of developers' optimised MD implementations, run in "
            "Apptainer containers on staged NVE workloads and validated against "
            "ML-PEG's reference calculators."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "optimized_inference_speed_metrics_table.json",
        extra_components=[
            read_plot(
                filename=DATA_PATH / "figure_scaling.json",
                id=f"{BENCHMARK_NAME}-scaling-figure",
            ),
            read_plot(
                filename=DATA_PATH / "figure_memory.json",
                id=f"{BENCHMARK_NAME}-memory-figure",
            ),
        ],
        info_path=DATA_PATH / "info.json",
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8068, debug=True)
