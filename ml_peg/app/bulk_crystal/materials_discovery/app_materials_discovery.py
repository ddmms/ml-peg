"""Run the materials-discovery benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_density_plot_for_model
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

BENCHMARK_NAME = "Materials discovery"
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "materials_discovery"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "bulk_crystal.html#materials-discovery-evaluation"
)
MODELS = get_model_names(current_models)

PLOT_CONFIGS = {
    "figure_formation_energy_density.json": ("Full MAE", "Unique MAE", "10k MAE"),
    "figure_hull_distance_density.json": (
        "Full F1",
        "Full DAF",
        "Unique F1",
        "Unique DAF",
        "10k F1",
        "10k DAF",
    ),
}


class MaterialsDiscoveryApp(BaseApp):
    """Matbench Discovery benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register benchmark-specific callbacks."""
        cell_to_plot: dict[str, dict] = {}
        for model in MODELS:
            model_plots = {}
            for filename, metric_names in PLOT_CONFIGS.items():
                plot_path = DATA_PATH / filename
                if not plot_path.is_file():
                    continue
                graph = read_density_plot_for_model(
                    filename=plot_path,
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-{plot_path.stem}",
                )
                if graph is not None:
                    model_plots.update(dict.fromkeys(metric_names, graph))
            if model_plots:
                cell_to_plot[model] = model_plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_to_plot,
        )


def get_app() -> MaterialsDiscoveryApp:
    """
    Return the materials-discovery benchmark app.

    Returns
    -------
    MaterialsDiscoveryApp
        Configured materials-discovery application.
    """
    return MaterialsDiscoveryApp(
        name=BENCHMARK_NAME,
        description=(
            "Classification and regression performance on the WBM materials "
            "discovery test set."
        ),
        docs_url=DOCS_URL,
        framework_ids="matbench-discovery",
        table_path=DATA_PATH / "materials_discovery_metrics_table.json",
        extra_components=[Div(id=f"{BENCHMARK_NAME}-figure-placeholder")],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8051, debug=True)
