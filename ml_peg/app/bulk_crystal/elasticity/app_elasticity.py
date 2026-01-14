"""Run elasticity benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_density_plot_for_model
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(None)
BENCHMARK_NAME = "Elasticity"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#elasticity"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "elasticity"


class ElasticityApp(BaseApp):
    """Elasticity benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # Build plots for models with data (read_density_plot_for_model
        # returns None for models without data)
        density_plots = {}
        for model in MODELS:
            plots = {
                "Bulk modulus MAE": read_density_plot_for_model(
                    filename=DATA_PATH / "figure_bulk_density.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-bulk-figure",
                ),
                "Shear modulus MAE": read_density_plot_for_model(
                    filename=DATA_PATH / "figure_shear_density.json",
                    model=model,
                    id=f"{BENCHMARK_NAME}-{model}-shear-figure",
                ),
            }
            # Filter out None values (models without data for that metric)
            model_plots = {k: v for k, v in plots.items() if v is not None}
            if model_plots:
                density_plots[model] = model_plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=density_plots,
        )


def get_app() -> ElasticityApp:
    """
    Get elasticity benchmark app layout and callback registration.

    Returns
    -------
    ElasticityApp
        Benchmark layout and callback registration.
    """
    return ElasticityApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance when predicting VRH bulk and shear moduli for crystalline "
            "materials compared against Materials Project reference data."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "elasticity_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    elasticity_app = get_app()
    full_app.layout = elasticity_app.layout
    elasticity_app.register_callbacks()
    full_app.run(port=8054, debug=True)
