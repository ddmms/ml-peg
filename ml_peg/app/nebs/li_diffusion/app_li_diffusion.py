"""Run Li diffusion app."""

from __future__ import annotations

import warnings

from dash import Dash
from dash.html import Div
import numpy as np

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    filter_table,
    plot_from_table_cell,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Li diffusion"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nebs.html#li-diffusion"
DATA_PATH = APP_ROOT / "data" / "nebs" / "li_diffusion"


class LiDiffusionApp(BaseApp):
    """Li diffusion benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_plots = {
            model: {
                "Path B error": read_plot(
                    DATA_PATH / model / "figure_neb_b.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-b",
                ),
                "Path C error": read_plot(
                    DATA_PATH / model / "figure_neb_c.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-c",
                ),
            }
            for model in MODELS
        }

        # Assets dir will be parent directory
        assets_dir = "/assets/nebs/li_diffusion"
        structs = {
            model: {
                "Path B error": f"{assets_dir}/{model}/b-neb-band.extxyz",
                "Path C error": f"{assets_dir}/{model}/c-neb-band.extxyz",
            }
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=scatter_plots,
        )

        for model in MODELS:
            for path in ("b", "c"):
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-figure-{path}",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=structs[model][f"Path {path.upper()} error"],
                    mode="traj",
                )

        # Ensure elements are loaded
        if not hasattr(self, "elements"):
            self.get_elements()

        filter_table(
            table_id=self.table_id,
            filter_func=self.filter_data,
            filter_kwargs={"test_elements": self.elements},
        )

    def get_elements(self) -> None:
        """Get element sets for filtering."""
        try:
            self.elements = set(self.info["elements"][0])
        except (AttributeError, KeyError, TypeError):
            self.elements = set()
            warnings.warn("Unable to read elements lists.", stacklevel=2)

    @staticmethod
    def filter_data(
        filter_elements: set[str], test_elements: set[str]
    ) -> dict[str, dict]:
        """
        Apply elements filter to data.

        Parameters
        ----------
        filter_elements
            Set of elements to filter out of data.
        test_elements
            Set of elements for each system.

        Returns
        -------
        dict[str, dict]
            Metric names and values for all models.
        """
        # Get overlap of deselected elements with each system's elements
        if bool(test_elements & filter_elements):
            return {
                "Path B error": dict.fromkeys(MODELS, np.nan),
                "Path C error": dict.fromkeys(MODELS, np.nan),
            }
        return None


def get_app() -> LiDiffusionApp:
    """
    Get Li diffusion benchmark app layout and callback registration.

    Returns
    -------
    LiDiffusionApp
        Benchmark layout and callback registration.
    """
    return LiDiffusionApp(
        name=BENCHMARK_NAME,
        description=("Performance in predicting energy barriers for Li diffision."),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "li_diffusion_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=DATA_PATH / "info.json",
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    # Construct layout and register callbacks
    li_diffusion_app = get_app()
    full_app.layout = li_diffusion_app.layout
    li_diffusion_app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
