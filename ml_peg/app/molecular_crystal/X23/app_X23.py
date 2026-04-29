"""Run X23 app."""

from __future__ import annotations

import warnings

from dash import Dash
from dash.dcc import Graph
from dash.html import Div
import numpy as np

from ml_peg.analysis.molecular_crystal.X23.analyse_X23 import get_metrics
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    filter_table,
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot

# Get all models
BENCHMARK_NAME = "X23 Lattice Energies"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_crystal.html#x23"
)
DATA_PATH = APP_ROOT / "data" / "molecular_crystal" / "X23"
INFO_PATH = DATA_PATH / "info.json"
ELEMENT_DROPDOWN_ID = f"{BENCHMARK_NAME}-element-dropdown"


class X23App(BaseApp):
    """X23 benchmark app layout and callbacks."""

    def load_data(self) -> None:
        """Load data required for filtering."""
        self.data = read_plot(
            DATA_PATH / "figure_lattice_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        if not hasattr(self, "data"):
            self.load_data()

        # Assets dir will be parent directory - individual files for each system
        structs_dir = DATA_PATH / "mock"
        if not structs_dir.exists():
            warnings.warn(f"Structures directory {structs_dir} not found", stacklevel=2)
        structs = [
            f"/assets/molecular_crystal/X23/mock/{struct_file.stem}.xyz"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": self.data},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )

        # Ensure data and elements are loaded
        if not hasattr(self, "data"):
            self.load_data()
        if not hasattr(self, "elements"):
            self.get_elements()

        filter_table(
            table_id=self.table_id,
            filter_func=self.filter_data,
            filter_kwargs={"data": self.data, "test_elements": self.elements},
        )

    def get_elements(self) -> None:
        """Get element sets for filtering from loaded info."""
        try:
            self.elements = [set(entry) for entry in self.info["elements"]]
        except (AttributeError, KeyError, TypeError):
            self.elements = []
            warnings.warn("Unable to read elements lists.", stacklevel=2)

    @staticmethod
    def filter_data(
        filter_elements: set[str], data: Graph, test_elements: list[set[str]]
    ) -> dict[str, dict]:
        """
        Apply elements filter to data.

        Parameters
        ----------
        filter_elements
            Set of elements to filter out of data.
        data
            Scatter plot to filter.
        test_elements
            List of element for each system.

        Returns
        -------
        dict[str, dict]
            Metric names and values for all models.
        """
        # Get overlap of deselected elements with each system's elements
        filtered_indices = [
            not bool(elements & filter_elements) for elements in test_elements
        ]

        results = {}
        ref_filtered = False

        for plot in data.figure.data:
            # Ignore unamed (parity) line
            if plot.name:
                results[plot.name] = np.array(plot.x)[filtered_indices].tolist()
                if not ref_filtered:
                    results["ref"] = np.array(plot.y)[filtered_indices].tolist()
                    ref_filtered = True

        return get_metrics(results)


def get_app() -> X23App:
    """
    Get X23 benchmark app layout and callback registration.

    Returns
    -------
    X23App
        Benchmark layout and callback registration.
    """
    return X23App(
        name=BENCHMARK_NAME,
        description="Lattice energies for 23 organic molecular crystals.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "x23_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    x23_app = get_app()
    full_app.layout = x23_app.layout
    x23_app.register_callbacks()

    # Run app
    full_app.run(port=8053, debug=True)
