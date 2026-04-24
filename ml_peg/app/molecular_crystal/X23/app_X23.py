"""Run X23 app."""

from __future__ import annotations

import warnings

from dash import Dash
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


# FILTER_PAYLOAD = _load_filter_payload()
# FILTER_ELEMENTS = [
#     element
#     for element in FILTER_PAYLOAD.get("elements", [])
#     if isinstance(element, str) and element
# ]
# FILTER_SYSTEMS = [
#     system
#     for system in FILTER_PAYLOAD.get("systems", [])
#     if isinstance(system, str) and system
# ]
# FILTER_SYSTEM_ELEMENTS = FILTER_PAYLOAD.get("system_elements", [])
# SYSTEM_TO_INDEX = {system: idx for idx, system in enumerate(FILTER_SYSTEMS)}


# def _figure_dict(graph) -> dict:
#     """Convert a Plotly graph component into a mutable figure dictionary."""
#     figure = getattr(graph, "figure", None)
#     if figure is None:
#         return {}
#     if hasattr(figure, "to_plotly_json"):
#         return figure.to_plotly_json()
#     if isinstance(figure, dict):
#         return figure
#     return {}


# def _customdata_system(point_customdata) -> str | None:
#     """Extract the system label from a Plotly customdata entry."""
#     if point_customdata is None:
#         return None
#     if isinstance(point_customdata, (list, tuple, np.ndarray)):
#         if not point_customdata:
#             return None
#         value = point_customdata[0]
#     else:
#         value = point_customdata
#     return str(value) if value is not None else None


# def _keep_mask(selected_elements: list[str] | None) -> np.ndarray:
#     """Return a mask of systems that do not contain deselected elements."""
#     n_systems = len(FILTER_SYSTEMS)
#     n_elements = len(FILTER_ELEMENTS)
#     if n_systems == 0:
#         return np.zeros(0, dtype=bool)

#     selected = {
#         element for element in (selected_elements or []) if isinstance(element, str)
#     }
#     selected_vector = np.array(
#         [element in selected for element in FILTER_ELEMENTS], dtype=bool
#     )
#     deselected_vector = ~selected_vector
#     if not deselected_vector.any() or n_elements == 0:
#         return np.ones(n_systems, dtype=bool)

#     incidence = np.zeros((n_systems, n_elements), dtype=bool)
#     element_to_idx = {element: idx for idx, element in enumerate(FILTER_ELEMENTS)}
#     for system_idx, element_list in enumerate(FILTER_SYSTEM_ELEMENTS[:n_systems]):
#         if not isinstance(element_list, list):
#             continue
#         for element in element_list:
#             element_idx = element_to_idx.get(element)
#             if element_idx is not None:
#                 incidence[system_idx, element_idx] = True

#     excluded = incidence[:, deselected_vector].any(axis=1)
#     return ~excluded


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

        filter_table(table_id=self.table_id)

    def get_elements(self) -> None:
        """Get element sets for filtering from loaded info."""
        try:
            self.elements = (set(entry) for entry in self.info["elements"])
        except KeyError:
            warnings.warn("Unable to read elements lists.", stacklevel=2)

    def filter_data(self, deselected_elements: set[str]) -> dict[str, dict]:
        """
        Apply elements filter to data.

        Parameters
        ----------
        deselected_elements
            Set of elements to filter out of data.

        Returns
        -------
        dict[str, dict]
            Metric names and values for all models.
        """
        if not hasattr(self, "data"):
            self.load_data()
        if not hasattr(self, "elements"):
            self.get_elements()

        # Get overlap of deselected elements with each system's elements)
        filtered_indices = [
            not bool(elements & deselected_elements) for elements in self.elements
        ]

        results = {}
        ref_filtered = False

        for plot in self.data.figure.data:
            if plot.name:
                results[plot.name] = np.array(plot.x)[filtered_indices]
                if not ref_filtered:
                    results["ref"] = np.array(plot.y)[filtered_indices]
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
