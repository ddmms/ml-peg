"""Run elemental slab oxygen adsorption app."""

from __future__ import annotations

from copy import deepcopy
import warnings

from dash import Dash
from dash.html import Div
import numpy as np

from ml_peg.analysis.surfaces.elemental_slab_oxygen_adsorption.analyse_elemental_slab_oxygen_adsorption import (  # noqa: E501
    get_metrics,
)
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Elemental Slab Oxygen Adsorption"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/surfaces.html#elemental-slab-oxygen-adsorption"
DATA_PATH = APP_ROOT / "data" / "surfaces" / "elemental_slab_oxygen_adsorption"
INFO_PATH = DATA_PATH / "info.json"


class ElementalSlabOxygenAdsorptionApp(BaseApp):
    """Elemental slab oxygen adsorption benchmark app layout and callbacks."""

    def set_data(self) -> None:
        """Set data for app."""
        self.data = read_plot(
            DATA_PATH / "figure_adsorption_energies.json", id=f"{BENCHMARK_NAME}-figure"
        )
        self.set_partial_elements()

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        if not hasattr(self, "data") or self.data is None:
            self.set_data()

        # Assets dir will be parent directory
        structs_dir = DATA_PATH / "mock"
        structs = [
            f"/assets/surfaces/elemental_slab_oxygen_adsorption/mock/{struct_file.stem}.xyz"
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
            mode="traj",
        )

    def set_partial_elements(self) -> None:
        """Get lists of element sets for partial filtering."""
        try:
            self.partial_elements = [
                set(elements) for elements in self.info["elements"]
            ]
        except (AttributeError, KeyError, TypeError, IndexError) as err:
            self.partial_elements = set()
            warnings.warn(
                f"Unable to read elements lists for {self.name}: {err}", stacklevel=2
            )

    def filter_table(
        self,
        filter_elements: list[str],
    ) -> dict[str, dict]:
        """
        Filter data by elements.

        Parameters
        ----------
        filter_elements
            List of elements to filter out of data.

        Returns
        -------
        dict[str, dict]
            Updated benchmark table.
        """
        # Ensure scatter data and partial elements are set
        if not hasattr(self, "data") or self.data is None:
            self.set_data()

        filter_elements = set(filter_elements) if filter_elements else set()
        table_data = deepcopy(self.table.data)

        # If full overlap, set to None as with basic filtering
        if not bool(self.elements - filter_elements):
            for row in table_data:
                for metric in self.metrics:
                    row[metric] = None
            return table_data

        # If no elements filtered, return original table data
        if not filter_elements:
            for current_row, original_row in zip(
                table_data, self.original_table.data, strict=True
            ):
                for metric in self.metrics:
                    current_row[metric] = original_row[metric]
            return table_data

        # Partial filtering
        # Get overlap of deselected elements with each system's elements
        filtered_indices = [
            not bool(elements & filter_elements) for elements in self.partial_elements
        ]

        results = {}
        ref_filtered = False

        for plot in self.data.figure.data:
            # Ignore unamed (parity) line
            if plot.name and len(plot.x) != 0:
                results[plot.name] = np.array(plot.x)[filtered_indices].tolist()
                if not ref_filtered:
                    results["ref"] = np.array(plot.y)[filtered_indices].tolist()
                    ref_filtered = True

        new_metrics = get_metrics(results)

        for row in table_data:
            model = row["MLIP"]
            for metric in self.metrics:
                row[metric] = new_metrics[metric].get(model, None)

        return table_data


def get_app() -> ElementalSlabOxygenAdsorptionApp:
    """
    Get elemental slab oxygen adsorption benchmark app layout and callback registration.

    Returns
    -------
    ElementalSlabOxygenAdsorptionApp
        Benchmark layout and callback registration.
    """
    return ElementalSlabOxygenAdsorptionApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting adsorption energies of oxygen "
            "on elemental slabs."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "elemental_slab_oxygen_adsorption_metrics_table.json",
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
    elemental_slab_oxygen_adsorption_app = get_app()
    full_app.layout = elemental_slab_oxygen_adsorption_app.layout
    elemental_slab_oxygen_adsorption_app.register_callbacks()

    # Run app
    full_app.run(port=8052, debug=True)
