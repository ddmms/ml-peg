"""Run GMTKN55 app."""

from __future__ import annotations

from pathlib import Path
import warnings

from dash import Dash
from dash.html import Div

from ml_peg.analysis.molecular.GMTKN55.analyse_GMTKN55 import get_metrics
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    filter_table,
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.filter import filter_parity
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "GMTKN55"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular.html#gmtkn55"
DATA_PATH = APP_ROOT / "data" / "molecular" / "GMTKN55"


class GMTKN55App(BaseApp):
    """GMTKN55 benchmark app layout and callbacks."""

    def load_data(self) -> None:
        """Load data required for filtering."""
        self.data = read_plot(
            DATA_PATH / "figure_rel_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        if not hasattr(self, "data"):
            self.load_data()

        # Assets dir will be parent directory - individual files for each polymorph
        structs_dir = DATA_PATH / "mock"
        structs = [
            f"/assets/molecular/GMTKN55/mock/{struct_file.stem}.xyz"
            for struct_file in sorted(
                structs_dir.glob("*.xyz"), key=lambda f: int(Path(f).stem)
            )
        ]

        scatter = self.data
        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Small systems": scatter,
                "Large systems": scatter,
                "Barrier heights": scatter,
                "Intramolecular NCIs": scatter,
                "Intermolecular NCIs": scatter,
                "WTMAD": scatter,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="traj",
        )

        # Ensure data and elements are loaded
        if not hasattr(self, "data"):
            self.load_data()
        if not hasattr(self, "elements"):
            self.get_elements()

        filter_table(
            table_id=self.table_id,
            filter_func=self.filter_data,
            filter_kwargs={
                "data": self.data,
                "test_elements": self.elements,
                "metric_getter": get_metrics,
                "mask_to_getter": True,
            },
        )

    def get_elements(self) -> None:
        """Get element sets for filtering."""
        try:
            self.elements = [set(entry) for entry in self.info["elements"]]
        except (AttributeError, KeyError, TypeError):
            self.elements = []
            warnings.warn("Unable to read elements lists.", stacklevel=2)

    @staticmethod
    def filter_data(*args, **kwargs) -> dict[str, dict]:
        """
        Filter data by elements.

        Parameters
        ----------
        *args
            Positional arguments for filter function.
        **kwargs
            Keyword arguments for filter function.

        Returns
        -------
        dict[str, dict]
            Filtered results to update table.
        """
        return filter_parity(*args, **kwargs)


def get_app() -> GMTKN55App:
    """
    Get GMTKN55 benchmark app layout and callback registration.

    Returns
    -------
    GMTKN55App
        Benchmark layout and callback registration.
    """
    return GMTKN55App(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting relative energies for 55 subsets of molecules, "
            "inclding intramolecular non-covalent interactions (NCIs), intermolecular "
            "NCIs, small systems, large systems and barrier heights."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "gmtkn55_metrics_table.json",
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
    gmtkn55_app = get_app()
    full_app.layout = gmtkn55_app.layout
    gmtkn55_app.register_callbacks()

    # Run app
    full_app.run(port=8051, debug=True)
