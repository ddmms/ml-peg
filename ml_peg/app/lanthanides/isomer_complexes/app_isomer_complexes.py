"""Run lanthanide isomer complex benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column, struct_from_scatter
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Lanthanide Isomer Complexes"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/lanthanides.html"
    "#isomer-complexes"
)
DATA_PATH = APP_ROOT / "data" / "lanthanides" / "isomer_complexes"


class IsomerComplexesApp(BaseApp):
    """Lanthanide isomer complex benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_isomer_complexes.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )

        struct_root = DATA_PATH / "structures"
        if struct_root.exists():
            structs = []
            for system_dir in sorted(struct_root.glob("*")):
                if not system_dir.is_dir():
                    continue
                for struct_file in sorted(system_dir.glob("*.xyz")):
                    structs.append(
                        f"assets/lanthanides/isomer_complexes/structures/"
                        f"{system_dir.name}/{struct_file.name}"
                    )

            if structs:
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-figure",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=structs,
                    mode="struct",
                )


def get_app() -> IsomerComplexesApp:
    """
    Get lanthanide isomer complex benchmark app layout and callback registration.

    Returns
    -------
    IsomerComplexesApp
        Benchmark layout and callback registration.
    """
    return IsomerComplexesApp(
        name=BENCHMARK_NAME,
        description=(
            "Relative energies of lanthanide isomer complexes compared to r2SCAN-3c."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "isomer_complexes_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    app_instance = get_app()
    full_app.layout = app_instance.layout
    app_instance.register_callbacks()

    # Run app
    full_app.run(port=8061, debug=True)
