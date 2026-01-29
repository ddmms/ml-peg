"""Run amorphous carbon melt-quench app."""

from __future__ import annotations

from dash import Dash, Input, Output, callback
from dash.html import Div, Iframe

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.weas import generate_weas_html

BENCHMARK_NAME = "Melt-quench carbon"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "amorphous_materials.html#amorphous-carbon-melt-quench"
)
DATA_PATH = APP_ROOT / "data" / "amorphous_materials" / "amorphous_carbon_melt_quench"


class AmorphousCarbonMeltQuenchApp(BaseApp):
    """Amorphous carbon melt-quench benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_sp3_vs_density.json",
            id="amorphous-carbon-melt-quench-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id="amorphous-carbon-melt-quench-figure-placeholder",
            column_to_plot={
                "MAE vs DFT": scatter,
                "MAE vs Expt": scatter,
            },
        )

        @callback(
            Output("amorphous-carbon-melt-quench-struct-placeholder", "children"),
            Input("amorphous-carbon-melt-quench-figure", "clickData"),
        )
        def show_structure(click_data) -> Div:
            if not click_data:
                return Div("Click on a model point to view the structure.")
            point = click_data["points"][0]
            struct_path = point.get("customdata")
            if not struct_path:
                return Div("No structure available for this point.")
            mode = "traj" if "trajectory_" in str(struct_path) else "struct"
            return Div(
                Iframe(
                    srcDoc=generate_weas_html(
                        struct_path,
                        mode=mode,
                        legend_items=[
                            ("sp1 (coord=2)", "green"),
                            ("sp2 (coord=3)", "blue"),
                            ("sp3 (coord=4)", "orange"),
                        ],
                        show_controls=True,
                        show_bounds=True,
                    ),
                    style={
                        "height": "550px",
                        "width": "100%",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                    },
                )
            )


def get_app() -> AmorphousCarbonMeltQuenchApp:
    """
    Get amorphous carbon melt-quench benchmark app layout and callbacks.

    Returns
    -------
    AmorphousCarbonMeltQuenchApp
        Benchmark layout and callback registration.
    """
    return AmorphousCarbonMeltQuenchApp(
        name=BENCHMARK_NAME,
        description=(
            "Melt-quench simulations of amorphous carbon; compare sp3 fraction versus "
            "density to DFT and experimental references."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "amorphous_carbon_melt_quench_metrics_table.json",
        extra_components=[
            Div(id="amorphous-carbon-melt-quench-figure-placeholder"),
            Div(id="amorphous-carbon-melt-quench-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent)

    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()

    full_app.run(port=8052, debug=True)
