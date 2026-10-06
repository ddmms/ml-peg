"""Run amorphous carbon melt-quench app."""

from __future__ import annotations

from dash.html import Div, Span

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_multi_scatters,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Melt-quench carbon"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "amorphous_materials.html#amorphous-carbon-melt-quench"
)
DATA_PATH = APP_ROOT / "data" / "amorphous_materials" / "amorphous_carbon_melt_quench"
INFO_PATH = DATA_PATH / "info.json"

# Coordination classes are written as stand-in elements, shown in WEAS default colors
LEGEND = Div(
    [
        Span(f"■ {label}", style={"color": color, "marginRight": "16px"})
        for label, color in (
            ("sp1 (coord=2)", "green"),
            ("sp2 (coord=3)", "blue"),
            ("sp3 (coord=4)", "orange"),
        )
    ]
)


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

        # Model traces carry structure paths as customdata. Reference traces have none
        struct_from_multi_scatters(
            scatter_id="amorphous-carbon-melt-quench-figure",
            struct_id="amorphous-carbon-melt-quench-struct-placeholder",
            structs=[
                list(trace.customdata or [])
                for trace in (scatter.figure.data if scatter.figure else [])
            ],
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
            LEGEND,
            Div(id="amorphous-carbon-melt-quench-struct-placeholder"),
        ],
        framework_ids="mace-mp",
        info_path=INFO_PATH,
    )
