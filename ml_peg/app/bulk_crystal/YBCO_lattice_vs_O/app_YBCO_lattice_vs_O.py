"""Run the YBCO lattice-parameters-vs-oxygen-content benchmark app."""

from __future__ import annotations

from dash.dcc import Graph
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_multi_scatters,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "YBCO lattice vs oxygen"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "bulk_crystal.html#ybco-lattice-parameters-vs-oxygen-content"
)
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "YBCO_lattice_vs_O"
INFO_PATH = DATA_PATH / "info.json"


def _model_structs(scatter: Graph, asset_dir: str) -> list[list[str]]:
    """
    Get relaxed structure paths for each model trace and point of a parity plot.

    Parameters
    ----------
    scatter
        Parity plot, with one trace per model and point labels as the last
        ``customdata`` entry.
    asset_dir
        Asset directory containing a subdirectory of structures per model.

    Returns
    -------
    list[list[str]]
        Structure paths indexed by trace then point, empty where unavailable.
    """
    structs = []
    for trace in (scatter.figure or {}).get("data", []):
        model = trace.get("name")
        structs.append(
            [
                f"/assets/{asset_dir}/{model}/{point[-1]}.xyz"
                if (DATA_PATH / model / f"{point[-1]}.xyz").exists()
                else ""
                for point in trace.get("customdata") or []
            ]
        )
    return structs


class YBCOLatticeApp(BaseApp):
    """YBCO lattice-vs-oxygen benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_a = read_plot(
            DATA_PATH / "figure_ybco_lattice_a.json", id=f"{BENCHMARK_NAME}-figure"
        )
        scatter_b = read_plot(
            DATA_PATH / "figure_ybco_lattice_b.json", id=f"{BENCHMARK_NAME}-figure"
        )
        scatter_c = read_plot(
            DATA_PATH / "figure_ybco_lattice_c.json", id=f"{BENCHMARK_NAME}-figure"
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "MAE a (PBE)": scatter_a,
                "MAE b (PBE)": scatter_b,
                "MAE c (PBE)": scatter_c,
            },
        )

        struct_from_multi_scatters(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=_model_structs(scatter_a, "bulk_crystal/YBCO_lattice_vs_O"),
            mode="struct",
        )


def get_app() -> YBCOLatticeApp:
    """
    Get YBCO lattice-vs-oxygen benchmark app layout and callback registration.

    Returns
    -------
    YBCOLatticeApp
        Benchmark layout and callback registration.
    """
    return YBCOLatticeApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance when predicting the YBa2Cu3O(6+x) lattice parameters a, b, c "
            "as a function of oxygen content (6.0-7.0), including the orthorhombic-to-"
            "tetragonal phase transition, against CP2K PBE."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "ybco_lattice_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
