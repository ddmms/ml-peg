"""Run TorsionNet500 dihedral scan benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "TorsionNet500"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/conformers.html#torsionnet500"
)
DATA_PATH = APP_ROOT / "data" / "conformers" / "TorsionNet500"


class TorsionNet500App(BaseApp):
    """TorsionNet500 dihedral scan benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_torsionnet500.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        struct_dir = DATA_PATH / "mock"
        if struct_dir.exists():
            labels = sorted([f.stem for f in struct_dir.glob("*.xyz")])
            structs = [
                f"/assets/conformers/TorsionNet500/mock/{label}.xyz" for label in labels
            ]
        else:
            structs = []

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Barrier Height MAE": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> TorsionNet500App:
    """
    Get TorsionNet500 benchmark app layout and callback registration.

    Returns
    -------
    TorsionNet500App
        Benchmark layout and callback registration.
    """
    return TorsionNet500App(
        name=BENCHMARK_NAME,
        framework_ids="mlip_audit",
        description=(
            "Performance in predicting torsion energy barriers for drug-like "
            "molecules from systematic dihedral scans. Reference data from "
            "wB97M-D3(BJ)/def2-TZVPPD calculations."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "torsionnet500_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8069, debug=True)
