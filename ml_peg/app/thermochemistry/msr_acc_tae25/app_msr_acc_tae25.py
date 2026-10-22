"""Run MSR_ACC_TAE25 app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column, struct_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "msr_acc_tae25"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/thermochemistry.html#msr_acc_tae25"
DATA_PATH = APP_ROOT / "data" / "thermochemistry" / "msr_acc_tae25"
INFO_PATH = DATA_PATH / "info.json"


class MSRACCTAE25APP(BaseApp):
    """MSR_ACC_TAE25 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "msr_acc_tae25_parity_plot.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        structs_dir = DATA_PATH / "mock"
        structs = [
            f"/assets/thermochemistry/msr_acc_tae25/mock/{struct_file.stem}.xyz"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"MAE": scatter},
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> MSRACCTAE25APP:
    """
    Get  benchmark app layout and callback registration.

    Returns
    -------
    MSRACCTAE25APP
        Benchmark layout and callback registration.
    """
    return MSRACCTAE25APP(
        name=BENCHMARK_NAME,
        description="Total atomization energy for closed-shell neutral molecules.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "msr_acc_tae25_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
