"""Run Al3Ti polymorph app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Al3Ti polymorphs"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/alloy_metallurgy.html#al3ti-polymorphs"
DATA_PATH = APP_ROOT / "data" / "alloy_metallurgy" / "al3ti_polymorphs"
INFO_PATH = DATA_PATH / "info.json"

POLYMORPHS = ("D022", "D023", "L12")


class Al3TiPolymorphApp(BaseApp):
    """Al3Ti Polymorph benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        formation_scatter = read_plot(
            DATA_PATH / "figure_formation_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        structs_dir = next(
            (
                DATA_PATH / model_name
                for model_name in MODELS
                if all(
                    (DATA_PATH / model_name / f"{name}.xyz").is_file()
                    for name in POLYMORPHS
                )
            ),
            None,
        )
        structs = (
            [
                f"/assets/alloy_metallurgy/al3ti_polymorphs/{structs_dir.name}/{name}.xyz"
                for name in POLYMORPHS
            ]
            if structs_dir is not None
            else []
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Formation Energy MAE": formation_scatter,
                "Relative Stability Ordering Accuracy": formation_scatter,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> Al3TiPolymorphApp:
    """
    Get Al3Ti Polymorph benchmark app layout and callback registration.

    Returns
    -------
    Al3TiPolymorphApp
        Benchmark layout and callback registration.
    """
    return Al3TiPolymorphApp(
        name=BENCHMARK_NAME,
        description=(
            "Formation energies, relaxed cell geometry, and stability ordering "
            "for Al3Ti polymorphs."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "al3ti_metrics.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
