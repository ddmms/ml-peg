"""Run lattice parameters app."""

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

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Lattice Parameters"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/carbon.html#lattice-parameters"
DATA_PATH = APP_ROOT / "data" / "carbon" / "lattice_parameters"
INFO_PATH = DATA_PATH / "info.json"


class LatticeParametersApp(BaseApp):
    """Lattice parameters benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_lattice_parameters_energy_above_graphite.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Energy above graphite MAE (D3)": scatter},
        )

        model_dir = DATA_PATH / MODELS[0]
        if model_dir.exists():
            labels = sorted(f.stem for f in model_dir.glob("*.xyz"))
            structs = [
                f"/assets/carbon/lattice_parameters/{MODELS[0]}/{label}.xyz"
                for label in labels
            ]
        else:
            structs = []

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> LatticeParametersApp:
    """
    Get lattice parameters benchmark app layout and callback registration.

    Returns
    -------
    LatticeParametersApp
        Benchmark layout and callback registration.
    """
    return LatticeParametersApp(
        name=BENCHMARK_NAME,
        description=(
            "Lattice parameters, neighbour bond lengths, and energy above"
            " graphite for carbon allotropes. Reference data is optB88-vdW."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "lattice_parameters_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
        framework_ids="gap-20",
    )
