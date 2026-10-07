"""Run elemental cohesive energy app."""

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
BENCHMARK_NAME = "Elemental cohesive energy"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#elemental-cohesive-energy"
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "elemental_cohesive_energy"
INFO_PATH = DATA_PATH / "info.json"


class ElementalCohesiveEnergyApp(BaseApp):
    """Elemental cohesive energy benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_id = f"{BENCHMARK_NAME}-figure"
        scatter_pbe = read_plot(
            DATA_PATH / "figure_cohesive_energies_pbe.json", id=scatter_id
        )
        scatter_exp = read_plot(
            DATA_PATH / "figure_cohesive_energies_exp.json", id=scatter_id
        )

        # Assets dir will be parent directory - individual files for each element
        structs_dir = DATA_PATH / MODELS[0]
        structs = [
            f"/assets/bulk_crystal/elemental_cohesive_energy/{MODELS[0]}/{struct_file.stem}.xyz"
            for struct_file in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "MAE (PBE)": scatter_pbe,
                "MAE (Experimental)": scatter_exp,
            },
        )

        struct_from_scatter(
            scatter_id=scatter_id,
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="traj",
        )


def get_app() -> ElementalCohesiveEnergyApp:
    """
    Get elemental cohesive energy benchmark app layout and callback registration.

    Returns
    -------
    ElementalCohesiveEnergyApp
        Benchmark layout and callback registration.
    """
    return ElementalCohesiveEnergyApp(
        name=BENCHMARK_NAME,
        description="Cohesive energies of 25 elemental crystals.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "elemental_cohesive_energy_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
