"""Run CPOSS209 app."""

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
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "CPOSS209 Lattice Energies"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_crystal.html#cposs209"
DATA_PATH = APP_ROOT / "data" / "molecular_crystal" / "CPOSS209"


class CPOSS209App(BaseApp):
    """CPOSS209 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        absolute_scatter = read_plot(
            DATA_PATH / "figure_absolute_lattice_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        relative_scatter = read_plot(
            DATA_PATH / "figure_relative_lattice_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        absolute_small_rigid_molecule_scatter = read_plot(
            DATA_PATH / "figure_absolute_lattice_energies_small_rigid_molecules.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        relative_small_rigid_molecule_scatter = read_plot(
            DATA_PATH / "figure_relative_lattice_energies_small_rigid_molecules.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        absolute_carbamazepine_scatter = read_plot(
            DATA_PATH / "figure_absolute_lattice_energies_carbamazepine_family.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        relative_carbamazepine_scatter = read_plot(
            DATA_PATH / "figure_relative_lattice_energies_carbamazepine_family.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        absolute_fenamate_scatter = read_plot(
            DATA_PATH / "figure_absolute_lattice_energies_fenamate_family.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        relative_fenamate_scatter = read_plot(
            DATA_PATH / "figure_relative_lattice_energies_fenamate_family.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        absolute_small_drug_molecule_scatter = read_plot(
            DATA_PATH
            / "figure_absolute_lattice_energies_small_drug_molecule_family.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        relative_small_drug_molecule_scatter = read_plot(
            DATA_PATH
            / "figure_relative_lattice_energies_small_drug_molecule_family.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Assets dir will be parent directory - individual files for each system
        structs_dir = DATA_PATH / MODELS[0]

        structs = [
            f"assets/molecular_crystal/CPOSS209/{MODELS[0]}/{struct_file.relative_to(structs_dir)}"
            for struct_file in sorted(structs_dir.glob("**/crystal*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Absolute MAE": absolute_scatter,
                "Relative MAE": relative_scatter,
                "Absolute MAE small rigid molecules": (
                    absolute_small_rigid_molecule_scatter
                ),
                "Relative MAE small rigid molecules": (
                    relative_small_rigid_molecule_scatter
                ),
                "Absolute MAE carbamazepine family": (absolute_carbamazepine_scatter),
                "Relative MAE carbamazepine family": (relative_carbamazepine_scatter),
                "Absolute MAE fenamate family": (absolute_fenamate_scatter),
                "Relative MAE fenamate family": (relative_fenamate_scatter),
                "Absolute MAE small drug molecules": (
                    absolute_small_drug_molecule_scatter
                ),
                "Relative MAE small drug molecules": (
                    relative_small_drug_molecule_scatter
                ),
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> CPOSS209App:
    """
    Get CPOSS209 benchmark app layout and callback registration.

    Returns
    -------
    CPOSS209App
        Benchmark layout and callback registration.
    """
    return CPOSS209App(
        name=BENCHMARK_NAME,
        description=(
            "Absolute and relative lattice energies for 209 organic molecular crystals."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "cposs209_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    cposs209_app = get_app()
    full_app.layout = cposs209_app.layout
    cposs209_app.register_callbacks()

    # Run app
    full_app.run(port=8053, debug=True)
