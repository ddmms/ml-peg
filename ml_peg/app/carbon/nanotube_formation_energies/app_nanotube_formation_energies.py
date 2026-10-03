"""Run nanotube formation energies app."""

from __future__ import annotations

import warnings

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "Nanotube Formation Energies"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/carbon.html"
    "#nanotube-formation-energies"
)
DATA_PATH = APP_ROOT / "data" / "carbon" / "nanotube_formation_energies"
INFO_PATH = DATA_PATH / "info.json"


class NanotubeFormationEnergiesApp(BaseApp):
    """Nanotube formation energies benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_nanotube_formation_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        # Assets dir will be parent directory - individual files for each system
        structs_dir = DATA_PATH / "mock"
        if not structs_dir.exists():
            warnings.warn(f"Structures directory {structs_dir} not found", stacklevel=2)
        structs = [
            f"/assets/carbon/nanotube_formation_energies/mock/{struct_file.name}"
            for struct_file in sorted(structs_dir.glob("*.extxyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Armchair strain energy MAE": scatter,
                "Zigzag strain energy MAE": scatter,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> NanotubeFormationEnergiesApp:
    """
    Get nanotube formation energies benchmark app layout and callback registration.

    Returns
    -------
    NanotubeFormationEnergiesApp
        Benchmark layout and callback registration.
    """
    return NanotubeFormationEnergiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Single-point strain energy relative to graphene for ten armchair"
            " and ten zigzag carbon nanotubes. Reference data is optB88-vdW."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "nanotube_formation_energies_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
        framework_ids="gap-20",
    )
