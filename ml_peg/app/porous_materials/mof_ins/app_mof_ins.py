"""Run MOF INS spectra benchmark app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "MOF INS spectra"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "porous_materials.html#mof-ins-spectra"
)
DATA_PATH = APP_ROOT / "data" / "porous_materials" / "mof_ins"


class MOFINSApp(BaseApp):
    """MOF INS benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        violin = read_plot(
            DATA_PATH / "figure_ins_wasserstein.json", id=f"{BENCHMARK_NAME}-figure"
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Wasserstein distance": violin},
        )


def get_app() -> MOFINSApp:
    """
    Get MOF INS benchmark app layout and callback registration.

    Returns
    -------
    MOFINSApp
        Benchmark layout and callback registration.
    """
    return MOFINSApp(
        name=BENCHMARK_NAME,
        description=(
            "Simulated inelastic neutron scattering spectra of metal-organic "
            "frameworks, scored by the Wasserstein-1 distance to digitised "
            "experimental spectra. The percentage of imaginary modes from the "
            "same phonon calculation is reported alongside it."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "mof_ins_metrics_table.json",
        extra_components=[Div(id=f"{BENCHMARK_NAME}-figure-placeholder")],
    )
