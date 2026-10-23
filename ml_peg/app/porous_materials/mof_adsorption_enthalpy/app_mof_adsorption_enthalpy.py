"""Run MOF adsorption enthalpy app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "MOF adsorption enthalpy"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "porous_materials.html#mof-adsorption-enthalpy"
)
DATA_PATH = APP_ROOT / "data" / "porous_materials" / "mof_adsorption_enthalpy"


class MOFAdsorptionEnthalpyApp(BaseApp):
    """MOF adsorption enthalpy benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        rigid = read_plot(
            DATA_PATH / "figure_enthalpy_rigid.json",
            id=f"{BENCHMARK_NAME}-rigid-figure",
        )
        nvt = read_plot(
            DATA_PATH / "figure_enthalpy_nvt.json",
            id=f"{BENCHMARK_NAME}-nvt-figure",
        )
        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Rigid MAE": rigid, "NVT MAE": nvt},
        )


def get_app() -> MOFAdsorptionEnthalpyApp:
    """
    Get MOF adsorption enthalpy benchmark app layout and callback registration.

    Returns
    -------
    MOFAdsorptionEnthalpyApp
        Benchmark layout and callback registration.
    """
    return MOFAdsorptionEnthalpyApp(
        name=BENCHMARK_NAME,
        description=(
            "Enthalpy of adsorption of CO2 in metal-organic frameworks from Widom "
            "insertion, compared against experiment."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "mof_adsorption_enthalpy_metrics_table.json",
        extra_components=[Div(id=f"{BENCHMARK_NAME}-figure-placeholder")],
    )
