"""Run battery electrolyte benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
MODELS = MODELS[:-1]

BENCHMARK_NAME = "Battery electrolyte"
DATA_PATH = APP_ROOT / "data" / "bulk_liquids" / "battery_electrolyte"
REF_PATH = CALCS_ROOT / "bulk_liquids" / "battery_electrolyte" / "data"


class BatteryElectrolyteApp(BaseApp):
    """Battery electrolyte benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        # parity_plots = {
        #     model: {
        #         "Intra-Forces": read_plot(
        #             DATA_PATH / f"intra_forces_parity_{model}.json",
        #             id=f"{BENCHMARK_NAME}-{model}-figure",
        #         ),
        #         "Inter-Forces": read_plot(
        #             DATA_PATH / f"inter_forces_parity_{model}.json",
        #             id=f"{BENCHMARK_NAME}-{model}-figure",
        #         ),
        #         "Inter-Energy": read_plot(
        #             DATA_PATH / f"inter_energy_parity_{model}.json",
        #             id=f"{BENCHMARK_NAME}-{model}-figure",
        #         ),
        #         "Intra-Virial": read_plot(
        #             DATA_PATH / f"intra_virial_parity_{model}.json",
        #             id=f"{BENCHMARK_NAME}-{model}-figure",
        #         ),
        #         "Inter-Virial": read_plot(
        #             DATA_PATH / f"inter_virial_parity_{model}.json",
        #             id=f"{BENCHMARK_NAME}-{model}-figure",
        #         ),
        #     }
        #     for model in MODELS
        # }

        # plot_from_table_cell(
        #     table_id=self.table_id,
        #     plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
        #     cell_to_plot=parity_plots,
        # )
        scatter_plots = {
            model: {
                "solvent": read_plot(
                    DATA_PATH / f"solvent_{model}_volscan_scatter.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-solventVS",
                ),
                "electrolyte": read_plot(
                    DATA_PATH / f"electrolyte_{model}_volscan_scatter.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-electrolyteVS",
                ),
            }
            for model in MODELS
        }

        # Assets dir will be parent directory
        ref_conf_dir = REF_PATH / "Volume_Scan_data"
        structs = {
                "solvent": f"{ref_conf_dir}/solvent_VS_PBED3.extxyz",
                "electrolyte": f"{ref_conf_dir}/electrolyte_VS_PBED3.extxyz",
            }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=scatter_plots,
        )

        for model in MODELS:
            for volscan in ("solvent", "electrolyte"):
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-figure-{volscan}VS",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=structs[volscan],
                    mode="traj",
                )


def get_app() -> BatteryElectrolyteApp:
    """
    Get elasticity benchmark app layout and callback registration.

    Returns
    -------
    ElasticityApp
        Benchmark layout and callback registration.
    """
    return BatteryElectrolyteApp(
        name=BENCHMARK_NAME,
        description=(
            "Evaluate model inter/intra property prediction "
            "for different densities of LIB full electrolyte"
            " and neat solvent configs"
        ),
        # docs_url=DOCS_URL,
        # table_path=DATA_PATH / "Inter_intra_metrics_table.json",
        # extra_components=[
        #     Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        # ],
        table_path=DATA_PATH / "ol_scan_rmses_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    BatteryElectrolyte_app = get_app()
    full_app.layout = BatteryElectrolyte_app.layout
    BatteryElectrolyte_app.register_callbacks()
    full_app.run(port=8054, debug=True)
