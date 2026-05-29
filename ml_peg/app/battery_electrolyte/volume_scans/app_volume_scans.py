"""Run Volume Scans Benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_cell,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

BENCHMARK_NAME = "Volume-Scans"
# DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/battery_electrolyte.html#volume_scans"
DATA_PATH = APP_ROOT / "data" / "battery_electrolyte" / "volume_scans"


class VolumeScansApp(BaseApp):
    """Volume Scans benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_plots = {
            model: {
                "Solvent": read_plot(
                    DATA_PATH / f"solvent_{model}_volscan_scatter.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-solventVS",
                ),
                "Electrolyte": read_plot(
                    DATA_PATH / f"electrolyte_{model}_volscan_scatter.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-electrolyteVS",
                ),
            }
            for model in MODELS
        }

        # Assets dir will be parent directory
        assets_dir = "assets/battery_electrolyte/volume_scans"
        structs = {
            model: {
                "Solvent": f"{assets_dir}/{model}/solvent_VS_{model}_D3.extxyz",
                "Electrolyte": f"{assets_dir}/{model}/electrolyte_VS_{model}_D3.extxyz",
            }
            for model in MODELS
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
                    structs=structs[model][volscan.capitalize()],
                    mode="traj",
                )


def get_app() -> VolumeScansApp:
    """
    Get Volume Scan benchmark app layout and callback registration.

    Returns
    -------
    VolumeScansApp
        Benchmark layout and callback registration.
    """
    return VolumeScansApp(
        name=BENCHMARK_NAME,
        description=(
            "Evaluate model energy predictions on "
            "battery solvent and electrolyte Volume Scans"
        ),
        # docs_url=DOCS_URL,
        table_path=DATA_PATH / "vol_scan_rmses_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(
        __name__,
        assets_folder=DATA_PATH.parent.parent,
        suppress_callback_exceptions=True,
    )

    # Construct layout and register callbacks
    VolumeScan_app = get_app()
    full_app.layout = VolumeScan_app.layout
    VolumeScan_app.register_callbacks()

    # Run app
    full_app.run(port=8054, debug=True)
