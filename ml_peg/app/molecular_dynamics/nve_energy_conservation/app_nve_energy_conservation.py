"""Run NVE energy conservation benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

BENCHMARK_NAME = "NVEEnergyConservation"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#nve-energy-conservation"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "nve_energy_conservation"

METRICS = ("Energy Drift", "Energy Drift Ratio", "Systems Completed", "NVE Score")


class NVEEnergyConservationApp(BaseApp):
    """NVE energy conservation benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        drift_plots = {
            model: dict.fromkeys(
                METRICS,
                read_plot(
                    DATA_PATH / f"{model}_drift_scatter.json",
                    id=f"{BENCHMARK_NAME}-{model}-figure-drift",
                ),
            )
            for model in MODELS
        }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=drift_plots,
        )


def get_app() -> NVEEnergyConservationApp:
    """
    Get NVE energy conservation benchmark app layout and callback registration.

    Returns
    -------
    NVEEnergyConservationApp
        Benchmark layout and callback registration.
    """
    return NVEEnergyConservationApp(
        name="NVE Energy Conservation",
        framework_ids="mlip_audit",
        description=(
            "Performance in conserving the total energy during microcanonical "
            "(NVE) molecular dynamics of a small molecule in vacuum, bulk water, "
            "and two solvated peptides. Without a thermostat the total energy is "
            "conserved by the dynamics, so any drift comes from the potential."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "nve_energy_conservation_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8070, debug=True)
