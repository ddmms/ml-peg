"""Run app for Si defects benchmark."""

from __future__ import annotations

from dataclasses import dataclass

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell, struct_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)

BENCHMARK_NAME = "Si defects"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/nebs.html#si-defects"
DATA_PATH = APP_ROOT / "data" / "nebs" / "si_defects"


@dataclass(frozen=True)
class _Case:
    """Definition of a single Si defects NEB dataset."""

    key: str
    label: str


CASES: tuple[_Case, ...] = (
    _Case(key="64_atoms", label="64"),
    _Case(key="216_atoms", label="216"),
    _Case(key="216_atoms_di_to_single", label="216 di-to-single"),
)


class SiDefectNebSinglepointsApp(BaseApp):
    """Si defects benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register interactive callbacks for plot and structure viewing."""
        scatter_plots: dict[str, dict] = {}

        for model in MODELS:
            model_plots: dict[str, object] = {}
            for case in CASES:
                energy_plot_path = (
                    DATA_PATH / f"figure_{model}_{case.key}_energy_error.json"
                )
                force_plot_path = (
                    DATA_PATH / f"figure_{model}_{case.key}_force_rms.json"
                )
                if energy_plot_path.exists():
                    model_plots[f"Energy MAE ({case.label})"] = read_plot(
                        energy_plot_path,
                        id=f"{BENCHMARK_NAME}-{model}-{case.key}-energy-figure",
                    )
                if force_plot_path.exists():
                    model_plots[f"Force MAE ({case.label})"] = read_plot(
                        force_plot_path,
                        id=f"{BENCHMARK_NAME}-{model}-{case.key}-force-figure",
                    )
            if model_plots:
                scatter_plots[model] = model_plots

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=scatter_plots,
        )

        for model in scatter_plots:
            for case in CASES:
                structs = (
                    f"assets/nebs/si_defects/{case.key}/{model}/{model}-neb-band.extxyz"
                )
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{case.key}-energy-figure",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=structs,
                    mode="traj",
                )
                struct_from_scatter(
                    scatter_id=f"{BENCHMARK_NAME}-{model}-{case.key}-force-figure",
                    struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
                    structs=structs,
                    mode="traj",
                )


def get_app() -> SiDefectNebSinglepointsApp:
    """
    Get Si defects app.

    Returns
    -------
    SiDefectNebSinglepointsApp
        App instance.
    """
    return SiDefectNebSinglepointsApp(
        name=BENCHMARK_NAME,
        description=(
            "Energy/force MAE of MLIPs on fixed Si interstitial migration NEB images, "
            "referenced to DFT singlepoints."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "si_defects_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Use APP_ROOT/data as assets root so `assets/nebs/...` resolves correctly.
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8060, debug=True)
