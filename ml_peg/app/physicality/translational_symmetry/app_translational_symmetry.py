"""Run translational symmetry app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Translational Symmetry"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#translational-symmetry"
DATA_PATH = APP_ROOT / "data" / "physicality" / "translational_symmetry"
INFO_PATH = DATA_PATH / "info.json"

# Must match the column names built in analyse_translational_symmetry.metrics:
# a mismatch leaves the table rendering correctly but silently stops cells
# opening their plot.
ENERGY_COLUMNS = [
    "Mean ΔE (1 Å)",
    "Max ΔE (1 Å)",
    "Mean ΔE (40 Å)",
    "Max ΔE (40 Å)",
    "Mean ΔE (1000 Å)",
    "Max ΔE (1000 Å)",
]
FORCE_COLUMNS = [
    "Mean ΔF (1 Å)",
    "Max ΔF (1 Å)",
    "Mean ΔF (40 Å)",
    "Max ΔF (40 Å)",
    "Mean ΔF (1000 Å)",
    "Max ΔF (1000 Å)",
]


class TranslationalSymmetryApp(BaseApp):
    """Translational symmetry benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        cell_to_plot = {}
        for model in MODELS:
            energy_path = DATA_PATH / f"{model}_energy_by_structure.json"
            force_path = DATA_PATH / f"{model}_force_by_structure.json"
            if not (energy_path.exists() and force_path.exists()):
                continue

            energy_plot = read_plot(
                energy_path, id=f"{BENCHMARK_NAME}-{model}-figure-energy"
            )
            force_plot = read_plot(
                force_path, id=f"{BENCHMARK_NAME}-{model}-figure-force"
            )
            cell_to_plot[model] = {
                **dict.fromkeys(ENERGY_COLUMNS, energy_plot),
                **dict.fromkeys(FORCE_COLUMNS, force_plot),
            }

        plot_from_table_cell(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            cell_to_plot=cell_to_plot,
        )


def get_app() -> TranslationalSymmetryApp:
    """
    Get translational symmetry benchmark app layout and callback registration.

    Returns
    -------
    TranslationalSymmetryApp
        Benchmark layout and callback registration.
    """
    return TranslationalSymmetryApp(
        name=BENCHMARK_NAME,
        description=(
            "Energy and force changes after a rigid translation, at two magnitudes."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "translational_symmetry_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
        info_path=INFO_PATH,
    )
