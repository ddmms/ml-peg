"""Run rotational symmetry app."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_cell
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Rotational Symmetry"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#rotational-symmetry"
DATA_PATH = APP_ROOT / "data" / "physicality" / "rotational_symmetry"
INFO_PATH = DATA_PATH / "info.json"

# Must match the column names built in analyse_rotational_symmetry.metrics:
# a mismatch leaves the table rendering correctly but silently stops cells
# opening their plot.
ENERGY_COLUMNS = [
    "Mean ΔE",
    "Max ΔE",
]
FORCE_COLUMNS = [
    "Mean ΔF",
    "Max ΔF",
]


class RotationalSymmetryApp(BaseApp):
    """Rotational symmetry benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        cell_to_plot = {}
        for model in MODELS:
            energy_path = DATA_PATH / f"{model}_energy_by_angle.json"
            force_path = DATA_PATH / f"{model}_force_by_angle.json"
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


def get_app() -> RotationalSymmetryApp:
    """
    Get rotational symmetry benchmark app layout and callback registration.

    Returns
    -------
    RotationalSymmetryApp
        Benchmark layout and callback registration.
    """
    return RotationalSymmetryApp(
        name=BENCHMARK_NAME,
        description=(
            "Energy and force changes between successive rigid rotations, "
            "over 105 orientation pairs."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "rotational_symmetry_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
        info_path=INFO_PATH,
    )
