"""Run split vacancy benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Split vacancy"
# TODO: change DOCS_URL
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/bulk_crystal.html#lattice-constants"
DATA_PATH = APP_ROOT / "data" / "bulk_crystal" / "split_vacancy"


class SplitVacancyApp(BaseApp):
    """Split vacancy benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_dft = read_plot(
            DATA_PATH / "figure_formation_energies_dft.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "MAE": scatter_dft,
                "Mean Spearman's Coefficient": scatter_dft,
                "RMSD": scatter_dft,
            },
        )


def get_app() -> SplitVacancyApp:
    """
    Get split vacancy benchmark app layout and callback registration.

    Returns
    -------
    SplitVacancyApp
        Benchmark layout and callback registration.
    """
    return SplitVacancyApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance predicting the formation energy of split "
            "vacancies from regular vacancies in XX systems."  # TODO: add number
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "split_vacancy_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    split_vacancy_app = get_app()
    full_app.layout = split_vacancy_app.layout
    split_vacancy_app.register_callbacks()
    full_app.run(port=8054, debug=True)
