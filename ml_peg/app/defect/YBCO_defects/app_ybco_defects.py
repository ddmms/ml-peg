"""Run the YBCO point-defect formation-energy benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "YBCO defects"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "defect.html#ybco-point-defect-formation-energies"
)
DATA_PATH = APP_ROOT / "data" / "defect" / "YBCO_defects"
INFO_PATH = DATA_PATH / "info.json"


class YBCODefectsApp(BaseApp):
    """YBCO defect formation-energy benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_ybco_defects.json", id=f"{BENCHMARK_NAME}-figure"
        )
        scatter_reactions = read_plot(
            DATA_PATH / "figure_ybco_reactions.json", id=f"{BENCHMARK_NAME}-figure"
        )

        # one structure per scatter point, from the first model
        structs_dir = DATA_PATH / MODELS[0]
        structs = [
            f"/assets/defect/YBCO_defects/{MODELS[0]}/{f.stem}.xyz"
            for f in sorted(structs_dir.glob("*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "RMSD vacancy": scatter,
                "RMSD antisite": scatter,
                "RMSD interstitial": scatter,
                "RMSD reactions": scatter_reactions,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> YBCODefectsApp:
    """
    Get YBCO defect benchmark app layout and callback registration.

    Returns
    -------
    YBCODefectsApp
        Benchmark layout and callback registration.
    """
    return YBCODefectsApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance when predicting YBa2Cu3O7 point-defect formation energies "
            "(O/Cu/Ba/Y vacancies, antisites and oxygen interstitials) against CP2K "
            "PBE, using elemental chemical potentials."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "ybco_defects_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    ybco_defects_app = get_app()
    full_app.layout = ybco_defects_app.layout
    ybco_defects_app.register_callbacks()
    full_app.run(port=8056, debug=True)
