"""Run the YBCO point-defect formation-energy benchmark app."""

from __future__ import annotations

from dash.dcc import Graph
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_multi_scatters,
)
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "YBCO defects"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "defects.html#ybco-point-defect-formation-energies"
)
DATA_PATH = APP_ROOT / "data" / "defects" / "YBCO_defects"
INFO_PATH = DATA_PATH / "info.json"


def _model_structs(scatter: Graph, asset_dir: str) -> list[list[str]]:
    """
    Get relaxed structure paths for each model trace and point of a parity plot.

    Parameters
    ----------
    scatter
        Parity plot, with one trace per model and point labels as the last
        ``customdata`` entry.
    asset_dir
        Asset directory containing a subdirectory of structures per model.

    Returns
    -------
    list[list[str]]
        Structure paths indexed by trace then point, empty where unavailable.
    """
    structs = []
    for trace in (scatter.figure or {}).get("data", []):
        model = trace.get("name")
        structs.append(
            [
                f"/assets/{asset_dir}/{model}/{point[-1]}.xyz"
                if (DATA_PATH / model / f"{point[-1]}.xyz").exists()
                else ""
                for point in trace.get("customdata") or []
            ]
        )
    return structs


class YBCODefectsApp(BaseApp):
    """YBCO defect formation-energy benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_ybco_defects.json", id=f"{BENCHMARK_NAME}-figure"
        )
        scatter_reactions = read_plot(
            DATA_PATH / "figure_ybco_reactions.json",
            id=f"{BENCHMARK_NAME}-reactions-figure",
        )

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

        struct_from_multi_scatters(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=_model_structs(scatter, "defects/YBCO_defects"),
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
