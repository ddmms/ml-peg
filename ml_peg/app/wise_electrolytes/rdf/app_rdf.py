"""Dash app for LiTFSI/H2O Li-O RDF / coordination number benchmark."""

from __future__ import annotations

from pathlib import Path

from dash import html

from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "wise_electrolytes_rdf"
APP_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = APP_ROOT / "data" / "wise_electrolytes" / "rdf"


class RdfApp(BaseApp):
    """Dash app for WiSE electrolyte Li-O RDF benchmark."""

    def register_callbacks(self) -> None:
        """Register interactive plot callbacks."""
        cn_bar = read_plot(
            DATA_PATH / "figure_cn_bar.json",
            id=f"{BENCHMARK_NAME}-cn-bar",
        )
        gr_plot = read_plot(
            DATA_PATH / "figure_gr.json",
            id=f"{BENCHMARK_NAME}-gr",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "CN Li-O_water Error": cn_bar,
                "CN Li-O_TFSI Error": cn_bar,
                "Score": gr_plot,
            },
        )


def get_app() -> RdfApp:
    """
    Return configured RDF benchmark app.

    Returns
    -------
    RdfApp
        Configured app instance.
    """
    return RdfApp(
        name="WiSE Li-O RDF",
        description=(
            "Li-O coordination numbers from radial distribution functions "
            "of 21 m LiTFSI/H2O electrolyte (NVT, 298 K). "
            "Reference: Watanabe et al., J. Phys. Chem. B 125, 7477 (2021)."
        ),
        docs_url="",
        table_path=DATA_PATH / "rdf_metrics_table.json",
        extra_components=[
            html.Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    app = get_app()
    app.run(port=8062, debug=True)
