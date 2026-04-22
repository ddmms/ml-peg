"""Dash app for LiTFSI/H2O electrolyte density benchmark."""

from __future__ import annotations

from pathlib import Path

from dash import html

from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "wise_electrolytes_density"
APP_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = APP_ROOT / "data" / "wise_electrolytes" / "density"


class DensityApp(BaseApp):
    """Dash app for WiSE electrolyte density benchmark."""

    def register_callbacks(self) -> None:
        """Register interactive plot callbacks."""
        bar_chart = read_plot(
            DATA_PATH / "figure_density_bar.json",
            id=f"{BENCHMARK_NAME}-bar",
        )
        timeseries = read_plot(
            DATA_PATH / "figure_density_timeseries.json",
            id=f"{BENCHMARK_NAME}-timeseries",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Density Error": bar_chart,
                "Density Error (%)": bar_chart,
                "Score": timeseries,
            },
        )


def get_app() -> DensityApp:
    """
    Return configured density benchmark app.

    Returns
    -------
    DensityApp
        Configured app instance.
    """
    return DensityApp(
        name="WiSE Density",
        description=(
            "NPT density of 21 m LiTFSI/H2O electrolyte at 298 K. "
            "Experimental reference: 1.72 g/cm³ (Maginn et al., 2021)."
        ),
        docs_url="",
        table_path=DATA_PATH / "density_metrics_table.json",
        extra_components=[
            html.Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    app = get_app()
    app.run(port=8060, debug=True)
