"""Dash app for the consolidated WiSE 21 m LiTFSI/H2O electrolyte benchmark."""

from __future__ import annotations

from pathlib import Path

from dash import html

from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "wise_electrolytes_litfsi_h2o_21m"
APP_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = APP_ROOT / "data" / "wise_electrolytes" / "litfsi_h2o_21m"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "wise_electrolytes.html#litfsi-h2o-21-m"
)


class LitfsiH2O21mApp(BaseApp):
    """Dash app for the consolidated WiSE LiTFSI/H2O 21 m benchmark."""

    def register_callbacks(self) -> None:
        """Register interactive plot callbacks for all metrics."""
        density_bar = read_plot(
            DATA_PATH / "figure_density_bar.json",
            id=f"{BENCHMARK_NAME}-density-bar",
        )
        density_timeseries = read_plot(
            DATA_PATH / "figure_density_timeseries.json",
            id=f"{BENCHMARK_NAME}-density-timeseries",
        )
        cn_bar = read_plot(
            DATA_PATH / "figure_cn_bar.json",
            id=f"{BENCHMARK_NAME}-cn-bar",
        )
        gr_plot = read_plot(
            DATA_PATH / "figure_gr.json",
            id=f"{BENCHMARK_NAME}-gr",
        )
        sq_plot = read_plot(
            DATA_PATH / "figure_xray_sq_comparison.json",
            id=f"{BENCHMARK_NAME}-sq",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Density Error": density_bar,
                "Density Error (%)": density_bar,
                "CN Li-O_water Error": cn_bar,
                "CN Li-O_TFSI Error": cn_bar,
                "S(q) R-factor": sq_plot,
                "First Peak Position Error": sq_plot,
                "Score": gr_plot,
            },
        )


def get_app() -> LitfsiH2O21mApp:
    """
    Return the configured WiSE LiTFSI/H2O 21 m benchmark app.

    Returns
    -------
    LitfsiH2O21mApp
        Configured app instance.
    """
    return LitfsiH2O21mApp(
        name="WiSE LiTFSI/H2O 21 m",
        description=(
            "Consolidated structural and thermodynamic benchmark for the "
            "21 m LiTFSI/H2O 'water-in-salt' electrolyte: NPT density vs "
            "experiment (Gilbert 2017), Li-O coordination numbers from RDFs "
            "(Watanabe 2021), and X-ray structure factor S(q) (Zhang 2021). "
            "Trajectories produced with LAMMPS+symmetrix on Adastra (MI250X)."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "litfsi_h2o_21m_metrics_table.json",
        extra_components=[
            html.Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    app = get_app()
    app.run(port=8060, debug=True)
