"""Dash app for LiTFSI/H2O X-ray structure factor benchmark."""

from __future__ import annotations

from pathlib import Path

from dash import html

from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

BENCHMARK_NAME = "wise_electrolytes_xray_sf"
APP_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = APP_ROOT / "data" / "wise_electrolytes" / "xray_sf"


class XraySfApp(BaseApp):
    """Dash app for WiSE electrolyte X-ray S(q) benchmark."""

    def register_callbacks(self) -> None:
        """Register interactive plot callbacks."""
        sq_plot = read_plot(
            DATA_PATH / "figure_xray_sq_comparison.json",
            id=f"{BENCHMARK_NAME}-sq",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "S(q) R-factor": sq_plot,
                "First Peak Position Error": sq_plot,
                "Score": sq_plot,
            },
        )


def get_app() -> XraySfApp:
    """
    Return configured X-ray S(q) benchmark app.

    Returns
    -------
    XraySfApp
        Configured app instance.
    """
    return XraySfApp(
        name="WiSE X-ray S(q)",
        description=(
            "X-ray structure factor S(q) of 21 m LiTFSI/H2O electrolyte from NVT MD. "
            "Computed via dynasor with Cromer-Mann form factors "
            "(Faber-Ziman normalization). "
            "Experimental reference: SAXS (Maginn et al., J. Phys. Chem. B, 2021)."
        ),
        docs_url="",
        table_path=DATA_PATH / "xray_sf_metrics_table.json",
        extra_components=[
            html.Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )


if __name__ == "__main__":
    app = get_app()
    app.run(port=8061, debug=True)
