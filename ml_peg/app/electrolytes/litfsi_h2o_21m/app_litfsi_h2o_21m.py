"""Dash app for the consolidated WiSE 21 m LiTFSI/H2O electrolyte benchmark."""

from __future__ import annotations

from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column, struct_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "electrolytes_litfsi_h2o_21m"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "electrolytes.html#litfsi-h2o-21-m"
)
DATA_PATH = APP_ROOT / "data" / "electrolytes" / "litfsi_h2o_21m"
INFO_PATH = DATA_PATH / "info.json"


class LitfsiH2O21mApp(BaseApp):
    """Dash app for the consolidated WiSE LiTFSI/H2O 21 m benchmark."""

    def register_callbacks(self) -> None:
        """Register interactive plot callbacks for all metrics."""
        density_bar = read_plot(
            DATA_PATH / "figure_density_bar.json",
            id=f"{BENCHMARK_NAME}-density-bar",
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
                "CN Li-O_total Error": cn_bar,
                "S(q) R-factor": sq_plot,
                "Score": gr_plot,
            },
        )

        # Clicking a bar of the density chart shows the last frame of that
        # model's NPT run, i.e. the cell at the density the model settles at.
        # The bars carry the registry name as customdata.
        structs = {
            model: f"/assets/electrolytes/litfsi_h2o_21m/{model}/npt_final.xyz"
            for model in MODELS
            if (DATA_PATH / model / "npt_final.xyz").exists()
        }
        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-density-bar",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
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
            "Structural and thermodynamic benchmark for the 21 m LiTFSI/H2O "
            "'water-in-salt' electrolyte: NPT density vs experiment "
            "(Gilbert 2017), Li-O coordination numbers from RDFs vs r2SCAN "
            "references at 21 m (Brugnoli 2026), and X-ray structure factor "
            "S(q) vs SAXS (Zhang 2021). Models not trained on dispersion are "
            "run with a D3(BJ) correction (-D3)."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "litfsi_h2o_21m_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    app = get_app()
    app.run(port=8060, debug=True)
