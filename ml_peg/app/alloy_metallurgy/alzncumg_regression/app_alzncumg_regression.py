"""Run Al-Cu-Mg-Zn metallurgy regression benchmark app."""

from __future__ import annotations

from dash import Dash
from dash.html import Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column, struct_from_scatter
from ml_peg.app.utils.load import read_plot
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Al-Zn-Cu-Mg regression"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/alloy_metallurgy.html#al-zn-cu-mg-regression"
DATA_PATH = APP_ROOT / "data" / "alloy_metallurgy" / "alzncumg_regression"


class AlZnCuMgRegressionApp(BaseApp):
    """Al-Cu-Mg-Zn metallurgy benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        formation_scatter = read_plot(
            DATA_PATH / "figure_formation_energy.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        volume_scatter = read_plot(
            DATA_PATH / "figure_volume_peratom.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        structs_dir = next(
            (
                DATA_PATH / model_name
                for model_name in MODELS
                if (DATA_PATH / model_name).exists()
            ),
            DATA_PATH / MODELS[0],
        )
        structs = [
            f"/assets/alloy_metallurgy/alzncumg_regression/{structs_dir.name}/{struct_file.name}"
            for struct_file in sorted(structs_dir.glob("OQMD_*.xyz"))
        ]

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Formation Energy MAE": formation_scatter,
                "Volume MAE": volume_scatter,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> AlZnCuMgRegressionApp:
    """
    Get Al-Cu-Mg-Zn metallurgy benchmark layout and callback registration.

    Returns
    -------
    AlZnCuMgRegressionApp
        Benchmark layout and callback registration.
    """
    return AlZnCuMgRegressionApp(
        name=BENCHMARK_NAME,
        description=(
            "Bulk formation-energy and volume errors for the first staged "
            "Al-Cu-Mg-Zn OQMD structure slice."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "alzncumg_regression_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    alzncumg_app = get_app()
    full_app.layout = alzncumg_app.layout
    alzncumg_app.register_callbacks()
    full_app.run(port=8054, debug=True)
