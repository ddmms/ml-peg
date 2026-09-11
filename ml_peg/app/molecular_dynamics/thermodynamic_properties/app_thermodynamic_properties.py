"""Run thermodynamic properties app."""

from __future__ import annotations

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
BENCHMARK_NAME = "Thermodynamic Properties"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "molecular_dynamics.html#thermodynamic-properties"
)
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "thermodynamic_properties"
INFO_PATH = DATA_PATH / "info.json"


class ThermodynamicPropertiesApp(BaseApp):
    """Thermodynamic properties benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        plots = {
            "density_MAE": read_plot(
                DATA_PATH / "figure_density.json",
                id=f"{BENCHMARK_NAME}-density-figure",
            ),
            "cp_MAE": read_plot(
                DATA_PATH / "figure_cp.json",
                id=f"{BENCHMARK_NAME}-cp-figure",
            ),
            "evaporation_enthalpy_MAE": read_plot(
                DATA_PATH / "figure_evaporation_enthalpy.json",
                id=f"{BENCHMARK_NAME}-evaporation-enthalpy-figure",
            ),
            "compressibility_MAE": read_plot(
                DATA_PATH / "figure_compressibility.json",
                id=f"{BENCHMARK_NAME}-compressibility-figure",
            ),
            "alpha_MAE": read_plot(
                DATA_PATH / "figure_alpha.json",
                id=f"{BENCHMARK_NAME}-alpha-figure",
            ),
        }

        model = plots["density_MAE"].figure.data[0].name
        model_dir = DATA_PATH / model

        if model_dir.exists():
            labels = sorted(f.stem for f in model_dir.glob("*-liq.xyz"))
            structs = [
                (
                    "/assets/molecular_dynamics/"
                    f"thermodynamic_properties/{model}/{label}.xyz"
                )
                for label in labels
            ]
        else:
            structs = []

        print("DEBUG model:", model)
        print("DEBUG structs:", structs)

        print("Density graph ID:", f"{BENCHMARK_NAME}-density-figure")
        print("Model:", MODELS[0])
        print("Model directory:", model_dir)
        print("Structures:", structs)

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot=plots,
        )

        print("DEBUG MODELS:", MODELS)
        print("DEBUG structs:", structs)
        print("DEBUG len(structs):", len(structs))

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-density-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs,
            mode="struct",
        )


def get_app() -> ThermodynamicPropertiesApp:
    """
    Get thermodynamic properties benchmark app layout and callback registration.

    Returns
    -------
    ThermodynamicPropertiesApp
        Benchmark layout and callback registration.
    """
    return ThermodynamicPropertiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting thermodynamic properties of organic liquids."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "thermodynamic_properties_metrics_table.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
        framework_ids="ml_peg",
    )
