"""Run BDEs app."""

from __future__ import annotations

from dash import Input, Output, callback
from dash.html import Div, Iframe

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    struct_from_scatter,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.weas import generate_weas_html
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Bond Dissociation Energies"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular.html#bdes"
DATA_PATH = APP_ROOT / "data" / "molecular" / "BDEs"


class BDEsApp(BaseApp):
    """BDEs benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter_bdes = read_plot(
            DATA_PATH / "figure.CYP3A4.dft_opt_geometry.BDEs.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        scatter_ranks = read_plot(
            DATA_PATH / "figure.CYP3A4.dft_opt_geometry.BDE_ranks.json",
            id=f"{BENCHMARK_NAME}-ranks-figure",
        )
        scatter_mlff_bdes = read_plot(
            DATA_PATH / "figure.CYP3A4.mlff_opt_geometry.BDEs.json",
            id=f"{BENCHMARK_NAME}-mlff-figure",
        )
        scatter_mlff_ranks = read_plot(
            DATA_PATH / "figure.CYP3A4.mlff_opt_geometry.BDE_ranks.json",
            id=f"{BENCHMARK_NAME}-mlff-ranks-figure",
        )

        def _get_structs(suffix: str) -> list[str]:
            """
            Return asset paths for the first model with saved structure files.

            Parameters
            ----------
            suffix
                Geometry suffix, e.g. ``"dft_opt"`` or ``"mlff_opt"``.

            Returns
            -------
            list[str]
                Asset URL paths for each structure file, or empty list if none found.
            """
            for model_name in MODELS:
                structs_dir = DATA_PATH / model_name / suffix
                xyz_files = sorted(structs_dir.glob("*.xyz"), key=lambda p: int(p.stem))
                if xyz_files:
                    return [
                        f"assets/molecular/BDEs/{model_name}/{suffix}/{f.stem}.xyz"
                        for f in xyz_files
                    ]
            return []

        def _get_structs_per_model(suffix: str) -> dict[str, list[str]]:
            """
            Return asset paths for each model that has saved structure files.

            Parameters
            ----------
            suffix
                Geometry suffix, e.g. ``"dft_opt"`` or ``"mlff_opt"``.

            Returns
            -------
            dict[str, list[str]]
                Mapping of model name to asset URL paths for its structure files.
            """
            result = {}
            for model_name in MODELS:
                structs_dir = DATA_PATH / model_name / suffix
                xyz_files = sorted(structs_dir.glob("*.xyz"), key=lambda p: int(p.stem))
                if xyz_files:
                    result[model_name] = [
                        f"assets/molecular/BDEs/{model_name}/{suffix}/{f.stem}.xyz"
                        for f in xyz_files
                    ]
            return result

        structs_dft = _get_structs("dft_opt")
        structs_mlff_by_model = _get_structs_per_model("mlff_opt")

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Direct BDE": scatter_bdes,
                "BDE rank": scatter_ranks,
                "Direct BDE (MLFF opt)": scatter_mlff_bdes,
                "BDE rank (MLFF opt)": scatter_mlff_ranks,
            },
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs_dft,
            mode="struct",
        )

        struct_from_scatter(
            scatter_id=f"{BENCHMARK_NAME}-ranks-figure",
            struct_id=f"{BENCHMARK_NAME}-struct-placeholder",
            structs=structs_dft,
            mode="struct",
        )

        def _show_mlff_struct_from_click(click_data):
            """
            Return structure viewer for the clicked MLFF scatter point.

            Parameters
            ----------
            click_data
                Clicked data point in scatter plot.

            Returns
            -------
            Div
                Visualised structure on plot click.
            """
            if not click_data:
                return Div("Click on a metric to view the structure.")
            point = click_data["points"][0]
            idx = point["pointNumber"]
            curve_num = point["curveNumber"]
            if curve_num >= len(MODELS):
                return Div("No structures available for this model.")
            model_name = MODELS[curve_num]
            structs = structs_mlff_by_model.get(model_name, [])
            if not structs:
                return Div("No structures available for this model.")
            return Div(
                Iframe(
                    srcDoc=generate_weas_html(structs[idx], "struct", 0),
                    style={
                        "height": "550px",
                        "width": "100%",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                    },
                )
            )

        @callback(
            Output(
                f"{BENCHMARK_NAME}-struct-placeholder", "children", allow_duplicate=True
            ),
            Input(f"{BENCHMARK_NAME}-mlff-figure", "clickData"),
            prevent_initial_call="initial_duplicate",
        )
        def show_mlff_struct(click_data):
            """
            Show structure for the clicked MLFF BDE scatter point.

            Parameters
            ----------
            click_data
                Clicked data point in scatter plot.

            Returns
            -------
            Div
                Visualised structure on plot click.
            """
            return _show_mlff_struct_from_click(click_data)

        @callback(
            Output(
                f"{BENCHMARK_NAME}-struct-placeholder", "children", allow_duplicate=True
            ),
            Input(f"{BENCHMARK_NAME}-mlff-ranks-figure", "clickData"),
            prevent_initial_call="initial_duplicate",
        )
        def show_mlff_rank_struct(click_data):
            """
            Show structure for the clicked MLFF BDE rank scatter point.

            Parameters
            ----------
            click_data
                Clicked data point in scatter plot.

            Returns
            -------
            Div
                Visualised structure on plot click.
            """
            return _show_mlff_struct_from_click(click_data)


def get_app() -> BDEsApp:
    """
    Get bond dissociation energy benchmark app.

    Returns
    -------
    BDEsApp
        Benchmark layout and callback registration.
    """
    return BDEsApp(
        name=BENCHMARK_NAME,
        description="Bond Dissociation Energies",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "metrics_table.CYP3A4.dft_opt_geometry.BDEs.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )
