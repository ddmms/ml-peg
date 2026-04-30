"""Run compression benchmark app."""

from __future__ import annotations

from dash import Dash, Input, Output, callback, dcc
from dash.dcc import Loading
from dash.exceptions import PreventUpdate
from dash.html import Div, Label
from plotly.io import read_json

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Compression"
DATA_PATH = APP_ROOT / "data" / "physicality" / "compression"
FIGURE_PATH = DATA_PATH / "figures"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#compression"
)


def _available_formulas(model_name: str) -> list[str]:
    """
    List unique formulas available for a given model.

    Parameters
    ----------
    model_name
        Selected model identifier.

    Returns
    -------
    list[str]
        Sorted list of unique formulas.
    """
    model_dir = FIGURE_PATH / model_name
    if not model_dir.exists():
        return []
    return sorted(p.stem for p in model_dir.glob("*.json"))


class CompressionApp(BaseApp):
    """Compression benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register dropdown-driven compression curve callbacks."""
        model_dropdown_id = f"{BENCHMARK_NAME}-model-dropdown"
        composition_dropdown_id = f"{BENCHMARK_NAME}-composition-dropdown"
        figure_id = f"{BENCHMARK_NAME}-figure"

        @callback(
            Output(composition_dropdown_id, "options"),
            Output(composition_dropdown_id, "value"),
            Input(model_dropdown_id, "value"),
        )
        def _update_composition_options(model_name: str):
            """
            Update composition dropdown options based on selected model.

            Parameters
            ----------
            model_name
                Currently selected model identifier.

            Returns
            -------
            tuple
                Dropdown options list and default value.
            """
            if not model_name:
                raise PreventUpdate
            formulas = _available_formulas(model_name)
            options = [{"label": f, "value": f} for f in formulas]
            default = formulas[0] if formulas else None
            return options, default

        @callback(
            Output(figure_id, "figure"),
            Input(model_dropdown_id, "value"),
            Input(composition_dropdown_id, "value"),
        )
        def _update_figure(model_name: str, composition: str | None):
            """
            Load pre-built energy-per-atom vs scale factor figure.

            Parameters
            ----------
            model_name
                Currently selected model identifier.
            composition
                Reduced chemical formula for the composition group.

            Returns
            -------
            Figure
                Plotly figure loaded from the pre-built JSON file.
            """
            if not model_name or not composition:
                raise PreventUpdate

            figure_file = FIGURE_PATH / model_name / f"{composition}.json"
            if not figure_file.exists():
                raise PreventUpdate

            return read_json(figure_file)


def get_app() -> CompressionApp:
    """
    Get compression benchmark app layout and callback registration.

    Returns
    -------
    CompressionApp
        Benchmark layout and callback registration.
    """
    model_options = [{"label": model, "value": model} for model in MODELS]
    default_model = model_options[0]["value"] if model_options else None

    extra_components = [
        Div(
            [
                Label("Select model:"),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-model-dropdown",
                    options=model_options,
                    value=default_model,
                    clearable=False,
                    style={"width": "300px", "marginBottom": "20px"},
                ),
                Label("Select composition:"),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-composition-dropdown",
                    options=[],
                    value=None,
                    clearable=False,
                    style={"width": "300px"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        Loading(
            dcc.Graph(
                id=f"{BENCHMARK_NAME}-figure",
                style={"height": "500px", "width": "100%", "marginTop": "20px"},
            ),
            type="circle",
        ),
    ]

    return CompressionApp(
        name=BENCHMARK_NAME,
        description=(
            "A handful of common prototype (and randomly generated via pyxtal)"
            "structures are isotropically scaled across a wide range."
            "A scale factor of 1.0 means that a pair of atoms in the"
            "structure is separated by the sum of their covalent radii. "
            "e.g. min(d_ij/(r_cov_i + r_cov_j)) = 1.0."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "compression_metrics_table.json",
        extra_components=extra_components,
    )


if __name__ == "__main__":
    dash_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    compression_app = get_app()
    dash_app.layout = compression_app.layout
    compression_app.register_callbacks()
    dash_app.run(port=8056, debug=True)
