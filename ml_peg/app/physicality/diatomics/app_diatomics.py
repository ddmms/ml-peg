"""Run diatomics app."""

from __future__ import annotations

from dash import Dash, dcc
from dash.dcc import Loading
from dash.html import Div, Label

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import register_image_gallery_callbacks
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
BENCHMARK_NAME = "diatomics"
DATA_PATH = APP_ROOT / "data" / "physicality" / "diatomics"
CURVE_PATH = DATA_PATH / "curves"


class DiatomicsApp(BaseApp):
    """Diatomics benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register dropdown-driven image callbacks."""
        register_image_gallery_callbacks(
            model_dropdown_id=f"{BENCHMARK_NAME}-model-dropdown",
            element_dropdown_id=f"{BENCHMARK_NAME}-element-dropdown",
            figure_id=f"{BENCHMARK_NAME}-figure",
            manifest_dir=CURVE_PATH,
            overview_label="Homonuclear diatomics",
            curve_dir=CURVE_PATH,
        )


def get_app() -> DiatomicsApp:
    """
    Get diatomics benchmark app layout and callback registration.

    Returns
    -------
    DiatomicsApp
        Benchmark layout and callback registration.
    """
    model_options = [{"label": model, "value": model} for model in MODELS]
    default_model = model_options[0]["value"]
    overview_label = "Homonuclear diatomics"

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
                Label("Select heteronuclear element:"),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-element-dropdown",
                    options=[{"label": overview_label, "value": overview_label}],
                    value=overview_label,
                    clearable=False,
                    style={"width": "300px"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        Loading(
            dcc.Graph(
                id=f"{BENCHMARK_NAME}-figure",
                style={"height": "700px", "width": "100%", "marginTop": "20px"},
            ),
            type="circle",
        ),
    ]

    return DiatomicsApp(
        name=BENCHMARK_NAME,
        description="Homonuclear diatomic explorer with periodic-table views.",
        docs_url=None,
        table_path=DATA_PATH / "diatomics_metrics_table.json",
        extra_components=extra_components,
    )


if __name__ == "__main__":
    dash_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    diatomics_app = get_app()
    dash_app.layout = diatomics_app.layout
    diatomics_app.register_callbacks()
    dash_app.run(port=8055, debug=True)
