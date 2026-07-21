"""Run diatomics app."""

from __future__ import annotations

from dash import Dash, dcc
from dash.dcc import Loading
from dash.html import Div, Label

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.physicality.diatomics.download_utils import (
    register_data_download_callbacks,
)
from ml_peg.app.utils.build_callbacks import register_image_gallery_callbacks
from ml_peg.app.utils.build_components import build_data_download_controls
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Diatomics"
DATA_PATH = APP_ROOT / "data" / "physicality" / "diatomics"
CURVE_PATH = DATA_PATH / "curves"
OVERVIEW_LABEL = "Homonuclear diatomics"
DATA_DOWNLOAD_ID = f"{BENCHMARK_NAME}-curve-data-download"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#diatomics"
)
INFO_PATH = DATA_PATH / "info.json"


class DiatomicsApp(BaseApp):
    """Diatomics benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register dropdown-driven image callbacks."""
        register_image_gallery_callbacks(
            model_dropdown_id=f"{BENCHMARK_NAME}-model-dropdown",
            element_dropdown_id=f"{BENCHMARK_NAME}-element-dropdown",
            figure_id=f"{BENCHMARK_NAME}-figure",
            manifest_dir=CURVE_PATH,
            overview_label=OVERVIEW_LABEL,
            curve_dir=CURVE_PATH,
        )
        register_data_download_callbacks(
            download_id=DATA_DOWNLOAD_ID,
            model_dropdown_id=f"{BENCHMARK_NAME}-model-dropdown",
            element_dropdown_id=f"{BENCHMARK_NAME}-element-dropdown",
            curve_path=CURVE_PATH,
            overview_label=OVERVIEW_LABEL,
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
                    options=[{"label": OVERVIEW_LABEL, "value": OVERVIEW_LABEL}],
                    value=OVERVIEW_LABEL,
                    clearable=False,
                    style={"width": "300px"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        build_data_download_controls(
            DATA_DOWNLOAD_ID,
            formats=(("CSV", "csv"), ("JSON", "json"), ("PNG", "png")),
        ),
        Loading(
            dcc.Graph(
                id=f"{BENCHMARK_NAME}-figure",
                style={"height": "700px", "width": "100%", "marginTop": "20px"},
                config={
                    "toImageButtonOptions": {
                        "format": "png",
                        "filename": "diatomics",
                        "width": 12000,
                        "height": 6000,
                        "scale": 1,
                    },
                },
            ),
            type="circle",
        ),
    ]

    return DiatomicsApp(
        name=BENCHMARK_NAME,
        framework_ids="mace-multihead",
        description=(
            "Diatomics explorer with periodic-table views. Metrics are averaged "
            "across all diatomic pairs (both homonuclear and heteronuclear)."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "diatomics_metrics_table.json",
        extra_components=extra_components,
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    dash_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    diatomics_app = get_app()
    dash_app.layout = diatomics_app.layout
    diatomics_app.register_callbacks()
    dash_app.run(port=8055, debug=True)
