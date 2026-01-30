"""Run iron properties app."""

from __future__ import annotations

from dash import Dash, dcc
from dash.dcc import Loading
from dash.html import Div, Label

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Iron Properties"
DATA_PATH = APP_ROOT / "data" / "physicality" / "iron_properties"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/physicality.html#iron-properties"
)


class IronPropertiesApp(BaseApp):
    """Iron properties benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks for curve visualization."""
        pass  # Curve visualization to be added via plot_from_table_column


def get_app() -> IronPropertiesApp:
    """
    Get iron properties benchmark app layout and callback registration.

    Returns
    -------
    IronPropertiesApp
        Benchmark layout and callback registration.
    """
    model_options = [{"label": model, "value": model} for model in MODELS]
    default_model = model_options[0]["value"] if model_options else None

    extra_components = [
        Div(
            [
                Label("Select model for curve visualization:"),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-model-dropdown",
                    options=model_options,
                    value=default_model,
                    clearable=False,
                    style={"width": "300px", "marginBottom": "20px"},
                ),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-curve-dropdown",
                    options=[
                        {"label": "EOS Curve", "value": "eos"},
                        {"label": "Bain Path", "value": "bain"},
                        {"label": "SFE {110}<111>", "value": "sfe_110"},
                        {"label": "SFE {112}<111>", "value": "sfe_112"},
                        {"label": "(100)[010] K-E Curve", "value": "crack_1"},
                        {"label": "(100)[001] K-E Curve", "value": "crack_2"},
                        {"label": "(110)[001] K-E Curve", "value": "crack_3"},
                        {"label": "(110)[1-10] K-E Curve", "value": "crack_4"},
                    ],
                    value="eos",
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

    return IronPropertiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Comprehensive BCC iron properties benchmark. "
            "Includes equation of state (lattice parameter, bulk modulus), "
            "elastic constants (C11, C12, C44), Bain path (BCC-FCC transformation), "
            "vacancy formation energy, surface energies (100, 110, 111, 112), "
            "generalized stacking fault energy curves for {110}<111> and {112}<111> slip systems, "
            "dislocation core energies for 5 dislocation types (edge, mixed, screw), "
            "and crack K-tests for 4 crack systems. "
            "This benchmark is computationally expensive and marked with @pytest.mark.slow."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "iron_properties_metrics_table.json",
        extra_components=extra_components,
    )


if __name__ == "__main__":
    dash_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    iron_properties_app = get_app()
    dash_app.layout = iron_properties_app.layout
    iron_properties_app.register_callbacks()
    dash_app.run(port=8060, debug=True)
