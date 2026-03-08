"""Run compression benchmark app."""

from __future__ import annotations

import json
from pathlib import Path

from dash import Dash, Input, Output, callback, dcc
from dash.dcc import Loading
from dash.exceptions import PreventUpdate
from dash.html import Div, Label
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Compression"
DATA_PATH = APP_ROOT / "data" / "physicality" / "compression"
CURVE_PATH = DATA_PATH / "curves"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "physicality.html#compression"
)


def _available_structures(model_name: str) -> list[str]:
    """
    List structure labels available for a given model.

    Parameters
    ----------
    model_name
        Selected model identifier.

    Returns
    -------
    list[str]
        Sorted list of structure labels with stored curve data.
    """
    model_dir = CURVE_PATH / model_name
    if not model_dir.exists():
        return []
    return sorted(p.stem for p in model_dir.glob("*.json"))


def _load_curve(model_name: str, structure: str) -> dict | None:
    """
    Load a single curve payload from disk.

    Parameters
    ----------
    model_name
        Model identifier.
    structure
        Structure label (filename stem).

    Returns
    -------
    dict | None
        Curve payload dict, or None if the file is missing / unreadable.
    """
    filepath = CURVE_PATH / model_name / f"{structure}.json"
    if not filepath.exists():
        return None
    try:
        with filepath.open(encoding="utf8") as fh:
            return json.load(fh)
    except Exception:
        return None


class CompressionApp(BaseApp):
    """Compression benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register dropdown-driven compression curve callbacks."""
        model_dropdown_id = f"{BENCHMARK_NAME}-model-dropdown"
        structure_dropdown_id = f"{BENCHMARK_NAME}-structure-dropdown"
        figure_id = f"{BENCHMARK_NAME}-figure"

        @callback(
            Output(structure_dropdown_id, "options"),
            Output(structure_dropdown_id, "value"),
            Input(model_dropdown_id, "value"),
        )
        def _update_structure_options(model_name: str):
            """
            Populate structure dropdown for the selected model.

            Parameters
            ----------
            model_name
                Selected model value from the model dropdown.

            Returns
            -------
            tuple[list[dict], str | None]
                Structure dropdown options and default selection.
            """
            if not model_name:
                raise PreventUpdate
            structures = _available_structures(model_name)
            options = [{"label": s, "value": s} for s in structures]
            default = structures[0] if structures else None
            return options, default

        @callback(
            Output(figure_id, "figure"),
            Input(model_dropdown_id, "value"),
            Input(structure_dropdown_id, "value"),
        )
        def _update_figure(model_name: str, structure: str | None):
            """
            Render energy-vs-volume and dE/dV curves for the selected structure.

            Parameters
            ----------
            model_name
                Selected model identifier.
            structure
                Selected structure label.

            Returns
            -------
            go.Figure
                Plotly figure with two subplots.
            """
            if not model_name or not structure:
                raise PreventUpdate

            payload = _load_curve(model_name, structure)
            if payload is None:
                raise PreventUpdate

            volumes = payload.get("volume_per_atom", [])
            energies = payload.get("energy_per_atom", [])
            de_dv = payload.get("dEdV", [])

            if not volumes or not energies:
                raise PreventUpdate

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    "Energy per atom vs Volume per atom",
                    "dE/dV vs Volume per atom",
                ),
            )

            fig.add_trace(
                go.Scatter(
                    x=volumes,
                    y=energies,
                    mode="lines+markers",
                    name="E/atom",
                    line={"color": "royalblue"},
                    marker={"size": 4},
                ),
                row=1,
                col=1,
            )

            if de_dv:
                fig.add_trace(
                    go.Scatter(
                        x=volumes,
                        y=de_dv,
                        mode="lines+markers",
                        name="dE/dV",
                        line={"color": "firebrick"},
                        marker={"size": 4},
                    ),
                    row=2,
                    col=1,
                )

            fig.update_xaxes(title_text="Volume per atom (ų)", row=2, col=1)
            fig.update_yaxes(title_text="Energy per atom (eV)", row=1, col=1)
            fig.update_yaxes(title_text="dE/dV (eV/ų)", row=2, col=1)

            fig.update_layout(
                title=f"{model_name} — {structure}",
                height=700,
                showlegend=True,
                template="plotly_white",
            )

            return fig


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
                Label("Select structure:"),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-structure-dropdown",
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
                style={"height": "700px", "width": "100%", "marginTop": "20px"},
            ),
            type="circle",
        ),
    ]

    return CompressionApp(
        name=BENCHMARK_NAME,
        description=(
            "Uniform crystal compression explorer. Structures are isotropically "
            "scaled and the energy per atom, its derivative dE/dV, and stress are "
            "recorded. Metrics are averaged across all structures."
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
