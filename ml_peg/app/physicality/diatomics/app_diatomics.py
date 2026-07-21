"""Run diatomics app."""

from __future__ import annotations

import csv
import io
import json

from dash import Dash, Input, Output, State, callback, dcc, no_update
from dash.dcc import Loading
from dash.exceptions import PreventUpdate
from dash.html import Div, Label

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    load_model_curves,
    register_image_gallery_callbacks,
    render_periodic_curve_gallery_png,
)
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
        register_data_download_callbacks()


def _safe_filename_stem(model_name: str, element_value: str | None) -> str:
    """
    Build a safe download filename stem for the current model and view.

    Parameters
    ----------
    model_name
        Name of the model being exported.
    element_value
        Current dropdown choice: the overview label, or an element symbol.

    Returns
    -------
    str
        Filename stem with anything other than letters, digits, ``-``, ``_``
        or ``.`` replaced by underscores.
    """
    view = "homonuclear" if element_value == OVERVIEW_LABEL else str(element_value)
    stem = f"{model_name}_diatomics_{view}"
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in stem)


def _serialise_selected_curves(
    model_name: str,
    element_value: str | None,
) -> tuple[dict, list[dict]]:
    """
    Build the JSON and CSV representations of the current selected view.

    Parameters
    ----------
    model_name
        Name of the model whose curves to export.
    element_value
        Current dropdown choice: the overview label, or an element symbol.

    Returns
    -------
    tuple[dict, list[dict]]
        A nested structure for the JSON file (model, view, selected element,
        and one entry per element pair) and a flat list of rows, one per
        distance point, for the CSV file.
    """
    selected_element, curves = load_model_curves(
        CURVE_PATH, model_name, element_value, OVERVIEW_LABEL
    )
    view = "homonuclear" if selected_element is None else "heteronuclear"
    pairs_payload: list[dict] = []
    rows: list[dict] = []

    for pair in sorted(curves):
        payload = curves[pair]
        distances = payload.get("distance") or []
        energies = payload.get("energy") or []
        forces = payload.get("force_parallel") or []
        shift = energies[-1] if energies else None
        shifted_energies = [
            energy - shift if shift is not None else None for energy in energies
        ]
        element_1 = payload.get("element_1")
        element_2 = payload.get("element_2")

        pair_payload = {
            "pair": pair,
            "element_1": element_1,
            "element_2": element_2,
            "distance": distances,
            "energy": energies,
            "shifted_energy": shifted_energies,
        }
        if forces:
            pair_payload["force_parallel"] = forces
        pairs_payload.append(pair_payload)

        for idx, distance in enumerate(distances):
            energy = energies[idx] if idx < len(energies) else None
            shifted_energy = (
                shifted_energies[idx] if idx < len(shifted_energies) else None
            )
            force_parallel = forces[idx] if idx < len(forces) else None
            rows.append(
                {
                    "model": model_name,
                    "view": view,
                    "selected_element": selected_element or "",
                    "pair": pair,
                    "element_1": element_1,
                    "element_2": element_2,
                    "distance": distance,
                    "energy": energy,
                    "shifted_energy": shifted_energy,
                    "force_parallel": force_parallel,
                }
            )

    return (
        {
            "model": model_name,
            "view": view,
            "selected_element": selected_element,
            "pairs": pairs_payload,
        },
        rows,
    )


def register_data_download_callbacks() -> None:
    """Register the current-view diatomics data download callback."""

    @callback(
        Output(f"{DATA_DOWNLOAD_ID}-download", "data"),
        Output(f"{DATA_DOWNLOAD_ID}-status", "children"),
        Input(f"{DATA_DOWNLOAD_ID}-button", "n_clicks"),
        State(f"{DATA_DOWNLOAD_ID}-format", "value"),
        State(f"{BENCHMARK_NAME}-model-dropdown", "value"),
        State(f"{BENCHMARK_NAME}-element-dropdown", "value"),
        prevent_initial_call=True,
        running=[(Output(f"{DATA_DOWNLOAD_ID}-button", "disabled"), True, False)],
    )
    def _download_data(
        n_clicks: int,
        download_format: str,
        model_name: str,
        element_value: str | None,
    ) -> tuple:
        """
        Turn the current diatomics view into a downloadable file.

        Parameters
        ----------
        n_clicks
            Number of times the download button has been clicked.
        download_format
            Chosen export format (``csv``, ``json`` or ``png``).
        model_name
            Name of the currently selected model.
        element_value
            Current dropdown choice: the overview label, or an element symbol.

        Returns
        -------
        tuple
            The file to download, and a status message (empty on success, or a
            short explanation when there is no data to export).
        """
        if not n_clicks or not model_name:
            raise PreventUpdate

        no_data_message = "No curve data for this selection."
        stem = _safe_filename_stem(model_name, element_value)
        fmt = (download_format or "csv").lower()
        if fmt == "png":
            try:
                png_bytes, _width, _height = render_periodic_curve_gallery_png(
                    curve_dir=CURVE_PATH,
                    model_name=model_name,
                    element_value=element_value,
                    overview_label=OVERVIEW_LABEL,
                )
            except PreventUpdate:
                return no_update, no_data_message
            return dcc.send_bytes(png_bytes, f"{stem}.png", type="image/png"), ""

        json_payload, rows = _serialise_selected_curves(model_name, element_value)
        if not rows:
            return no_update, no_data_message
        if fmt == "json":
            return (
                dcc.send_string(
                    json.dumps(json_payload, indent=2),
                    f"{stem}.json",
                    type="application/json",
                ),
                "",
            )

        buffer = io.StringIO()
        fieldnames = [
            "model",
            "view",
            "selected_element",
            "pair",
            "element_1",
            "element_2",
            "distance",
            "energy",
            "shifted_energy",
            "force_parallel",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return dcc.send_string(buffer.getvalue(), f"{stem}.csv", type="text/csv"), ""


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
