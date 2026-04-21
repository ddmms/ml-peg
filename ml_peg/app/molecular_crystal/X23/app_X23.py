"""Run X23 app."""

from __future__ import annotations

from copy import deepcopy
import json

from dash import Dash, Input, Output, State, callback, dcc
from dash.exceptions import PreventUpdate
from dash.html import Div, Iframe, Label
import numpy as np

from ml_peg.analysis.utils.utils import (
    calc_metric_scores,
    calc_table_scores,
    get_table_style,
)
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.utils import (
    build_level_of_theory_warnings,
    clean_thresholds,
    filter_rows_by_models,
    format_metric_columns,
    format_tooltip_headers,
    get_scores,
)
from ml_peg.app.utils.weas import generate_weas_html
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "X23 Lattice Energies"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_crystal.html#x23"
)
DATA_PATH = APP_ROOT / "data" / "molecular_crystal" / "X23"
FILTER_DATA_PATH = DATA_PATH / "x23_element_filter_data.json"
ELEMENT_DROPDOWN_ID = f"{BENCHMARK_NAME}-element-dropdown"


def _load_filter_payload() -> dict:
    """Load element-filter payload for X23."""
    if not FILTER_DATA_PATH.exists():
        return {}
    with FILTER_DATA_PATH.open(encoding="utf8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    return payload


FILTER_PAYLOAD = _load_filter_payload()
FILTER_ELEMENTS = [
    element
    for element in FILTER_PAYLOAD.get("elements", [])
    if isinstance(element, str) and element
]
FILTER_SYSTEMS = [
    system
    for system in FILTER_PAYLOAD.get("systems", [])
    if isinstance(system, str) and system
]
FILTER_SYSTEM_ELEMENTS = FILTER_PAYLOAD.get("system_elements", [])
SYSTEM_TO_INDEX = {system: idx for idx, system in enumerate(FILTER_SYSTEMS)}


def _figure_dict(graph) -> dict:
    """Convert a Plotly graph component into a mutable figure dictionary."""
    figure = getattr(graph, "figure", None)
    if figure is None:
        return {}
    if hasattr(figure, "to_plotly_json"):
        return figure.to_plotly_json()
    if isinstance(figure, dict):
        return figure
    return {}


def _customdata_system(point_customdata) -> str | None:
    """Extract the system label from a Plotly customdata entry."""
    if point_customdata is None:
        return None
    if isinstance(point_customdata, (list, tuple, np.ndarray)):
        if not point_customdata:
            return None
        value = point_customdata[0]
    else:
        value = point_customdata
    return str(value) if value is not None else None


def _keep_mask(selected_elements: list[str] | None) -> np.ndarray:
    """Return a mask of systems that do not contain deselected elements."""
    n_systems = len(FILTER_SYSTEMS)
    n_elements = len(FILTER_ELEMENTS)
    if n_systems == 0:
        return np.zeros(0, dtype=bool)

    selected = {
        element for element in (selected_elements or []) if isinstance(element, str)
    }
    selected_vector = np.array(
        [element in selected for element in FILTER_ELEMENTS], dtype=bool
    )
    deselected_vector = ~selected_vector
    if not deselected_vector.any() or n_elements == 0:
        return np.ones(n_systems, dtype=bool)

    incidence = np.zeros((n_systems, n_elements), dtype=bool)
    element_to_idx = {element: idx for idx, element in enumerate(FILTER_ELEMENTS)}
    for system_idx, element_list in enumerate(FILTER_SYSTEM_ELEMENTS[:n_systems]):
        if not isinstance(element_list, list):
            continue
        for element in element_list:
            element_idx = element_to_idx.get(element)
            if element_idx is not None:
                incidence[system_idx, element_idx] = True

    excluded = incidence[:, deselected_vector].any(axis=1)
    return ~excluded


class X23App(BaseApp):
    """X23 benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register callbacks to app."""
        scatter = read_plot(
            DATA_PATH / "figure_lattice_energies.json",
            id=f"{BENCHMARK_NAME}-figure",
        )
        baseline_scatter = _figure_dict(scatter)

        system_to_struct = {
            system: f"/assets/molecular_crystal/X23/{MODELS[0]}/{system}.xyz"
            for system in FILTER_SYSTEMS
        }

        @callback(
            Output(f"{BENCHMARK_NAME}-struct-placeholder", "children"),
            Input(f"{BENCHMARK_NAME}-figure", "clickData"),
            prevent_initial_call=True,
        )
        def show_struct(click_data):
            """Show the clicked X23 structure based on the system label."""
            if not click_data:
                return Div("Click on a point to view the structure.")

            point = click_data["points"][0]
            system = _customdata_system(point.get("customdata"))
            if system is None:
                raise PreventUpdate

            struct = system_to_struct.get(system)
            if not struct:
                raise PreventUpdate

            return Div(
                Iframe(
                    srcDoc=generate_weas_html(struct, mode="struct", index=0),
                    style={
                        "height": "550px",
                        "width": "100%",
                        "border": "1px solid #ddd",
                        "borderRadius": "5px",
                    },
                )
            )

        baseline_rows = deepcopy(self.table.data)
        model_levels = getattr(self.table, "model_levels_of_theory", None)
        metric_levels = getattr(self.table, "metric_levels_of_theory", None)
        model_configs = getattr(self.table, "model_configs", None)

        n_systems = len(FILTER_SYSTEMS)

        @callback(
            Output(f"{BENCHMARK_NAME}-figure", "figure", allow_duplicate=True),
            Input(ELEMENT_DROPDOWN_ID, "value"),
            prevent_initial_call=True,
        )
        def update_parity_figure(selected_elements):
            """Filter the parity figure to systems that match the element mask."""
            if not baseline_scatter:
                raise PreventUpdate
            if n_systems == 0:
                raise PreventUpdate

            keep_mask = _keep_mask(selected_elements)

            fig = json.loads(json.dumps(baseline_scatter))
            traces = fig.get("data", [])
            for trace in traces:
                customdata = trace.get("customdata")
                if not customdata:
                    continue
                x_vals = trace.get("x") or []
                y_vals = trace.get("y") or []
                if len(x_vals) != len(customdata) or len(y_vals) != len(customdata):
                    continue

                kept_x = []
                kept_y = []
                kept_customdata = []
                for idx, point_customdata in enumerate(customdata):
                    system = _customdata_system(point_customdata)
                    if system is None:
                        continue
                    system_idx = SYSTEM_TO_INDEX.get(system)
                    if system_idx is None:
                        continue
                    if not keep_mask[system_idx]:
                        continue
                    kept_x.append(x_vals[idx])
                    kept_y.append(y_vals[idx])
                    kept_customdata.append(point_customdata)

                trace["x"] = kept_x
                trace["y"] = kept_y
                trace["customdata"] = kept_customdata

            return fig

        @callback(
            Output(self.table_id, "data", allow_duplicate=True),
            Output(self.table_id, "style_data_conditional", allow_duplicate=True),
            Output(self.table_id, "tooltip_data", allow_duplicate=True),
            Output(self.table_id, "columns", allow_duplicate=True),
            Output(self.table_id, "tooltip_header", allow_duplicate=True),
            Output(f"{self.table_id}-computed-store", "data", allow_duplicate=True),
            Output(f"{self.table_id}-raw-data-store", "data", allow_duplicate=True),
            Input(ELEMENT_DROPDOWN_ID, "value"),
            State(f"{self.table_id}-weight-store", "data"),
            State(f"{self.table_id}-thresholds-store", "data"),
            State(f"{self.table_id}-normalized-toggle", "value"),
            State("selected-models-store", "data"),
            State(f"{self.table_id}-raw-tooltip-store", "data"),
            State(self.table_id, "columns"),
            prevent_initial_call=True,
        )
        def update_table_for_elements(
            selected_elements,
            stored_weights,
            stored_threshold,
            toggle_value,
            selected_models,
            raw_tooltips,
            current_columns,
        ):
            """Recompute X23 MAE values after applying element-based filtering."""
            if n_systems == 0:
                raise PreventUpdate
            if current_columns is None:
                raise PreventUpdate

            keep_mask = _keep_mask(selected_elements)

            model_to_mae: dict[str, float | None] = {}
            for trace in baseline_scatter.get("data", []):
                model_id = trace.get("name")
                x_vals = trace.get("x") or []
                y_vals = trace.get("y") or []
                customdata = trace.get("customdata") or []
                if not model_id or not x_vals or not y_vals:
                    continue

                trace_keep_mask: list[bool] = []
                for point_customdata in customdata:
                    system = _customdata_system(point_customdata)
                    system_idx = SYSTEM_TO_INDEX.get(system) if system else None
                    trace_keep_mask.append(
                        bool(system_idx is not None and keep_mask[system_idx])
                    )

                if len(trace_keep_mask) != len(x_vals):
                    model_to_mae[model_id] = None
                    continue

                x_arr = np.asarray(x_vals, dtype=float)
                y_arr = np.asarray(y_vals, dtype=float)
                keep_arr = np.asarray(trace_keep_mask, dtype=bool)
                valid = np.isfinite(x_arr) & np.isfinite(y_arr) & keep_arr
                if not valid.any():
                    model_to_mae[model_id] = None
                    continue

                model_to_mae[model_id] = float(
                    np.abs(x_arr[valid] - y_arr[valid]).mean()
                )

            filtered_raw_rows = []
            for row in baseline_rows:
                updated_row = row.copy()
                model_id = updated_row.get("id") or updated_row.get("MLIP")
                updated_row["MAE"] = model_to_mae.get(model_id)
                updated_row["Score"] = None
                filtered_raw_rows.append(updated_row)

            thresholds = clean_thresholds(stored_threshold)
            show_normalized = bool(toggle_value) and toggle_value[0] == "norm"

            metrics_data = calc_table_scores(
                filtered_raw_rows, stored_weights, thresholds
            )
            scored_rows = calc_metric_scores(filtered_raw_rows, thresholds)
            display_rows = get_scores(
                metrics_data, scored_rows, thresholds, toggle_value
            )

            filtered_rows = filter_rows_by_models(display_rows, selected_models)
            filtered_scores = filter_rows_by_models(scored_rows, selected_models)
            style = (
                get_table_style(filtered_rows, scored_data=filtered_scores)
                if filtered_rows
                else []
            )

            warning_styles, tooltip_data = build_level_of_theory_warnings(
                filtered_rows,
                model_levels,
                metric_levels,
                model_configs,
            )
            style = style + warning_styles
            columns = format_metric_columns(
                current_columns, thresholds, show_normalized
            )
            tooltips = format_tooltip_headers(raw_tooltips, thresholds, show_normalized)

            return (
                filtered_rows,
                style,
                tooltip_data,
                columns,
                tooltips,
                scored_rows,
                metrics_data,
            )


def get_app() -> X23App:
    """
    Get X23 benchmark app layout and callback registration.

    Returns
    -------
    X23App
        Benchmark layout and callback registration.
    """
    element_options = [
        {"label": element, "value": element} for element in FILTER_ELEMENTS
    ]
    scatter = read_plot(
        DATA_PATH / "figure_lattice_energies.json",
        id=f"{BENCHMARK_NAME}-figure",
    )
    baseline_scatter = _figure_dict(scatter)

    return X23App(
        name=BENCHMARK_NAME,
        description="Lattice energies for 23 organic molecular crystals.",
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "x23_metrics_table.json",
        extra_components=[
            Div(
                [
                    Label("Element filter (included elements):"),
                    dcc.Dropdown(
                        id=ELEMENT_DROPDOWN_ID,
                        options=element_options,
                        value=FILTER_ELEMENTS,
                        multi=True,
                        clearable=False,
                        placeholder="Select elements",
                        style={"width": "360px", "marginBottom": "20px"},
                    ),
                ]
            ),
            dcc.Graph(
                id=f"{BENCHMARK_NAME}-figure",
                figure=baseline_scatter if baseline_scatter else scatter.figure,
                style={"height": "700px", "width": "100%", "marginTop": "20px"},
            ),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
    )


if __name__ == "__main__":
    # Create Dash app
    full_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)

    # Construct layout and register callbacks
    x23_app = get_app()
    full_app.layout = x23_app.layout
    x23_app.register_callbacks()

    # Run app
    full_app.run(port=8053, debug=True)
