"""Run Al3Ti polymorph app."""

from __future__ import annotations

from copy import deepcopy

from ase.io import read
from dash import Input, Output, State, callback
from dash.html import H4, Div, Iframe

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_with_download_controls,
    register_plot_settings_callbacks,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.plot_helpers import PARITY_LINE_NAME
from ml_peg.app.utils.register_callbacks import register_plot_download_callbacks
from ml_peg.app.utils.weas import generate_weas_html
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names()
BENCHMARK_NAME = "Al3Ti polymorphs"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/alloy_metallurgy.html#al3ti-polymorphs"
DATA_PATH = APP_ROOT / "data" / "alloy_metallurgy" / "al3ti_polymorphs"
INFO_PATH = DATA_PATH / "info.json"

POLYMORPHS = ("D022", "D023", "L12")
POLYMORPH_LABELS = ("D0₂₂", "D0₂₃", "L1₂")
ASSET_PREFIX = "/assets/alloy_metallurgy/al3ti_polymorphs"
PLOT_ID = f"{BENCHMARK_NAME}-figure"
PLOT_PATHS = {
    "Formation Energy MAE": "figure_formation_energies.json",
    "Volume MAE": "figure_volumes_per_atom.json",
    "Lattice Parameter MAPE": "figure_lattice_parameters.json",
}


def highlight_model(figure: dict, model_name: str) -> dict:
    """
    Emphasize one model's markers while retaining plot guides.

    Parameters
    ----------
    figure
        Saved comparison figure.
    model_name
        Raw model name stored in the selected table row ID.

    Returns
    -------
    dict
        Figure with the selected traces drawn last.
    """
    result = deepcopy(figure)
    if not any(trace.get("name") == model_name for trace in result["data"]):
        return result
    selected = []
    others = []
    for trace in result["data"]:
        if trace.get("name") == model_name:
            trace["opacity"] = 1
            trace["marker"] = {
                **trace.get("marker", {}),
                "size": 13,
                "line": {"color": "#222222", "width": 2},
            }
            selected.append(trace)
        else:
            if trace.get("name") in MODELS and trace.get("name") != PARITY_LINE_NAME:
                trace["opacity"] = 0.25
            others.append(trace)
    result["data"] = others + selected
    return result


def structure_grid(model_name: str) -> Div:
    """
    Show available relaxed polymorphs for one model.

    Parameters
    ----------
    model_name
        Raw model name stored in the selected table row ID.

    Returns
    -------
    Div
        Responsive row of structure viewers or unavailable messages.
    """
    cards = []
    for name, label in zip(POLYMORPHS, POLYMORPH_LABELS, strict=True):
        path = DATA_PATH / model_name / f"{name}.xyz"
        try:
            if not path.is_file():
                raise FileNotFoundError(path)
            read(path)
        except Exception:
            content = Div("Structure unavailable.")
        else:
            content = Iframe(
                srcDoc=generate_weas_html(f"{ASSET_PREFIX}/{model_name}/{name}.xyz"),
                style={"width": "100%", "height": "550px", "border": "0"},
            )
        cards.append(
            Div(
                [H4(label), content],
                style={
                    "minWidth": 0,
                    "padding": "8px",
                    "border": "1px solid var(--mlpeg-border)",
                },
            )
        )
    return Div(
        cards,
        style={
            "display": "grid",
            "gridTemplateColumns": "repeat(auto-fit, minmax(260px, 1fr))",
            "gap": "12px",
        },
    )


class Al3TiPolymorphApp(BaseApp):
    """Al3Ti Polymorph benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register the table-driven plot and structure callbacks."""
        register_plot_download_callbacks()
        register_plot_settings_callbacks()

        @callback(
            Output(f"{BENCHMARK_NAME}-figure-placeholder", "children"),
            Output(f"{BENCHMARK_NAME}-struct-placeholder", "children"),
            Input(self.table_id, "active_cell"),
            State(self.table_id, "data"),
        )
        def show_visualizations(active_cell, table_data):
            """
            Display the selected model's metric plot and structures.

            Parameters
            ----------
            active_cell
                Selected table cell, or None when no cell is selected.
            table_data
                Current table rows used to validate the selected model.

            Returns
            -------
            tuple[Div, Div]
                Plot controls and structure viewers for the selected model.
            """
            if not active_cell:
                graph = read_plot(
                    DATA_PATH / PLOT_PATHS["Formation Energy MAE"], PLOT_ID
                )
                return plot_with_download_controls(graph), Div()

            model_name = active_cell.get("row_id")
            metric = active_cell.get("column_id")
            if model_name not in MODELS or not any(
                row.get("id") == model_name for row in (table_data or [])
            ):
                return Div("No data available for this model."), Div()

            if metric == "Relative Stability Ordering Accuracy":
                plot_path = (
                    DATA_PATH / model_name / "figure_formation_energy_ordering.json"
                )
            else:
                filename = PLOT_PATHS.get(metric)
                plot_path = DATA_PATH / filename if filename else None

            if plot_path is None or not plot_path.is_file():
                plot = Div("Plot unavailable for this model.")
            else:
                graph = read_plot(plot_path, PLOT_ID)
                if metric in PLOT_PATHS:
                    graph.figure = highlight_model(graph.figure, model_name)
                plot = plot_with_download_controls(graph)
            return plot, structure_grid(model_name)


def get_app() -> Al3TiPolymorphApp:
    """
    Get Al3Ti Polymorph benchmark app layout and callback registration.

    Returns
    -------
    Al3TiPolymorphApp
        Benchmark layout and callback registration.
    """
    return Al3TiPolymorphApp(
        name=BENCHMARK_NAME,
        description=(
            "Formation energies, relaxed cell geometry, and stability ordering "
            "for Al3Ti polymorphs."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "al3ti_metrics.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )
