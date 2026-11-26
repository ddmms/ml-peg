"""Run aluminosilicates densities app."""

from __future__ import annotations

from pathlib import Path

from dash import Dash, Input, Output, State, callback, dcc
from dash.html import Div, Iframe

from ml_peg.analysis.molecular_dynamics.aluminosilicates_densities import (
    analyse_aluminosilicates_densities as density_analysis,
)
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.weas import generate_weas_html

COMPOSITIONS = density_analysis.COMPOSITIONS
MODELS = density_analysis.MODELS
BENCHMARK_NAME = "Aluminosilicates Densities"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#aluminosilicates-densities"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "aluminosilicates_densities"
INFO_PATH = DATA_PATH / "info.json"


def render_structure(filepath: Path) -> Iframe | Div:
    """
    Show a structure file in the shared WEAS viewer.

    Parameters
    ----------
    filepath
        Path to the structure file to be displayed.

    Returns
    -------
    Iframe or Div
        WEAS viewer for the structure, or a message if the file is missing.
    """
    if not filepath.exists():
        return Div(
            f"File not found: {filepath.name}",
            style={"color": "#e53e3e", "padding": "15px"},
        )

    # WEAS fetches the file in the browser, so it needs the URL served by the app.
    asset_url = f"/assets/{filepath.relative_to(APP_ROOT / 'data').as_posix()}"
    return Iframe(
        srcDoc=generate_weas_html(asset_url, "struct"),
        style={
            "width": "100%",
            "height": "520px",
            "border": "1px solid #cbd5e1",
            "borderRadius": "8px",
        },
    )


class AluminosilicatesDensitiesApp(BaseApp):
    """Aluminosilicates densities benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """
        Register callbacks to app.

        Returns
        -------
        None
            This method does not return a value.
        """
        scatter = read_plot(
            DATA_PATH / "figure_density_parity.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Density MAE (albite)": scatter,
                "Density MAE (anorthite)": scatter,
                "Density MAE (sanidine)": scatter,
                "Density MAE (mean)": scatter,
                "Density MAPE (albite)": scatter,
                "Density MAPE (anorthite)": scatter,
                "Density MAPE (sanidine)": scatter,
                "Density MAPE (mean)": scatter,
            },
        )

        @callback(
            Output(f"{BENCHMARK_NAME}-struct-placeholder", "children"),
            Input(f"{BENCHMARK_NAME}-figure", "clickData"),
            Input(f"{BENCHMARK_NAME}-replica-select", "value"),
            State(f"{BENCHMARK_NAME}-figure", "figure"),
        )
        def update_displayed_structure(click_data, replica_idx, figure):
            """
            Update the displayed 3D atomic structure based on user click.

            Parameters
            ----------
            click_data : dict or None
                Interaction data from clicking points on the density parity plot.
            replica_idx : int
                Index of the structure replica to display.
            figure : dict or None
                Current figure, used to read the model name of the clicked trace.

            Returns
            -------
            Iframe or Div
                The 3D viewer component for the selected structure, or a hint if
                no valid point is selected.
            """
            hint = Div(
                "Click a point in the plot to display its structure.",
                style={"color": "#64748b", "padding": "15px"},
            )

            if not (click_data and click_data.get("points") and figure):
                return hint

            point = click_data["points"][0]
            curve_idx = point.get("curveNumber")
            point_idx = point.get("pointIndex", point.get("pointNumber"))
            traces = figure.get("data", [])
            if curve_idx is None or point_idx is None:
                return hint
            if not 0 <= curve_idx < len(traces):
                return hint

            trace = traces[curve_idx]
            model = trace.get("name")  # the parity line has no name

            customdata = point.get("customdata")
            if customdata is None:
                trace_custom = trace.get("customdata") or []
                if point_idx < len(trace_custom):
                    customdata = trace_custom[point_idx]
            comp = customdata[0] if customdata else None

            if model not in MODELS or comp not in COMPOSITIONS:
                return hint

            replica_idx = replica_idx if replica_idx is not None else 0
            filename = f"{model}_{comp}_replica{replica_idx}_quenched.xyz"
            return render_structure(DATA_PATH / model / comp / filename)


def get_app() -> AluminosilicatesDensitiesApp:
    """
    Get aluminosilicates densities benchmark app layout and callback registration.

    Returns
    -------
    AluminosilicatesDensitiesApp
        The initialized app instance for the aluminosilicates densities benchmark.
    """
    return AluminosilicatesDensitiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting melt-quenched glass densities "
            "from MD simulations. "
            "Reference values are experimental densities."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "aluminosilicates_densities_metrics_table.json",
        extra_components=[
            dcc.Dropdown(
                id=f"{BENCHMARK_NAME}-replica-select",
                options=[{"label": f"Replica {i}", "value": i} for i in range(3)],
                value=0,
                clearable=False,
                style={"width": "200px", "marginBottom": "10px"},
            ),
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
    )


if __name__ == "__main__":
    full_app = Dash(
        __name__,
        assets_folder=DATA_PATH.parent.parent,
        suppress_callback_exceptions=True,
    )
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8064, debug=True)
