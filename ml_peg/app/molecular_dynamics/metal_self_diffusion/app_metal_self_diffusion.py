"""Run the metal self-diffusion benchmark app."""

from __future__ import annotations

from dash.dcc import Graph, Tab, Tabs
from dash.html import H3, Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_from_table_column,
    plot_with_download_controls,
)
from ml_peg.app.utils.load import read_json, read_plot

BENCHMARK_NAME = "MetalSelfDiffusion"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#metal-self-diffusion"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "metal_self_diffusion"


class MetalSelfDiffusionApp(BaseApp):
    """Metal self-diffusion metrics and element-specific diagnostic plots."""

    def register_callbacks(self) -> None:
        """Show element plots when the diffusion metric column is clicked."""
        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={"Diffusion coefficient log error": load_element_plots()},
        )


def load_vacf_plot(element: str) -> Graph | None:
    """
    Load VACFs and select among the temperatures present in the saved plot.

    Parameters
    ----------
    element
        Element symbol identifying the saved plot.

    Returns
    -------
    Graph or None
        VACF graph with temperature selection, or None when the plot is missing.
    """
    path = DATA_PATH / f"{element}_vacfs.json"
    if not path.is_file():
        return None
    graph = read_plot(path, id=f"{BENCHMARK_NAME}-{element}-vacfs")
    graph.mathjax = True
    traces = graph.figure["data"]
    if not traces:
        return graph
    temperatures = sorted({trace["meta"]["temperature"] for trace in traces})
    steps = []
    for temperature in temperatures:
        visible = [trace["meta"]["temperature"] == temperature for trace in traces]
        steps.append(
            {
                "label": f"{temperature:g} K",
                "method": "restyle",
                "args": [{"visible": visible}],
            }
        )
    if temperatures:
        for trace, visible in zip(traces, steps[0]["args"][0]["visible"], strict=True):
            trace["visible"] = visible
    graph.figure["layout"]["sliders"] = [
        {
            "steps": steps,
            "active": 0,
            "currentvalue": {"prefix": "Temperature: "},
            "pad": {"t": 45},
        }
    ]
    margin = graph.figure["layout"].setdefault("margin", {})
    margin["b"] = max(margin.get("b", 0), 100)
    return graph


def load_diffusion_plot(element: str) -> Graph | None:
    """
    Load diffusion points, standard errors, and saved Arrhenius fits.

    Parameters
    ----------
    element
        Element symbol identifying the saved plot.

    Returns
    -------
    Graph or None
        Diffusion graph with error bars, or None when the plot is missing.
    """
    path = DATA_PATH / f"{element}_diffusion.json"
    if not path.is_file():
        return None
    graph = read_plot(path, id=f"{BENCHMARK_NAME}-{element}-diffusion")
    graph.mathjax = True
    traces = graph.figure["data"]
    errors_path = DATA_PATH / f"{element}_diffusion_errors.json"
    errors = read_json(errors_path) if errors_path.is_file() else {}
    for trace in traces:
        if trace["mode"] != "markers":
            continue
        if trace["name"] in errors:
            trace["error_y"] = {
                "type": "data",
                "array": [
                    error * 1e9 if error is not None else None
                    for error in errors[trace["name"]]
                ],
                "visible": True,
            }
    return graph


def load_element_plots() -> Div:
    """
    Build element tabs containing VACF and diffusion plots.

    Returns
    -------
    Div
        Element tabs with plots and their download controls.
    """
    info = read_json(DATA_PATH / "info.json")
    tabs = []
    for element in info["filenames"]:
        plots = []
        for title, graph in (
            ("Velocity autocorrelation functions", load_vacf_plot(element)),
            ("Diffusion constant versus temperature", load_diffusion_plot(element)),
        ):
            plots.extend(
                [
                    H3(title),
                    plot_with_download_controls(graph)
                    if graph is not None
                    else Div("No data available for this element."),
                ]
            )
        tabs.append(Tab(label=element, value=element, children=plots))

    return Div(
        Tabs(
            id=f"{BENCHMARK_NAME}-elements",
            value=info["filenames"][0] if tabs else None,
            children=tabs,
        )
    )


def get_app() -> MetalSelfDiffusionApp:
    """
    Build the metrics table with a placeholder for element plots.

    Returns
    -------
    MetalSelfDiffusionApp
        Benchmark application with plots revealed by metric-table clicks.
    """
    return MetalSelfDiffusionApp(
        name=BENCHMARK_NAME,
        description=(
            "Liquid-metal self-diffusion coefficients and their temperature "
            "dependence compared with experiment. Click the diffusion metric, "
            "then select an element to view "
            "velocity autocorrelation functions and diffusion curves across models."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "metal_self_diffusion_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
        ],
    )
