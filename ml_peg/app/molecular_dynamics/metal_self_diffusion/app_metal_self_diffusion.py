"""Run the metal self-diffusion benchmark app."""

from __future__ import annotations

import json

from dash import Dash
from dash.dcc import Tab, Tabs
from dash.html import H3, Div

from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import (
    plot_with_download_controls,
    register_plot_settings_callbacks,
)
from ml_peg.app.utils.load import read_plot
from ml_peg.app.utils.register_callbacks import register_plot_download_callbacks

BENCHMARK_NAME = "MetalSelfDiffusion"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#metal-self-diffusion"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "metal_self_diffusion"


class MetalSelfDiffusionApp(BaseApp):
    """Metal self-diffusion metrics and element-specific diagnostic plots."""

    def register_callbacks(self) -> None:
        """Register the standard plot settings and download controls."""
        register_plot_settings_callbacks()
        register_plot_download_callbacks()


def get_app() -> MetalSelfDiffusionApp:
    """Build the metrics table and element tabs for VACF and diffusion plots."""
    info = json.loads((DATA_PATH / "info.json").read_text())
    tabs = []
    for element in info["filenames"]:
        plots = []
        for suffix, title in (
            ("vacfs", "Velocity autocorrelation functions"),
            ("diffusion", "Diffusion constant versus temperature"),
        ):
            path = DATA_PATH / f"{element}_{suffix}.json"
            if path.is_file():
                graph = read_plot(path, id=f"{BENCHMARK_NAME}-{element}-{suffix}")
                graph.mathjax = True
            plots.extend(
                [
                    H3(title),
                    plot_with_download_controls(graph)
                    if path.is_file()
                    else Div("No data available for this element."),
                ]
            )
        tabs.append(Tab(label=element, value=element, children=plots))

    return MetalSelfDiffusionApp(
        name=BENCHMARK_NAME,
        description=(
            "Liquid-metal self-diffusion coefficients and their temperature "
            "dependence compared with experiment. Select an element to view "
            "velocity autocorrelation functions and diffusion curves across models."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "metal_self_diffusion_metrics_table.json",
        info_path=DATA_PATH / "info.json",
        extra_components=[
            Tabs(
                id=f"{BENCHMARK_NAME}-elements",
                value=info["filenames"][0] if tabs else None,
                children=tabs,
            ),
        ],
    )
