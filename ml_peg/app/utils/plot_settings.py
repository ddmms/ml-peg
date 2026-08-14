"""Components for the shared Plotly settings menu."""

from __future__ import annotations

from dash.dcc import Checklist, Dropdown, Store
from dash.dcc import Input as DCC_Input
from dash.html import Button, Details, Div, Label, Summary


def _plot_setting_id(control: str, graph_id: str) -> dict[str, str]:
    """
    Build a pattern-matching ID for one graph's plot setting control.

    Parameters
    ----------
    control
        Name of the plot setting control.
    graph_id
        String ID of the graph controlled by the menu.

    Returns
    -------
    dict[str, str]
        Dash pattern-matching component ID.
    """
    return {"type": f"plot-settings-{control}", "index": graph_id}


def _build_plot_axis_controls(axis: str, graph_id: str) -> Div:
    """
    Build scale, range, direction, and tick controls for one axis.

    Parameters
    ----------
    axis
        Axis identifier, either ``"x"`` or ``"y"``.
    graph_id
        String ID of the graph controlled by the menu.

    Returns
    -------
    Div
        Section containing controls for the selected axis.
    """
    label = axis.upper()
    return Div(
        [
            Div(
                [
                    Label(f"{label} axis", className="plot-settings-axis-label"),
                    Dropdown(
                        id=_plot_setting_id(f"{axis}-scale", graph_id),
                        options=[
                            {"label": "Linear", "value": "linear"},
                            {"label": "Log", "value": "log"},
                        ],
                        value="linear",
                        clearable=False,
                        searchable=False,
                        className="plot-settings-scale",
                    ),
                    DCC_Input(
                        id=_plot_setting_id(f"{axis}-min", graph_id),
                        type="number",
                        placeholder="Min",
                        debounce=True,
                        className="plot-settings-number",
                    ),
                    DCC_Input(
                        id=_plot_setting_id(f"{axis}-max", graph_id),
                        type="number",
                        placeholder="Max",
                        debounce=True,
                        className="plot-settings-number",
                    ),
                    Button(
                        f"Autoscale {label}",
                        id=_plot_setting_id(f"{axis}-autoscale", graph_id),
                        n_clicks=0,
                        className="plot-settings-secondary-button",
                    ),
                ],
                className="plot-settings-axis-row",
            ),
            Div(
                [
                    Label("Direction", className="plot-settings-row-label"),
                    Checklist(
                        id=_plot_setting_id(f"{axis}-reverse", graph_id),
                        options=[{"label": "Reverse axis", "value": "reverse"}],
                        value=[],
                        className="plot-settings-checklist",
                    ),
                ],
                className="plot-settings-option-row",
            ),
            Div(
                [
                    Label("Ticks", className="plot-settings-row-label"),
                    Dropdown(
                        id=_plot_setting_id(f"{axis}-tick-format", graph_id),
                        options=[
                            {"label": "Automatic ticks", "value": "auto"},
                            {"label": "Decimal", "value": "decimal"},
                            {"label": "Scientific", "value": "scientific"},
                        ],
                        value="auto",
                        clearable=False,
                        searchable=False,
                        className="plot-settings-tick-format",
                    ),
                    DCC_Input(
                        id=_plot_setting_id(f"{axis}-tick-precision", graph_id),
                        type="number",
                        min=0,
                        max=10,
                        step=1,
                        value=2,
                        placeholder="Precision",
                        className="plot-settings-number",
                    ),
                    DCC_Input(
                        id=_plot_setting_id(f"{axis}-tick-spacing", graph_id),
                        type="number",
                        min=0,
                        placeholder="Tick spacing",
                        debounce=True,
                        className="plot-settings-number",
                    ),
                ],
                className="plot-settings-tick-row",
            ),
        ],
        className="plot-settings-section",
    )


def _build_plot_size_controls(graph_id: str) -> Div:
    """
    Build responsive, preset, and custom figure-size controls.

    Parameters
    ----------
    graph_id
        String ID of the graph controlled by the menu.

    Returns
    -------
    Div
        Section containing figure-size controls.
    """
    return Div(
        [
            Label("Figure size", className="plot-settings-section-label"),
            Dropdown(
                id=_plot_setting_id("size-preset", graph_id),
                options=[
                    {"label": "Responsive", "value": "responsive"},
                    {"label": "Square (700 × 700)", "value": "square"},
                    {"label": "Wide (1000 × 600)", "value": "wide"},
                    {"label": "Custom", "value": "custom"},
                ],
                value="responsive",
                clearable=False,
                searchable=False,
                className="plot-settings-size-preset",
            ),
            DCC_Input(
                id=_plot_setting_id("width", graph_id),
                type="number",
                min=200,
                max=3000,
                step=10,
                placeholder="Width (px)",
                className="plot-settings-number",
            ),
            DCC_Input(
                id=_plot_setting_id("height", graph_id),
                type="number",
                min=200,
                max=3000,
                step=10,
                placeholder="Height (px)",
                className="plot-settings-number",
            ),
        ],
        className="plot-settings-size-row plot-settings-section",
    )


def build_plot_settings_controls(graph_id: str) -> Details:
    """
    Build plot display controls for a Plotly graph.

    Parameters
    ----------
    graph_id
        String ID of the graph controlled by the menu.

    Returns
    -------
    Details
        Collapsible figure, axis, and tick controls.
    """
    return Details(
        [
            Summary("Plot settings", className="plot-settings-summary"),
            Div(
                [
                    _build_plot_size_controls(graph_id),
                    _build_plot_axis_controls("x", graph_id),
                    _build_plot_axis_controls("y", graph_id),
                    Div(
                        "For explicit limits, enter both minimum and maximum. "
                        "Log tick spacing is measured in powers of ten.",
                        className="plot-settings-help",
                    ),
                    Div(
                        [
                            Button(
                                "Apply",
                                id=_plot_setting_id("apply", graph_id),
                                n_clicks=0,
                                className="plot-settings-apply",
                            ),
                            Button(
                                "Reset all",
                                id=_plot_setting_id("reset", graph_id),
                                n_clicks=0,
                                className="plot-settings-reset",
                            ),
                        ],
                        className="plot-settings-actions",
                    ),
                    Div(
                        id=_plot_setting_id("message", graph_id),
                        className="plot-settings-message",
                        role="alert",
                    ),
                    Store(id=_plot_setting_id("result", graph_id)),
                    Store(
                        id=_plot_setting_id("graph-id", graph_id),
                        data=graph_id,
                    ),
                ],
                className="plot-settings-body",
            ),
        ],
        className="plot-settings",
    )
