"""Display comparison figures prepared by water benchmark analysis."""

from __future__ import annotations

from dash import dcc, html


def render_subplot_component(click_data: dict) -> html.Div:
    """
    Display the precomputed Plotly comparison attached to a clicked bar.

    Parameters
    ----------
    click_data : dict
        Plotly click event containing the comparison figure in the point's customdata.

    Returns
    -------
    dash.html.Div
        Container holding the comparison graph or a message when none is available.
    """
    points = (click_data or {}).get("points", [])
    if not points:
        return html.Div("Click on a bar to see comparison.")
    custom = points[0].get("customdata", [])
    if len(custom) < 2 or "figure" not in custom[1]:
        return html.Div("Comparison unavailable. Regenerate the benchmark analysis.")
    return html.Div(dcc.Graph(figure=custom[1]["figure"]))
