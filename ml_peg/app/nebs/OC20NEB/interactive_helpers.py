"""Utilities for building OC20NEB interactive assets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import matplotlib

matplotlib.use("Agg")
from dash import dcc, html
from dash.html import Iframe

from ml_peg.app.utils.plot_helpers import figure_from_dict
from ml_peg.app.utils.weas import generate_weas_html


def lookup_system_entry(model_entry: Mapping[str, Any], point_id: str | int):
    """
    Find the dataset entry (metric or stability) linked to ``point_id``.

    Parameters
    ----------
    model_entry
        Dictionary describing a single model's metrics/stability info.
    point_id
        Identifier extracted from the scatter click.

    Returns
    -------
    dict | None
        Matching point metadata or ``None`` when not found.
    """

    def _match(points: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
        """
        Match selection IDs across metric/stability buckets.

        Parameters
        ----------
        points
            Iterable of point dictionaries to probe.

        Returns
        -------
        dict | None
            Matching point metadata or ``None`` when not found.
        """
        target = str(point_id)
        return next(
            (point for point in points if str(point.get("reaction")) == target),
            None,
        )

    metrics = model_entry.get("metrics", {})
    for metric_data in metrics.values():
        match = _match(metric_data.get("points", []))
        if match:
            return match

    return None


def render_neb_profile(
    selection_context: Mapping[str, Any],
    scatter_id: str,
) -> html.Div | None:
    """
    Render the pre-built NEB profile figure for a selection.

    The figure itself is generated during analysis (see
    ``analyse_OC20NEB.py::_build_neb_profile_figure``) and stored under the
    point's ``profile_figure`` key, so this just deserializes and displays it.
    The returned ``dcc.Graph`` uses ``scatter_id`` so that downstream callbacks
    wired to that ID can detect clicks on individual NEB images.

    Parameters
    ----------
    selection_context
        Dictionary containing ``model`` and resolved ``selection`` data.
    scatter_id
        Dash component ID assigned to the rendered ``dcc.Graph``.

    Returns
    -------
    dash.html.Div | None
        Component containing the interactive profile graph, or ``None`` if
        no pre-built figure is available.
    """
    selected = selection_context.get("selection") or {}
    label = selected.get("label") or selected.get("reaction", "")
    figure_dict = selected.get("profile_figure")

    if not figure_dict:
        return None

    fig = figure_from_dict(figure_dict)

    return html.Div(
        [
            html.H4(label),
            html.P(
                "Click on a point to view the DFT and MLIP geometries side-by-side.",
                style={"color": "#666", "fontSize": "13px", "margin": "4px 0 8px"},
            ),
            dcc.Graph(
                id=scatter_id,
                figure=fig,
                config={"displayModeBar": True, "scrollZoom": False},
                style={"height": "420px"},
            ),
        ]
    )


def render_geometry_comparison(
    click_context: Mapping[str, Any],
) -> html.Div | None:
    """
    Render DFT and MLIP geometries side-by-side for a clicked NEB image.

    Parameters
    ----------
    click_context
        Dictionary containing ``ref_profile``, ``pred_profile`` (trajectory
        file paths) and ``image_index`` (the clicked NEB image number).

    Returns
    -------
    dash.html.Div | None
        Side-by-side WEAS structure viewers, or ``None`` when data is missing.
    """
    ref_path = click_context.get("ref_profile")
    pred_path = click_context.get("pred_profile")
    image_index = click_context.get("image_index", 0)
    system_label = click_context.get("system_label", "")

    if not ref_path or not pred_path:
        return None

    iframe_style = {
        "height": "450px",
        "width": "100%",
        "border": "1px solid #ddd",
        "borderRadius": "5px",
    }
    label_style = {
        "textAlign": "center",
        "fontWeight": "bold",
        "marginBottom": "6px",
        "fontSize": "14px",
    }

    try:
        ref_html = generate_weas_html(ref_path, "traj", image_index)
        pred_html = generate_weas_html(pred_path, "traj", image_index)
    except Exception:
        return None

    return html.Div(
        [
            html.H5(
                f"NEB Image {image_index}"
                + (f" — {system_label}" if system_label else ""),
                style={"marginBottom": "12px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.P("DFT (Reference)", style=label_style),
                            Iframe(srcDoc=ref_html, style=iframe_style),
                        ],
                        style={"flex": "1", "minWidth": 0},
                    ),
                    html.Div(
                        [
                            html.P("MLIP (Predicted)", style=label_style),
                            Iframe(srcDoc=pred_html, style=iframe_style),
                        ],
                        style={"flex": "1", "minWidth": 0},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "16px",
                    "alignItems": "stretch",
                },
            ),
        ]
    )
