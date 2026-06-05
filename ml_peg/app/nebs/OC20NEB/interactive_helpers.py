"""Utilities for building OC20NEB interactive assets."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ase.io import read
import matplotlib

matplotlib.use("Agg")
from dash import dcc, html
from dash.html import Iframe
import numpy as np
import plotly.graph_objects as go

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
    reference_label: str,
    scatter_id: str,
) -> html.Div | None:
    """
    Render an interactive Plotly NEB profile for a selection.

    The returned ``dcc.Graph`` uses ``scatter_id`` so that downstream callbacks
    wired to that ID can detect clicks on individual NEB images.

    Parameters
    ----------
    selection_context
        Dictionary containing ``model`` and resolved ``selection`` data.
    reference_label
        Legend label for the reference trace.
    scatter_id
        Dash component ID assigned to the rendered ``dcc.Graph``.

    Returns
    -------
    dash.html.Div | None
        Component containing the interactive profile graph, or ``None`` if
        no data is available.
    """
    model_display = selection_context.get("model")
    selected = selection_context.get("selection") or {}
    data_paths = selected.get("data_paths")
    label = selected.get("label") or selected.get("reaction", "")

    if not data_paths:
        return None

    fig = _build_neb_profile_figure(
        paths=data_paths,
        model_label=model_display,
        system_label=label,
        reference_label=reference_label,
    )
    if fig is None:
        return None

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


def _build_neb_profile_figure(
    *,
    paths: Mapping[str, str | None],
    model_label: str,
    system_label: str,
    reference_label: str = "Reference",
) -> go.Figure | None:
    """
    Build an interactive Plotly figure showing DFT and MLIP NEB energy profiles.

    Each data point is individually clickable; ``pointNumber`` maps directly to
    the NEB image index in the trajectory files.

    Parameters
    ----------
    paths
        Mapping with ``"ref_profile"`` and ``"pred_profile"`` trajectory paths.
    model_label
        Display name for the predicted model trace.
    system_label
        Title displayed above the plot.
    reference_label
        Legend label for the reference (DFT) trace.

    Returns
    -------
    go.Figure | None
        Interactive Plotly figure or ``None`` when trajectory files are missing
        or unreadable.
    """
    ref_path = paths.get("ref_profile")
    pred_path = paths.get("pred_profile")
    if not ref_path or not pred_path:
        return None

    try:
        ref_profile = read(ref_path, ":")
        pred_profile = read(pred_path, ":")
    except Exception:
        return None

    try:
        ref_energies = np.array([at.info["DFT_energy"] for at in ref_profile])
        pred_energies = np.array([at.get_potential_energy() for at in pred_profile])
    except Exception:
        return None

    ref_shifted = ref_energies - ref_energies[0]
    pred_shifted = pred_energies - pred_energies[0]
    image_indices = list(range(len(ref_shifted)))

    hover_ref = [
        f"<b>Image {i}</b><br>ΔE = {e:.3f} eV<br><i>Click to view geometry</i>"
        for i, e in enumerate(ref_shifted)
    ]
    hover_pred = [
        f"<b>Image {i}</b><br>ΔE = {e:.3f} eV<br><i>Click to view geometry</i>"
        for i, e in enumerate(pred_shifted)
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=image_indices,
            y=ref_shifted.tolist(),
            mode="lines+markers",
            name=reference_label,
            line={"color": "#1f77b4", "width": 2},
            marker={"size": 10, "symbol": "circle", "color": "#1f77b4"},
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_ref,
        )
    )

    pred_color = "#d62728"
    fig.add_trace(
        go.Scatter(
            x=list(range(len(pred_shifted))),
            y=pred_shifted.tolist(),
            mode="lines+markers",
            name=model_label or "MLIP",
            line={"color": pred_color, "width": 2, "dash": "dash"},
            marker={"size": 10, "symbol": "circle", "color": pred_color},
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover_pred,
        )
    )

    fig.update_layout(
        title={"text": system_label, "font": {"size": 15}},
        xaxis={
            "title": "NEB Image Index",
            "tickmode": "linear",
            "dtick": 1,
            "gridcolor": "#eee",
        },
        yaxis={"title": "ΔEnergy (eV)", "gridcolor": "#eee"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "right",
            "x": 1,
        },
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="closest",
        margin={"l": 60, "r": 20, "t": 60, "b": 50},
        clickmode="event",
    )

    return fig


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
