"""Utilities for building OC20NEB interactive assets."""

from __future__ import annotations

import base64
from collections.abc import Iterable, Mapping
from io import BytesIO
from typing import Any

from ase.io import read
import matplotlib

matplotlib.use("Agg")
from dash import html
import matplotlib.pyplot as plt
import numpy as np


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
):
    """
    Render a Matplotlib dispersion PNG or fallback image for a selection.

    Parameters
    ----------
    selection_context
        Dictionary containing ``model`` and resolved ``selection`` data.
    reference_label
        Legend label for the reference trace.

    Returns
    -------
    dash.html.Div | None
        Component containing the image preview, or ``None`` if missing.
    """
    model_display = selection_context.get("model")
    selected = selection_context.get("selection") or {}
    data_paths = selected.get("data_paths")
    label = selected.get("label") or selected.get("reaction", "")
    image_src = None
    if data_paths:
        image_src = render_neb_profile_png(
            paths=data_paths,
            model_label=model_display,
            system_label=label,
            reference_label=reference_label,
            prediction_label=model_display,
        )
    elif selected.get("image"):
        image_src = f"/{selected['image']}"
    if not image_src:
        return None
    return html.Div(
        [
            html.H4(label),
            html.Img(
                src=image_src,
                style={"maxWidth": "100%", "border": "1px solid #ccc"},
            ),
        ]
    )


def render_neb_profile_png(
    *,
    paths: Mapping[str, str | None],
    model_label: str,
    system_label: str,
    reference_label: str = "Reference",
    prediction_label: str = "Prediction",
) -> str | None:
    """
    Render NEB profiles of Reference and Predicted as a PNG data URI.

    Parameters
    ----------
    paths
        Mapping with ``"ref_profile"``, ``"pred_profile"``.
    model_label
        Label for the predicted model trace.
    system_label
        Title displayed above the plot.
    reference_label
        Legend label for the reference trace.
    prediction_label
        Legend label for the predicted trace.

    Returns
    -------
    str | None
        Base64-encoded data URI or ``None`` when assets are missing.
    """
    ref_profile = read(paths["ref_profile"], ":")
    pred_profile = read(paths["pred_profile"], ":")

    ref_energies = np.array([at.info["DFT_energy"] for at in ref_profile])
    pred_energies = np.array([at.get_potential_energy() for at in pred_profile])

    fig = plt.figure(figsize=(7, 6))

    fontsize = 18
    plt.plot(ref_energies - ref_energies[0], c="b", marker="o", label="RPBE")
    plt.plot(
        pred_energies - pred_energies[0],
        c="r",
        linestyle="--",
        marker="o",
        label=f"{model_label}",
    )
    plt.xlabel("# Image", fontsize=fontsize)
    plt.ylabel(r"$\Delta$Energy (eV)", fontsize=fontsize)
    plt.tick_params(axis="both", which="major", labelsize=fontsize)
    plt.title(f"{system_label}", fontsize=fontsize)
    plt.legend(fontsize=fontsize, frameon=False)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
