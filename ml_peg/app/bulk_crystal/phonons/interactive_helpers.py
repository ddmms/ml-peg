"""Utilities for building phonon interactive assets."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from io import BytesIO
import json
from pathlib import Path
import pickle
from typing import Any

import matplotlib

matplotlib.use("Agg")
from dash import dcc, html
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np

from ml_peg.app.utils.plot_helpers import build_violin_distribution


def _load_band(calc_root: Path, rel_path: str | None) -> dict[str, Any] | None:
    """
    Load a band-structure dictionary from disk.

    Parameters
    ----------
    calc_root
        Base directory containing serialized outputs.
    rel_path
        Relative path to the pickled ``*_band_structure`` file.

    Returns
    -------
    dict[str, Any] | None
        Parsed band-structure data or ``None`` when the file is missing.
    """
    if not rel_path:
        return None
    try:
        full_path = calc_root / rel_path
        with full_path.open("rb") as handle:
            band_data = pickle.load(handle)

        base_stem = full_path.stem.replace("_band_structure", "")
        labels_path = full_path.parent / f"{base_stem}_labels.json"
        connections_path = full_path.parent / f"{base_stem}_connections.json"

        if labels_path.exists():
            with labels_path.open() as file_obj:
                band_data["labels"] = json.load(file_obj)
        if connections_path.exists():
            with connections_path.open() as file_obj:
                band_data["path_connections"] = json.load(file_obj)

        return band_data
    except OSError:
        return None


def _load_dos(calc_root: Path, rel_path: str | None):
    """
    Load DOS (frequency_points, total_dos) arrays from disk.

    Parameters
    ----------
    calc_root
        Base directory containing serialized outputs.
    rel_path
        Relative path to the pickled DOS file.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        Frequency points and DOS values, or ``None`` when missing.
    """
    if not rel_path:
        return None
    try:
        with (calc_root / rel_path).open("rb") as handle:
            data = pickle.load(handle)
        return data["frequency_points"], data["total_dos"]
    except OSError:
        return None


def _build_xticks(distances, labels, connections):
    """
    Construct Brillouin-path ticks for Matplotlib.

    Parameters
    ----------
    distances
        Sequence of cumulative k-point distances for each path segment.
    labels
        High-symmetry point labels associated with each segment.
    connections
        Boolean mask indicating whether consecutive segments are connected.

    Returns
    -------
    tuple[list[float], list[str]]
        Matplotlib tick positions and corresponding labels.
    """
    xticks, xticklabels = [], []
    cumulative_dist, index = 0.0, 0
    connections = [True] + connections
    for seg_dist, connected in zip(distances, connections, strict=False):
        start, end = labels[index], labels[index + 1]
        pos_start = cumulative_dist
        pos_end = cumulative_dist + (seg_dist[-1] - seg_dist[0])
        xticks.append(pos_start)
        xticklabels.append(f"{start}|{end}" if not connected else start)
        index += 2 if not connected else 1
        cumulative_dist = pos_end
    xticks.append(cumulative_dist)
    xticklabels.append(labels[-1])
    return xticks, xticklabels


def _flatten_band_errors(errors: Mapping[str, Any]) -> tuple[list[float], list[str]]:
    """
    Convert band-error mapping into aligned ``(values, labels)`` lists.

    Parameters
    ----------
    errors
        Mapping from Materials Project ID to error series (scalar/composite).

    Returns
    -------
    tuple[list[float], list[str]]
        Flattened values and matching MP-ID labels for violin plotting.
    """
    values: list[float] = []
    labels: list[str] = []
    for mp_id, series in errors.items():
        if series is None:
            continue
        if isinstance(series, (int | float | np.floating | np.integer)):
            values.append(float(series))
            labels.append(str(mp_id))
            continue
        if isinstance(series, np.ndarray):
            flattened = series.ravel().tolist()
            values.extend(flattened)
            labels.extend([str(mp_id)] * len(flattened))
            continue
        if isinstance(series, (list | tuple)):
            numeric_vals = [
                float(val)
                for val in series
                if isinstance(val, (int | float | np.floating | np.integer))
            ]
            values.extend(numeric_vals)
            labels.extend([str(mp_id)] * len(numeric_vals))
            continue
        try:
            values.append(float(series))
            labels.append(str(mp_id))
        except (TypeError, ValueError):
            continue
    return values, labels


def build_bz_violin_content(
    model_display: str,
    column_id: str,  # Dash passes column_id to every handler; unused for BZ plots.
    *,
    models_data: Mapping[str, Any],
    scatter_id: str,
    instructions: str,
    yaxis_title: str,
    hovertemplate: str,
):
    """
    Build the Brillouin-zone error distribution content.

    Parameters
    ----------
    model_display
        Table entry label for the selected model.
    column_id
        Column identifier (unused).
    models_data
        Interactive dataset keyed by model name.
    scatter_id
        Graph identifier used for the violin plot.
    instructions
        Instructional text displayed above the violin plot.
    yaxis_title
        Label describing the violin axis.
    hovertemplate
        Plotly hover template applied to each violin point.

    Returns
    -------
    tuple
        ``(content, meta)`` pair consumed by the shared callback helper.
    """
    band_errors = models_data.get(model_display, {}).get("band_errors", {})
    values, labels = _flatten_band_errors(band_errors)
    fig = build_violin_distribution(
        values,
        labels,
        title=f"{model_display} – Brillouin zone error distribution",
        yaxis_title=yaxis_title,
        hovertemplate=hovertemplate,
    )
    graph = dcc.Graph(id=scatter_id, figure=fig)
    meta = {"model": model_display, "type": "bz"}
    content = html.Div([html.P(instructions), graph])
    return content, meta


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
    target = str(point_id)
    for metric_data in model_entry.get("metrics", {}).values():
        for point in metric_data.get("points", []):
            if str(point.get("id")) == target:
                return point
    for point in model_entry.get("stability", {}).get("points", []):
        if str(point.get("id")) == target:
            return point
    return None


def render_dispersion_component(
    selection_context: Mapping[str, Any],
    scatter_meta: Mapping[str, Any],  # Unused but kept for signature parity.
    *,
    calc_root: Path,
    frequency_scale: float,
    frequency_unit: str,
    reference_label: str,
):
    """
    Render a Matplotlib dispersion PNG or fallback image for a selection.

    Parameters
    ----------
    selection_context
        Dictionary containing ``model`` and resolved ``selection`` data.
    scatter_meta
        Metadata describing the scatter context (unused).
    calc_root
        Base directory containing serialized band/DOS assets.
    frequency_scale
        Multiplicative factor applied to raw frequencies.
    frequency_unit
        Unit label displayed on the y-axis after scaling.
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
    label = selected.get("label") or selected.get("id", "")
    image_src = None
    if data_paths:
        image_src = render_band_dos_png(
            calc_root=calc_root,
            paths=data_paths,
            model_label=model_display,
            system_label=label,
            frequency_scale=frequency_scale,
            frequency_unit=frequency_unit,
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


def render_band_dos_png(
    *,
    calc_root: Path,
    paths: Mapping[str, str | None],
    model_label: str,
    system_label: str,
    frequency_scale: float = 1.0,
    frequency_unit: str = "THz",
    reference_label: str = "Reference",
    prediction_label: str = "Prediction",
) -> str | None:
    """
    Render a band-structure + DOS comparison as a PNG data URI.

    Parameters
    ----------
    calc_root
        Base directory containing serialized band/DOS assets.
    paths
        Mapping with ``"ref_band"``, ``"pred_band"``, ``"ref_dos"``, ``"pred_dos"``.
    model_label
        Label for the predicted model trace.
    system_label
        Title displayed above the plot.
    frequency_scale
        Multiplicative factor applied to raw frequencies (e.g. THz -> K).
    frequency_unit
        Unit label displayed on the y-axis after scaling.
    reference_label
        Legend label for the reference trace.
    prediction_label
        Legend label for the predicted trace.

    Returns
    -------
    str | None
        Base64-encoded data URI or ``None`` when assets are missing.
    """
    calc_root = Path(calc_root)
    ref_band = _load_band(calc_root, paths.get("ref_band"))
    pred_band = _load_band(calc_root, paths.get("pred_band"))
    ref_dos = _load_dos(calc_root, paths.get("ref_dos"))
    pred_dos = _load_dos(calc_root, paths.get("pred_dos"))
    if not all([ref_band, pred_band, ref_dos, pred_dos]):
        return None

    fig = plt.figure(figsize=(9, 5))
    gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
    ax1 = fig.add_axes([0.12, 0.07, 0.67, 0.85])
    ax2 = fig.add_axes([0.82, 0.07, 0.17, 0.85])

    distances_ref = ref_band["distances"]
    frequencies_ref = [
        np.asarray(segment) * frequency_scale for segment in ref_band["frequencies"]
    ]
    distances_pred = pred_band["distances"]
    frequencies_pred = [
        np.asarray(segment) * frequency_scale for segment in pred_band["frequencies"]
    ]
    dos_freqs_ref, dos_values_ref = ref_dos
    dos_freqs_ref = np.asarray(dos_freqs_ref) * frequency_scale
    dos_freqs_pred, dos_values_pred = pred_dos
    dos_freqs_pred = np.asarray(dos_freqs_pred) * frequency_scale

    pred_label_added = False
    for dist_segment, freq_segment in zip(
        distances_pred, frequencies_pred, strict=False
    ):
        for band in freq_segment.T:
            ax1.plot(
                dist_segment,
                band,
                lw=1,
                linestyle="--",
                color="red",
                label=prediction_label if not pred_label_added else None,
            )
            pred_label_added = True

    ax2.plot(dos_values_pred, dos_freqs_pred, lw=1.2, color="red", linestyle="--")

    ref_label_added = False
    for dist_segment, freq_segment in zip(distances_ref, frequencies_ref, strict=False):
        for band in freq_segment.T:
            ax1.plot(
                dist_segment,
                band,
                lw=1,
                linestyle="-",
                color="blue",
                label=reference_label if not ref_label_added else None,
            )
            ref_label_added = True

    ax2.plot(dos_values_ref, dos_freqs_ref, lw=1.2, color="blue")

    labels = ref_band.get("labels", [])
    connections = ref_band.get("path_connections", [])
    if labels and connections:
        xticks, xticklabels = _build_xticks(distances_ref, labels, connections)
        for x_val in xticks:
            ax1.axvline(x=x_val, color="k", linewidth=1)
        ax1.set_xticks(xticks, xticklabels)
        ax1.set_xlim(xticks[0], xticks[-1])

    ax1.axhline(0, color="k", linewidth=1)
    ax2.axhline(0, color="k", linewidth=1)
    ax1.set_ylabel(f"Frequency ({frequency_unit})", fontsize=16)
    ax1.set_xlabel("Wave Vector", fontsize=16)
    ax1.tick_params(axis="both", which="major", labelsize=14)

    pred_flat = np.concatenate(frequencies_pred).flatten()
    ref_flat = np.concatenate(frequencies_ref).flatten()
    all_freqs = np.concatenate([pred_flat, ref_flat])
    ax1.set_ylim(all_freqs.min() - 0.4, all_freqs.max() + 0.4)
    ax2.set_ylim(ax1.get_ylim())

    plt.setp(ax2.get_yticklabels(), visible=False)
    ax2.set_xlabel("DOS")

    handles, labels = ax1.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    if by_label:
        fig.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper center",
            bbox_to_anchor=(0.8, 1.02),
            frameon=False,
            ncol=2,
            fontsize=14,
        )

    ax1.grid(True, linestyle=":", linewidth=0.5)
    ax2.grid(True, linestyle=":", linewidth=0.5)
    fig.suptitle(system_label, x=0.4, fontsize=14)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
