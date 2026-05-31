"""Utilities for diamond phonon interactive assets (bands-only)."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
import pickle
from typing import Any

import matplotlib

matplotlib.use("Agg")
from dash import html
import matplotlib.pyplot as plt
import numpy as np

CM1_TO_THZ = 1.0 / 33.35640951981521

# FCC diamond BZ path: Γ→X→W→K→Γ→L→U→W→L→K→X
HS_LABELS = [r"$\Gamma$", "X", "W", "K", r"$\Gamma$", "L", "U", "W", "L", "K", "X"]


def _detect_hs_boundaries(qpoints: np.ndarray) -> list[int]:
    """
    Return indices of high-symmetry points along the q-path.

    Detects direction changes by comparing consecutive step directions.
    Always includes the first and last index.

    Parameters
    ----------
    qpoints
        Array of q-point coordinates, shape ``(N, 3)``.

    Returns
    -------
    list[int]
        Indices of high-symmetry points including the first and last.
    """
    dq = np.diff(qpoints, axis=0)
    dq_norm = np.linalg.norm(dq, axis=1)
    dq_unit = dq / (dq_norm[:, None] + 1e-12)
    cosang = np.sum(dq_unit[1:] * dq_unit[:-1], axis=1)
    turns = list(np.where(cosang < 0.95)[0] + 1)
    return [0] + turns + [len(qpoints) - 1]


def render_dispersion_component(
    selection_context: Mapping[str, Any],
    calc_root: Path,
    frequency_scale: float,
    frequency_unit: str,
    reference_label: str,
    reference_band_npz: Path | None = None,
):
    """
    Return a Dash component containing a dispersion PNG, or None.

    Parameters
    ----------
    selection_context
        Selection context; expects ``selection["band_npz"]`` relative to ``calc_root``.
    calc_root
        Root directory used to resolve relative file paths.
    frequency_scale
        Multiplicative scale applied to frequencies.
    frequency_unit
        Y-axis unit label.
    reference_label
        Legend label for the reference overlay.
    reference_band_npz
        Optional absolute path to the DFT reference ``.npz`` (frequencies in cm⁻¹).

    Returns
    -------
    html.Div | None
        Component with an embedded PNG (data URI), or None if unavailable.
    """
    model_display = selection_context.get("model")
    selected = selection_context.get("selection") or {}
    band_npz = selected.get("band_npz")
    label = selected.get("label") or selected.get("id", "")

    if not band_npz:
        return None

    image_src = _render_png(
        calc_root=Path(calc_root),
        band_npz=str(band_npz),
        reference_band_npz=reference_band_npz,
        frequency_scale=float(frequency_scale),
        frequency_unit=str(frequency_unit),
        reference_label=str(reference_label),
        prediction_label=str(model_display)
        if model_display is not None
        else "Prediction",
    )

    if not image_src:
        return None

    return html.Div(
        [
            html.H4(str(label)) if label else None,
            html.Img(
                src=image_src,
                style={
                    "width": "100%",
                    "maxWidth": "820px",
                    "height": "auto",
                    "display": "block",
                    "borderRadius": "8px",
                    "border": "1px solid #ddd",
                },
            ),
        ]
    )


def _render_png(
    *,
    calc_root: Path,
    band_npz: str,
    reference_band_npz: Path | None,
    frequency_scale: float,
    frequency_unit: str,
    reference_label: str,
    prediction_label: str,
) -> str | None:
    """
    Render a dispersion PNG from a pickle band structure with optional DFT overlay.

    Parameters
    ----------
    calc_root
        Root directory used to resolve ``band_npz``.
    band_npz
        Relative path to the pickled band-structure file.
    reference_band_npz
        Absolute path to the DFT reference ``.npz`` (frequencies in cm⁻¹).
    frequency_scale
        Multiplicative scale applied to all frequencies.
    frequency_unit
        Y-axis unit label.
    reference_label
        Legend label for the DFT reference trace.
    prediction_label
        Legend label for the MLIP prediction trace.

    Returns
    -------
    str | None
        Base64 PNG data URI, or None on failure.
    """
    pred_path = calc_root / band_npz
    if not pred_path.exists():
        return None

    try:
        with pred_path.open("rb") as f:
            band_data = pickle.load(f)
    except Exception:
        return None

    distances = band_data.get("distances", [])
    frequencies = band_data.get("frequencies", [])
    if not distances or not frequencies:
        return None

    s_pred = np.concatenate(distances)
    freqs_pred = np.vstack(frequencies) * frequency_scale

    if not np.isfinite(freqs_pred).all():
        return None

    s_ref: np.ndarray | None = None
    freqs_ref: np.ndarray | None = None
    q_ref: np.ndarray | None = None

    if reference_band_npz is not None and Path(reference_band_npz).exists():
        try:
            obj = np.load(Path(reference_band_npz), allow_pickle=False)
            s_ref = np.asarray(obj["distance"], dtype=float)
            q_ref = np.asarray(obj["qpoints"], dtype=float)
            freqs_ref_cm1 = np.asarray(obj["freqs_cm1"], dtype=float)
            if freqs_ref_cm1.ndim == 2 and np.isfinite(freqs_ref_cm1).all():
                freqs_ref = freqs_ref_cm1 * CM1_TO_THZ * frequency_scale
            else:
                s_ref = None
                q_ref = None
        except Exception:
            s_ref = None
            q_ref = None

    x = s_ref if (s_ref is not None and len(s_ref) == len(s_pred)) else s_pred

    # Detect HS boundaries: prefer DFT q-points, fall back to MLIP q-points.
    boundary_indices: list[int] | None = None
    if q_ref is not None and len(q_ref) == len(s_pred):
        boundary_indices = _detect_hs_boundaries(q_ref)
    elif "qpoints" in band_data:
        q_pred = np.concatenate(band_data["qpoints"])
        if len(q_pred) == len(s_pred):
            boundary_indices = _detect_hs_boundaries(q_pred)

    fig, ax = plt.subplots(figsize=(10, 7))

    for j in range(freqs_pred.shape[1]):
        ax.plot(
            x,
            freqs_pred[:, j],
            color="#1f77b4",
            lw=2.0,
            label=prediction_label if j == 0 else None,
        )

    if freqs_ref is not None and s_ref is not None:
        x_ref = x if len(s_ref) == len(s_pred) else s_ref
        for j in range(freqs_ref.shape[1]):
            ax.plot(
                x_ref,
                freqs_ref[:, j],
                color="k",
                lw=2.0,
                ls="--",
                label=reference_label if j == 0 else None,
            )

    if boundary_indices is not None and len(boundary_indices) == len(HS_LABELS):
        sym_pos = [float(x[i]) for i in boundary_indices]
        tick_labels = HS_LABELS
    else:
        sym_pos = [float(x[0]), float(x[-1])]
        tick_labels = [r"$\Gamma$", "X"]

    for xpos in sym_pos:
        ax.axvline(xpos, color="k", lw=1.0, alpha=0.6)
    ax.set_xticks(sym_pos, tick_labels, fontsize=12)
    ax.set_xlim(sym_pos[0], sym_pos[-1])
    ax.set_xlabel("Wave vector", fontsize=18)
    ax.set_ylabel(f"Frequency ({frequency_unit})", fontsize=18)
    ax.axhline(0.0, color="k", lw=1.5)
    ax.grid(axis="x")

    all_freqs = [freqs_pred]
    if freqs_ref is not None:
        all_freqs.append(freqs_ref)
    all_flat = np.concatenate([f.ravel() for f in all_freqs])
    ax.set_ylim(float(np.nanmin(all_flat)), float(np.nanmax(all_flat)))

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc=1, fontsize=14)

    fig.tight_layout()
    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=200)
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
