"""
Utilities for diamond phonon interactive assets (bands-only).

This module only supports rendering a dispersion preview from:

- Predicted phonopy band.yaml (THz)
- Reference dft_band.npz with keys: distance, qpoints, frequencies (cm^-1)
"""

from __future__ import annotations

import base64
from collections.abc import Mapping
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
from dash import html
import matplotlib.pyplot as plt
import numpy as np

THZ_TO_CM1 = 33.35640951981521
HS_LABELS = [r"$\Gamma$", "X", "W", "K", r"$\Gamma$", "L", "U", "W", "L", "K", "X"]


def render_dispersion_component(
    selection_context: Mapping[str, Any],
    calc_root: Path,
    frequency_scale: float,
    frequency_unit: str,
    reference_label: str,
    reference_band_npz: Path | None = None,
):
    """
    Render a Matplotlib dispersion PNG (prediction + reference overlay).

    Parameters
    ----------
    selection_context
        Mapping containing the current selection. Must contain a ``selection`` entry
        with a ``band_yaml`` path.
    calc_root
        Root directory for calculation outputs.
    frequency_scale
        Multiplicative scale applied to both predicted and reference frequencies.
    frequency_unit
        Unit label shown on the y-axis.
    reference_label
        Label used for the reference curve in the legend.
    reference_band_npz
        Optional path to the reference ``dft_band.npz`` file, relative to
        ``calc_root``.

    Returns
    -------
    dash.html.Div | None
        A Dash component containing the rendered PNG, or ``None`` if no band file
        is selected or rendering fails.
    """
    model_display = selection_context.get("model")
    selected = selection_context.get("selection") or {}

    band_yaml = selected.get("band_yaml")
    label = selected.get("label") or selected.get("id", "")

    if not band_yaml:
        return None

    image_src = render_band_yaml_png(
        calc_root=calc_root,
        band_yaml=str(band_yaml),
        reference_band_npz=reference_band_npz,
        system_label=str(label),
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
            html.H4(label),
            html.Img(
                src=image_src,
                style={"maxWidth": "100%", "border": "1px solid #ccc"},
            ),
        ]
    )


def render_band_yaml_png(
    *,
    calc_root: Path,
    band_yaml: str,
    reference_band_npz: Path | None,
    system_label: str,
    frequency_scale: float,
    frequency_unit: str,
    reference_label: str,
    prediction_label: str,
) -> str | None:
    """
    Render dispersion by parsing a phonopy band.yaml and overlaying a DFT reference.

    Parameters
    ----------
    calc_root
        Root directory for calculation outputs.
    band_yaml
        Path to the predicted phonopy ``band.yaml`` relative to ``calc_root``.
    reference_band_npz
        Optional path to the reference ``dft_band.npz`` relative to ``calc_root``.
    system_label
        Label identifying the system being plotted.
    frequency_scale
        Multiplicative scale applied to both predicted and reference frequencies.
    frequency_unit
        Unit label shown on the y-axis.
    reference_label
        Legend label for the reference curves.
    prediction_label
        Legend label for the prediction curves.

    Returns
    -------
    str | None
        A base64-encoded PNG image (``data:image/png;base64,...``), or ``None`` if
        the input data could not be loaded.
    """
    import yaml  # type: ignore

    def _detect_symmetry_boundaries(q_ref: np.ndarray) -> list[int]:
        """
        Detect symmetry-point boundaries along the reference q-point path.

        Parameters
        ----------
        q_ref
            Reference q-point coordinates of shape ``(Nq, 3)``.

        Returns
        -------
        list[int]
            Indices of q-points corresponding to symmetry boundaries.
        """
        dq = np.diff(q_ref, axis=0)
        dq_norm = np.linalg.norm(dq, axis=1)
        eps = 1e-12
        dq_unit = dq / (dq_norm[:, None] + eps)
        cosang = np.sum(dq_unit[1:] * dq_unit[:-1], axis=1)
        cand = np.where(cosang < 0.95)[0] + 1

        boundaries = [0]
        for i in cand:
            if i + 1 < len(q_ref) and np.allclose(q_ref[i], q_ref[i + 1], atol=1e-10):
                boundaries.append(int(i))
            else:
                boundaries.append(int(i))

        boundaries = sorted(set(boundaries))
        if boundaries[0] != 0:
            boundaries = [0] + boundaries
        if boundaries[-1] != len(q_ref) - 1:
            boundaries.append(len(q_ref) - 1)
        return boundaries

    pred_path = Path(calc_root) / band_yaml
    if not pred_path.exists():
        return None

    try:
        with pred_path.open("r", encoding="utf8") as f:
            y = yaml.safe_load(f)
    except Exception:
        return None

    phonon = y.get("phonon", None)
    if not isinstance(phonon, list) or not phonon:
        return None

    s_pred = np.asarray([p.get("distance", np.nan) for p in phonon], float)

    freqs_pred_thz = np.asarray(
        [[b.get("frequency", np.nan) for b in p.get("band", [])] for p in phonon],
        float,
    )
    if freqs_pred_thz.ndim != 2 or not np.isfinite(freqs_pred_thz).all():
        return None

    freqs_pred_cm1 = freqs_pred_thz * THZ_TO_CM1 * frequency_scale

    s_ref = None
    q_ref = None
    freqs_ref_cm1 = None
    sym_pos = None

    if reference_band_npz is not None:
        ref_path = Path(calc_root) / reference_band_npz
        if ref_path.exists():
            obj = np.load(ref_path, allow_pickle=False)

            s_ref = np.asarray(obj["distance"], float)
            q_ref = np.asarray(obj["qpoints"], float)
            freqs_ref_cm1 = np.asarray(obj["frequencies"], float)

            if freqs_ref_cm1.ndim == 3 and freqs_ref_cm1.shape[0] == 1:
                freqs_ref_cm1 = freqs_ref_cm1[0]

            if freqs_ref_cm1.ndim != 2:
                s_ref = q_ref = freqs_ref_cm1 = None
            else:
                freqs_ref_cm1 = freqs_ref_cm1 * frequency_scale
                bounds = _detect_symmetry_boundaries(q_ref)
                sym_pos = s_ref[bounds]

    use_ref_x = s_ref is not None and len(s_ref) == len(s_pred)
    x = s_ref if use_ref_x else s_pred

    fig, ax = plt.subplots(figsize=(8, 5))

    for band in freqs_pred_cm1.T:
        ax.plot(x, band, "--", lw=1.0, color="red")

    if freqs_ref_cm1 is not None:
        for band in freqs_ref_cm1.T:
            ax.plot(x if use_ref_x else s_ref, band, "-", lw=1.0, color="blue")

    if sym_pos is not None and len(sym_pos) == len(HS_LABELS):
        for xpos in sym_pos:
            ax.axvline(float(xpos), color="k", lw=0.8)
        ax.set_xticks(sym_pos)
        ax.set_xticklabels(HS_LABELS)
        ax.set_xlim(float(x[0]), float(x[-1]))

    ax.set_xlabel("Wave vector")
    ax.set_ylabel(rf"Frequency ({frequency_unit})")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.grid(True, linestyle=":", linewidth=0.5)
    ax.set_ylim(bottom=0.0)

    from matplotlib.lines import Line2D

    handles = []
    if freqs_ref_cm1 is not None:
        handles.append(
            Line2D([0], [0], color="blue", lw=1.0, ls="-", label=reference_label)
        )
    handles.append(
        Line2D([0], [0], color="red", lw=1.0, ls="--", label=prediction_label)
    )
    ax.legend(handles=handles, frameon=False)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"
