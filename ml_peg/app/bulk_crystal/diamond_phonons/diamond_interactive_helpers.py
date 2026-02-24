"""
Utilities for diamond phonon interactive assets (bands-only).

Supports rendering a dispersion preview from:
- Predicted Phonopy band.yaml (THz)
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
    Return a Dash component containing a dispersion PNG, or None.

    Parameters
    ----------
    selection_context : Mapping[str, Any]
        Selection context; expects selection["band_yaml"] and optional label/id.
    calc_root : Path
        Root directory used to resolve files.
    frequency_scale : float
        Multiplicative scale applied to frequencies.
    frequency_unit : str
        Y-axis unit label.
    reference_label : str
        Legend label for the reference overlay.
    reference_band_npz : Path | None
        Optional reference .npz (distance, qpoints, frequencies in cm^-1).

    Returns
    -------
    html.Div | None
        Component with an embedded PNG (data URI), or None if unavailable.
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


def render_band_yaml_png(
    *,
    calc_root: Path,
    band_yaml: str,
    reference_band_npz: Path | None,
    frequency_scale: float,
    frequency_unit: str,
    reference_label: str,
    prediction_label: str,
) -> str | None:
    """
    Render a dispersion PNG from band.yaml with optional reference overlay.

    Parameters
    ----------
    calc_root : Path
        Root directory used to resolve files.
    band_yaml : str
        Predicted Phonopy band.yaml path (relative to calc_root).
    reference_band_npz : Path | None
        Optional reference .npz (distance, qpoints, frequencies in cm^-1).
    frequency_scale : float
        Multiplicative scale applied to frequencies.
    frequency_unit : str
        Y-axis unit label.
    reference_label : str
        Legend label for the reference overlay.
    prediction_label : str
        Legend label for the prediction.

    Returns
    -------
    str | None
        Base64 PNG data URI, or None on failure.
    """
    import yaml  # type: ignore

    def _detect_symmetry_boundaries(q_ref: np.ndarray) -> list[int]:
        """
        Return indices of k-path corners (including first and last).

        Parameters
        ----------
        q_ref : np.ndarray
            Array of q-points with shape (N, 3) along the band path.

        Returns
        -------
        list[int]
            Sorted indices of symmetry boundaries (path corners), including
            indices 0 and N-1.
        """
        dq = np.diff(q_ref, axis=0)
        dq_norm = np.linalg.norm(dq, axis=1)
        eps = 1e-12
        dq_unit = dq / (dq_norm[:, None] + eps)
        cosang = np.sum(dq_unit[1:] * dq_unit[:-1], axis=1)
        cand = np.where(cosang < 0.95)[0] + 1

        boundaries = [0]
        for i in cand:
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

    s_pred = np.asarray([p.get("distance", np.nan) for p in phonon], dtype=float)

    freqs_pred_thz = np.asarray(
        [[b.get("frequency", np.nan) for b in p.get("band", [])] for p in phonon],
        dtype=float,
    )
    if freqs_pred_thz.ndim != 2 or not np.isfinite(freqs_pred_thz).all():
        return None

    freqs_pred_thz = freqs_pred_thz * float(frequency_scale)

    s_ref: np.ndarray | None = None
    q_ref: np.ndarray | None = None
    freqs_ref_thz: np.ndarray | None = None
    sym_pos: np.ndarray | None = None

    if reference_band_npz is not None:
        ref_path = Path(calc_root) / reference_band_npz
        if ref_path.exists():
            obj = np.load(ref_path, allow_pickle=False)

            s_ref = np.asarray(obj["distance"], dtype=float)
            q_ref = np.asarray(obj["qpoints"], dtype=float)
            freqs_ref_cm1 = np.asarray(obj["frequencies"], dtype=float)

            if freqs_ref_cm1.ndim == 3 and freqs_ref_cm1.shape[0] == 1:
                freqs_ref_cm1 = freqs_ref_cm1[0]

            if freqs_ref_cm1.ndim == 2 and np.isfinite(freqs_ref_cm1).all():
                freqs_ref_thz = (freqs_ref_cm1 / THZ_TO_CM1) * float(frequency_scale)
                bounds = _detect_symmetry_boundaries(q_ref)
                sym_pos = s_ref[bounds]
            else:
                s_ref = None
                q_ref = None
                freqs_ref_thz = None
                sym_pos = None

    use_ref_x = s_ref is not None and len(s_ref) == len(s_pred)
    x = s_ref if use_ref_x else s_pred

    fig, ax = plt.subplots(figsize=(10, 7))

    for j in range(freqs_pred_thz.shape[1]):
        ax.plot(
            x,
            freqs_pred_thz[:, j],
            color="#1f77b4",
            lw=2.0,
            label=prediction_label if j == 0 else None,
        )

    if freqs_ref_thz is not None and s_ref is not None:
        x_ref = x if use_ref_x else s_ref
        for j in range(freqs_ref_thz.shape[1]):
            ax.plot(
                x_ref,
                freqs_ref_thz[:, j],
                color="k",
                lw=2.0,
                ls="--",
                label=reference_label if j == 0 else None,
            )

    if sym_pos is not None and len(sym_pos) == len(HS_LABELS):
        for xpos in sym_pos:
            ax.axvline(float(xpos), color="k", lw=1.0, alpha=0.6)
        ax.set_xticks(sym_pos)
        ax.set_xticklabels(HS_LABELS)
        ax.set_xlim(float(x[0]), float(x[-1]))

    ax.set_xlabel("Wave vector", fontsize=18)
    ax.set_ylabel(f"Frequency ({frequency_unit})", fontsize=18)
    ax.axhline(0.0, color="k", lw=1.5)
    ax.grid(axis="x")
    ax.set_ylim(
        float(
            min(
                np.nanmin(freqs_pred_thz),
                np.nanmin(freqs_ref_thz) if freqs_ref_thz is not None else np.inf,
            )
        ),
        float(
            max(
                np.nanmax(freqs_pred_thz),
                np.nanmax(freqs_ref_thz) if freqs_ref_thz is not None else -np.inf,
            )
        ),
    )

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
