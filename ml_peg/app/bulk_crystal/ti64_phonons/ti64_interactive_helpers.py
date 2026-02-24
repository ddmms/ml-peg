"""Helpers for Ti64 phonon interactive dispersion/DOS rendering."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from functools import lru_cache
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
from dash import html
import matplotlib.pyplot as plt


def _load_json(path: Path) -> Any:
    """
    Load a JSON file from disk.

    Parameters
    ----------
    path
        Path to a JSON file.

    Returns
    -------
    Any
        Parsed JSON payload.
    """
    with path.open("r", encoding="utf8") as f:
        return json.load(f)


def lookup_system_entry(
    model_entry: Mapping[str, Any],
    point_id: str | int,
    *,
    data_root: Path,
    assets_prefix: str,
) -> dict[str, Any] | None:
    """
    Resolve a scatter (model, point) selection into a data-path payload.

    Parameters
    ----------
    model_entry
        Model entry from the interactive dataset (contains ``model`` and ``metrics``).
    point_id
        Point identifier selected in the scatter.
    data_root
        Root directory for app data (kept for API compatibility; unused here).
    assets_prefix
        Assets prefix for app data (kept for API compatibility; unused here).

    Returns
    -------
    dict[str, Any] or None
        Selection dictionary containing ``model``, ``system``, and ``data_paths`` if the
        point is found; otherwise ``None``.
    """
    _ = data_root
    _ = assets_prefix

    model_name = (
        model_entry.get("model") or model_entry.get("id") or model_entry.get("name")
    )
    if not isinstance(model_name, str) or not model_name.strip():
        return None

    system_name = str(point_id)

    metrics = model_entry.get("metrics", {})
    if not isinstance(metrics, dict):
        return None

    # Search all metrics for a matching point id
    for _metric_id, metric_payload in metrics.items():
        if not isinstance(metric_payload, dict):
            continue
        points = metric_payload.get("points", [])
        if not isinstance(points, list):
            continue
        for p in points:
            if not isinstance(p, dict):
                continue
            if str(p.get("id")) != system_name:
                continue

            data_paths = p.get("data_paths")
            if not isinstance(data_paths, dict):
                return None

            # must include at least the calc npz path
            npz_rel = data_paths.get("npz")
            if not isinstance(npz_rel, str) or not npz_rel:
                return None

            meta_rel = data_paths.get("meta")

            return {
                "model": model_name,
                "system": system_name,
                "data_paths": {
                    "npz": npz_rel,
                    "meta": meta_rel,
                },
                "label": p.get("label"),
                "ref": p.get("ref"),
                "pred": p.get("pred"),
            }

    return None


def resample_dft_to_ml_grid(
    dft_x: np.ndarray, dft_freqs: np.ndarray, n_ml: int
) -> np.ndarray:
    """
    Resample DFT frequencies onto a uniform ML grid spanning the same path.

    Parameters
    ----------
    dft_x
        DFT path coordinate array of shape ``(n_dft,)``.
    dft_freqs
        DFT frequencies array of shape ``(n_dft, n_branches)``.
    n_ml
        Number of ML q-points.

    Returns
    -------
    numpy.ndarray
        DFT frequencies interpolated onto the ML grid, shape
        ``(n_ml, n_branches)``.
    """
    dft_x = np.asarray(dft_x, dtype=float).reshape(-1)
    dft_freqs = np.asarray(dft_freqs, dtype=float)

    ml_x = np.linspace(dft_x[0], dft_x[-1], int(n_ml), dtype=float)

    out = np.empty((ml_x.size, dft_freqs.shape[1]), dtype=float)
    for j in range(dft_freqs.shape[1]):
        out[:, j] = np.interp(ml_x, dft_x, dft_freqs[:, j])
    return out


def gaussian_broadened_dos(
    freqs_flat: np.ndarray,
    weights_flat: np.ndarray,
    grid: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """
    Compute a Gaussian-broadened DOS on a target frequency grid.

    Parameters
    ----------
    freqs_flat
        Flattened frequencies (e.g. THz), shape ``(n_modes,)``.
    weights_flat
        Flattened weights matching ``freqs_flat``, shape ``(n_modes,)``.
    grid
        Frequency grid to evaluate the DOS on.
    sigma
        Gaussian broadening (same units as ``grid``).

    Returns
    -------
    numpy.ndarray
        DOS values evaluated on ``grid``.
    """
    f = np.asarray(freqs_flat, dtype=float).reshape(-1)
    w = np.asarray(weights_flat, dtype=float).reshape(-1)
    x = np.asarray(grid, dtype=float).reshape(-1)

    diff = f[:, None] - x[None, :]
    return np.sum(w[:, None] * np.exp(-0.5 * (diff / sigma) ** 2), axis=0)


def gaussian_smooth_on_grid(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Smooth 1D data on a uniform grid using a Gaussian kernel.

    Parameters
    ----------
    x
        Grid values.
    y
        Values defined on ``x``.
    sigma
        Gaussian kernel standard deviation in the same units as ``x``.

    Returns
    -------
    numpy.ndarray
        Smoothed values on the same grid.
    """
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)

    if x.size < 3:
        return y

    dx = float(np.median(np.diff(x)))
    if dx <= 0:
        return y

    sigma_pts = sigma / dx
    half = int(max(3, np.ceil(4.0 * sigma_pts)))
    t = np.arange(-half, half + 1, dtype=float)

    k = np.exp(-0.5 * (t / sigma_pts) ** 2)
    k /= np.sum(k)

    return np.convolve(y, k, mode="same")


@lru_cache(maxsize=512)
def _render_npz_to_data_uri(
    npz_path_str: str, meta_path_str: str | None
) -> tuple[str | None, dict[str, Any]]:
    """
    Render a dispersion/DOS PNG from a calc-stage NPZ and optional meta JSON.

    Parameters
    ----------
    npz_path_str
        Path to the calc-stage NPZ file (string for cache key stability).
    meta_path_str
        Optional path to a metadata JSON file (string for cache key stability).

    Returns
    -------
    tuple[str | None, dict[str, Any]]
        ``(data_uri, extras)`` where ``data_uri`` is a PNG data URI or ``None`` if the
        NPZ file is missing. ``extras`` contains values used to build the caption.
    """
    npz_path = Path(npz_path_str)
    if not npz_path.exists():
        return None, {}

    data = np.load(npz_path, allow_pickle=True)

    labels = None
    if meta_path_str:
        meta_path = Path(meta_path_str)
        if meta_path.exists():
            try:
                meta = _load_json(meta_path)
                labels = meta.get("labels", None) if isinstance(meta, dict) else None
                if not isinstance(labels, list):
                    labels = None
            except Exception:
                labels = None

    # Dispersion
    dft_x = np.asarray(data["dft_x"], dtype=float)
    dft_freq = np.asarray(data["dft_frequencies"], dtype=float)
    ml_freq = np.asarray(data["ml_frequencies"], dtype=float)

    dft_on_ml = resample_dft_to_ml_grid(dft_x, dft_freq, n_ml=ml_freq.shape[0])

    omega_avg_ref = float(np.mean(dft_on_ml))
    omega_avg_pred = float(np.mean(ml_freq))

    y_lo = float(min(np.min(dft_on_ml), np.min(ml_freq)))
    y_hi = float(max(np.max(dft_on_ml), np.max(ml_freq)))

    # DOS
    required = [
        "pdos_frequency_points",
        "pdos_projected",
        "q_weights",
        "q_frequencies_dft",
    ]
    has_dos = all(k in data.files for k in required)

    dos_grid = None
    dft_dos_plot = None
    ml_dos_plot = None

    if has_dos:
        fgrid = np.asarray(data["pdos_frequency_points"], dtype=float).reshape(-1)
        pdos_proj = np.asarray(data["pdos_projected"], dtype=float)

        if pdos_proj.ndim == 2 and pdos_proj.shape[0] == fgrid.size:
            ml_dos = np.mean(pdos_proj, axis=1)
        else:
            ml_dos = np.mean(pdos_proj, axis=0)
        ml_dos = np.asarray(ml_dos, dtype=float).reshape(-1)

        area_ml = float(np.trapz(ml_dos, fgrid))
        ml_dos_n = ml_dos / area_ml if area_ml > 0 else ml_dos

        q_w = np.asarray(data["q_weights"], dtype=float)
        q_f = np.asarray(data["q_frequencies_dft"], dtype=float)

        weights_tile = np.tile(q_w[:, None], (1, q_f.shape[1])).reshape(-1, order="F")
        freqs_flat = q_f.reshape(-1, order="F")  # THz

        dft_dos = gaussian_broadened_dos(freqs_flat, weights_tile, fgrid, sigma=0.05)
        area_dft = float(np.trapz(dft_dos, fgrid))
        dft_dos_n = dft_dos / area_dft if area_dft > 0 else dft_dos

        order = np.argsort(fgrid)
        fgrid = fgrid[order]
        ml_dos_n = ml_dos_n[order]
        dft_dos_n = dft_dos_n[order]

        ml_dos_n = gaussian_smooth_on_grid(fgrid, ml_dos_n, sigma=0.05)
        dft_dos_n = gaussian_smooth_on_grid(fgrid, dft_dos_n, sigma=0.05)

        df = float(np.median(np.diff(fgrid))) if fgrid.size > 2 else 0.01
        dos_grid = np.arange(y_lo, y_hi + df, df)

        ml_dos_plot = np.interp(dos_grid, fgrid, ml_dos_n, left=0.0, right=0.0)

        dft_dos_plot = gaussian_broadened_dos(
            freqs_flat,
            weights_tile,
            dos_grid,
            sigma=0.05,
        )
        area_dft_plot = float(np.trapz(dft_dos_plot, dos_grid))
        dft_dos_plot = (
            dft_dos_plot / area_dft_plot if area_dft_plot > 0 else dft_dos_plot
        )

        ml_dos_plot = gaussian_smooth_on_grid(dos_grid, ml_dos_plot, sigma=0.05)
        dft_dos_plot = gaussian_smooth_on_grid(dos_grid, dft_dos_plot, sigma=0.05)

    fig, (a0, a1) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [4, 1], "wspace": 0.05},
        figsize=(10, 8),
    )

    n_br = ml_freq.shape[1] if ml_freq.ndim == 2 else 1
    x = np.arange(ml_freq.shape[0], dtype=float)

    ml_color = "C0"
    dft_color = "k"

    if n_br == 1 and ml_freq.ndim == 1:
        a0.plot(x, ml_freq, lw=2.0, color=ml_color, label="ML")
        a0.plot(x, dft_on_ml, lw=2.0, ls="--", color=dft_color, label="DFT")
    else:
        for j in range(n_br):
            a0.plot(
                x,
                ml_freq[:, j],
                lw=2.0,
                color=ml_color,
                label="ML" if j == 0 else None,
            )
            a0.plot(
                x,
                dft_on_ml[:, j],
                lw=2.0,
                ls="--",
                color=dft_color,
                label="DFT" if j == 0 else None,
            )

    if "ml_normal_ticks" in data.files and labels is not None:
        ticks = np.asarray(data["ml_normal_ticks"], dtype=float)
        if np.max(ticks) <= ml_freq.shape[0] - 1 + 1e-6:
            xt = ticks
        else:
            xt = (
                (ticks - ticks.min())
                / (ticks.max() - ticks.min() + 1e-12)
                * (ml_freq.shape[0] - 1)
            )
        a0.set_xticks(xt)
        a0.set_xticklabels(labels)
        a0.set_xlim(float(xt[0]), float(xt[-1]))

    a0.set_ylabel("Frequency (THz)", fontsize=20)
    a0.axhline(0.0, color="k", lw=1.5)
    a0.grid(axis="x")
    a0.set_ylim(y_lo, y_hi)

    handles, lbls = a0.get_legend_handles_labels()
    by_label = dict(zip(lbls, handles, strict=False))
    if by_label:
        a0.legend(by_label.values(), by_label.keys(), loc=1, fontsize=18)

    if (
        has_dos
        and dos_grid is not None
        and dft_dos_plot is not None
        and ml_dos_plot is not None
    ):
        a1.fill_betweenx(dos_grid, 0.0, dft_dos_plot, color="k", alpha=0.25)
        a1.plot(dft_dos_plot, dos_grid, color="k", lw=2.0)
        a1.plot(ml_dos_plot, dos_grid, lw=2.0)
        a1.set_xlabel("DOS", fontsize=20)
        a1.grid(True, linestyle=":", linewidth=0.6)
        plt.setp(a1.get_yticklabels(), visible=False)
        a1.set_ylim(y_lo, y_hi)
    else:
        a1.axis("off")

    bio = BytesIO()
    fig.savefig(bio, format="png", dpi=200)
    plt.close(fig)

    data_uri = "data:image/png;base64," + base64.b64encode(bio.getvalue()).decode(
        "ascii"
    )
    return (
        data_uri,
        {
            "omega_avg_ref_thz": omega_avg_ref,
            "omega_avg_pred_thz": omega_avg_pred,
        },
    )


def render_dispersion_component(
    selection_context: dict[str, Any],
    *,
    calc_root: Path,
    **_: Any,
):
    """
    Render the dispersion + DOS panel for a selected scatter point.

    Parameters
    ----------
    selection_context
        Selection payload produced by the scatter/table callbacks.
    calc_root
        Root directory containing calculation artifacts referenced by ``data_paths``.
    **_
        Additional keyword arguments accepted for callback API compatibility.

    Returns
    -------
    Any or None
        A Dash component containing the rendered image and caption, or ``None`` if the
        required artifacts are unavailable.
    """
    selected = (
        selection_context.get("selection")
        if isinstance(selection_context, dict)
        else None
    )
    if not isinstance(selected, dict):
        selected = selection_context if isinstance(selection_context, dict) else {}

    data_paths = selected.get("data_paths")
    if not isinstance(data_paths, dict):
        return None

    npz_rel = data_paths.get("npz")
    if not isinstance(npz_rel, str) or not npz_rel:
        return None

    meta_rel = data_paths.get("meta")
    meta_rel_str = meta_rel if isinstance(meta_rel, str) and meta_rel else None

    npz_path = (calc_root / npz_rel).resolve()
    meta_path = (calc_root / meta_rel_str).resolve() if meta_rel_str else None

    src, extras = _render_npz_to_data_uri(
        str(npz_path),
        str(meta_path) if meta_path else None,
    )
    if not src:
        return None

    caption_bits = []
    oref = extras.get("omega_avg_ref_thz")
    opred = extras.get("omega_avg_pred_thz")
    if oref is not None and opred is not None:
        caption_bits.append(f"ω_avg: DFT {oref:.3f} | ML {opred:.3f} THz")

    caption = " | ".join(caption_bits)
    title = selected.get("system") or selected.get("label") or selected.get("id") or ""

    return html.Div(
        [
            html.H4(title) if title else None,
            html.Img(
                src=src,
                style={
                    "width": "100%",
                    "maxWidth": "820px",
                    "height": "auto",
                    "display": "block",
                    "borderRadius": "8px",
                    "border": "1px solid #ddd",
                },
            ),
            html.Div(
                caption,
                style={"marginTop": "8px", "fontSize": "0.95rem", "opacity": 0.85},
            )
            if caption
            else None,
        ]
    )
