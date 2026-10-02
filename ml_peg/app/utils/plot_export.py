"""Shared helpers for exporting Matplotlib figures to downloadable formats."""

from __future__ import annotations

import base64
import io
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def figure_to_bytes(
    fig: Figure,
    fmt: str,
    *,
    dpi: int | None = None,
    bbox_inches: str | None = None,
) -> bytes:
    """
    Render a Matplotlib figure to raw bytes in the requested format.

    The figure is not closed; callers remain responsible for ``plt.close``.

    Parameters
    ----------
    fig
        Matplotlib figure to render.
    fmt
        Output format passed to ``savefig`` (e.g. ``"png"`` or ``"svg"``).
    dpi
        Output resolution in dots per inch. Omitted from ``savefig`` when None.
    bbox_inches
        Bounding-box mode (e.g. ``"tight"``). Omitted from ``savefig`` when None.

    Returns
    -------
    bytes
        Encoded figure contents.
    """
    buf = io.BytesIO()
    save_kwargs: dict = {"format": fmt}
    if dpi is not None:
        save_kwargs["dpi"] = dpi
    if bbox_inches is not None:
        save_kwargs["bbox_inches"] = bbox_inches
    fig.savefig(buf, **save_kwargs)
    return buf.getvalue()


def bytes_to_data_uri(data: bytes, mime: str) -> str:
    """
    Base64-encode raw bytes into a ``data:`` URI.

    Parameters
    ----------
    data
        Raw bytes to encode.
    mime
        MIME type for the URI (e.g. ``"image/png"``).

    Returns
    -------
    str
        ``data:<mime>;base64,<encoded>`` URI string.
    """
    encoded = base64.b64encode(data).decode()
    return f"data:{mime};base64,{encoded}"
