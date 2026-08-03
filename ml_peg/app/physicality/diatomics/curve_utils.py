"""Load, render, and display diatomic curves."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path

from dash import Input, Output, callback
from dash.exceptions import PreventUpdate
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from ml_peg.analysis.utils.periodic_table import (
    PERIODIC_TABLE_COLS,
    PERIODIC_TABLE_POSITIONS,
    PERIODIC_TABLE_ROWS,
)
from ml_peg.app.utils.plot_export import bytes_to_data_uri, figure_to_bytes


def load_model_curves(
    curve_dir: str | Path,
    model_name: str,
    element_value: str | None,
    overview_label: str,
) -> tuple[str | None, dict[str, dict]]:
    """
    Load a model's diatomic energy curves for the currently selected view.

    Parameters
    ----------
    curve_dir
        Directory holding one subfolder of curve files per model.
    model_name
        Name of the model whose curves to load.
    element_value
        Current dropdown choice: the overview label, or an element symbol.
    overview_label
        The dropdown value that means "show the homonuclear overview".

    Returns
    -------
    tuple[str | None, dict[str, dict]]
        The selected element (``None`` for the homonuclear overview) and a
        mapping from element-pair name (e.g. ``"H-O"``) to that curve's data.
        The mapping is empty when the model has no curves for this view.
    """
    selected_element = None if element_value == overview_label else element_value
    model_curve_dir = Path(curve_dir) / model_name
    if not model_curve_dir.exists():
        return selected_element, {}

    filtered: dict[str, dict] = {}
    for curve_file in model_curve_dir.glob("*.json"):
        try:
            payload = json.loads(curve_file.read_text())
        except Exception:
            continue
        pair = payload.get("pair") or curve_file.stem
        try:
            first, second = pair.split("-")
        except ValueError:
            first = second = pair
        if selected_element is None:
            if first == second:
                filtered[pair] = payload
        elif selected_element in (first, second):
            filtered[pair] = payload

    return selected_element, filtered


def render_periodic_curve_gallery_png(
    *,
    curve_dir: str | Path,
    model_name: str,
    element_value: str | None,
    overview_label: str,
    dpi: int = 600,
) -> tuple[bytes, float, float]:
    """
    Draw the periodic-table grid of diatomic curves for one model and view.

    Parameters
    ----------
    curve_dir
        Directory holding one subfolder of curve files per model.
    model_name
        Name of the model to plot.
    element_value
        Current dropdown choice: the overview label, or an element symbol.
    overview_label
        The dropdown value that means "show the homonuclear overview".
    dpi
        Resolution of the rendered image, in dots per inch.

    Returns
    -------
    tuple[bytes, float, float]
        The PNG image bytes, and the image's width and height in pixels.
    """
    selected_element, filtered = load_model_curves(
        curve_dir, model_name, element_value, overview_label
    )
    if not filtered:
        raise PreventUpdate

    import matplotlib as mpl

    try:
        mpl.use("Agg")
    except Exception:
        pass

    fig, axes = plt.subplots(
        PERIODIC_TABLE_ROWS,
        PERIODIC_TABLE_COLS,
        figsize=(30, 15),
        constrained_layout=True,
    )
    axes = axes.reshape(PERIODIC_TABLE_ROWS, PERIODIC_TABLE_COLS)
    for ax in axes.ravel():
        ax.axis("off")

    has_data = False
    for pair, payload in filtered.items():
        first, second = pair.split("-") if "-" in pair else (pair, pair)
        other = second if selected_element == first else first
        pos = PERIODIC_TABLE_POSITIONS.get(other)
        if pos is None:
            continue
        x_vals = payload.get("distance") or []
        y_vals = payload.get("energy") or []
        if not x_vals or not y_vals:
            continue
        try:
            x = [float(v) for v in x_vals]
            y = [float(v) for v in y_vals]
        except Exception:
            continue

        shift = y[-1]
        y_shifted = [yy - shift for yy in y]
        row, col = pos
        ax = axes[row, col]
        ax.axis("on")
        ax.plot(x, y_shifted, linewidth=1, zorder=1)
        ax.axhline(0, color="grey", linewidth=0.5, zorder=0)
        ax.set_title(f"{first}-{second}, shift: {shift:.4f}", fontsize=8)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_yticks([-20, -10, 0, 10, 20])
        ax.set_xlim(0, 6)
        ax.set_ylim(-20, 20)
        if selected_element and (first == second == selected_element):
            for spine in ax.spines.values():
                spine.set_edgecolor("crimson")
                spine.set_linewidth(2)
        has_data = True

    if not has_data:
        plt.close(fig)
        raise PreventUpdate

    title = (
        f"Heteronuclear diatomics for {selected_element}: {model_name}"
        if selected_element
        else f"Homonuclear diatomics: {model_name}"
    )
    fig.suptitle(title, fontsize=32, fontweight="bold")
    png_bytes = figure_to_bytes(fig, "png", dpi=dpi)
    width, height = fig.get_size_inches() * dpi
    plt.close(fig)
    return png_bytes, float(width), float(height)


def register_image_gallery_callbacks(
    model_dropdown_id: str,
    element_dropdown_id: str,
    figure_id: str,
    manifest_dir: str | Path,
    curve_dir: str | Path,
    overview_label: str = "All",
) -> None:
    """
    Register callbacks to display pre-rendered images stored per model.

    Parameters
    ----------
    model_dropdown_id
        Dash component ID for the model selector.
    element_dropdown_id
        Dash component ID for the element selector.
    figure_id
        Dash component ID for the output ``dcc.Graph``.
    manifest_dir
        Directory containing per-model ``manifest.json`` files.
    curve_dir
        Directory holding one subfolder of curve files per model. Element
        selections are drawn on the fly from these curves instead of relying on
        pre-generated element images.
    overview_label
        Dropdown label representing the overview image. Default is ``"All"``.
    """
    curve_base = Path(curve_dir) if curve_dir else None

    def _data_url(path: Path) -> tuple[str, float, float]:
        """
        Convert an image file into a base64 data URL.

        Parameters
        ----------
        path
            Path to the image file.

        Returns
        -------
        tuple[str, float, float]
            Data URL string suitable for Plotly layout images, plus image width/height
            (falls back to ``1.0`` when unavailable).
        """
        width, height = 1.0, 1.0
        suffix = path.suffix.lower()
        try:
            from PIL import Image

            with Image.open(path) as im:
                width, height = float(im.width), float(im.height)
        except Exception:
            if suffix == ".svg":
                try:
                    import xml.etree.ElementTree as ET

                    root = ET.fromstring(path.read_text())
                    viewbox = root.attrib.get("viewBox")
                    if viewbox:
                        parts = [float(v) for v in viewbox.strip().split()[-2:]]
                        if len(parts) == 2:
                            width, height = parts
                    else:
                        w_attr = root.attrib.get("width")
                        h_attr = root.attrib.get("height")
                        if w_attr and h_attr:
                            width = float(str(w_attr).replace("px", ""))
                            height = float(str(h_attr).replace("px", ""))
                except Exception:
                    width, height = 1.0, 1.0
            else:
                width, height = 1.0, 1.0
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }.get(suffix, "application/octet-stream")
        encoded = base64.b64encode(path.read_bytes()).decode()
        return f"data:{mime};base64,{encoded}", width, height

    def _data_url_from_bytes(
        data: bytes,
        mime: str = "image/png",
        width: float | None = None,
        height: float | None = None,
    ) -> tuple[str, float, float]:
        """
        Build a data URL from raw bytes and infer image dimensions.

        Parameters
        ----------
        data
            Raw image bytes.
        mime
            MIME type string for the encoded image.
        width, height
            Optional known image dimensions. When supplied, the bytes are not reopened
            with PIL to infer dimensions.

        Returns
        -------
        tuple[str, float, float]
            Data URL string and inferred image width/height (falls back to 1.0).
        """
        if width is None or height is None:
            width, height = 1.0, 1.0
            try:
                from PIL import Image

                with Image.open(io.BytesIO(data)) as im:
                    width, height = float(im.width), float(im.height)
            except Exception:
                pass
        return bytes_to_data_uri(data, mime), width, height

    def _image_figure(src: str, width: float, height: float) -> go.Figure:
        """
        Build a Plotly figure that displays the supplied image.

        Parameters
        ----------
        src
            Base64-encoded data URL for the image.
        width
            Image width in pixels (used for aspect ratio).
        height
            Image height in pixels (used for aspect ratio).

        Returns
        -------
        go.Figure
            Figure containing the image without axes, matching legacy behaviour.
        """
        aspect = width / height if height else 1.0
        aspect = aspect if math.isfinite(aspect) and aspect > 0 else 1.0
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, aspect],
                y=[0, 1],
                mode="markers",
                marker={"opacity": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_layout_image(
            {
                "source": src,
                "xref": "x",
                "yref": "y",
                "x": 0,
                "y": 0,
                "sizex": aspect,
                "sizey": 1,
                "xanchor": "left",
                "yanchor": "bottom",
                "layer": "below",
            }
        )
        fig.update_layout(
            xaxis={
                "visible": False,
                "range": [0, aspect],
                "constrain": "domain",
            },
            yaxis={
                "visible": False,
                "range": [0, 1],
                "scaleanchor": "x",
                "scaleratio": 1,
            },
            margin={"l": 0, "r": 0, "t": 0, "b": 0},
            autosize=True,
            dragmode="pan",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    @callback(
        Output(element_dropdown_id, "options"),
        Output(element_dropdown_id, "value"),
        Input(model_dropdown_id, "value"),
    )
    def _update_options(model_name: str):
        """
        Populate element dropdown options for the selected model.

        Parameters
        ----------
        model_name
            Selected model value from the dropdown.

        Returns
        -------
        tuple[list[dict], str]
            Dropdown options and default selection.
        """
        if not model_name:
            raise PreventUpdate
        element_opts: list[str] = []
        model_curve_dir = curve_base / model_name
        if model_curve_dir.exists():
            for curve_file in model_curve_dir.glob("*.json"):
                try:
                    payload = json.loads(curve_file.read_text())
                except Exception:
                    continue
                pair = payload.get("pair") or curve_file.stem
                try:
                    first, second = pair.split("-")
                except ValueError:
                    first = second = pair
                element_opts.extend([first, second])

        options = [{"label": overview_label, "value": overview_label}] + [
            {"label": element, "value": element}
            for element in sorted({opt for opt in element_opts if opt})
        ]
        return options, overview_label

    @callback(
        Output(figure_id, "figure"),
        Input(model_dropdown_id, "value"),
        Input(element_dropdown_id, "value"),
    )
    def _update_figure(model_name: str, element_value: str | None):
        """
        Return figure for overview or selected-element view.

        Parameters
        ----------
        model_name
            Selected model identifier.
        element_value
            Selected element dropdown value (overview label or element symbol).

        Returns
        -------
        go.Figure
            Image figure ready for display.
        """
        if not model_name:
            raise PreventUpdate

        png_bytes, width, height = render_periodic_curve_gallery_png(
            curve_dir=curve_base,
            model_name=model_name,
            element_value=element_value,
            overview_label=overview_label,
            dpi=200,
        )
        src, width, height = _data_url_from_bytes(
            png_bytes,
            mime="image/png",
            width=width,
            height=height,
        )
        return _image_figure(src, width, height)
