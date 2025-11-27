"""Helpers to create callbacks for Dash app."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Literal

from dash import Input, Output, callback
from dash.dcc import Graph
from dash.exceptions import PreventUpdate
from dash.html import Div, Iframe
import plotly.graph_objects as go

from ml_peg.analysis.utils.decorators import (
    PERIODIC_TABLE_COLS,
    PERIODIC_TABLE_POSITIONS,
    PERIODIC_TABLE_ROWS,
)
from ml_peg.app.utils.weas import generate_weas_html


def plot_from_table_column(
    table_id: str, plot_id: str, column_to_plot: dict[str, Graph]
) -> None:
    """
    Attach callback to show plot when a table column is clicked.

    Parameters
    ----------
    table_id
        ID for Dash table being clicked.
    plot_id
        ID for Dash plot placeholder Div.
    column_to_plot
        Dictionary relating table headers (keys) and plot to show (values).
    """

    @callback(Output(plot_id, "children"), Input(table_id, "active_cell"))
    def show_plot(active_cell) -> Div:
        """
        Register callback to show plot when a table column is clicked.

        Parameters
        ----------
        active_cell
            Clicked cell in Dash table.

        Returns
        -------
        Div
            Message explaining interactivity, or plot on table click.
        """
        if not active_cell:
            return Div("Click on a metric to view plot.")
        column_id = active_cell.get("column_id", None)
        if column_id:
            if column_id in column_to_plot:
                return Div(column_to_plot[column_id])
            raise PreventUpdate
        raise ValueError("Invalid column_id")


def plot_from_table_cell(
    table_id: str,
    plot_id: str,
    cell_to_plot: dict[str, dict[Graph]],
) -> None:
    """
    Attach callback to show plot when a table cell is clicked.

    Parameters
    ----------
    table_id
        ID for Dash table being clicked.
    plot_id
        ID for Dash plot placeholder Div.
    cell_to_plot
        Nested dictionary of model names, column names, and plot to show.
    """

    @callback(Output(plot_id, "children"), Input(table_id, "active_cell"))
    def show_plot(active_cell) -> Div:
        """
        Register callback to show plot when a table cell is clicked.

        Parameters
        ----------
        active_cell
            Clicked cell in Dash table.

        Returns
        -------
        Div
            Message explaining interactivity, or plot on cell click.
        """
        if not active_cell:
            return Div("Click on a metric to view plot.")
        column_id = active_cell.get("column_id", None)
        row_id = active_cell.get("row_id", None)

        if row_id in cell_to_plot and column_id in cell_to_plot[row_id]:
            return Div(cell_to_plot[row_id][column_id])
        return Div("Click on a metric to view plot.")


def struct_from_scatter(
    scatter_id: str,
    struct_id: str,
    structs: str | list[str],
    mode: Literal["struct", "traj"] = "struct",
) -> None:
    """
    Attach callback to show a structure when a scatter point is clicked.

    Parameters
    ----------
    scatter_id
        ID for Dash scatter being clicked.
    struct_id
        ID for Dash plot placeholder Div where structures will be visualised.
    structs
        List of structure filenames in same order as scatter data to be visualised.
    mode
        Whether to display a single structure ("struct"), or trajectory from an initial
        image ("traj"). Default is "struct".
    """

    @callback(
        Output(struct_id, "children", allow_duplicate=True),
        Input(scatter_id, "clickData"),
        prevent_initial_call="initial_duplicate",
    )
    def show_struct(click_data):
        """
        Register callback to show structure when a scatter point is clicked.

        Parameters
        ----------
        click_data
            Clicked data point in scatter plot.

        Returns
        -------
        Div
            Visualised structure on plot click.
        """
        if not click_data:
            return None
        idx = click_data["points"][0]["pointNumber"]

        if isinstance(structs, str):
            struct = structs
            index = idx
        else:
            struct = structs[idx]
            index = 0

        return Div(
            Iframe(
                srcDoc=generate_weas_html(struct, mode, index),
                style={
                    "height": "550px",
                    "width": "100%",
                    "border": "1px solid #ddd",
                    "borderRadius": "5px",
                },
            )
        )


def struct_from_table(
    table_id: str,
    struct_id: str,
    column_to_struct: dict[str, str],
    mode: Literal["struct", "traj"] = "struct",
) -> None:
    """
    Attach callback to show a structure when a table is clicked.

    Parameters
    ----------
    table_id
        ID for Dash table being clicked.
    struct_id
        ID for Dash plot placeholder Div where structures will be visualised.
    column_to_struct
        Dictionary of structure filenames indexed by table column.
    mode
        Whether to display a single structure ("struct"), or trajectory from an initial
        image ("traj"). Default is "struct".
    """

    @callback(
        Output(struct_id, "children", allow_duplicate=True),
        Input(table_id, "active_cell"),
        prevent_initial_call="initial_duplicate",
    )
    def show_struct(active_cell):
        """
        Register callback to show structure when a table is clicked.

        Parameters
        ----------
        active_cell
            Clicked cell in Dash table.

        Returns
        -------
        Div
            Visualised structure on plot click.
        """
        if not active_cell:
            return Div("Click on a metric to view the structure.")

        column_id = active_cell.get("column_id", None)
        if column_id:
            if column_id in column_to_struct:
                struct = column_to_struct[column_id]

                return Div(
                    Iframe(
                        srcDoc=generate_weas_html(struct, mode),
                        style={
                            "height": "550px",
                            "width": "100%",
                            "border": "1px solid #ddd",
                            "borderRadius": "5px",
                        },
                    )
                )

            raise PreventUpdate
        raise ValueError("Invalid column_id")


def register_image_gallery_callbacks(
    model_dropdown_id: str,
    element_dropdown_id: str,
    figure_id: str,
    manifest_dir: str | Path,
    overview_label: str = "All",
    curve_dir: str | Path | None = None,
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
    overview_label
        Dropdown label representing the overview image. Default is ``"All"``.
    curve_dir
        Directory of per-model curve JSON payloads. Element selections are rendered on
        the fly from these payloads instead of relying on pre-generated element images.
    """
    curve_base = Path(curve_dir) if curve_dir else None
    if curve_base is None:
        raise ValueError(
            "curve_dir must be provided to render diatomic plots dynamically."
        )

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
        data: bytes, mime: str = "image/png"
    ) -> tuple[str, float, float]:
        """
        Build a data URL from raw bytes and infer image dimensions.

        Parameters
        ----------
        data
            Raw image bytes.
        mime
            MIME type string for the encoded image.

        Returns
        -------
        tuple[str, float, float]
            Data URL string and inferred image width/height (falls back to 1.0).
        """
        width, height = 1.0, 1.0
        try:
            from PIL import Image

            with Image.open(io.BytesIO(data)) as im:
                width, height = float(im.width), float(im.height)
        except Exception:
            pass
        encoded = base64.b64encode(data).decode()
        return f"data:{mime};base64,{encoded}", width, height

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

        model_curve_dir = curve_base / model_name
        if not model_curve_dir.exists():
            raise PreventUpdate

        curves: dict[str, dict] = {}
        for curve_file in model_curve_dir.glob("*.json"):
            try:
                payload = json.loads(curve_file.read_text())
            except Exception:
                continue
            pair = payload.get("pair") or curve_file.stem
            curves[pair] = payload

        if not curves:
            raise PreventUpdate

        # Decide which pairs to render: overview -> homonuclear only, otherwise
        # all pairs involving the selected element.
        selected_element = None if element_value == overview_label else element_value
        filtered: dict[str, dict] = {}
        for pair, payload in curves.items():
            try:
                first, second = pair.split("-")
            except ValueError:
                first = second = pair
            if selected_element is None:
                if first == second:
                    filtered[pair] = payload
            else:
                if selected_element in (first, second):
                    filtered[pair] = payload

        if not filtered:
            raise PreventUpdate

        import matplotlib as mpl

        # Ensure a non-interactive backend for server-side rendering
        try:
            mpl.use("Agg")
        except Exception:
            pass
        import matplotlib.pyplot as plt

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
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=200)
        plt.close(fig)
        src, width, height = _data_url_from_bytes(buf.getvalue(), mime="image/png")
        return _image_figure(src, width, height)
