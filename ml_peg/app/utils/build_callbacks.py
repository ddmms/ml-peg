"""Helpers to create callbacks for Dash app."""

from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
import pickle
from typing import Literal

import matplotlib

matplotlib.use("Agg")
from dash import Input, Output, State, callback, callback_context, dcc, html
from dash.dcc import Graph
from dash.exceptions import PreventUpdate
from dash.html import Div, Iframe
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
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
            return Div("Click on a metric to view the structure.")
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
        Directory of per-model curve JSON payloads. Element selections are rendered on
        the fly from these payloads instead of relying on pre-generated element images.
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


def register_phonon_callbacks(
    *,
    table_id: str,
    table_data: list[dict],
    plot_container_id: str,
    dispersion_container_id: str,
    scatter_meta_store_id: str,
    last_cell_store_id: str,
    scatter_graph_id: str,
    interactive_data: dict,
    calc_root: Path,
) -> None:
    """
    Register callbacks for phonon scatter ↔ dispersion interactivity.

    Parameters
    ----------
    table_id
        Dash table identifier emitting `active_cell` events.
    table_data
        Pre-rendered table rows used to look up model names.
    plot_container_id
        Div ID hosting the scatter/violin/stability plots.
    dispersion_container_id
        Div ID that displays the phonon dispersion PNG.
    scatter_meta_store_id
        Store component ID that tracks the latest scatter metadata.
    last_cell_store_id
        Store component ID used to toggle table-cell interactions.
    scatter_graph_id
        Graph ID used for scatter callbacks.
    interactive_data
        Serialized dataset produced by ``interactive_payload`` analysis fixture.
    calc_root
        Base path where band/DOS assets live.
    """
    calc_root = Path(calc_root)
    metric_labels = interactive_data.get("metrics", {})
    label_to_key = {label: key for key, label in metric_labels.items()}
    bz_column = interactive_data.get("bz_column", "Avg BZ MAE")
    stability_column = interactive_data.get(
        "stability_column", "Stability Classification (F1)"
    )

    def _metric_points(model_name: str, metric_key: str) -> list[dict]:
        """
        Return scatter entries for a given model/metric pair.

        Parameters
        ----------
        model_name
            Display name of the model.
        metric_key
            Internal metric key (e.g. ``"max_freq"``).

        Returns
        -------
        list[dict]
            Scatter entries with ``ref``, ``pred``, and metadata.
        """
        return (
            interactive_data["models"]
            .get(model_name, {})
            .get("metrics", {})
            .get(metric_key, {})
            .get("points", [])
        )

    def _band_errors(model_name: str) -> dict[str, list[float]]:
        """
        Return per-system band errors for a model.

        Parameters
        ----------
        model_name
            Display name of the model.

        Returns
        -------
        dict[str, list[float]]
            Mapping of mp-id to BZ MAE samples.
        """
        return interactive_data["models"].get(model_name, {}).get("band_errors", {})

    def _stability_points(model_name: str) -> list[dict]:
        """
        Return stability scatter entries for a model.

        Parameters
        ----------
        model_name
            Display name of the model.

        Returns
        -------
        list[dict]
            Scatter entries storing ``ref``/``pred`` ω_min values.
        """
        return (
            interactive_data["models"]
            .get(model_name, {})
            .get("stability", {})
            .get("points", [])
        )

    def _system_payload(model_name: str, point_id: str | int) -> dict | None:
        """
        Fetch metadata for a specific system ID.

        Parameters
        ----------
        model_name
            Display name of the model.
        point_id
            Identifier (typically mp-id) of the system.

        Returns
        -------
        dict | None
            First matching metadata entry with ``data_paths``.
        """
        if point_id is None:
            return None
        model_entry = interactive_data["models"].get(model_name, {})
        target = str(point_id)
        for metric_data in model_entry.get("metrics", {}).values():
            for point in metric_data.get("points", []):
                if str(point.get("id")) == target:
                    return point
        for point in model_entry.get("stability", {}).get("points", []):
            if str(point.get("id")) == target:
                return point
        return None

    def _build_metric_scatter(model_name: str, metric_key: str):
        """
        Build Plotly scatter figure for the requested metric.

        Parameters
        ----------
        model_name
            Display name of the model.
        metric_key
            Internal metric identifier.

        Returns
        -------
        go.Figure
            Scatter figure showing ref vs pred values.
        """
        points = _metric_points(model_name, metric_key)
        fig = go.Figure()
        if not points:
            fig.update_layout(title=f"No data for {model_name}")
            return fig
        refs = [point["ref"] for point in points]
        preds = [point["pred"] for point in points]
        hover = [point.get("label") or point.get("id") for point in points]
        custom = [[point.get("id") or idx] for idx, point in enumerate(points)]
        fig.add_trace(
            go.Scatter(
                x=refs,
                y=preds,
                mode="markers",
                text=hover,
                customdata=custom,
                hovertemplate=(
                    "System: %{text}<br>Reference: %{x:.3f}<br>"
                    "Prediction: %{y:.3f}<extra></extra>"
                ),
                name=model_name,
            )
        )
        lower = min(min(refs), min(preds))
        upper = max(max(refs), max(preds))
        fig.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                showlegend=False,
                line={"color": "#8c8c8c", "dash": "dash"},
            )
        )
        mae_value = interactive_data["models"][model_name]["metrics"][metric_key].get(
            "mae"
        )
        if mae_value is not None:
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                text=f"MAE: {mae_value:.3f}",
                showarrow=False,
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor="#444",
            )
        fig.update_layout(
            title=f"{model_name} – {metric_labels.get(metric_key, metric_key)}",
            xaxis_title="Reference",
            yaxis_title="Prediction",
            xaxis={"scaleanchor": "y", "scaleratio": 1, "showgrid": True},
            yaxis={"showgrid": True},
            plot_bgcolor="#ffffff",
        )
        return fig

    def _build_bz_violin(model_name: str):
        """
        Build BZ MAE violin plot for a model.

        Parameters
        ----------
        model_name
            Display name of the model.

        Returns
        -------
        go.Figure
            Violin plot of per-structure BZ MAE values.
        """
        band_errors = _band_errors(model_name)
        values: list[float] = []
        labels: list[str] = []
        for mp_id, series in band_errors.items():
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
        fig = go.Figure()
        if not values:
            fig.update_layout(title="No dispersion errors available.")
            return fig
        customdata = [[label] for label in labels]
        fig.add_trace(
            go.Violin(
                y=values,
                text=labels,
                customdata=customdata,
                points="all",
                jitter=0.05,
                box_visible=True,
                meanline_visible=True,
                fillcolor="#636EFA",
                line_color="#636EFA",
                opacity=0.6,
                hovertemplate="System: %{text}<br>|Error|: %{y:.3f} K<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"{model_name} – Brillouin zone error distribution",
            yaxis_title="|Error| / K",
            plot_bgcolor="#ffffff",
        )
        return fig

    def _build_stability_scatter(model_name: str):
        """
        Build stability scatter plot for a model.

        Parameters
        ----------
        model_name
            Display name of the model.

        Returns
        -------
        go.Figure
            Scatter figure comparing predicted vs reference ω_min.
        """
        points = _stability_points(model_name)
        fig = go.Figure()
        if not points:
            fig.update_layout(title="No stability data available.")
            return fig
        colours = {
            "TP": "#2ca02c",
            "TN": "#1f77b4",
            "FP": "#ff7f0e",
            "FN": "#d62728",
        }
        for label, colour in colours.items():
            subset = [point for point in points if point.get("class") == label]
            if not subset:
                continue
            refs = [point["ref"] for point in subset]
            preds = [point["pred"] for point in subset]
            hover = [point.get("label") or point.get("id") for point in subset]
            custom = [[point.get("id") or idx] for idx, point in enumerate(subset)]
            fig.add_trace(
                go.Scatter(
                    x=refs,
                    y=preds,
                    mode="markers",
                    name=label,
                    marker={"color": colour},
                    text=hover,
                    customdata=custom,
                    hovertemplate=(
                        "System: %{text}<br>Reference: %{x:.3f} K<br>"
                        "Prediction: %{y:.3f} K<extra></extra>"
                    ),
                )
            )
        combined_refs = [point["ref"] for point in points]
        combined_preds = [point["pred"] for point in points]
        lower = min(min(combined_refs), min(combined_preds))
        upper = max(max(combined_refs), max(combined_preds))
        fig.add_trace(
            go.Scatter(
                x=[lower, upper],
                y=[lower, upper],
                mode="lines",
                showlegend=False,
                line={"color": "#8c8c8c", "dash": "dash"},
            )
        )
        fig.update_layout(
            title=f"{model_name} – Stability classification",
            xaxis_title="Reference ω_min / K",
            yaxis_title="Predicted ω_min / K",
            plot_bgcolor="#ffffff",
        )
        return fig

    def _load_band(rel_path: str):
        """
        Load a band-structure dictionary from ``rel_path``.

        Parameters
        ----------
        rel_path
            Path relative to ``calc_root`` pointing to npz payload.

        Returns
        -------
        dict | None
            Parsed band-structure dictionary or ``None`` if missing.
        """
        if not rel_path:
            return None
        try:
            full_path = calc_root / rel_path
            with full_path.open("rb") as handle:
                band_data = pickle.load(handle)

            # Load corresponding labels/connections
            base_stem = full_path.stem.replace("_band_structure", "")
            labels_path = full_path.parent / f"{base_stem}_labels.json"
            connections_path = full_path.parent / f"{base_stem}_connections.json"

            try:
                import json

                if labels_path.exists():
                    with labels_path.open() as f:
                        band_data["labels"] = json.load(f)
                if connections_path.exists():
                    with connections_path.open() as f:
                        band_data["path_connections"] = json.load(f)
            except Exception:
                pass  # If labels/connections don't exist, continue without them

            return band_data
        except OSError:
            return None

    def _load_dos(rel_path: str):
        """
        Load DOS (frequency, total_dos) tuple from ``rel_path``.

        Parameters
        ----------
        rel_path
            Relative path to the DOS npz payload.

        Returns
        -------
        tuple | None
            ``(frequency_points, total_dos)`` arrays or ``None``.
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
            Distance arrays for each path segment.
        labels
            High-symmetry labels along the path.
        connections
            Boolean list denoting whether segments connect.

        Returns
        -------
        tuple[list, list]
            Tick positions and labels.
        """
        xticks, xticklabels = [], []
        cumulative_dist, i = 0.0, 0
        connections = [True] + connections
        for seg_dist, connected in zip(distances, connections, strict=False):
            start, end = labels[i], labels[i + 1]
            pos_start = cumulative_dist
            pos_end = cumulative_dist + (seg_dist[-1] - seg_dist[0])
            xticks.append(pos_start)
            xticklabels.append(f"{start}|{end}" if not connected else start)
            i += 2 if not connected else 1
            cumulative_dist = pos_end
        xticks.append(cumulative_dist)
        xticklabels.append(labels[-1])
        return xticks, xticklabels

    def _render_dispersion_image(model_name: str, system_label: str, paths: dict):
        """
        Render dispersion comparison figure to a base64 data URI.

        Parameters
        ----------
        model_name
            Display name of the model.
        system_label
            Identifier shown above the plot.
        paths
            Mapping containing ``ref_band``, ``pred_band``, ``ref_dos``, ``pred_dos``.

        Returns
        -------
        str | None
            Base64 encoded PNG or ``None`` if assets are missing.
        """
        ref_band = _load_band(paths.get("ref_band"))
        pred_band = _load_band(paths.get("pred_band"))
        ref_dos = _load_dos(paths.get("ref_dos"))
        pred_dos = _load_dos(paths.get("pred_dos"))
        if not all([ref_band, pred_band, ref_dos, pred_dos]):
            return None

        fig = plt.figure(figsize=(9, 5))
        gridspec.GridSpec(1, 2, width_ratios=[4, 1], wspace=0.05)
        ax1 = fig.add_axes([0.12, 0.07, 0.67, 0.85])
        ax2 = fig.add_axes([0.82, 0.07, 0.17, 0.85])

        distances_ref = ref_band["distances"]
        frequencies_ref = ref_band["frequencies"]
        distances_pred = pred_band["distances"]
        frequencies_pred = pred_band["frequencies"]
        dos_freqs_ref, dos_values_ref = ref_dos
        dos_freqs_pred, dos_values_pred = pred_dos

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
                    label=model_name if not pred_label_added else None,
                )
                pred_label_added = True

        ax2.plot(dos_values_pred, dos_freqs_pred, lw=1.2, color="red", linestyle="--")

        ref_label_added = False
        for dist_segment, freq_segment in zip(
            distances_ref, frequencies_ref, strict=False
        ):
            for band in freq_segment.T:
                ax1.plot(
                    dist_segment,
                    band,
                    lw=1,
                    linestyle="-",
                    color="blue",
                    label="PBE" if not ref_label_added else None,
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
        ax1.set_ylabel("Frequency (THz)", fontsize=16)
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

        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    @callback(
        Output(plot_container_id, "children"),
        Output(scatter_meta_store_id, "data"),
        Output(last_cell_store_id, "data"),
        Input(table_id, "active_cell"),
        State(last_cell_store_id, "data"),
        prevent_initial_call=True,
    )
    def update_plot(active_cell, last_cell):
        """
        Update plot container when a table cell is clicked.

        Parameters
        ----------
        active_cell
            Dash ``active_cell`` payload from the metrics table.
        last_cell
            Previously clicked cell stored in ``last_cell_store_id``.

        Returns
        -------
        tuple
            Plot container children, scatter meta, and new ``last_cell`` value.
        """
        if not active_cell:
            raise PreventUpdate
        if last_cell and last_cell == active_cell:
            return (
                html.Div("Click on a metric to view scatter plots."),
                None,
                None,
            )
        row = active_cell.get("row")
        column = active_cell.get("column_id")
        if row is None or column is None or row < 0 or row >= len(table_data):
            raise PreventUpdate
        model_display = table_data[row]["MLIP"]
        model_data = interactive_data["models"].get(model_display)
        if model_data is None:
            return (
                html.Div(f"No data for {model_display}"),
                None,
                active_cell,
            )
        if column in label_to_key:
            metric_key = label_to_key[column]
            figure = _build_metric_scatter(model_display, metric_key)
            graph = dcc.Graph(id=scatter_graph_id, figure=figure)
            meta = {
                "model": model_display,
                "type": "metric",
                "metric": metric_key,
            }
            content = html.Div(
                [html.P("Click a data point to preview its dispersion plot."), graph]
            )
            return content, meta, active_cell
        if column == bz_column:
            figure = _build_bz_violin(model_display)
            graph = dcc.Graph(id=scatter_graph_id, figure=figure)
            meta = {"model": model_display, "type": "bz"}
            content = html.Div(
                [
                    html.P(
                        "Click a violin sample to preview the phonon dispersion plot."
                    ),
                    graph,
                ]
            )
            return content, meta, active_cell
        if column == stability_column:
            scatter = _build_stability_scatter(model_display)
            confusion = go.Figure()
            confusion_data = model_data.get("stability", {}).get("confusion") or []
            if confusion_data:
                confusion_array = np.array(confusion_data, dtype=float)
                max_val = confusion_array.max(initial=0.0)
                total = confusion_array.sum()
                confusion.add_trace(
                    go.Heatmap(
                        z=confusion_array,
                        y=["DFT stable", "DFT unstable"],
                        x=["MLIP stable", "MLIP unstable"],
                        colorscale="Blues",
                        showscale=True,
                        hovertemplate=(
                            "DFT %{y}<br>MLIP %{x}<br>Count: %{z:.0f}<extra></extra>"
                        ),
                    )
                )
                for y_idx, y_label in enumerate(["DFT stable", "DFT unstable"]):
                    for x_idx, x_label in enumerate(["MLIP stable", "MLIP unstable"]):
                        cell_val = confusion_array[y_idx, x_idx]
                        pct = (cell_val / total * 100) if total else 0.0
                        color = (
                            "#ffffff"
                            if max_val and cell_val >= 0.6 * max_val
                            else "#111111"
                        )
                        confusion.add_annotation(
                            x=x_label,
                            y=y_label,
                            text=f"{cell_val:.0f} ({pct:.1f}%)",
                            showarrow=False,
                            font={"color": color, "size": 14},
                        )
                confusion.update_layout(
                    title=f"{model_display} – Stability confusion matrix",
                    plot_bgcolor="#ffffff",
                )
            else:
                confusion.update_layout(
                    title="No stability confusion data available.",
                    plot_bgcolor="#ffffff",
                )
            scatter_graph = dcc.Graph(id=scatter_graph_id, figure=scatter)
            confusion_graph = dcc.Graph(
                id=f"{scatter_graph_id}-confusion", figure=confusion
            )
            meta = {"model": model_display, "type": "stability"}
            content = html.Div(
                [
                    html.P("Click a data point to preview the phonon dispersion plot."),
                    scatter_graph,
                    confusion_graph,
                ]
            )
            return content, meta, active_cell
        raise PreventUpdate

    @callback(
        Output(dispersion_container_id, "children"),
        Input(scatter_graph_id, "clickData"),
        Input(scatter_meta_store_id, "data"),
        prevent_initial_call=True,
    )
    def show_dispersion(click_data, scatter_meta):
        """
        Render the phonon dispersion figure for a clicked scatter point.

        Parameters
        ----------
        click_data
            Plotly click payload from the scatter graph.
        scatter_meta
            Metadata describing which model/metric is active.

        Returns
        -------
        dash.html.Div
            Container showing the dispersion PNG or a help message.
        """
        trigger = callback_context.triggered_id
        if trigger is None:
            raise PreventUpdate
        if trigger == scatter_meta_store_id:
            return html.Div("Click on a data point to preview the phonon dispersion.")
        if trigger != scatter_graph_id or not scatter_meta:
            raise PreventUpdate
        if not click_data:
            raise PreventUpdate
        point = click_data["points"][0]
        custom = point.get("customdata") or []
        point_id = custom[0] if custom else point.get("text") or point.get("label")
        if point_id is None:
            return html.Div("No dispersion plot available for this point.")
        model_display = scatter_meta.get("model")
        meta_type = scatter_meta.get("type")
        selected = None
        if meta_type == "metric":
            metric_points = _metric_points(model_display, scatter_meta.get("metric"))
            selected = next(
                (entry for entry in metric_points if entry.get("id") == point_id),
                None,
            )
            if selected is None and isinstance(point_id, int):
                if 0 <= point_id < len(metric_points):
                    selected = metric_points[point_id]
        elif meta_type == "stability":
            metric_points = _stability_points(model_display)
            selected = next(
                (entry for entry in metric_points if entry.get("id") == point_id),
                None,
            )
        elif meta_type == "bz":
            selected = _system_payload(model_display, point_id)
        if not selected:
            return html.Div("No dispersion plot available for this point.")

        image_src = None
        data_paths = selected.get("data_paths")
        if data_paths:
            image_src = _render_dispersion_image(
                model_display,
                selected.get("label") or selected.get("id", ""),
                data_paths,
            )
        elif selected.get("image"):
            image_src = f"/{selected['image']}"

        if not image_src:
            return html.Div("No dispersion plot available for this point.")

        title = selected.get("label") or selected.get("id", "")
        return html.Div(
            [
                html.H4(title),
                html.Img(
                    src=image_src,
                    style={"maxWidth": "100%", "border": "1px solid #ccc"},
                ),
            ]
        )
