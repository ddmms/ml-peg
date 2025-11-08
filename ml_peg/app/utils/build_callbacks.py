"""Helpers to create callbacks for Dash app."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Literal

from dash import Input, Output, callback
from dash.dcc import Graph
from dash.exceptions import PreventUpdate
from dash.html import Div, Iframe
import plotly.graph_objects as go

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
    """
    base_dir = Path(manifest_dir)

    def _load_manifest(model_name: str) -> dict:
        """
        Read the manifest JSON for the selected model.

        Parameters
        ----------
        model_name
            Model identifier matching a subdirectory under ``manifest_dir``.

        Returns
        -------
        dict
            Parsed manifest describing overview and per-element assets.
        """
        manifest_path = base_dir / model_name / "manifest.json"
        if not manifest_path.exists():
            raise PreventUpdate
        with manifest_path.open("r", encoding="utf8") as fh:
            return json.load(fh)

    def _data_url(path: Path) -> str:
        """
        Convert an image file into a base64 data URL.

        Parameters
        ----------
        path
            Path to the image file.

        Returns
        -------
        str
            Data URL string suitable for Plotly layout images.
        """
        suffix = path.suffix.lower()
        mime = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".svg": "image/svg+xml",
        }.get(suffix, "application/octet-stream")
        return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    def _image_figure(src: str) -> go.Figure:
        """
        Build a Plotly figure that displays the supplied image.

        Parameters
        ----------
        src
            Base64-encoded data URL for the image.

        Returns
        -------
        go.Figure
            Figure containing the image without axes, matching legacy behaviour.
        """
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
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
                "sizex": 1,
                "sizey": 1,
                "xanchor": "left",
                "yanchor": "bottom",
                "layer": "below",
            }
        )
        fig.update_layout(
            xaxis={
                "visible": False,
                "range": [0, 1],
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
        manifest = _load_manifest(model_name)
        element_opts = manifest.get("elements", {})
        options = [{"label": overview_label, "value": overview_label}] + [
            {"label": element, "value": element} for element in sorted(element_opts)
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

        manifest = _load_manifest(model_name)
        if element_value in (None, overview_label):
            rel_path = manifest.get("overview")
        else:
            rel_path = manifest.get("elements", {}).get(element_value or "")

        if not rel_path:
            raise PreventUpdate

        file_path = base_dir / model_name / rel_path
        if not file_path.exists():
            raise PreventUpdate

        return _image_figure(_data_url(file_path))
