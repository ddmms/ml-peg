"""Data-download callback and helpers for the diatomics benchmark app."""

from __future__ import annotations

import csv
import io
import json
from pathlib import Path

from dash import Input, Output, State, callback, dcc, no_update
from dash.exceptions import PreventUpdate

from ml_peg.app.utils.build_callbacks import (
    load_model_curves,
    render_periodic_curve_gallery_png,
)


def _safe_filename_stem(
    model_name: str,
    element_value: str | None,
    overview_label: str,
) -> str:
    """
    Build a safe download filename stem for the current model and view.

    Parameters
    ----------
    model_name
        Name of the model being exported.
    element_value
        Current dropdown choice: the overview label, or an element symbol.
    overview_label
        The dropdown value that means "show the homonuclear overview".

    Returns
    -------
    str
        Filename stem with anything other than letters, digits, ``-``, ``_``
        or ``.`` replaced by underscores.
    """
    view = "homonuclear" if element_value == overview_label else str(element_value)
    stem = f"{model_name}_diatomics_{view}"
    return "".join(char if char.isalnum() or char in "-_." else "_" for char in stem)


def _serialise_selected_curves(
    curve_path: Path,
    model_name: str,
    element_value: str | None,
    overview_label: str,
) -> tuple[dict, list[dict]]:
    """
    Build the JSON and CSV representations of the current selected view.

    Parameters
    ----------
    curve_path
        Directory holding one subfolder of curve files per model.
    model_name
        Name of the model whose curves to export.
    element_value
        Current dropdown choice: the overview label, or an element symbol.
    overview_label
        The dropdown value that means "show the homonuclear overview".

    Returns
    -------
    tuple[dict, list[dict]]
        A nested structure for the JSON file (model, view, selected element,
        and one entry per element pair) and a flat list of rows, one per
        distance point, for the CSV file.
    """
    selected_element, curves = load_model_curves(
        curve_path, model_name, element_value, overview_label
    )
    view = "homonuclear" if selected_element is None else "heteronuclear"
    pairs: list[dict] = []
    rows: list[dict] = []

    for pair in sorted(curves):
        curve = curves[pair]
        distances = curve.get("distance") or []
        energies = curve.get("energy") or []
        forces = curve.get("force_parallel") or []
        shift = energies[-1] if energies else None
        shifted_energies = [
            energy - shift if shift is not None else None for energy in energies
        ]
        element_1 = curve.get("element_1")
        element_2 = curve.get("element_2")

        pair_entry = {
            "pair": pair,
            "element_1": element_1,
            "element_2": element_2,
            "distance": distances,
            "energy": energies,
            "shifted_energy": shifted_energies,
        }
        if forces:
            pair_entry["force_parallel"] = forces
        pairs.append(pair_entry)

        for idx, distance in enumerate(distances):
            energy = energies[idx] if idx < len(energies) else None
            shifted_energy = (
                shifted_energies[idx] if idx < len(shifted_energies) else None
            )
            force_parallel = forces[idx] if idx < len(forces) else None
            rows.append(
                {
                    "model": model_name,
                    "view": view,
                    "selected_element": selected_element or "",
                    "pair": pair,
                    "element_1": element_1,
                    "element_2": element_2,
                    "distance": distance,
                    "energy": energy,
                    "shifted_energy": shifted_energy,
                    "force_parallel": force_parallel,
                }
            )

    return (
        {
            "model": model_name,
            "view": view,
            "selected_element": selected_element,
            "pairs": pairs,
        },
        rows,
    )


def register_data_download_callbacks(
    *,
    download_id: str,
    model_dropdown_id: str,
    element_dropdown_id: str,
    curve_path: Path,
    overview_label: str,
) -> None:
    """
    Register the callback that downloads the current diatomics view.

    Parameters
    ----------
    download_id
        Prefix shared by the download dropdown, button, status and
        ``dcc.Download`` component IDs.
    model_dropdown_id
        Component ID of the model selector.
    element_dropdown_id
        Component ID of the element selector.
    curve_path
        Directory holding one subfolder of curve files per model.
    overview_label
        The dropdown value that means "show the homonuclear overview".
    """

    @callback(
        Output(f"{download_id}-download", "data"),
        Output(f"{download_id}-status", "children"),
        Input(f"{download_id}-button", "n_clicks"),
        State(f"{download_id}-format", "value"),
        State(model_dropdown_id, "value"),
        State(element_dropdown_id, "value"),
        prevent_initial_call=True,
        running=[(Output(f"{download_id}-button", "disabled"), True, False)],
    )
    def _download_data(
        n_clicks: int,
        download_format: str,
        model_name: str,
        element_value: str | None,
    ) -> tuple:
        """
        Turn the current diatomics view into a downloadable file.

        Parameters
        ----------
        n_clicks
            Number of times the download button has been clicked.
        download_format
            Chosen export format (``csv``, ``json`` or ``png``).
        model_name
            Name of the currently selected model.
        element_value
            Current dropdown choice: the overview label, or an element symbol.

        Returns
        -------
        tuple
            The file to download, and a status message (empty on success, or a
            short explanation when there is no data to export).
        """
        if not n_clicks or not model_name:
            raise PreventUpdate

        no_data_message = "No curve data for this selection."
        stem = _safe_filename_stem(model_name, element_value, overview_label)
        fmt = (download_format or "csv").lower()
        if fmt == "png":
            try:
                png_bytes, _width, _height = render_periodic_curve_gallery_png(
                    curve_dir=curve_path,
                    model_name=model_name,
                    element_value=element_value,
                    overview_label=overview_label,
                )
            except PreventUpdate:
                return no_update, no_data_message
            return dcc.send_bytes(png_bytes, f"{stem}.png", type="image/png"), ""

        json_data, rows = _serialise_selected_curves(
            curve_path, model_name, element_value, overview_label
        )
        if not rows:
            return no_update, no_data_message
        if fmt == "json":
            return (
                dcc.send_string(
                    json.dumps(json_data, indent=2),
                    f"{stem}.json",
                    type="application/json",
                ),
                "",
            )

        buffer = io.StringIO()
        fieldnames = [
            "model",
            "view",
            "selected_element",
            "pair",
            "element_1",
            "element_2",
            "distance",
            "energy",
            "shifted_energy",
            "force_parallel",
        ]
        writer = csv.DictWriter(buffer, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        return dcc.send_string(buffer.getvalue(), f"{stem}.csv", type="text/csv"), ""
