"""Functions to WEAS visualisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


def generate_weas_html(
    filename: str | Path,
    mode: Literal["struct", "traj"] = "struct",
    index: int = 0,
    *,
    color_by: str | None = None,
    color_ramp: list[str] | None = None,
    legend_items: list[tuple[str, str]] | None = None,
    show_controls: bool = False,
    show_bounds: bool = False,
) -> str:
    """
    Generate HTML for WEAS.

    Parameters
    ----------
    filename
        Path of structure file.
    mode
        Whether viewing a single structure (or set of structures) ("struct"), or
        if different views of the a trajectory are being selected ("traj").
    index
        Frame of structure file to load, or of trajectory to view. In "struct" mode,
        all structures will be loaded by default. In "traj" mode, the first frame will
        be loaded by default.
    color_by
        Optional atom attribute name to color by.
    color_ramp
        Optional color ramp for attribute coloring.
    legend_items
        Optional legend entries as ``(label, color)`` pairs.
    show_controls
        Whether to display viewer controls.
    show_bounds
        Whether to show the periodic cell bounds.

    Returns
    -------
    str
        HTML for WEAS to visualise structure.
    """
    if mode == "struct":
        frame = 0
        atoms_txt = f"atoms[{index}" if index else "atoms"
    elif mode == "traj":
        frame = index
        atoms_txt = "atoms"

    color_by_js = f'editor.avr.color_by = "{color_by}";' if color_by is not None else ""
    color_ramp_js = (
        f"editor.avr.color_ramp = {json.dumps(color_ramp)};"
        if color_ramp is not None
        else ""
    )

    legend_html = ""
    if legend_items:
        legend_rows = "\n".join(
            (
                "<div style='display:flex; align-items:center; gap:8px;'>"
                f"<span style='display:inline-block; width:12px; height:12px; "
                f"background:{color}; border-radius:2px;'></span>"
                f"<span>{label}</span></div>"
            )
            for label, color in legend_items
        )
        legend_html = (
            "<div id='legend' style='position:absolute; top:36px; right:24px; "
            "font-size:15px; background:rgba(255,255,255,0.85); padding:6px 8px; "
            "border-radius:6px; z-index:10;'>\n"
            f"{legend_rows}\n"
            "</div>"
        )
    bounds_js = (
        "editor.avr.showCell = true;"
        "editor.avr.showAxis = false;"
        "editor.avr.boundary = [[0, 1], [0, 1], [0, 1]];"
        if show_bounds
        else ""
    )

    return f"""
    <!doctype html>
    <html lang="en">
    <body>
        <div id="viewer-wrapper" style="position: relative; width: 100%; height: 500px">
            <div id="viewer" style="width: 100%; height: 100%"></div>
            {legend_html}
        </div>

        <script type="module">

        async function fetchFile(filename) {{
            const response = await fetch(`${{filename}}`);
            if (!response.ok) {{
            throw new Error(`Failed to load file for structure: ${{filename}}`);
            }}
            return await response.text();
        }}

        import {{ WEAS, parseXYZ, parseCIF, parseCube, parseXSF }} from 'https://unpkg.com/weas/dist/index.mjs';
        const domElement = document.getElementById("viewer");

        // hide the buttons
        const guiConfig = {{
            buttons: {{
                enabled: {str(show_controls).lower()},
            }},
        }};
        const editor = new WEAS({{ domElement, viewerConfig: {{ _modelStyle: 1 }}, guiConfig}});

        let structureData;
        const filename = "{str(filename)}";
        console.log("filename: ", filename);
        structureData = await fetchFile(filename);
        console.log("structureData: ", structureData);

        if (filename.endsWith(".xyz") || filename.endsWith(".extxyz")) {{

            const atoms = parseXYZ(structureData);
            editor.avr.atoms = {atoms_txt};
            editor.avr.modelStyle = 1;
            {color_by_js}
            {color_ramp_js}
            {bounds_js}

        }} else if (filename.endsWith(".cif")) {{

            const atoms = parseCIF(structureData);
            editor.avr.atoms = {atoms_txt};
            editor.avr.showBondedAtoms = true;
            editor.avr.colorType = "VESTA";
            editor.avr.boundary = [[-0.01, 1.01], [-0.01, 1.01], [-0.01, 1.01]];
            editor.avr.modelStyle = 2;

        }} else {{
            document.getElementById("viewer").innerText = "Unsupported file format.";
        }}

        editor.avr.currentFrame = {frame};
        editor.avr.drawModels();
        editor.render();

        </script>
    </body>
    </html>
    """  # noqa: E501
