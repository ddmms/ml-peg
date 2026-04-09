"""Functions to WEAS visualisation."""

from __future__ import annotations

from pathlib import Path
from typing import Literal


def generate_weas_html(
    filename: str | Path,
    mode: Literal["struct", "traj"] = "struct",
    index: int = 0,
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

    return f"""
    <!doctype html>
    <html lang="en">
    <body>
        <div id="weas-title"
             style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
                    font-size: 12px;
                    color: #444;
                    margin: 0 0 8px 0;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;"></div>
        <div id="viewer" style="position: relative; width: 100%; height: 500px"></div>

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
                enabled: false,
            }},
        }};
        const editor = new WEAS({{ domElement, viewerConfig: {{ _modelStyle: 1 }}, guiConfig}});

        let structureData;
        const filename = "{str(filename)}";
        const title = document.getElementById("weas-title");
        if (title) {{
            const basename = filename.split(/[/\\\\]/).pop() || filename;
            title.textContent = `Viewing: ${{basename}}`;
            title.title = basename;
        }}
        console.log("filename: ", filename);
        structureData = await fetchFile(filename);
        console.log("structureData: ", structureData);

        if (filename.endsWith(".xyz") || filename.endsWith(".extxyz")) {{

            const atoms = parseXYZ(structureData);
            editor.avr.atoms = {atoms_txt};
            editor.avr.modelStyle = 1;

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
        editor.render();

        </script>
    </body>
    </html>
    """  # noqa: E501
