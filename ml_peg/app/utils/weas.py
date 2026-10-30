"""Functions to WEAS visualisation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal


def generate_weas_html(
    filename: str | Path,
    mode: Literal["struct", "traj"] = "struct",
    index: int = 0,
    frame_indices: list[int] | None = None,
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
    frame_indices
        Optional zero-based XYZ frames to display. Selects raw frame blocks before
        parsing, so unselected frames do not create viewer objects.

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

    # In traj mode, report the current frame to the parent page on every change
    # (play, step, slider) so a linked plot can highlight the matching point.
    # WEAS has no frame-change event, so poll editor.avr.currentFrame via rAF.
    frame_reporter = (
        """
        let __mlPegLastFrame = editor.avr.currentFrame;
        function __mlPegReportFrame() {
            const f = editor.avr.currentFrame;
            if (f !== __mlPegLastFrame) {
                __mlPegLastFrame = f;
                window.parent.postMessage(
                    {type: "ml-peg-weas-frame", frame: f}, "*"
                );
            }
            requestAnimationFrame(__mlPegReportFrame);
        }
        requestAnimationFrame(__mlPegReportFrame);
        """
        if mode == "traj"
        else ""
    )

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

        // WEAS calls download/upload "export"/"import" in the browser bundle.
        const guiConfig = {{
            buttons: {{
                enabled: true,
                fullscreen: true,
                undo: false,
                redo: false,
                export: true,
                import: false,
                measurement: false,
            }},
        }};
        const editor = new WEAS({{ domElement, viewerConfig: {{ _modelStyle: 1 }}, guiConfig}});
        const originalExportImage = editor.tjs.exportImage.bind(editor.tjs);
        editor.tjs.exportImage = function(resolution = 3) {{
            return originalExportImage(resolution);
        }};

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

            const selectedFrames = {json.dumps(frame_indices)};
            if (selectedFrames !== null) {{
                // Fetch one cacheable trajectory, but parse only unique selected frames.
                const wanted = new Set(selectedFrames);
                const lines = structureData.trimEnd().split(/\\r?\\n/);
                const blocks = [];
                let frame = 0;
                for (let start = 0; start < lines.length; frame++) {{
                    const count = Number(lines[start].trim());
                    if (!Number.isInteger(count) || count < 0) {{
                        throw new Error("Invalid XYZ atom count");
                    }}
                    const end = start + count + 2;
                    if (wanted.has(frame)) blocks.push(lines.slice(start, end).join("\\n"));
                    start = end;
                }}
                structureData = blocks.join("\\n") + "\\n";
                title.textContent = `Viewing: ${{wanted.size}} unique structures in density bin`;
            }}
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
        {frame_reporter}
        </script>
    </body>
    </html>
    """  # noqa: E501
