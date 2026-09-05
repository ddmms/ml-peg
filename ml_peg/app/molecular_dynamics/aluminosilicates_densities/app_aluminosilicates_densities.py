"""Run aluminosilicates densities app."""

from __future__ import annotations

from pathlib import Path
import re

from dash import Dash, Input, Output, callback, dcc
from dash.html import Div, Iframe

from ml_peg.analysis.molecular_dynamics.aluminosilicates_densities import (
    analyse_aluminosilicates_densities as density_analysis,
)
from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.app.utils.build_callbacks import plot_from_table_column
from ml_peg.app.utils.load import read_plot

COMPOSITIONS = density_analysis.COMPOSITIONS
MODELS = density_analysis.MODELS
BENCHMARK_NAME = "Aluminosilicates Densities"
DOCS_URL = "https://ddmms.github.io/ml-peg/user_guide/benchmarks/molecular_dynamics.html#aluminosilicates-densities"
DATA_PATH = APP_ROOT / "data" / "molecular_dynamics" / "aluminosilicates_densities"
INFO_PATH = DATA_PATH / "info.json"


def parse_xyz_cell(xyz_text: str) -> tuple[float, float, float] | None:
    """
    Extract unit cell dimensions (a, b, c) from the Extended XYZ header.

    Parameters
    ----------
    xyz_text : str
        The raw text content of the Extended XYZ file.

    Returns
    -------
    tuple[float, float, float] or None
        Unit cell lengths (a, b, c) rounded to two decimal places, or None
        if the lattice specifications are missing or invalid.
    """
    lines = xyz_text.strip().splitlines()
    if len(lines) > 1:
        match = re.search(r'Lattice="([^"]+)"', lines[1])
        if match:
            vals = [float(x) for x in match.group(1).split()]
            if len(vals) == 9:
                ax, ay, az, bx, by, bz, cx, cy, cz = vals
                a = (ax**2 + ay**2 + az**2) ** 0.5
                b = (bx**2 + by**2 + bz**2) ** 0.5
                c = (cx**2 + cy**2 + cz**2) ** 0.5
                return round(a, 2), round(b, 2), round(c, 2)
    return None


def render_rich_3d_structure(filepath: Path) -> Iframe | Div:
    """
    3D structure viewer with control toolbar.

    Supports style, color scheme, unit cell toggle, and cell dimensions.

    Parameters
    ----------
    filepath : Path
        Path to the structure file to be displayed.

    Returns
    -------
    Iframe or Div
        The HTML or Bokeh component containing the 3D structure viewer.
    """
    if not filepath.exists():
        return Div(
            f"File not found: {filepath.name}",
            style={"color": "#e53e3e", "padding": "15px"},
        )

    xyz_text = filepath.read_text()
    cell_dims = parse_xyz_cell(xyz_text)
    a, b, c = cell_dims
    cell_str = (
        f"a = {a} \u00c5 | b = {b} \u00c5 | c = {c} \u00c5"
        if cell_dims
        else "Cell dimensions N/A"
    )
    file_name = filepath.name
    xyz_json = (
        xyz_text.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("\n", "\\n")
        .replace("\r", "")
    )

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/3Dmol/2.0.4/3Dmol-min.js"></script>
      <style>
        body {{
          margin: 0;
          padding: 0;
          font-family: system-ui, sans-serif;
          background: #ffffff;
          overflow: hidden;
        }}
        #toolbar {{
          position: absolute; top: 10px; left: 10px; z-index: 100;
          background: rgba(255, 255, 255, 0.95); padding: 8px 12px;
          border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15);
          display: flex; gap: 12px; align-items: center; font-size: 13px;
        }}
        select, button {{
          padding: 4px 8px; border-radius: 4px; border: 1px solid #cbd5e1;
          background: #fff; cursor: pointer; font-size: 12px;
        }}
        button:hover {{ background: #f1f5f9; }}
        #box-info {{
          position: absolute; bottom: 10px; left: 10px; z-index: 100;
          background: rgba(15, 23, 42, 0.85); color: #fff; padding: 6px 12px;
          border-radius: 6px; font-size: 12px; font-family: monospace;
        }}
      </style>
    </head>
    <body>
      <div id="toolbar">
        <label><b>Style:</b>
          <select id="styleSelect" onchange="updateStyle()">
            <option value="ballStick">Ball & Stick</option>
            <option value="spacefill">Spacefill (Spheres)</option>
            <option value="sticks">Sticks</option>
            <option value="wireframe">Wireframe</option>
          </select>
        </label>
        <label><b>Colors:</b>
          <select id="colorSelect" onchange="updateStyle()">
            <option value="cpk">CPK (Elements)</option>
            <option value="index">By Atom Index</option>
          </select>
        </label>
        <button onclick="toggleUnitCell()">Unit Cell Box</button>
        <button onclick="resetCamera()">Reset View</button>
      </div>

    <div id="box-info">
        <b>File:</b> {file_name} &nbsp;|&nbsp; <b>Cell:</b> {cell_str}
    </div>
    <div id="container" style="width: 100vw; height: 100vh;"></div>

      <script>
        let viewer = null;
        let showBox = true;

        document.addEventListener("DOMContentLoaded", function() {{
          let container = document.getElementById("container");
          viewer = $3Dmol.createViewer(container, {{backgroundColor: "white"}});
          let xyzData = `{xyz_json}`;
          viewer.addModel(xyzData, "xyz");
          viewer.addUnitCell();
          updateStyle();
          viewer.zoomTo();
          viewer.render();
        }});

        function updateStyle() {{
          if (!viewer) return;
          let styleType = document.getElementById("styleSelect").value;
          let colorType = document.getElementById("colorSelect").value;
          let styleSpec = {{}};

          if (styleType === "ballStick") {{
            styleSpec = {{ sphere: {{ scale: 0.25 }}, stick: {{ radius: 0.12 }} }};
          }} else if (styleType === "spacefill") {{
            styleSpec = {{ sphere: {{ scale: 0.8 }} }};
          }} else if (styleType === "sticks") {{
            styleSpec = {{ stick: {{ radius: 0.25 }} }};
          }} else if (styleType === "wireframe") {{
            styleSpec = {{ line: {{}} }};
          }}

          if (colorType === "index") {{
            styleSpec.colorscheme = "rainbow";
          }}

          viewer.setStyle({{}}, styleSpec);
          viewer.render();
        }}

        function toggleUnitCell() {{
          if (!viewer) return;
          showBox = !showBox;
          if (showBox) {{
            viewer.addUnitCell();
          }} else {{
            viewer.removeAllShapes();
          }}
          viewer.render();
        }}

        function resetCamera() {{
          if (!viewer) return;
          viewer.zoomTo();
          viewer.render();
        }}
      </script>
    </body>
    </html>
    """
    return Iframe(
        srcDoc=html_content,
        style={
            "width": "100%",
            "height": "520px",
            "border": "1px solid #cbd5e1",
            "borderRadius": "8px",
        },
    )


class AluminosilicatesDensitiesApp(BaseApp):
    """Aluminosilicates densities benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """
        Register callbacks to app.

        Returns
        -------
        None
            This method does not return a value.
        """
        scatter = read_plot(
            DATA_PATH / "figure_density_parity.json",
            id=f"{BENCHMARK_NAME}-figure",
        )

        plot_from_table_column(
            table_id=self.table_id,
            plot_id=f"{BENCHMARK_NAME}-figure-placeholder",
            column_to_plot={
                "Density MAE (albite)": scatter,
                "Density MAE (anorthite)": scatter,
                "Density MAE (sanidine)": scatter,
                "Density MAE (mean)": scatter,
            },
        )

        @callback(
            Output(f"{BENCHMARK_NAME}-struct-placeholder", "children"),
            Input(f"{BENCHMARK_NAME}-figure", "clickData"),
            Input(f"{BENCHMARK_NAME}-replica-select", "value"),
        )
        def update_displayed_structure(click_data, replica_idx):
            """
            Update the displayed 3D atomic structure based on user click.

            Parameters
            ----------
            click_data : dict or None
                Interaction data from clicking points on the density parity plot.
            replica_idx : int
                Index of the structure replica to display.

            Returns
            -------
            Iframe or Div
                The 3D viewer component for the selected structure.
            """
            model = MODELS[0]
            comp = COMPOSITIONS[0]

            if click_data and "points" in click_data and len(click_data["points"]) > 0:
                pt = click_data["points"][0]
                curve_idx = pt.get("curveNumber", 0)
                point_idx = pt.get("pointIndex", pt.get("pointNumber", 0))

                if 0 <= curve_idx < len(MODELS):
                    model = MODELS[curve_idx]
                if 0 <= point_idx < len(COMPOSITIONS):
                    comp = COMPOSITIONS[point_idx]

            replica_idx = replica_idx if replica_idx is not None else 0

            filename = f"{model}_{comp}_replica{replica_idx}_quenched.xyz"
            filepath = DATA_PATH / model / comp / filename

            return render_rich_3d_structure(filepath)


def get_app() -> AluminosilicatesDensitiesApp:
    """
    Get aluminosilicates densities benchmark app layout and callback registration.

    Returns
    -------
    AluminosilicatesDensitiesApp
        The initialized app instance for the aluminosilicates densities benchmark.
    """
    return AluminosilicatesDensitiesApp(
        name=BENCHMARK_NAME,
        description=(
            "Performance in predicting melt-quenched glass densities "
            "from MD simulations. "
            "Reference values are experimental densities."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "aluminosilicates_densities_metrics_table.json",
        extra_components=[
            dcc.Dropdown(
                id=f"{BENCHMARK_NAME}-replica-select",
                options=[{"label": f"Replica {i}", "value": i} for i in range(3)],
                value=0,
                clearable=False,
                style={"width": "200px", "marginBottom": "10px"},
            ),
            Div(id=f"{BENCHMARK_NAME}-figure-placeholder"),
            Div(id=f"{BENCHMARK_NAME}-struct-placeholder"),
        ],
        info_path=INFO_PATH,
        framework_ids="mace-polar-1",
    )


if __name__ == "__main__":
    full_app = Dash(
        __name__,
        assets_folder=DATA_PATH.parent.parent,
        suppress_callback_exceptions=True,
    )
    benchmark_app = get_app()
    full_app.layout = benchmark_app.layout
    benchmark_app.register_callbacks()
    full_app.run(port=8064, debug=True)
