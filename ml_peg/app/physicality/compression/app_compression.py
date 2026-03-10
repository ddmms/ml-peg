"""Run compression benchmark app."""

from __future__ import annotations

import json
import re
from pathlib import Path

from ase.formula import Formula

from dash import Dash, Input, Output, callback, dcc
from dash.dcc import Loading
from dash.exceptions import PreventUpdate
from dash.html import Div, Label
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


from ml_peg.app import APP_ROOT
from ml_peg.app.base_app import BaseApp
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)
BENCHMARK_NAME = "Compression"
DATA_PATH = APP_ROOT / "data" / "physicality" / "compression"
CURVE_PATH = DATA_PATH / "curves"
DOCS_URL = (
    "https://ddmms.github.io/ml-peg/user_guide/benchmarks/"
    "physicality.html#compression"
)

def _chemical_formula_from_label(label: str) -> str:
    """convert a label like "C2H4_pyxtal_0" to a reduce chemical formula string like "CH2"."""
    formula_str = label.split("_")[0]
    f = Formula(formula_str)
    return str(f.reduce()[0])

def _available_formulas(model_name: str) -> list[str]:
    """
    List unique formulas available for a given model.
    
    Parameters
    ----------
    model_name
        Selected model identifier.

    Returns
    -------
    list[str]
        Sorted list of unique formulas.
    """
    model_dir = CURVE_PATH / model_name
    if not model_dir.exists():
        return []
    groups: set[str] = set()
    for p in model_dir.glob("*.json"):
        groups.add(_chemical_formula_from_label(p.stem))
    return sorted(groups)

def _load_curves_for_formula(
    model_name: str, formula: str
) -> list[tuple[str, dict]]:
    """
    Load all curve payloads whose reduced chemical formula matches *formula*.

    Parameters
    ----------
    model_name
        Model identifier.
    formula
        Reduced chemical formula (e.g. ``"CH2"``).

    Returns
    -------
    list[tuple[str, dict]]
        List of ``(label, payload)`` tuples for every matching structure.
    """
    model_dir = CURVE_PATH / model_name
    if not model_dir.exists():
        return []
    results: list[tuple[str, dict]] = []
    for p in sorted(model_dir.glob("*.json")):
        if _chemical_formula_from_label(p.stem) == formula:
            try:
                with p.open(encoding="utf8") as fh:
                    results.append((p.stem, json.load(fh)))
            except Exception:
                continue
    return results


# Fraction of the y-axis occupied by the linear region (±linthresh).
LINEAR_FRAC: float = 0.6

# Separate linear thresholds for energy and pressure symlog axes
LINTHRESH_ENERGY: float = 10.0   # eV/atom
LINTHRESH_PRESSURE: float = 100.0  # GPa

# Conversion factor: 1 eV/ų = 160.21766 GPa


def _symlog(
    values: list[float] | np.ndarray,
    linthresh: float = 10.0,
    linear_frac: float = LINEAR_FRAC,
    decades: int = 4,
) -> list[float]:
    """
    Apply a symmetric-log transform to *values*.

    Linear within ``[-linthresh, linthresh]``, logarithmic outside.
    The linear region is scaled so that it occupies *linear_frac* of the
    total axis range (assuming *decades* log decades on each side).

    Parameters
    ----------
    values
        Raw data values.
    linthresh
        Linear threshold.
    linear_frac
        Fraction of the full axis height reserved for the linear region.
    decades
        Number of log decades shown on each side (used to compute the
        compression factor for the logarithmic tails).

    Returns
    -------
    list[float]
        Transformed values.
    """
    lin_half = linear_frac / 2.0
    log_scale = (1.0 - linear_frac) / (2.0 * max(decades, 1))

    arr = np.asarray(values, dtype=float)
    out = np.where(
        np.abs(arr) <= linthresh,
        lin_half * arr / linthresh,
        np.sign(arr)
        * (lin_half + log_scale * np.log10(np.abs(arr) / linthresh)),
    )
    return out.tolist()


def _symlog_ticks(
    linthresh: float = 10.0,
    decades: int = 4,
    linear_frac: float = LINEAR_FRAC,
) -> tuple[list[float], list[str]]:
    """
    Generate tick positions and labels for a symlog axis.

    Parameters
    ----------
    linthresh
        Linear threshold matching ``_symlog``.
    decades
        Number of decades to show on each side of zero.
    linear_frac
        Must match the value passed to ``_symlog``.

    Returns
    -------
    tuple[list[float], list[str]]
        Tick values (in transformed space) and their display labels.
    """
    ticks_val: list[float] = []
    ticks_txt: list[str] = []

    kw = {"linthresh": linthresh, "linear_frac": linear_frac, "decades": decades}

    # Negative decades
    for d in range(decades, 0, -1):
        raw = -(linthresh * 10 ** d)
        ticks_val.append(_symlog([raw], **kw)[0])
        ticks_txt.append(f"{raw:.0e}")
    # Linear region
    for v in [-linthresh, -linthresh / 2, 0, linthresh / 2, linthresh]:
        ticks_val.append(_symlog([v], **kw)[0])
        ticks_txt.append(f"{v:g}")
    # Positive decades
    for d in range(1, decades + 1):
        raw = linthresh * 10 ** d
        ticks_val.append(_symlog([raw], **kw)[0])
        ticks_txt.append(f"{raw:.0e}")

    return ticks_val, ticks_txt


# A palette for overlaid RSS curves
_PALETTE = [
    "royalblue", "firebrick", "seagreen", "darkorange", "mediumpurple",
    "deeppink", "teal", "goldenrod", "slategray", "crimson",
]


class CompressionApp(BaseApp):
    """Compression benchmark app layout and callbacks."""

    def register_callbacks(self) -> None:
        """Register dropdown-driven compression curve callbacks."""
        model_dropdown_id = f"{BENCHMARK_NAME}-model-dropdown"
        composition_dropdown_id = f"{BENCHMARK_NAME}-composition-dropdown"
        figure_id = f"{BENCHMARK_NAME}-figure"

        @callback(
            Output(composition_dropdown_id, "options"),
            Output(composition_dropdown_id, "value"),
            Input(model_dropdown_id, "value"),
        )
        def _update_composition_options(model_name: str):
            """
            Populate composition dropdown for the selected model.

            Parameters
            ----------
            model_name
                Selected model value from the model dropdown.

            Returns
            -------
            tuple[list[dict], str | None]
                Composition dropdown options and default selection.
            """
            if not model_name:
                raise PreventUpdate
            formulas = _available_formulas(model_name)
            options = [{"label": f, "value": f} for f in formulas]
            default = formulas[0] if formulas else None
            return options, default

        @callback(
            Output(figure_id, "figure"),
            Input(model_dropdown_id, "value"),
            Input(composition_dropdown_id, "value"),
        )
        def _update_figure(model_name: str, composition: str | None):
            """
            Render energy-per-atom and pressure curves vs linear scale
            factor for all structures sharing the selected composition,
            using symlog y-axes.

            Parameters
            ----------
            model_name
                Selected model identifier.
            composition
                Selected composition label.

            Returns
            -------
            go.Figure
                Plotly figure with two subplots.
            """
            if not model_name or not composition:
                raise PreventUpdate

            curves = _load_curves_for_formula(model_name, composition)
            if not curves:
                raise PreventUpdate

            decades = 4
            e_tick_vals, e_tick_text = _symlog_ticks(
                LINTHRESH_ENERGY, decades=decades, linear_frac=LINEAR_FRAC,
            )
            p_tick_vals, p_tick_text = _symlog_ticks(
                LINTHRESH_PRESSURE, decades=decades, linear_frac=LINEAR_FRAC,
            )

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.08,
                subplot_titles=(
                    "Energy per atom vs Scale factor",
                    "Pressure vs Scale factor",
                ),
            )

            for idx, (label, payload) in enumerate(curves):
                scales = payload.get("scale", [])
                energies = payload.get("energy_per_atom", [])
                pressures = payload.get("pressure", [])
                color = _PALETTE[idx % len(_PALETTE)]
                # Use the full composition + RSS index as the legend entry
                short_label = label

                if not scales or not energies:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=scales,
                        y=_symlog(energies, LINTHRESH_ENERGY, LINEAR_FRAC, decades),
                        mode="lines+markers",
                        name=f"E — {short_label}",
                        line={"color": color},
                        marker={"size": 3},
                        legendgroup=short_label,
                        customdata=energies,
                        hovertemplate=(
                            "scale: %{x:.3f}<br>"
                            "E/atom: %{customdata:.4f} eV<extra></extra>"
                        ),
                    ),
                    row=1,
                    col=1,
                )

                if pressures:
                    fig.add_trace(
                        go.Scatter(
                            x=scales,
                            y=_symlog(pressures, LINTHRESH_PRESSURE, LINEAR_FRAC, decades),
                            mode="lines+markers",
                            name=f"P — {short_label}",
                            line={"color": color},
                            marker={"size": 3},
                            legendgroup=short_label,
                            showlegend=False,
                            customdata=pressures,
                            hovertemplate=(
                                "scale: %{x:.3f}<br>"
                                "P: %{customdata:.2f} GPa<extra></extra>"
                            ),
                        ),
                        row=2,
                        col=1,
                    )

            # Apply symlog tick formatting to both y-axes
            fig.update_yaxes(
                tickvals=e_tick_vals,
                ticktext=e_tick_text,
                title_text="Energy per atom (eV, symlog)",
                row=1, col=1,
            )
            fig.update_yaxes(
                tickvals=p_tick_vals,
                ticktext=p_tick_text,
                title_text="Pressure (GPa, symlog)",
                row=2, col=1,
            )
            fig.update_xaxes(title_text="Scale factor", row=2, col=1)

            fig.update_layout(
                title=f"{model_name} — {composition}",
                height=700,
                showlegend=True,
                template="plotly_white",
            )

            return fig


def get_app() -> CompressionApp:
    """
    Get compression benchmark app layout and callback registration.

    Returns
    -------
    CompressionApp
        Benchmark layout and callback registration.
    """
    model_options = [{"label": model, "value": model} for model in MODELS]
    default_model = model_options[0]["value"] if model_options else None

    extra_components = [
        Div(
            [
                Label("Select model:"),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-model-dropdown",
                    options=model_options,
                    value=default_model,
                    clearable=False,
                    style={"width": "300px", "marginBottom": "20px"},
                ),
                Label("Select composition:"),
                dcc.Dropdown(
                    id=f"{BENCHMARK_NAME}-composition-dropdown",
                    options=[],
                    value=None,
                    clearable=False,
                    style={"width": "300px"},
                ),
            ],
            style={"marginBottom": "20px"},
        ),
        Loading(
            dcc.Graph(
                id=f"{BENCHMARK_NAME}-figure",
                style={"height": "700px", "width": "100%", "marginTop": "20px"},
            ),
            type="circle",
        ),
    ]

    return CompressionApp(
        name=BENCHMARK_NAME,
        description=(
            "Uniform crystal compression explorer. Structures are isotropically "
            "scaled and the energy per atom, its derivative dE/dV, and stress are "
            "recorded. Metrics are averaged across all structures."
        ),
        docs_url=DOCS_URL,
        table_path=DATA_PATH / "compression_metrics_table.json",
        extra_components=extra_components,
    )


if __name__ == "__main__":
    dash_app = Dash(__name__, assets_folder=DATA_PATH.parent.parent)
    compression_app = get_app()
    dash_app.layout = compression_app.layout
    compression_app.register_callbacks()
    dash_app.run(port=8056, debug=True)
