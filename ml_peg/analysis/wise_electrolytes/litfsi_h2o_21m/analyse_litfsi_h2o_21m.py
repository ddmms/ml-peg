"""Analyse the WiSE 21 m LiTFSI/H2O electrolyte benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from scipy.interpolate import interp1d

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

# --- Paths -------------------------------------------------------------------

CALC_PATH = CALCS_ROOT / "wise_electrolytes" / "litfsi_h2o_21m" / "outputs"
OUT_PATH = APP_ROOT / "data" / "wise_electrolytes" / "litfsi_h2o_21m"
EXP_SAXS_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "saxs_maginn.txt"
)

MODELS = load_models(current_models)

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# --- Physical constants and reference values ---------------------------------

# Box geometry (same for all models: p64_w170 NVT cell)
L_BOX = 27.4938  # Å (cubic NVT cell at experimental density)
V_BOX = L_BOX**3  # Å³

# Integration cutoff: first minimum of Li-O_total g(r) (r2SCAN AIMD)
R_CUT = 2.83  # Å
DR = 0.02  # Å (bin width used in calc_rdf.py)

# Experimental references
RHO_EXP = 1.7126     # g/cm³ — Gilbert et al., JCED 62, 2056 (2017)
CN_EXP_WATER = 2.0   # Watanabe et al., JPCB 125, 7477 (2021)
CN_EXP_TFSI = 2.0


# =============================================================================
# Density helpers and figures
# =============================================================================


def load_density_results() -> dict[str, dict]:
    """
    Load density.json for each registered model that has calc output.

    Returns
    -------
    dict[str, dict]
        Per-model density results keyed by registry model name.
    """
    results = {}
    for model in MODELS:
        json_path = CALC_PATH / model / "density.json"
        if json_path.exists():
            with open(json_path) as f:
                results[model] = json.load(f)
    return results


def build_density_bar_chart(data: dict[str, dict]) -> go.Figure:
    """
    Build bar chart of density per model.

    Parameters
    ----------
    data : dict[str, dict]
        Per-model density results.

    Returns
    -------
    go.Figure
        Plotly bar chart figure.
    """
    fig = go.Figure()
    for model in MODELS:
        if model not in data:
            continue
        fig.add_trace(
            go.Bar(
                x=[model],
                y=[data[model]["rho_mean"]],
                error_y={"type": "data", "array": [data[model]["rho_std"]]},
                name=model,
            )
        )

    fig.add_hline(
        y=RHO_EXP,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Exp. ({RHO_EXP} g/cm\u00b3, Gilbert 2017)",
    )
    fig.update_layout(
        title="Electrolyte Density (LiTFSI/H2O, 21 m)",
        yaxis_title="Density / g cm⁻³",
        showlegend=False,
    )
    return fig


def build_density_timeseries(data: dict[str, dict]) -> go.Figure:
    """
    Build density vs time plot for all models.

    Parameters
    ----------
    data : dict[str, dict]
        Per-model density results.

    Returns
    -------
    go.Figure
        Plotly timeseries figure.
    """
    fig = go.Figure()
    for model in MODELS:
        if model not in data:
            continue
        d = data[model]
        fig.add_trace(
            go.Scatter(
                x=d["time_full"],
                y=d["density_full"],
                mode="lines",
                name=model,
                opacity=0.8,
            )
        )

    fig.add_hline(
        y=RHO_EXP,
        line_dash="dash",
        line_color="black",
        annotation_text=f"Exp. ({RHO_EXP} g/cm\u00b3)",
    )
    fig.update_layout(
        title="NPT Density vs Time (LiTFSI/H2O)",
        xaxis_title="Time / ps",
        yaxis_title="Density / g cm⁻³",
    )
    return fig


# =============================================================================
# RDF / coordination number helpers and figures
# =============================================================================


def compute_cn_from_gr(
    gr: np.ndarray,
    r: np.ndarray,
    n_neighbor: int,
    volume: float,
    r_cut: float,
    dr: float,
) -> float:
    """
    Integrate g(r) to coordination number: CN = 4pi rho int_0^r_cut g(r) r^2 dr.

    Parameters
    ----------
    gr : np.ndarray
        Radial distribution function values.
    r : np.ndarray
        Radial distance values in angstrom.
    n_neighbor : int
        Number of neighbor atoms of the relevant species in the simulation box.
    volume : float
        Volume of the simulation box in angstrom^3.
    r_cut : float
        Integration cutoff distance in angstrom.
    dr : float
        Bin width in angstrom.

    Returns
    -------
    float
        Coordination number obtained by integrating g(r) up to ``r_cut``.
    """
    rho = n_neighbor / volume
    mask = r <= r_cut
    return float(np.sum(4 * np.pi * rho * gr[mask] * r[mask] ** 2 * dr))


def load_rdf_results() -> dict[str, dict]:
    """
    Load rdf.json and rdf.npz for each registered model that has calc output.

    Returns
    -------
    dict[str, dict]
        Mapping from registry model name to a dict containing coordination
        numbers, errors, radial distances, g(r) arrays, and frame count.
    """
    results = {}
    for model in MODELS:
        json_path = CALC_PATH / model / "rdf.json"
        npz_path = CALC_PATH / model / "rdf.npz"
        if not json_path.exists() or not npz_path.exists():
            continue

        with open(json_path) as f:
            meta = json.load(f)

        data = np.load(npz_path)
        r = data["r"]

        n_ow = meta["n_O_water"]
        n_of = meta["n_O_TFSI"]

        cn_w = compute_cn_from_gr(data["gr_LiO_water"], r, n_ow, V_BOX, R_CUT, DR)
        cn_f = compute_cn_from_gr(data["gr_LiO_TFSI"], r, n_of, V_BOX, R_CUT, DR)

        results[model] = {
            "cn_water": cn_w,
            "cn_tfsi": cn_f,
            "cn_total": cn_w + cn_f,
            "err_water": abs(cn_w - CN_EXP_WATER),
            "err_tfsi": abs(cn_f - CN_EXP_TFSI),
            "r": r,
            "gr_water": data["gr_LiO_water"],
            "gr_tfsi": data["gr_LiO_TFSI"],
            "gr_total": data["gr_LiO_total"],
            "n_frames": meta["n_frames_used"],
        }
    return results


def build_cn_bar_chart(data: dict[str, dict]) -> go.Figure:
    """
    Bar chart of CN_water and CN_TFSI per model with experimental reference.

    Parameters
    ----------
    data : dict[str, dict]
        Per-model RDF results as returned by :func:`load_rdf_results`.

    Returns
    -------
    go.Figure
        Plotly figure with grouped bars and experimental reference lines.
    """
    fig = go.Figure()

    x_models = [m for m in MODELS if m in data]
    cn_water_vals = [data[m]["cn_water"] for m in x_models]
    cn_tfsi_vals = [data[m]["cn_tfsi"] for m in x_models]

    fig.add_trace(
        go.Bar(
            name="Li-O<sub>water</sub>",
            x=x_models,
            y=cn_water_vals,
            marker_color="steelblue",
        )
    )
    fig.add_trace(
        go.Bar(
            name="Li-O<sub>TFSI</sub>",
            x=x_models,
            y=cn_tfsi_vals,
            marker_color="coral",
        )
    )

    fig.add_hline(
        y=CN_EXP_WATER,
        line_dash="dash",
        line_color="steelblue",
        annotation_text=f"Exp. O<sub>water</sub> ({CN_EXP_WATER})",
        annotation_position="top right",
    )
    fig.add_hline(
        y=CN_EXP_TFSI,
        line_dash="dash",
        line_color="coral",
        annotation_text=f"Exp. O<sub>TFSI</sub> ({CN_EXP_TFSI})",
        annotation_position="bottom right",
    )

    fig.update_layout(
        title="Li⁺ Coordination Numbers (LiTFSI/H₂O, 21 m)",
        yaxis_title="Coordination Number",
        barmode="group",
        legend={"x": 0.01, "y": 0.99},
    )
    return fig


def build_gr_plot(data: dict[str, dict]) -> go.Figure:
    """
    G(r) plot: Li-O_water and Li-O_TFSI for all models.

    Parameters
    ----------
    data : dict[str, dict]
        Per-model RDF results as returned by :func:`load_rdf_results`.

    Returns
    -------
    go.Figure
        Plotly figure with g(r) curves and integration cutoff line.
    """
    fig = go.Figure()

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
    ]

    for i, model in enumerate(MODELS):
        if model not in data:
            continue
        d = data[model]
        c = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=d["r"],
                y=d["gr_water"],
                mode="lines",
                name=f"{model} O<sub>w</sub>",
                line={"color": c, "dash": "solid"},
                legendgroup=model,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d["r"],
                y=d["gr_tfsi"],
                mode="lines",
                name=f"{model} O<sub>TFSI</sub>",
                line={"color": c, "dash": "dot"},
                legendgroup=model,
            )
        )

    fig.add_vline(
        x=R_CUT,
        line_dash="dash",
        line_color="gray",
        annotation_text=f"r_cut={R_CUT} Å",
        annotation_position="top right",
    )

    fig.update_layout(
        title="Li-O Radial Distribution Functions (LiTFSI/H₂O, 21 m)",
        xaxis_title="r / Å",
        yaxis_title="g(r)",
        xaxis_range=[1.0, 6.0],
    )
    return fig


# =============================================================================
# X-ray S(q) helpers and figures
# =============================================================================


def load_experimental_sq() -> tuple[np.ndarray, np.ndarray]:
    """
    Load experimental SAXS S(q) data.

    Returns
    -------
    q_exp : np.ndarray
        Scattering vector values in inverse angstroms.
    sq_exp : np.ndarray
        Experimental structure factor values.
    """
    data = np.loadtxt(EXP_SAXS_PATH)
    return data[:, 0], data[:, 1]


def load_computed_sq(model: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load computed S(q) for a model.

    Parameters
    ----------
    model : str
        Registry name of the MLIP model.

    Returns
    -------
    tuple of (np.ndarray, np.ndarray) or None
        Scattering vector and structure factor arrays, or None if the
        output file does not exist.
    """
    json_path = CALC_PATH / model / "xray_sq.json"
    if not json_path.exists():
        return None
    with open(json_path) as f:
        d = json.load(f)
    return np.array(d["q"]), np.array(d["Sq"])


def compute_r_factor(
    q_exp: np.ndarray,
    sq_exp: np.ndarray,
    q_calc: np.ndarray,
    sq_calc: np.ndarray,
) -> float:
    """
    Compute R-factor: sum|S_exp - S_calc| / sum|S_exp|.

    Parameters
    ----------
    q_exp : np.ndarray
        Experimental scattering vector values.
    sq_exp : np.ndarray
        Experimental structure factor values.
    q_calc : np.ndarray
        Calculated scattering vector values.
    sq_calc : np.ndarray
        Calculated structure factor values.

    Returns
    -------
    float
        R-factor value, or NaN if no overlapping valid data exists.
    """
    valid_calc = ~np.isnan(sq_calc)
    q_min_calc = float(q_calc[valid_calc].min())
    q_max_exp = float(q_exp.max())

    f = interp1d(
        q_calc[valid_calc],
        sq_calc[valid_calc],
        kind="linear",
        bounds_error=False,
        fill_value=np.nan,
    )

    mask = (q_exp >= q_min_calc) & (q_exp <= q_max_exp)
    sq_interp = f(q_exp[mask])
    sq_e = sq_exp[mask]

    ok = ~np.isnan(sq_interp)
    if not ok.any():
        return float("nan")

    return float(np.sum(np.abs(sq_e[ok] - sq_interp[ok])) / np.sum(np.abs(sq_e[ok])))


def find_first_peak(
    q: np.ndarray, sq: np.ndarray, q_min: float = 0.5, q_max: float = 2.0
) -> float:
    """
    Find position of first peak in S(q).

    Parameters
    ----------
    q : np.ndarray
        Scattering vector values.
    sq : np.ndarray
        Structure factor values.
    q_min : float, optional
        Lower bound of the search window in inverse angstroms (default 0.5).
    q_max : float, optional
        Upper bound of the search window in inverse angstroms (default 2.0).

    Returns
    -------
    float
        Position of the first peak, or NaN if fewer than 3 valid points
        exist in the search window.
    """
    mask = (q >= q_min) & (q <= q_max) & ~np.isnan(sq)
    if mask.sum() < 3:
        return float("nan")
    q_sub = q[mask]
    sq_sub = sq[mask]
    return float(q_sub[np.argmax(sq_sub)])


def build_sq_comparison_plot(
    q_exp: np.ndarray,
    sq_exp: np.ndarray,
    model_data: dict[str, tuple[np.ndarray, np.ndarray]],
) -> go.Figure:
    """
    Build S(q) overlay plot comparing computed and experimental data.

    Parameters
    ----------
    q_exp : np.ndarray
        Experimental scattering vector values.
    sq_exp : np.ndarray
        Experimental structure factor values.
    model_data : dict of str to tuple of (np.ndarray, np.ndarray)
        Mapping from model name to (q, S(q)) arrays.

    Returns
    -------
    go.Figure
        Plotly figure with experimental and computed S(q) traces.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=q_exp.tolist(),
            y=sq_exp.tolist(),
            mode="lines",
            name="Exp. (Zhang et al. 2021)",
            line={"color": "black", "width": 2},
        )
    )

    for model, (q, sq) in model_data.items():
        fig.add_trace(
            go.Scatter(
                x=q.tolist(),
                y=sq.tolist(),
                mode="lines",
                name=model,
                opacity=0.8,
            )
        )

    fig.update_layout(
        title="X-ray Structure Factor S(q) — LiTFSI/H2O (21 m)",
        xaxis_title="q / Å⁻¹",
        yaxis_title="S(q) (Faber-Ziman)",
        legend={"x": 0.60, "y": 0.98},
    )
    return fig


# =============================================================================
# Pytest fixtures
# =============================================================================


@pytest.fixture
def density_results() -> dict[str, dict]:
    """
    Load density results and write density plot JSONs.

    Returns
    -------
    dict[str, dict]
        Per-model density results keyed by registry model name.
    """
    data = load_density_results()
    if not data:
        pytest.skip("No density data found")

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    fig_bar = build_density_bar_chart(data)
    with open(OUT_PATH / "figure_density_bar.json", "w") as f:
        f.write(fig_bar.to_json())

    fig_ts = build_density_timeseries(data)
    with open(OUT_PATH / "figure_density_timeseries.json", "w") as f:
        f.write(fig_ts.to_json())

    return data


@pytest.fixture
def rdf_results() -> dict[str, dict]:
    """
    Load RDF results and write RDF plot JSONs.

    Returns
    -------
    dict[str, dict]
        Per-model RDF results keyed by registry model name.
    """
    data = load_rdf_results()
    if not data:
        pytest.skip("No RDF data found")

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    fig_cn = build_cn_bar_chart(data)
    with open(OUT_PATH / "figure_cn_bar.json", "w") as f:
        f.write(fig_cn.to_json())

    fig_gr = build_gr_plot(data)
    with open(OUT_PATH / "figure_gr.json", "w") as f:
        f.write(fig_gr.to_json())

    return data


@pytest.fixture
def xray_sf_results() -> dict[str, dict]:
    """
    Compute per-model R-factor and peak-position errors and write S(q) plot.

    Returns
    -------
    dict[str, dict]
        Per-model results keyed by registry model name, containing
        ``r_factor``, ``peak_calc``, ``peak_exp``, ``peak_position_error``.
    """
    q_exp, sq_exp = load_experimental_sq()
    peak_exp = find_first_peak(q_exp, sq_exp)

    model_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    results: dict[str, dict] = {}

    for model in MODELS:
        data = load_computed_sq(model)
        if data is None:
            continue
        q_calc, sq_calc = data
        model_data[model] = (q_calc, sq_calc)

        r_factor = compute_r_factor(q_exp, sq_exp, q_calc, sq_calc)
        peak_calc = find_first_peak(q_calc, sq_calc)
        peak_error = (
            abs(peak_calc - peak_exp) if not np.isnan(peak_calc) else float("nan")
        )

        results[model] = {
            "r_factor": r_factor,
            "peak_calc": peak_calc,
            "peak_exp": peak_exp,
            "peak_position_error": peak_error,
        }

    if not results:
        pytest.skip("No S(q) data found")

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    fig = build_sq_comparison_plot(q_exp, sq_exp, model_data)
    with open(OUT_PATH / "figure_xray_sq_comparison.json", "w") as f:
        f.write(fig.to_json())

    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "litfsi_h2o_21m_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    density_results: dict[str, dict],
    rdf_results: dict[str, dict],
    xray_sf_results: dict[str, dict],
) -> dict[str, dict]:
    """
    Build per-metric dicts consumed by the ``build_table`` decorator.

    Parameters
    ----------
    density_results
        Per-model density results.
    rdf_results
        Per-model RDF / coordination-number results.
    xray_sf_results
        Per-model R-factor and peak-position results.

    Returns
    -------
    dict[str, dict]
        Metric names mapped to ``{model_name: value}`` dicts. The metric
        order matches ``metrics.yml`` (Density Error, Density Error (%),
        CN Li-O_water Error, CN Li-O_TFSI Error, S(q) R-factor,
        First Peak Position Error).
    """
    return {
        "Density Error": {
            model: d["rho_abs_error"] for model, d in density_results.items()
        },
        "Density Error (%)": {
            model: abs(d["rho_error_pct"]) for model, d in density_results.items()
        },
        "CN Li-O_water Error": {
            model: d["err_water"] for model, d in rdf_results.items()
        },
        "CN Li-O_TFSI Error": {
            model: d["err_tfsi"] for model, d in rdf_results.items()
        },
        "S(q) R-factor": {
            model: r["r_factor"] for model, r in xray_sf_results.items()
        },
        "First Peak Position Error": {
            model: r["peak_position_error"] for model, r in xray_sf_results.items()
        },
    }


def test_litfsi_h2o_21m(metrics: dict[str, dict]) -> None:
    """
    Run the consolidated WiSE LiTFSI/H2O benchmark.

    Parameters
    ----------
    metrics
        Per-metric values for all models.
    """
    assert metrics, "No models produced data for the WiSE LiTFSI/H2O benchmark"
