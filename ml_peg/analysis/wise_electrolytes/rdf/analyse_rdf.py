"""Analyse Li-O RDF benchmark for LiTFSI/H2O electrolyte."""
# Reads rdf.json and rdf.npz files from calc outputs and generates
# scoring tables and g(r) plots for the ml-peg dashboard.
#
# Reference coordination numbers (Watanabe et al., J. Phys. Chem. B 125, 7477, 2021):
#   CN(Li-O_water) = 2.0  }  ~18.5 m LiTFSI/H2O,
#   CN(Li-O_TFSI)  = 2.0  }  neutron diffraction, 6Li/7Li + H/D substitution
#
# Integration cutoff: 2.83 Å (first minimum of Li-O_total g(r) from r2SCAN AIMD).
# Trajectories: NVT 50 ps (dt=0.1 ps/frame, 501 frames, equilibrated 50-100 ps window).
# Box length: 27.4938 Å (cubic, p64_w170 cell: 64 LiTFSI + 170 H2O = 1534 atoms).

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.utils.utils import load_metrics_config

# --- Paths -------------------------------------------------------------------

CALCS_ROOT = Path(__file__).resolve().parents[3] / "calcs"
CALC_PATH = CALCS_ROOT / "wise_electrolytes" / "rdf" / "outputs"
APP_ROOT = Path(__file__).resolve().parents[3] / "app"
OUT_PATH = APP_ROOT / "data" / "wise_electrolytes" / "rdf"

MODELS = [
    "matpes-r2scan",
    "mace-mpa-0-medium",
    "mace-omat-0-medium",
    "mace-mp-0b3",
    "mace-mh-1-omat",
    "mace-mh-1-omol",
]

# --- Physical constants and reference values ---------------------------------

# Box geometry (same for all models: p64_w170 NVT cell)
L_BOX = 27.4938  # Å (cubic NVT cell at experimental density)
V_BOX = L_BOX**3  # Å³

# Integration cutoff: first minimum of Li-O_total g(r)
R_CUT = 2.83  # Å
DR = 0.02  # Å (bin width used in calc_rdf.py)

# Experimental reference: Watanabe et al., JPCB 2021, ~18.5 m LiTFSI/H2O
CN_EXP_WATER = 2.0
CN_EXP_TFSI = 2.0

# Map benchmark model names → ml-peg registry names (for app summary table)
MODEL_NAME_MAP = {
    "matpes-r2scan": "mace-matpes-r2scan",
    "mace-mpa-0-medium": "mace-mpa-0",
    "mace-omat-0-medium": "mace-omat-0",
}

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


# --- Helpers -----------------------------------------------------------------


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
    Load rdf.json and rdf.npz for each model, recompute CN at r_cut=2.83 A.

    Returns
    -------
    dict[str, dict]
        Mapping from model name to a dict containing coordination numbers,
        errors, radial distances, g(r) arrays, and frame count.
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


def normalize_metric(value: float, good: float, bad: float) -> float:
    """
    Map value linearly: good to 1.0, bad to 0.0, clipped to [0, 1].

    Parameters
    ----------
    value : float
        The metric value to normalize.
    good : float
        Threshold value that maps to a score of 1.0.
    bad : float
        Threshold value that maps to a score of 0.0.

    Returns
    -------
    float
        Normalized score clipped to the range [0, 1].
    """
    if good == bad:
        return 1.0 if value == good else 0.0
    t = (value - bad) / (good - bad)
    return max(0.0, min(1.0, float(t)))


# --- Plot builders -----------------------------------------------------------


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

    # Experimental reference lines
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

    # Mark integration cutoff
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


# --- Metrics table -----------------------------------------------------------


def build_metrics_table(data: dict[str, dict]) -> dict:
    """
    Build metrics table JSON for ml-peg app.

    Parameters
    ----------
    data : dict[str, dict]
        Per-model RDF results as returned by :func:`load_rdf_results`.

    Returns
    -------
    dict
        Table payload with keys ``data``, ``columns``, ``tooltip_header``,
        ``thresholds``, and ``reference``.
    """
    rows = []
    for model in MODELS:
        if model not in data:
            continue
        d = data[model]

        score_w = normalize_metric(
            d["err_water"],
            DEFAULT_THRESHOLDS["CN Li-O_water Error"]["good"],
            DEFAULT_THRESHOLDS["CN Li-O_water Error"]["bad"],
        )
        score_f = normalize_metric(
            d["err_tfsi"],
            DEFAULT_THRESHOLDS["CN Li-O_TFSI Error"]["good"],
            DEFAULT_THRESHOLDS["CN Li-O_TFSI Error"]["bad"],
        )

        w_w = DEFAULT_WEIGHTS.get("CN Li-O_water Error", 1.0)
        w_f = DEFAULT_WEIGHTS.get("CN Li-O_TFSI Error", 1.0)
        total_w = w_w + w_f
        overall = (score_w * w_w + score_f * w_f) / total_w if total_w > 0 else 0.0

        rows.append(
            {
                "MLIP": model,
                "id": MODEL_NAME_MAP.get(model, model),
                "CN Li-O_water": round(d["cn_water"], 3),
                "CN Li-O_TFSI": round(d["cn_tfsi"], 3),
                "CN Li-O_water Error": round(d["err_water"], 3),
                "CN Li-O_TFSI Error": round(d["err_tfsi"], 3),
                "Score": round(overall, 3),
            }
        )

    rows.sort(key=lambda r: r["Score"], reverse=True)

    columns = [
        {"name": c, "id": c}
        for c in [
            "MLIP",
            "CN Li-O_water Error",
            "CN Li-O_TFSI Error",
            "Score",
        ]
    ]

    return {
        "data": rows,
        "columns": columns,
        "tooltip_header": DEFAULT_TOOLTIPS,
        "thresholds": {k: dict(v) for k, v in DEFAULT_THRESHOLDS.items()},
        "weights": dict(DEFAULT_WEIGHTS),
        "model_name_map": MODEL_NAME_MAP,
        "reference": {
            "CN_water": CN_EXP_WATER,
            "CN_TFSI": CN_EXP_TFSI,
            "citation": "Watanabe et al., J. Phys. Chem. B 125, 7477 (2021)",
            "concentration": "~18.5 m LiTFSI/H2O",
            "method": "neutron diffraction with 6Li/7Li + H/D isotopic substitution",
            "r_cut_angstrom": R_CUT,
        },
    }


# --- Pytest interface --------------------------------------------------------


@pytest.fixture
def rdf_analysis():
    """
    Run full RDF analysis: table + plots.

    Returns
    -------
    dict
        Metrics table payload written to ``rdf_metrics_table.json``.
    """
    data = load_rdf_results()
    if not data:
        pytest.skip("No RDF data found")

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    table = build_metrics_table(data)
    with open(OUT_PATH / "rdf_metrics_table.json", "w") as f:
        json.dump(table, f, indent=2)

    fig_cn = build_cn_bar_chart(data)
    with open(OUT_PATH / "figure_cn_bar.json", "w") as f:
        f.write(fig_cn.to_json())

    fig_gr = build_gr_plot(data)
    with open(OUT_PATH / "figure_gr.json", "w") as f:
        f.write(fig_gr.to_json())

    return table


def test_rdf(rdf_analysis) -> None:
    """
    Run RDF benchmark -- generates table and plots.

    Parameters
    ----------
    rdf_analysis : dict
        Metrics table payload provided by the ``rdf_analysis`` fixture.
    """
    table = rdf_analysis
    assert len(table["data"]) > 0, "No models produced RDF data"
