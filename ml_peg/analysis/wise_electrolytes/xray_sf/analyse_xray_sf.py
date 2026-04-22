"""Analyse X-ray structure factor benchmark for LiTFSI/H2O electrolyte."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import pytest
from scipy.interpolate import interp1d

from ml_peg.analysis.utils.utils import load_metrics_config

# --- Paths -------------------------------------------------------------------

CALCS_ROOT = Path(__file__).resolve().parents[3] / "calcs"
CALC_PATH = CALCS_ROOT / "wise_electrolytes" / "xray_sf" / "outputs"
APP_ROOT = Path(__file__).resolve().parents[3] / "app"
OUT_PATH = APP_ROOT / "data" / "wise_electrolytes" / "xray_sf"

# Experimental data bundled with the benchmark
EXP_PATH = Path(__file__).resolve().parents[1] / "data" / "saxs_maginn.txt"

MODELS = [
    "matpes-r2scan",
    "mace-mpa-0-medium",
    "mace-omat-0-medium",
    "mace-mp-0b3",
    "mace-mh-1-omat",
    "mace-mh-1-omol",
]

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
    data = np.loadtxt(EXP_PATH)
    return data[:, 0], data[:, 1]


def load_computed_sq(model: str) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load computed S(q) for a model.

    Parameters
    ----------
    model : str
        Name of the MLIP model.

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

    The q range used is the overlap between experimental and calculated data:
    q_min is the minimum q of the calculated S(q), set by the simulation cell
    size (2*pi/L), and q_max is the maximum q of the experimental data.

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


def normalize_metric(value: float, good: float, bad: float) -> float:
    """
    Normalize a metric value to [0, 1] where 0 is good and 1 is bad.

    Parameters
    ----------
    value : float
        Raw metric value.
    good : float
        Threshold corresponding to the best achievable score (0).
    bad : float
        Threshold corresponding to the worst score (1).

    Returns
    -------
    float
        Normalized score clipped to [0, 1].
    """
    if bad == good:
        return 0.0 if value <= good else 1.0
    score = (value - good) / (bad - good)
    return max(0.0, min(1.0, score))


# --- Plot builders -----------------------------------------------------------


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


def build_metrics_table(results: dict[str, dict]) -> dict:
    """
    Build metrics table as a JSON-serialisable dictionary.

    Parameters
    ----------
    results : dict of str to dict
        Per-model results containing ``r_factor``, ``peak_position_error``,
        ``peak_calc``, and ``peak_exp`` entries.

    Returns
    -------
    dict
        Dictionary with ``data`` (list of row dicts), ``columns``,
        ``tooltip_header``, and ``thresholds`` keys.
    """
    rows = []
    for model in MODELS:
        if model not in results:
            continue
        r = results[model]
        r_factor = r["r_factor"]
        peak_err = r["peak_position_error"]

        score_r = normalize_metric(
            r_factor,
            DEFAULT_THRESHOLDS["S(q) R-factor"]["good"],
            DEFAULT_THRESHOLDS["S(q) R-factor"]["bad"],
        )
        score_p = normalize_metric(
            peak_err,
            DEFAULT_THRESHOLDS["First Peak Position Error"]["good"],
            DEFAULT_THRESHOLDS["First Peak Position Error"]["bad"],
        )

        w_r = DEFAULT_WEIGHTS.get("S(q) R-factor", 1.0)
        w_p = DEFAULT_WEIGHTS.get("First Peak Position Error", 1.0)
        total_w = w_r + w_p
        overall = (
            1.0 - (score_r * w_r + score_p * w_p) / total_w if total_w > 0 else 0.0
        )

        rows.append(
            {
                "MLIP": model,
                "id": MODEL_NAME_MAP.get(model, model),
                "S(q) R-factor": round(r_factor, 4),
                "First Peak Position Error": round(peak_err, 4),
                "First Peak (calc)": round(r["peak_calc"], 2),
                "First Peak (exp)": round(r["peak_exp"], 2),
                "Score": round(overall, 3),
            }
        )

    rows.sort(key=lambda r: r["Score"], reverse=True)

    columns = [
        {"name": c, "id": c}
        for c in [
            "MLIP",
            "S(q) R-factor",
            "First Peak Position Error",
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
    }


# --- Pytest interface --------------------------------------------------------


@pytest.fixture
def xray_sf_analysis():
    """
    Run full X-ray S(q) analysis and write outputs.

    Returns
    -------
    dict
        Metrics table dictionary produced by ``build_metrics_table``.
    """
    q_exp, sq_exp = load_experimental_sq()
    peak_exp = find_first_peak(q_exp, sq_exp)

    model_data = {}
    results = {}

    for model in MODELS:
        data = load_computed_sq(model)
        if data is None:
            continue
        q_calc, sq_calc = data
        model_data[model] = (q_calc, sq_calc)

        r_factor = compute_r_factor(
            q_exp, sq_exp, q_calc, sq_calc
        )  # range: [q_min_calc, q_max_exp]
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

    table = build_metrics_table(results)
    with open(OUT_PATH / "xray_sf_metrics_table.json", "w") as f:
        json.dump(table, f, indent=2)

    fig = build_sq_comparison_plot(q_exp, sq_exp, model_data)
    with open(OUT_PATH / "figure_xray_sq_comparison.json", "w") as f:
        f.write(fig.to_json())

    with open(OUT_PATH / "xray_sf_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return table


def test_xray_sf(xray_sf_analysis) -> None:
    """
    Run X-ray structure factor benchmark.

    Parameters
    ----------
    xray_sf_analysis : dict
        Metrics table fixture providing analysis results.
    """
    table = xray_sf_analysis
    assert len(table["data"]) > 0, "No models produced S(q) data"
