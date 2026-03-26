"""Analyse equation of state benchmark."""

from __future__ import annotations

from pathlib import Path

from ase.eos import EquationOfState, birchmurnaghan
import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "equation_of_state" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "equation_of_state"

DATA_PATH = Path(__file__).parent / "../../../../inputs/bulk_crystal/equation_of_state/"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

ELEMENTS = [f.name.split("_")[0] for f in DATA_PATH.glob("*DFT*")]


def _fit_bm_clean(volumes: np.ndarray, energies: np.ndarray) -> tuple | None:
    """
    Fit a Birch-Murnaghan EOS, ignoring non-finite points.

    Parameters
    ----------
    volumes
        Per-atom volumes in Å³.
    energies
        Per-atom energies in eV.

    Returns
    -------
    tuple or None
        ``eos.eos_parameters`` = ``(E0, B0, BP, V0)`` on success, or ``None``
        if fewer than 4 finite points exist or fitting fails.
    """
    mask = np.isfinite(volumes) & np.isfinite(energies)
    v = np.asarray(volumes)[mask]
    e = np.asarray(energies)[mask]
    if v.size < 4:
        return None
    try:
        eos = EquationOfState(v, e, eos="birchmurnaghan")
        eos.fit()
        return eos.eos_parameters
    except Exception:
        return None


def calc_delta(
    data_f: dict[str, float],
    data_w: dict[str, float],
    useasymm: bool,
    vi: float,
    vf: float,
) -> tuple[float, float, float]:
    """
    Calculate the Delta metric (meV/atom) between two EOS curves.

    Expects B0 in ASE-native units (eV/A^3), as returned directly by
    ``EquationOfState.eos_parameters``. No GPa conversion is applied.

    Adapted from ``calcDelta.py``, supplementary material from:
    K. Lejaeghere, V. Van Speybroeck, G. Van Oost, and S. Cottenier:
    "Error estimates for solid-state density-functional theory predictions:
    an overview by means of the ground-state elemental crystals"
    Crit. Rev. Solid State (2014). Open access: http://arxiv.org/abs/1204.2733

    Parameters
    ----------
    data_f
        Reference EOS parameters: ``{"V0": float, "B0": float, "BP": float}``.
        B0 in eV/A^3 (ASE units).
    data_w
        Model EOS parameters with the same keys.
    useasymm
        If ``True``, normalise Delta1 by the reference V0/B0 only; if
        ``False`` use the average of reference and model values.
    vi
        Lower volume integration bound (A^3/atom).
    vf
        Upper volume integration bound (A^3/atom).

    Returns
    -------
    tuple[float, float, float]
        ``(delta, deltarel, delta1)`` where ``delta`` is in meV/atom.
    """
    v0w = data_w["V0"]
    b0w = data_w["B0"]
    b1w = data_w["BP"]

    v0f = data_f["V0"]
    b0f = data_f["B0"]
    b1f = data_f["BP"]

    vref = 30.0
    bref = 100.0  # eV/A^3, matching ASE-native units

    a3f = 9.0 * v0f**3.0 * b0f / 16.0 * (b1f - 4.0)
    a2f = 9.0 * v0f ** (7.0 / 3.0) * b0f / 16.0 * (14.0 - 3.0 * b1f)
    a1f = 9.0 * v0f ** (5.0 / 3.0) * b0f / 16.0 * (3.0 * b1f - 16.0)
    a0f = 9.0 * v0f * b0f / 16.0 * (6.0 - b1f)

    a3w = 9.0 * v0w**3.0 * b0w / 16.0 * (b1w - 4.0)
    a2w = 9.0 * v0w ** (7.0 / 3.0) * b0w / 16.0 * (14.0 - 3.0 * b1w)
    a1w = 9.0 * v0w ** (5.0 / 3.0) * b0w / 16.0 * (3.0 * b1w - 16.0)
    a0w = 9.0 * v0w * b0w / 16.0 * (6.0 - b1w)

    x = [0.0] * 7
    x[0] = (a0f - a0w) ** 2
    x[1] = 6.0 * (a1f - a1w) * (a0f - a0w)
    x[2] = -3.0 * (2.0 * (a2f - a2w) * (a0f - a0w) + (a1f - a1w) ** 2.0)
    x[3] = -2.0 * (a3f - a3w) * (a0f - a0w) - 2.0 * (a2f - a2w) * (a1f - a1w)
    x[4] = -3.0 / 5.0 * (2.0 * (a3f - a3w) * (a1f - a1w) + (a2f - a2w) ** 2.0)
    x[5] = -6.0 / 7.0 * (a3f - a3w) * (a2f - a2w)
    x[6] = -1.0 / 3.0 * (a3f - a3w) ** 2.0

    y = [0.0] * 7
    y[0] = (a0f + a0w) ** 2 / 4.0
    y[1] = 3.0 * (a1f + a1w) * (a0f + a0w) / 2.0
    y[2] = -3.0 * (2.0 * (a2f + a2w) * (a0f + a0w) + (a1f + a1w) ** 2.0) / 4.0
    y[3] = -(a3f + a3w) * (a0f + a0w) / 2.0 - (a2f + a2w) * (a1f + a1w) / 2.0
    y[4] = -3.0 / 20.0 * (2.0 * (a3f + a3w) * (a1f + a1w) + (a2f + a2w) ** 2.0)
    y[5] = -3.0 / 14.0 * (a3f + a3w) * (a2f + a2w)
    y[6] = -1.0 / 12.0 * (a3f + a3w) ** 2.0

    fi = 0.0
    ff = 0.0
    gi = 0.0
    gf = 0.0
    for n in range(7):
        fi += x[n] * vi ** (-(2.0 * n - 3.0) / 3.0)
        ff += x[n] * vf ** (-(2.0 * n - 3.0) / 3.0)
        gi += y[n] * vi ** (-(2.0 * n - 3.0) / 3.0)
        gf += y[n] * vf ** (-(2.0 * n - 3.0) / 3.0)

    delta = 1000.0 * np.sqrt((ff - fi) / (vf - vi))
    deltarel = 100.0 * np.sqrt((ff - fi) / (gf - gi))
    if useasymm:
        delta1 = delta / v0w / b0w * vref * bref
    else:
        delta1 = delta / (v0w + v0f) / (b0w + b0f) * 4.0 * vref * bref

    return delta, deltarel, delta1


def _phase_metrics_from_eos_mev(
    dft_data: pd.DataFrame, model_data: pd.DataFrame
) -> tuple[float, float]:
    """
    Compute phase-stability metrics relative to the reference phase.

    Parameters
    ----------
    dft_data
        DFT reference DataFrame with columns ``V/atom_{phase}`` and
        ``Delta_{phase}_E`` for each phase.
    model_data
        Model results DataFrame with columns ``V/atom`` and ``{phase}_E``
        for each phase.

    Returns
    -------
    tuple[float, float]
        ``(PhaseDiffEOS_MAE_meV, CorrectStability_pct)``. Returns
        ``(nan, nan)`` if any BM fit fails.
    """
    phases = [
        col.split("_")[1]
        for col in dft_data.columns
        if col.startswith("Delta_") and col.endswith("_E")
    ]

    dft_fits = [
        _fit_bm_clean(
            dft_data[f"V/atom_{phase}"].values,
            dft_data[f"Delta_{phase}_E"].values,
        )
        for phase in phases
    ]
    model_fits = [
        _fit_bm_clean(model_data["V/atom"].values, model_data[f"{phase}_E"].values)
        for phase in phases
    ]

    if any(fit is None for fit in dft_fits + model_fits):
        return np.nan, np.nan

    ref_volumes = dft_data[f"V/atom_{phases[0]}"].values
    ref_volumes = ref_volumes[np.isfinite(ref_volumes)]
    v_grid = np.linspace(ref_volumes.min(), ref_volumes.max(), 80)

    dft_deltas = np.vstack(
        [
            birchmurnaghan(v_grid, *dft_fit) - birchmurnaghan(v_grid, *dft_fits[0])
            for dft_fit in dft_fits[1:]
        ]
    )
    model_deltas = np.vstack(
        [
            birchmurnaghan(v_grid, *model_fit) - birchmurnaghan(v_grid, *model_fits[0])
            for model_fit in model_fits[1:]
        ]
    )

    mae_ev = float(np.mean(np.abs(dft_deltas - model_deltas)))
    correct_stability_pct = 100.0 * float(np.mean(np.all(model_deltas > 0, axis=0)))

    return 1000.0 * mae_ev, correct_stability_pct


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def eos_stats() -> dict[tuple[str, str], dict[str, float]]:
    """
    Compute all three metrics for every model-element pair.

    Returns
    -------
    dict[tuple[str, str], dict[str, float]]
        Mapping of ``(model_name, element)`` to ``{"Delta",
        "PhaseDiffEOS_MAE_meV", "CorrectStability_pct"}``.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, str], dict[str, float]] = {}

    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            continue

        for element in ELEMENTS:
            model_csv = model_dir / f"{element}_eos_results.csv"
            dft_csv = DATA_PATH / f"{element}_eos_DFT.csv"
            if not model_csv.exists() or not dft_csv.exists():
                continue

            model_data = pd.read_csv(model_csv)
            dft_data = pd.read_csv(dft_csv, comment="#")

            phases = [
                col.split("_")[1]
                for col in dft_data.columns
                if col.startswith("Delta_") and col.endswith("_E")
            ]
            # BM fit for reference phase (Delta metric)
            ref_phase = phases[0]  # Assuming the first phase is the reference
            ref_params = _fit_bm_clean(
                model_data["V/atom"].values, model_data[f"{ref_phase}_E"].values
            )
            dft_ref_params = _fit_bm_clean(
                dft_data[f"V/atom_{ref_phase}"].values,
                dft_data[f"Delta_{ref_phase}_E"].values,
            )

            if ref_params is not None and dft_ref_params is not None:
                # eos_parameters: (E0, B0, BP, V0)
                model_bm = {
                    "V0": ref_params[-1],
                    "B0": ref_params[-3],
                    "BP": ref_params[-2],
                }
                dft_bm = {
                    "V0": dft_ref_params[-1],
                    "B0": dft_ref_params[-3],
                    "BP": dft_ref_params[-2],
                }
                volumes = model_data["V/atom"]
                delta, _, _ = calc_delta(
                    dft_bm,
                    model_bm,
                    useasymm=False,
                    vi=float(volumes.iloc[0]),
                    vf=float(volumes.iloc[-1]),
                )
            else:
                delta = np.nan

            phase_diff_mae, correct_stability = _phase_metrics_from_eos_mev(
                dft_data, model_data
            )

            results[(model_name, element)] = {
                "Δ": delta,
                "Phase energy": phase_diff_mae,
                "Phase stability": correct_stability,
            }

    return results


@pytest.fixture
def delta(
    eos_stats: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    """
    Mean Delta (meV/atom) across elements for each model.

    Parameters
    ----------
    eos_stats
        Per-(model, element) metric values.

    Returns
    -------
    dict[str, float]
        Mean Delta per model.
    """
    results: dict[str, float] = {}
    for model_name in MODELS:
        values = [
            eos_stats[(model_name, el)]["Δ"]
            for el in ELEMENTS
            if (model_name, el) in eos_stats
        ]
        results[model_name] = float(np.nanmean(values)) if values else None
    return results


@pytest.fixture
def phase_diff_eos_mae(
    eos_stats: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    """
    Mean PhaseDiffEOS MAE (meV/atom) across elements for each model.

    Parameters
    ----------
    eos_stats
        Per-(model, element) metric values.

    Returns
    -------
    dict[str, float]
        Mean PhaseDiffEOS_MAE_meV per model.
    """
    results: dict[str, float] = {}
    for model_name in MODELS:
        values = [
            eos_stats[(model_name, el)]["Phase energy"]
            for el in ELEMENTS
            if (model_name, el) in eos_stats
        ]
        results[model_name] = float(np.nanmean(values)) if values else None
    return results


@pytest.fixture
def correct_stability(
    eos_stats: dict[tuple[str, str], dict[str, float]],
) -> dict[str, float]:
    """
    Mean CorrectStability (%) across elements for each model.

    Parameters
    ----------
    eos_stats
        Per-(model, element) metric values.

    Returns
    -------
    dict[str, float]
        Mean Phase stability per model.
    """
    results: dict[str, float] = {}
    for model_name in MODELS:
        values = [
            eos_stats[(model_name, el)]["Phase stability"]
            for el in ELEMENTS
            if (model_name, el) in eos_stats
        ]
        results[model_name] = float(np.nanmean(values)) if values else None
    return results


@pytest.fixture
@build_table(
    filename=OUT_PATH / "eos_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    delta: dict[str, float],
    phase_diff_eos_mae: dict[str, float],
    correct_stability: dict[str, float],
) -> dict[str, dict]:
    """
    All EOS benchmark metrics.

    Parameters
    ----------
    delta
        Mean Delta per model (meV/atom).
    phase_diff_eos_mae
        Mean PhaseDiffEOS MAE per model (meV/atom).
    correct_stability
        Mean Phase stability per model (%).

    Returns
    -------
    dict[str, dict]
        Mapping of metric name to per-model value dicts.
    """
    return {
        "Δ": delta,
        "Phase energy": phase_diff_eos_mae,
        "Phase stability": correct_stability,
    }


def test_equation_of_state(metrics: dict[str, dict]) -> None:
    """
    Run EOS benchmark analysis.

    Parameters
    ----------
    metrics
        All EOS benchmark metric values.
    """
    return
