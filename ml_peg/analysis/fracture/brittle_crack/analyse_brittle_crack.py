"""Analyse the brittle_crack benchmark: model S-curves against DFT reference paths."""

from __future__ import annotations

import json
from pathlib import Path
from warnings import warn

from ase.io import read, write
from ase.units import J, m
import numpy as np
import plotly.graph_objects as go
import pytest

from ml_peg.analysis.fracture.brittle_crack import s_curves as sc
from ml_peg.analysis.utils.decorators import build_table
from ml_peg.analysis.utils.utils import load_metrics_config, write_struct_info
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.fracture.brittle_crack import crack_tools as ct
from ml_peg.calcs.utils.utils import download_s3_data
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
BENCHMARK_DIR = "brittle_crack"
CALC_PATH = CALCS_ROOT / "fracture" / BENCHMARK_DIR / "outputs"
OUT_PATH = APP_ROOT / "data" / "fracture" / BENCHMARK_DIR
S3_KEY = "inputs/fracture/brittle_crack/brittle_crack.zip"
S3_FILENAME = "brittle_crack.zip"
CASES = ct.load_cases()
TERMS = {"K": "E_K", "E": "E_E", "D": "E_D"}
FAILED_KEYS = ("E_K", "E_E", "E_D", "coverage", "coverage_D")


class TopologyMismatchError(ValueError):
    """Frames are not on the case topology (outputs from another cases.yml)."""


METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def reference_frames(name: str) -> list:
    """
    Load the DFT reference path of a case from the S3 data (cached after download).

    Parameters
    ----------
    name
        Case name.

    Returns
    -------
    list
        Reference frames with info ``K_over_KG`` and ``REF_energy``.
    """
    try:
        path = download_s3_data(key=S3_KEY, filename=S3_FILENAME) / BENCHMARK_DIR
        return read(path / f"{name}.extxyz", ":")
    except Exception as exc:
        raise FileNotFoundError(
            f"reference {name}: download of {S3_KEY} failed ({exc})"
        ) from exc


def _gamma_jm2(consts: dict) -> float:
    """
    Return the surface energy in J/m^2.

    Parameters
    ----------
    consts
        Constants with ``gamma`` in eV/A^2.

    Returns
    -------
    float
        Surface energy (J/m^2).
    """
    return consts["gamma"] / (J / m**2)


def _track(frames, cluster, crk, consts: dict, k_g: float, pair) -> sc.Track:
    """
    Build the energy-norm track of a path on its cluster.

    Parameters
    ----------
    frames
        Path frames.
    cluster
        Cluster with array ``reference`` and info ``tip_x``, ``tip_y``.
    crk
        Crack of the path's material.
    consts
        Constants (``a0``).
    k_g
        Griffith K (eV/A^2.5).
    pair
        Coordinate bond.

    Returns
    -------
    Track
        The track.
    """
    ref = np.asarray(cluster.arrays["reference"])
    tip = np.array([cluster.info["tip_x"], cluster.info["tip_y"]])

    def field(k_abs: float) -> np.ndarray:
        """
        Return the K-field displacements at absolute load ``k_abs``.

        Parameters
        ----------
        k_abs
            Load (eV/A^2.5).

        Returns
        -------
        numpy.ndarray
            Displacements (N, 3).
        """
        ux, uy = crk.displacements(ref[:, 0], ref[:, 1], tip[0], tip[1], k_abs)
        return np.column_stack([ux, uy, np.zeros_like(ux)])

    return sc.Track(
        ref,
        tip,
        field,
        consts["a0"],
        float(cluster.cell[2, 2]),
        [f.positions for f in frames],
        [f.info["K_over_KG"] * k_g for f in frames],
        pair,
    )


def case_errors(model_frames, ref_frames, case: dict, consts: dict, ref_consts: dict):
    """
    Compute a model's errors on a case against the DFT reference path.

    Parameters
    ----------
    model_frames
        Traced model frames (possibly empty).
    ref_frames
        DFT reference frames.
    case
        Case settings.
    consts
        Model constants (``a0``, ``C11``, ``C12``, ``C44``, ``gamma``).
    ref_consts
        Reference constants (the case topology).

    Returns
    -------
    dict
        ``E_K``, ``E_E``, ``E_D``, ``coverage`` and ``coverage_D``.
    """
    if not len(model_frames):
        return {
            "E_K": np.nan,
            "E_E": np.nan,
            "E_D": None,
            "coverage": 0.0,
            "coverage_D": 0.0,
        }
    from matscipy.fracture_mechanics.clusters import cluster_like

    ref_cl = ct.reference_cluster(case)
    for what, frames in (("model", model_frames), ("reference", ref_frames)):
        bad = [f for f in frames if list(f.numbers) != list(ref_cl.numbers)]
        if bad:
            raise TopologyMismatchError(
                f"{what} frames are not on the {case['label']} topology "
                f"({len(bad[0])} atoms vs {len(ref_cl)}); "
                "outputs from another cases.yml?"
            )
    pair = tuple(case["coordinate"])
    crk_r, crk_m = ct.crack(case, ref_consts), ct.crack(case, consts)
    kg_r, kg_m = crk_r.k1g(ref_consts["gamma"]), crk_m.k1g(consts["gamma"])
    ref = sc.curve_from_frames(
        ref_frames, pair, ref_consts["a0"], _gamma_jm2(ref_consts), kg_r
    )
    mod = sc.curve_from_frames(
        model_frames, pair, consts["a0"], _gamma_jm2(consts), kg_m
    )
    out = sc.errors(mod, ref)
    mod_cl = cluster_like(ref_cl, crk_m, kg_m, consts["a0"])
    ed = sc.ed_on_grid(
        _track(model_frames, mod_cl, crk_m, consts, kg_m, pair),
        _track(ref_frames, ref_cl, crk_r, ref_consts, kg_r, pair),
        ref_cl.numbers,
        sc.window(ref),
    )
    out.update(E_D=ed["E_D"], coverage_D=ed["coverage"])
    return out


def model_errors(
    consts: dict, frames: dict, refs: dict, model: str = ""
) -> dict[str, dict]:
    """
    Compute a model's errors on every case; a failed calculation gives NaN errors.

    A trace that runs but stops short of the window is scored by its coverage; a
    calculation that failed (constants, cleavage energy or tracer error) is not scored,
    so the benchmark score is NaN rather than rewarding the failure. An error while
    scoring one case warns and gives NaN for that case; frames on the wrong topology
    (stale outputs) still raise.

    Parameters
    ----------
    consts
        Model constants from the calculation (``constants.json``); ``gamma`` and
        ``status`` per case.
    frames
        Traced frames by case name.
    refs
        DFT reference frames by case name.
    model
        Model name, for warnings.

    Returns
    -------
    dict[str, dict]
        Errors by case name.
    """
    out = {}
    for name, case in CASES.items():
        gamma = (consts.get("gamma") or {}).get(name)
        status = str((consts.get("status") or {}).get(name, ""))
        ok = "a0" in consts and gamma is not None and np.isfinite(gamma)
        if not ok or status.startswith("failed"):
            out[name] = dict.fromkeys(FAILED_KEYS, np.nan)
            continue
        try:
            out[name] = case_errors(
                frames.get(name, []),
                refs[name],
                case,
                {**consts, "gamma": gamma},
                ct.topology_constants(case),
            )
        except TopologyMismatchError:
            raise
        except Exception as exc:
            warn(f"{model} {name}: {exc}", stacklevel=2)
            out[name] = dict.fromkeys(FAILED_KEYS, np.nan)
    return out


def raw_metrics(errs: dict) -> dict[str, float]:
    """
    Return the coverage-padded raw metric values for the table.

    The padding uses the thresholds loaded from metrics.yml (thresholds changed later
    in the app do not re-pad). A failed calculation (NaN coverage) stays NaN.

    Parameters
    ----------
    errs
        Errors by case name (``case_errors`` output).

    Returns
    -------
    dict[str, float]
        Raw value by metric name.
    """
    out = {}
    for name, e in errs.items():
        label = CASES[name]["label"]
        for term, key in TERMS.items():
            metric = f"{label} {term} error"
            th = DEFAULT_THRESHOLDS[metric]
            cov = e["coverage_D"] if term == "D" else e["coverage"]
            if np.isnan(cov):  # failed calculation: not scored
                out[metric] = np.nan
                continue
            out[metric] = sc.padded(e[key], cov, th["good"], th["bad"])
    return out


def _model_constants(model: str) -> dict | None:
    """
    Load a model's constants written by the calculation.

    Parameters
    ----------
    model
        Model name.

    Returns
    -------
    dict | None
        Constants, or None if the calculation has not run.
    """
    path = CALC_PATH / model / "constants.json"
    return json.loads(path.read_text()) if path.exists() else None


def s_curve_figure(
    model: str, name: str, ref_x, ref_k, mod_x, mod_k, window_x
) -> go.Figure:
    """
    Build the S-curve figure of a model on a case, with the DFT path and the window.

    Only the model's points carry ``customdata`` (the model label), so clicking a
    model point opens its traced structure and a DFT point does not.

    Parameters
    ----------
    model
        Model name.
    name
        Case name.
    ref_x, ref_k
        DFT coordinate (A) and K/K_G.
    mod_x, mod_k
        Model coordinate (A) and K/K_G.
    window_x
        Scored window (A) on the coordinate.

    Returns
    -------
    plotly.graph_objects.Figure
        The figure.
    """
    case = CASES[name]
    pair = tuple(case["coordinate"])
    hover = "<b>bond: </b>%{x:.3f} Å<br><b>K/K_G: </b>%{y:.4f}<br>"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=ref_x, y=ref_k, name="DFT", mode="lines", hovertemplate=hover)
    )
    fig.add_trace(
        go.Scatter(
            x=mod_x,
            y=mod_k,
            name=model,
            mode="lines+markers",
            customdata=[[model] for _ in mod_x],
            hovertemplate=hover,
        )
    )
    fig.add_vrect(
        x0=window_x[0],
        x1=window_x[1],
        annotation_text="scored window",
        annotation_position="top",
        opacity=0.15,
        line_width=0,
    )
    fig.update_layout(
        title={"text": f"{case['label']}: K/K_G along the crack path"},
        xaxis={"title": {"text": f"bond {pair} (Å)"}},
        yaxis={"title": {"text": "K / K_G (own)"}},
    )
    return fig


def _write_plot(model: str, name: str, ref_frames, model_frames) -> None:
    """
    Write the S-curve figure of a model on a case.

    Parameters
    ----------
    model
        Model name.
    name
        Case name.
    ref_frames
        DFT reference frames.
    model_frames
        Model frames.
    """
    pair = tuple(CASES[name]["coordinate"])
    ref_x = [sc.pair_distance(f, pair) for f in ref_frames]
    fig = s_curve_figure(
        model,
        name,
        ref_x,
        [f.info["K_over_KG"] for f in ref_frames],
        [sc.pair_distance(f, pair) for f in model_frames],
        [f.info["K_over_KG"] for f in model_frames],
        (min(ref_x), max(ref_x)),
    )
    path = OUT_PATH / f"figure_{model}_{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_json(path)


def reference_structures() -> list[Path]:
    """
    Write the reference clusters of all cases, for the structure (element) info.

    Model traces may be empty (a failed trace), so element info is taken from the
    reference topology, which every model shares.

    Returns
    -------
    list[Path]
        Paths of the written clusters.
    """
    out = OUT_PATH / "structs"
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, case in CASES.items():
        path = out / f"{name}.extxyz"
        atoms = ct.reference_cluster(case)
        for key in list(atoms.arrays):
            if key not in ("numbers", "positions"):
                del atoms.arrays[key]
        atoms.info.clear()
        write(path, atoms, format="extxyz")
        paths.append(path)
    return paths


@pytest.fixture
def case_results() -> dict[str, dict[str, dict]]:
    """
    Compute errors for all available (model, case) pairs and write plots and frames.

    Returns
    -------
    dict[str, dict[str, dict]]
        Errors by model and case name.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    refs = {name: reference_frames(name) for name in CASES}
    results = {}
    for model in MODELS:
        consts = _model_constants(model)
        if consts is None:
            continue
        frames = {}
        for name in CASES:
            path = CALC_PATH / model / f"{name}.extxyz"
            frames[name] = (
                read(path, ":") if path.exists() and path.stat().st_size else []
            )
        results[model] = model_errors(consts, frames, refs, model=model)
        for name in CASES:
            _write_plot(model, name, refs[name], frames[name])
            if frames[name]:
                out = OUT_PATH / name / model
                out.mkdir(parents=True, exist_ok=True)
                write(out / f"{model}-trace.extxyz", frames[name])
    return results


@pytest.fixture
def metrics_dict(case_results: dict) -> dict[str, dict[str, float]]:
    """
    Build the raw metric values for the benchmark table.

    Parameters
    ----------
    case_results
        Errors by model and case name.

    Returns
    -------
    dict[str, dict[str, float]]
        Metric values by metric name and model.
    """
    metrics: dict[str, dict[str, float]] = {k: {} for k in DEFAULT_THRESHOLDS}
    for model, errs in case_results.items():
        for metric, value in raw_metrics(errs).items():
            metrics[metric][model] = value
    return metrics


@pytest.fixture
@build_table(
    filename=OUT_PATH / f"{BENCHMARK_DIR}_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(metrics_dict: dict[str, dict[str, float]]) -> dict[str, dict]:
    """
    Build the benchmark table JSON.

    Parameters
    ----------
    metrics_dict
        Metric values for all models.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return metrics_dict


def test_brittle_crack(metrics: dict[str, dict]) -> None:
    """
    Run the brittle_crack analysis.

    Parameters
    ----------
    metrics
        Benchmark metrics table.
    """
    write_struct_info(data_path=reference_structures(), out_path=OUT_PATH, index=0)
