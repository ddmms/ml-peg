"""
Analyse the YBCO lattice-parameters-vs-oxygen-content benchmark.

MAE of a, b, c vs the CP2K PBE reference, per parameter, in Angstrom.
"""

from __future__ import annotations

import json
from pathlib import Path

from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_parity
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "YBCO_lattice_vs_O" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "YBCO_lattice_vs_O"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

# Oxygen contents 6.00 .. 7.00 and the CP2K PBE reference lattice parameters (per unit
# cell, Angstrom) for a, b, c.
CONC = [round(6.0 + 0.1 * i, 2) for i in range(11)]
DFT = {
    "a": [
        3.9015415,
        3.8977335,
        3.89126025,
        3.88368875,
        3.8803395,
        3.8759885,
        3.87319575,
        3.8724425,
        3.86051275,
        3.86377375,
        3.8648375,
    ],
    "b": [
        3.9015805,
        3.90924125,
        3.92012725,
        3.93113625,
        3.9397595,
        3.94236175,
        3.9451875,
        3.94247975,
        3.94947875,
        3.9471525,
        3.940615,
    ],
    "c": [
        12.2686525,
        12.2411155,
        12.166553,
        12.139196,
        12.0902685,
        12.0576865,
        12.0337165,
        11.991617,
        11.99,
        11.966463,
        11.9447255,
    ],
}
PARAMS = ("a", "b", "c")
PARAM_IDX = {"a": 0, "b": 1, "c": 2}

# parity points: (a, b, c) per oxygen content
POINT_LABELS = [f"{p} @ YBCO{c:.2f}" for c in CONC for p in PARAMS]
REF_FLAT = [DFT[p][i] for i, _ in enumerate(CONC) for p in PARAMS]

# info.json (elements, for filtering) from the mock outputs
OUT_PATH.mkdir(parents=True, exist_ok=True)
try:
    get_struct_info(
        calc_path=CALC_PATH,
        glob_pattern="*-traj.extxyz",
        index="-1",
        info_keys=["name"],
        write_info=True,
        write_structs=False,
        out_path=OUT_PATH,
    )
except ValueError:
    with (OUT_PATH / "info.json").open("w", encoding="utf8") as f:
        json.dump({"elements": [], "name": []}, f, indent=1)


def _read_params(model_name: str) -> dict[float, tuple[float, float, float]]:
    """
    Read relaxed per-unit-cell a, b, c for each oxygen content for one model.

    Parameters
    ----------
    model_name
        Name of the model whose outputs to read.

    Returns
    -------
    dict[float, tuple[float, float, float]]
        Mapping of oxygen content to (a, b, c) in Angstrom.
    """
    out: dict[float, tuple[float, float, float]] = {}
    model_dir = CALC_PATH / model_name
    if not model_dir.exists():
        return out
    for conc in CONC:
        traj = model_dir / f"YBCO{conc:.2f}-traj.extxyz"
        if not traj.is_file():
            continue
        atoms = read(traj, index="-1")
        na, nb, nc = atoms.info.get("reps", (4, 4, 2))
        la, lb, lc = atoms.cell.lengths()
        out[conc] = (la / na, lb / nb, lc / nc)
    return out


@pytest.fixture
@plot_parity(
    filename=OUT_PATH / "figure_ybco_lattice.json",
    title="YBCO lattice parameters vs oxygen content",
    x_label="Predicted lattice parameter / Å",
    y_label="CP2K PBE lattice parameter / Å",
    hoverdata={"Point": POINT_LABELS},
)
def ybco_lattice() -> dict[str, list]:
    """
    Get DFT and predicted lattice parameters (a, b, c across oxygen content).

    Returns
    -------
    dict[str, list]
        Reference and predicted lattice parameters, flattened over (a, b, c) x content.
    """
    results = {"ref": REF_FLAT} | {mlip: [] for mlip in MODELS}
    for model_name in MODELS:
        params = _read_params(model_name)
        model_dir = CALC_PATH / model_name
        for conc in CONC:
            for p in PARAMS:
                if conc in params:
                    results[model_name].append(params[conc][PARAM_IDX[p]])
                else:
                    results[model_name].append(np.nan)
            # copy structure to app dir for visualisation
            src = model_dir / f"YBCO{conc:.2f}-traj.extxyz"
            if src.is_file():
                structs_dir = OUT_PATH / model_name
                structs_dir.mkdir(parents=True, exist_ok=True)
                write(structs_dir / f"YBCO{conc:.2f}.xyz", read(src, index="-1"))
    return results


def _param_errors(ybco_lattice: dict[str, list], param: str | None) -> dict[str, float]:
    """
    Mean absolute error vs DFT, optionally restricted to one lattice parameter.

    Parameters
    ----------
    ybco_lattice
        Reference and predicted lattice parameters.
    param
        One of "a", "b", "c" to restrict to, or None for all parameters.

    Returns
    -------
    dict[str, float]
        Mean absolute error per model, in Angstrom.
    """
    if param is None:
        mask = [True] * len(POINT_LABELS)
    else:
        mask = [lbl.startswith(f"{param} ") for lbl in POINT_LABELS]
    ref = [v for v, m in zip(ybco_lattice["ref"], mask, strict=True) if m]
    results = {}
    for model_name in MODELS:
        pred = [v for v, m in zip(ybco_lattice[model_name], mask, strict=True) if m]
        results[model_name] = mae(ref, pred)
    return results


@pytest.fixture
def ybco_lattice_errors(ybco_lattice) -> dict[str, float]:
    """
    MAE calculated over all lattice parameters. See :func:`_param_errors`.

    Parameters
    ----------
    ybco_lattice
        Reference and predicted lattice parameters.

    Returns
    -------
    dict[str, float]
        MAE per model, in Angstrom.
    """
    return _param_errors(ybco_lattice, None)


@pytest.fixture
def ybco_a_errors(ybco_lattice) -> dict[str, float]:
    """
    MAE for parameter a. See :func:`_param_errors`.

    Parameters
    ----------
    ybco_lattice
        Reference and predicted lattice parameters.

    Returns
    -------
    dict[str, float]
        MAE per model, in Angstrom.
    """
    return _param_errors(ybco_lattice, "a")


@pytest.fixture
def ybco_b_errors(ybco_lattice) -> dict[str, float]:
    """
    MAE for parameter b. See :func:`_param_errors`.

    Parameters
    ----------
    ybco_lattice
        Reference and predicted lattice parameters.

    Returns
    -------
    dict[str, float]
        MAE per model, in Angstrom.
    """
    return _param_errors(ybco_lattice, "b")


@pytest.fixture
def ybco_c_errors(ybco_lattice) -> dict[str, float]:
    """
    MAE for parameter c. See :func:`_param_errors`.

    Parameters
    ----------
    ybco_lattice
        Reference and predicted lattice parameters.

    Returns
    -------
    dict[str, float]
        MAE per model, in Angstrom.
    """
    return _param_errors(ybco_lattice, "c")


@pytest.fixture
@build_table(
    filename=OUT_PATH / "ybco_lattice_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
)
def metrics(
    ybco_a_errors: dict[str, float],
    ybco_b_errors: dict[str, float],
    ybco_c_errors: dict[str, float],
) -> dict[str, dict]:
    """
    Get all YBCO lattice-parameter metrics (per-parameter MAE).

    Parameters
    ----------
    ybco_a_errors
        Mean absolute error for a.
    ybco_b_errors
        Mean absolute error for b.
    ybco_c_errors
        Mean absolute error for c.

    Returns
    -------
    dict[str, dict]
        Metric names and values for all models.
    """
    return {
        "MAE a (PBE)": ybco_a_errors,
        "MAE b (PBE)": ybco_b_errors,
        "MAE c (PBE)": ybco_c_errors,
    }


def test_ybco_lattice_vs_O(metrics: dict[str, dict]) -> None:  # noqa: N802
    """
    Run the YBCO lattice-vs-oxygen analysis.

    Parameters
    ----------
    metrics
        All YBCO lattice-parameter metrics.
    """
    return
