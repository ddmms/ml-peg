"""Analyse Si interstitial NEB DFT singlepoints (energies + forces)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ase.atoms import Atoms
from ase.io import read, write
import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_scatter
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)

BENCHMARK_DIR = "si_defect_neb_singlepoints"
CALC_PATH = CALCS_ROOT / "nebs" / BENCHMARK_DIR / "outputs"
OUT_PATH = APP_ROOT / "data" / "nebs" / BENCHMARK_DIR

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


@dataclass(frozen=True)
class _Case:
    """Definition of a single Si interstitial NEB dataset."""

    key: str
    label: str


CASES: tuple[_Case, ...] = (
    _Case(key="64_atoms", label="64"),
    _Case(key="216_atoms", label="216"),
    _Case(key="216_atoms_di_to_single", label="216 di-to-single"),
)


@dataclass(frozen=True)
class _Results:
    """Computed errors and assets for one (case, model) pair."""

    energy_mae: float
    force_mae: float
    x: list[float]
    energy_error: list[float]
    force_rms: list[float]
    structs: list[Atoms]


def _load_structs(case_key: str, model: str) -> list[Atoms] | None:
    """
    Load MLIP-evaluated NEB frames (with DFT reference stored).

    Parameters
    ----------
    case_key
        Case key.
    model
        Model name.

    Returns
    -------
    list[ase.atoms.Atoms] | None
        Frames if the calculation output exists, otherwise ``None``.
    """
    path = CALC_PATH / case_key / model / f"{BENCHMARK_DIR}.extxyz"
    if not path.exists():
        return None
    structs = read(path, index=":")
    if not isinstance(structs, list) or not structs:
        raise ValueError(f"Unexpected output content: {path}")
    return structs


def _compute(structs: list[Atoms]) -> _Results:
    """
    Compute per-image errors and global MAEs.

    Parameters
    ----------
    structs
        Ordered NEB frames with ``ref_energy_ev``, ``ref_forces``, ``pred_energy_ev``,
        and ``pred_forces``.

    Returns
    -------
    _Results
        Computed errors and frames for downstream plotting/app.
    """
    ref_e = np.asarray([float(s.info["ref_energy_ev"]) for s in structs])
    pred_e = np.asarray([float(s.info["pred_energy_ev"]) for s in structs])
    # Compare *relative* energies along the NEB (shift image 0 to 0 eV for both
    # reference and predictions). This removes any constant energy offset.
    ref_e_rel = ref_e - ref_e[0]
    pred_e_rel = pred_e - pred_e[0]
    energy_error = pred_e_rel - ref_e_rel

    ref_f = np.asarray([s.arrays["ref_forces"] for s in structs], dtype=float)
    pred_f = np.asarray([s.arrays["pred_forces"] for s in structs], dtype=float)
    force_error = pred_f - ref_f

    x = [float(s.info.get("image_index", idx)) for idx, s in enumerate(structs)]
    return _Results(
        energy_mae=mae(ref_e_rel, pred_e_rel),
        force_mae=mae(ref_f.reshape(-1), pred_f.reshape(-1)),
        x=x,
        energy_error=[float(v) for v in energy_error],
        force_rms=[float(v) for v in np.sqrt(np.mean(force_error**2, axis=(1, 2)))],
        structs=structs,
    )


def _write_assets(case_key: str, model: str, results: _Results) -> None:
    """
    Write structure trajectory and per-image CSV for the app.

    Parameters
    ----------
    case_key
        Case key.
    model
        Model name.
    results
        Computed errors and frames.
    """
    out_dir = OUT_PATH / case_key / model
    out_dir.mkdir(parents=True, exist_ok=True)
    write(out_dir / f"{model}-neb-band.extxyz", results.structs)
    pd.DataFrame(
        {
            "image": results.x,
            "energy_error_ev": results.energy_error,
            "force_rms_ev_per_ang": results.force_rms,
        }
    ).to_csv(out_dir / f"{model}-per_image_errors.csv", index=False)


def _write_plots(case_key: str, case_label: str, model: str, results: _Results) -> None:
    """
    Write Dash/Plotly JSON plots for one (case, model) pair.

    Parameters
    ----------
    case_key
        Case key.
    case_label
        Human-friendly case label.
    model
        Model name.
    results
        Computed errors and frames.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    @plot_scatter(
        filename=OUT_PATH / f"figure_{model}_{case_key}_energy_error.json",
        title=f"Energy error along NEB ({case_label})",
        x_label="Image",
        y_label="Energy error / eV",
        show_line=True,
    )
    def plot_energy() -> dict[str, tuple[list[float], list[float]]]:
        """
        Plot per-image energy errors for a single model and case.

        Returns
        -------
        dict[str, tuple[list[float], list[float]]]
            Mapping of model name to x/y arrays.
        """
        return {model: [results.x, results.energy_error]}

    @plot_scatter(
        filename=OUT_PATH / f"figure_{model}_{case_key}_force_rms.json",
        title=f"Force RMS error along NEB ({case_label})",
        x_label="Image",
        y_label="Force RMS error / (eV/Å)",
        show_line=True,
    )
    def plot_forces() -> dict[str, tuple[list[float], list[float]]]:
        """
        Plot per-image force RMS errors for a single model and case.

        Returns
        -------
        dict[str, tuple[list[float], list[float]]]
            Mapping of model name to x/y arrays.
        """
        return {model: [results.x, results.force_rms]}

    plot_energy()
    plot_forces()


@pytest.fixture
def case_results() -> dict[tuple[str, str], _Results]:
    """
    Compute results for all available (case, model) combinations.

    Returns
    -------
    dict[tuple[str, str], _Results]
        Results indexed by ``(case_key, model_name)``.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    results: dict[tuple[str, str], _Results] = {}
    for case in CASES:
        for model in MODELS:
            structs = _load_structs(case.key, model)
            if structs is None:
                continue
            computed = _compute(structs)
            _write_assets(case.key, model, computed)
            _write_plots(case.key, case.label, model, computed)
            results[(case.key, model)] = computed
    return results


@pytest.fixture
def metrics_dict(
    case_results: dict[tuple[str, str], _Results],
) -> dict[str, dict[str, float]]:
    """
    Build raw metric dict for the benchmark table.

    Parameters
    ----------
    case_results
        Results indexed by ``(case_key, model_name)``.

    Returns
    -------
    dict[str, dict[str, float]]
        Metric values for all models.
    """
    metrics: dict[str, dict[str, float]] = {}
    for case in CASES:
        energy_key = f"Energy MAE ({case.label})"
        force_key = f"Force MAE ({case.label})"
        metrics[energy_key] = {}
        metrics[force_key] = {}
        for model in MODELS:
            result = case_results.get((case.key, model))
            if result is None:
                continue
            metrics[energy_key][model] = result.energy_mae
            metrics[force_key][model] = result.force_mae
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


def test_si_defect_neb_singlepoints(metrics: dict[str, dict]) -> None:
    """
    Run analysis for Si interstitial NEB DFT singlepoints.

    Parameters
    ----------
    metrics
        Benchmark metrics table.
    """
    return
