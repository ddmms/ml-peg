"""Analyse Ti64 phonons benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil
from typing import Any

from ase.units import kJ, mol
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.bulk_crystal.ti64_phonons.calc_ti64_phonons import CASES, TP_ON
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "bulk_crystal" / "ti64_phonons" / "outputs"
REF_PATH = CALC_PATH / "DFT"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "ti64_phonons"

SCATTER_FILENAME = OUT_PATH / "ti64_phonons_interactive.json"

METRICS_YML = Path(__file__).with_name("metrics.yml")
THRESHOLDS, METRIC_TOOLTIPS, WEIGHTS = load_metrics_config(METRICS_YML)

CASE_NAMES = [spec["case_name"] for spec in CASES]
EV_TO_KJMOL = mol / kJ

OMEGA_METRIC_ID = "omega_avg_thz_mae"
METRIC_ID_TO_LABEL: dict[str, str] = {
    "dispersion_rmse_thz_avg": "Dispersion RMSE (mean)",
    "dispersion_rmse_thz_max": "Dispersion RMSE (max)",
    "deltaF_0K_eV_per_atom_avg": "ΔF (0 K) mean",
    "deltaF_2000K_eV_per_atom_avg": "ΔF (2000 K) mean",
    OMEGA_METRIC_ID: "ω_avg MAE",
}

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
    include_filenames=True,
    write_info=True,
    write_structs=False,
    out_path=OUT_PATH,
    model_name="DFT",
)


def _load_band(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load concatenated distances and frequencies from a pickled band dict.

    Parameters
    ----------
    path
        Path to a pickled phonopy-style band-structure dict.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        ``(distances, frequencies)`` with shapes ``(nq,)`` and
        ``(nq, n_bands)``, or ``None`` when unavailable.
    """
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            band = pickle.load(handle)
        distances = np.concatenate([np.asarray(seg) for seg in band["distances"]])
        freqs = np.vstack([np.asarray(seg) for seg in band["frequencies"]])
        return distances, freqs
    except Exception as exc:
        print(f"Failed to load band structure from {path}: {exc}")
        return None


def _load_thermal(path: Path) -> dict[str, Any] | None:
    """
    Load thermal properties JSON.

    Parameters
    ----------
    path
        Path to a ``*_thermal_properties.json`` file.

    Returns
    -------
    dict[str, Any] | None
        Parsed JSON mapping, or ``None`` when unavailable.
    """
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf8"))
    except Exception as exc:
        print(f"Failed to load thermal properties from {path}: {exc}")
        return None


def _interp_ref_bands(
    ref_dist: np.ndarray,
    ref_freqs: np.ndarray,
    pred_dist: np.ndarray,
) -> np.ndarray:
    """
    Interpolate reference bands onto the model's band-path grid.

    Parameters
    ----------
    ref_dist
        Reference path distances, shape ``(n_ref,)``.
    ref_freqs
        Reference frequencies, shape ``(n_ref, n_bands)``.
    pred_dist
        Model path distances, shape ``(n_pred,)``.

    Returns
    -------
    np.ndarray
        Reference frequencies on the model grid, shape ``(n_pred, n_bands)``.
    """
    # Bands are computed along the same fractional k-path but the distance
    # scales can differ slightly; map the reference distances onto the model
    # span before interpolating.
    ref_x = ref_dist * (pred_dist[-1] / ref_dist[-1]) if ref_dist[-1] else ref_dist
    out = np.empty((len(pred_dist), ref_freqs.shape[1]), dtype=float)
    for branch in range(ref_freqs.shape[1]):
        out[:, branch] = np.interp(pred_dist, ref_x, ref_freqs[:, branch])
    return out


def _free_energy_errors(
    ref_thermal: dict[str, Any], pred_thermal: dict[str, Any]
) -> tuple[float, float] | None:
    """
    Absolute free-energy errors at the first and last temperature (eV/atom).

    Parameters
    ----------
    ref_thermal
        Reference thermal properties (free energy in kJ/mol per cell).
    pred_thermal
        Model thermal properties (free energy in kJ/mol per cell).

    Returns
    -------
    tuple[float, float] | None
        ``(ΔF_first, ΔF_last)`` in eV/atom, or ``None`` when data is invalid.
    """
    n_atoms = ref_thermal.get("n_atoms") or pred_thermal.get("n_atoms")
    if not n_atoms:
        return None

    ref_temps = np.asarray(ref_thermal["temperatures"], dtype=float)
    ref_f = np.asarray(ref_thermal["free_energy"], dtype=float)
    pred_temps = np.asarray(pred_thermal["temperatures"], dtype=float)
    pred_f = np.asarray(pred_thermal["free_energy"], dtype=float)

    if not (np.isfinite(ref_f).all() and np.isfinite(pred_f).all()):
        return None

    ref_on_pred = np.interp(pred_temps, ref_temps, ref_f)
    delta_ev_per_atom = np.abs(ref_on_pred - pred_f) / EV_TO_KJMOL / n_atoms
    return float(delta_ev_per_atom[0]), float(delta_ev_per_atom[-1])


@pytest.fixture
def ti64_stats() -> dict[str, dict[str, Any]]:
    """
    Aggregate Ti64 benchmark statistics per model.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of model name to per-case metrics and scatter points.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    # Pre-load reference data once and copy structures for the app viewer.
    ref_cache: dict[str, dict[str, Any]] = {}
    for case in CASE_NAMES:
        ref_band = _load_band(REF_PATH / f"{case}_band_structure.npz")
        if ref_band is None:
            print(f"Missing DFT reference for {case}, skipping case.")
            continue
        ref_cache[case] = {
            "band": ref_band,
            "thermal": _load_thermal(REF_PATH / f"{case}_thermal_properties.json"),
        }
        ref_struct_src = REF_PATH / f"{case}.xyz"
        if ref_struct_src.exists():
            (OUT_PATH / "DFT").mkdir(parents=True, exist_ok=True)
            shutil.copy2(ref_struct_src, OUT_PATH / "DFT" / f"{case}.xyz")

    if not ref_cache:
        print(f"ERROR: no DFT reference data found in {REF_PATH}")
        return {}

    stats: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        if not model_dir.exists():
            print(f"Model directory not found: {model_dir}")
            continue

        rmse_by_case: dict[str, float] = {}
        df0_by_case: dict[str, float] = {}
        df2000_by_case: dict[str, float] = {}
        points: list[dict[str, Any]] = []

        for case, ref_data in ref_cache.items():
            pred_band_path = model_dir / f"{case}_band_structure.npz"
            pred_band = _load_band(pred_band_path)
            if pred_band is None:
                continue

            ref_dist, ref_freqs = ref_data["band"]
            pred_dist, pred_freqs = pred_band
            if (
                ref_freqs.shape[1] != pred_freqs.shape[1]
                or not np.isfinite(pred_freqs).all()
            ):
                print(f"{model_name}/{case}: invalid band data, skipping case.")
                continue

            ref_on_pred = _interp_ref_bands(ref_dist, ref_freqs, pred_dist)
            rmse_by_case[case] = float(
                np.sqrt(np.mean((ref_on_pred - pred_freqs) ** 2))
            )

            data_paths = {
                "ref_band": str(
                    (REF_PATH / f"{case}_band_structure.npz").relative_to(
                        CALC_PATH.parent
                    )
                ),
                "ref_dos": str(
                    (REF_PATH / f"{case}_dos.npz").relative_to(CALC_PATH.parent)
                ),
                "pred_band": str(pred_band_path.relative_to(CALC_PATH.parent)),
                "pred_dos": str(
                    (model_dir / f"{case}_dos.npz").relative_to(CALC_PATH.parent)
                ),
            }
            structure_paths = None
            pred_struct_src = model_dir / f"{case}.xyz"
            if pred_struct_src.exists() and (REF_PATH / f"{case}.xyz").exists():
                (OUT_PATH / model_name).mkdir(parents=True, exist_ok=True)
                shutil.copy2(pred_struct_src, OUT_PATH / model_name / f"{case}.xyz")
                structure_paths = {
                    "ref": f"/assets/bulk_crystal/ti64_phonons/DFT/{case}.xyz",
                    "pred": (
                        f"/assets/bulk_crystal/ti64_phonons/{model_name}/{case}.xyz"
                    ),
                }
            points.append(
                {
                    "id": case,
                    "label": case,
                    "ref": float(np.mean(ref_on_pred)),
                    "pred": float(np.mean(pred_freqs)),
                    "data_paths": data_paths,
                    "structure_paths": structure_paths,
                }
            )

            if case in TP_ON and ref_data["thermal"] is not None:
                pred_thermal = _load_thermal(
                    model_dir / f"{case}_thermal_properties.json"
                )
                if pred_thermal is not None:
                    errors = _free_energy_errors(ref_data["thermal"], pred_thermal)
                    if errors is not None:
                        df0_by_case[case], df2000_by_case[case] = errors

        rmse_vals = list(rmse_by_case.values())
        omega_errors = [abs(p["ref"] - p["pred"]) for p in points]
        stats[model_name] = {
            "metrics": {
                "dispersion_rmse_thz_avg": float(np.mean(rmse_vals))
                if rmse_vals
                else None,
                "dispersion_rmse_thz_max": float(np.max(rmse_vals))
                if rmse_vals
                else None,
                "deltaF_0K_eV_per_atom_avg": float(np.mean(list(df0_by_case.values())))
                if df0_by_case
                else None,
                "deltaF_2000K_eV_per_atom_avg": float(
                    np.mean(list(df2000_by_case.values()))
                )
                if df2000_by_case
                else None,
                OMEGA_METRIC_ID: float(np.mean(omega_errors)) if omega_errors else None,
            },
            "points": points,
        }

    return stats


@pytest.fixture
@build_table(
    filename=OUT_PATH / "ti64_phonons_metrics_table.json",
    thresholds=THRESHOLDS,
    metric_tooltips=METRIC_TOOLTIPS,
    weights=WEIGHTS,
)
def metrics(
    ti64_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """
    Build the Ti64 metrics table for the Dash app.

    Parameters
    ----------
    ti64_stats
        Per-model statistics from :func:`ti64_stats`.

    Returns
    -------
    dict[str, dict[str, float | None]]
        Mapping of metric label to per-model values.
    """
    return {
        label: {
            model: ti64_stats.get(model, {}).get("metrics", {}).get(metric_id)
            for model in MODELS
        }
        for metric_id, label in METRIC_ID_TO_LABEL.items()
    }


@pytest.fixture
@cell_to_scatter(
    filename=SCATTER_FILENAME,
    x_label="Predicted ω_avg (THz)",
    y_label="Reference ω_avg (THz)",
)
def interactive_dataset(ti64_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Build the interactive scatter dataset for the Ti64 phonons Dash app.

    Parameters
    ----------
    ti64_stats
        Per-model statistics from :func:`ti64_stats`.

    Returns
    -------
    dict[str, Any]
        Interactive dataset written to JSON by the decorator.
    """
    dataset: dict[str, Any] = {
        "metrics": {OMEGA_METRIC_ID: METRIC_ID_TO_LABEL[OMEGA_METRIC_ID]},
        "models": {},
    }

    for model_name, model_data in ti64_stats.items():
        if not model_data["points"]:
            continue
        dataset["models"][model_name] = {
            "metrics": {
                OMEGA_METRIC_ID: {
                    "points": model_data["points"],
                    "mae": model_data["metrics"][OMEGA_METRIC_ID],
                }
            },
        }

    return dataset


def test_ti64_phonons_analysis(
    metrics: dict[str, Any],
    interactive_dataset: dict[str, Any],
) -> None:
    """
    Generate JSON artifacts for the Ti64 phonons benchmark.

    Parameters
    ----------
    metrics
        Table fixture output (decorator writes JSON).
    interactive_dataset
        Scatter fixture output (decorator writes JSON).
    """
    assert isinstance(metrics, dict)
    assert isinstance(interactive_dataset, dict)

    table_path = OUT_PATH / "ti64_phonons_metrics_table.json"
    assert table_path.exists()
    assert SCATTER_FILENAME.exists()
