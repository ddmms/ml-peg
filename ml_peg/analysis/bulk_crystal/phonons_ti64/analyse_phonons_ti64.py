"""Analyse Ti64 phonons benchmark."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.bulk_crystal.phonons.thermal_utils import EV_TO_KJMOL
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "bulk_crystal" / "phonons_ti64" / "outputs"
REF_PATH = CALC_PATH / "DFT"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "phonons_ti64"

SCATTER_FILENAME = OUT_PATH / "phonons_ti64_interactive.json"

METRICS_YML = Path(__file__).with_name("metrics.yml")
THRESHOLDS, METRIC_TOOLTIPS, WEIGHTS = load_metrics_config(METRICS_YML)

OMEGA_AVG_METRIC_ID = "omega_avg_thz_mae"
OMEGA_MAX_METRIC_ID = "omega_max_thz_mae"
FREE_ENERGY_0K_METRIC_ID = "deltaF_0K_eV_per_atom_avg"
FREE_ENERGY_2000K_METRIC_ID = "deltaF_2000K_eV_per_atom_avg"
METRIC_ID_TO_LABEL: dict[str, str] = {
    OMEGA_AVG_METRIC_ID: "ω_avg MAE",
    OMEGA_MAX_METRIC_ID: "ω_max MAE",
    FREE_ENERGY_0K_METRIC_ID: "ΔF (0 K) mean",
    FREE_ENERGY_2000K_METRIC_ID: "ΔF (2000 K) mean",
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


def _load_pickle(path: Path) -> Any | None:
    """
    Load a pickled file, returning None when missing or unreadable.

    Parameters
    ----------
    path
        Path to the pickle file.

    Returns
    -------
    Any | None
        Unpickled object, or ``None`` when the file is missing or unreadable.
    """
    if not path.exists():
        return None
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        print(f"Failed to load {path}: {exc}")
        return None


def _load_json(path: Path) -> Any | None:
    """
    Load a JSON file, returning None when missing or unreadable.

    Parameters
    ----------
    path
        Path to the JSON file.

    Returns
    -------
    Any | None
        Parsed JSON data, or ``None`` when the file is missing or unreadable.
    """
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf8") as handle:
            return json.load(handle)
    except Exception as exc:
        print(f"Failed to load {path}: {exc}")
        return None


def _band_arrays(band: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """
    Concatenate a band dict's segments into distance and frequency arrays.

    Parameters
    ----------
    band
        Phonopy-style band-structure dict with ``distances`` and
        ``frequencies`` segment lists.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(distances, frequencies)`` with shapes ``(nq,)`` and
        ``(nq, n_bands)``.
    """
    distances = np.concatenate([np.asarray(seg) for seg in band["distances"]])
    freqs = np.vstack([np.asarray(seg) for seg in band["frequencies"]])
    return distances, freqs


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
    # scales can differ slightly. Map the reference distances onto the model
    # span before interpolating.
    ref_x = ref_dist * (pred_dist[-1] / ref_dist[-1]) if ref_dist[-1] else ref_dist
    out = np.empty((len(pred_dist), ref_freqs.shape[1]), dtype=float)
    for branch in range(ref_freqs.shape[1]):
        out[:, branch] = np.interp(pred_dist, ref_x, ref_freqs[:, branch])
    return out


def _free_energy_comparisons(
    ref_thermal: dict[str, Any], pred_thermal: dict[str, Any]
) -> dict[str, dict[str, float]] | None:
    """
    Compare free energies at the first and last temperature (eV/atom).

    Parameters
    ----------
    ref_thermal
        Reference thermal properties (free energy in kJ/mol per cell).
    pred_thermal
        Model thermal properties (free energy in kJ/mol per cell).

    Returns
    -------
    dict[str, dict[str, float]] | None
        Reference, prediction, and absolute error for the 0 K and 2000 K
        metrics, or ``None`` when data is invalid.
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
    ref_ev_per_atom = ref_on_pred / EV_TO_KJMOL / n_atoms
    pred_ev_per_atom = pred_f / EV_TO_KJMOL / n_atoms
    return {
        FREE_ENERGY_0K_METRIC_ID: {
            "ref": float(ref_ev_per_atom[0]),
            "pred": float(pred_ev_per_atom[0]),
            "error": float(abs(pred_ev_per_atom[0] - ref_ev_per_atom[0])),
        },
        FREE_ENERGY_2000K_METRIC_ID: {
            "ref": float(ref_ev_per_atom[-1]),
            "pred": float(pred_ev_per_atom[-1]),
            "error": float(abs(pred_ev_per_atom[-1] - ref_ev_per_atom[-1])),
        },
    }


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

    # Discover cases from the reference outputs, pre-load them once, and copy
    # structures for the app viewer. Free-energy references only exist for the
    # thermodynamics-enabled subset of cases.
    case_names = sorted(
        path.name.removesuffix("_band_structure.npz")
        for path in REF_PATH.glob("*_band_structure.npz")
    )
    ref_cache: dict[str, dict[str, Any]] = {}
    for case in case_names:
        ref_band = _load_pickle(REF_PATH / f"{case}_band_structure.npz")
        if ref_band is None:
            print(f"Missing DFT reference for {case}, skipping case.")
            continue
        ref_cache[case] = {
            "band": ref_band,
            "thermal": _load_json(REF_PATH / f"{case}_thermal_properties.json"),
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

        frequency_points: dict[str, list[dict[str, Any]]] = {
            OMEGA_AVG_METRIC_ID: [],
            OMEGA_MAX_METRIC_ID: [],
        }
        free_energy_points: dict[str, list[dict[str, Any]]] = {
            FREE_ENERGY_0K_METRIC_ID: [],
            FREE_ENERGY_2000K_METRIC_ID: [],
        }

        for case, ref_data in ref_cache.items():
            pred_band_path = model_dir / f"{case}_band_structure.npz"
            pred_band = _load_pickle(pred_band_path)
            if pred_band is None:
                continue

            ref_dist, ref_freqs = _band_arrays(ref_data["band"])
            pred_dist, pred_freqs = _band_arrays(pred_band)
            if (
                ref_freqs.shape[1] != pred_freqs.shape[1]
                or not np.isfinite(pred_freqs).all()
            ):
                print(f"{model_name}/{case}: invalid band data, skipping case.")
                continue

            ref_on_pred = _interp_ref_bands(ref_dist, ref_freqs, pred_dist)

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
                    "ref": f"/assets/bulk_crystal/phonons_ti64/DFT/{case}.xyz",
                    "pred": (
                        f"/assets/bulk_crystal/phonons_ti64/{model_name}/{case}.xyz"
                    ),
                }
            point_metadata = {
                "id": case,
                "label": case,
                "data_paths": data_paths,
                "structure_paths": structure_paths,
            }
            frequency_points[OMEGA_AVG_METRIC_ID].append(
                point_metadata
                | {
                    "ref": float(np.mean(ref_on_pred)),
                    "pred": float(np.mean(pred_freqs)),
                }
            )
            frequency_points[OMEGA_MAX_METRIC_ID].append(
                point_metadata
                | {
                    "ref": float(np.max(ref_on_pred)),
                    "pred": float(np.max(pred_freqs)),
                }
            )

            if ref_data["thermal"] is not None:
                pred_thermal = _load_json(model_dir / f"{case}_thermal_properties.json")
                if pred_thermal is not None:
                    comparisons = _free_energy_comparisons(
                        ref_data["thermal"], pred_thermal
                    )
                    if comparisons is not None:
                        for metric_id, comparison in comparisons.items():
                            free_energy_points[metric_id].append(
                                point_metadata
                                | {
                                    "ref": comparison["ref"],
                                    "pred": comparison["pred"],
                                }
                            )

        stats[model_name] = {
            "metrics": {
                **{
                    metric_id: (
                        float(
                            np.mean(
                                [abs(point["pred"] - point["ref"]) for point in values]
                            )
                        )
                        if values
                        else None
                    )
                    for metric_id, values in frequency_points.items()
                },
                **{
                    metric_id: (
                        float(
                            np.mean(
                                [abs(point["pred"] - point["ref"]) for point in values]
                            )
                        )
                        if values
                        else None
                    )
                    for metric_id, values in free_energy_points.items()
                },
            },
            "frequency_points": frequency_points,
            "free_energy_points": free_energy_points,
        }

    return stats


@pytest.fixture
@build_table(
    filename=OUT_PATH / "phonons_ti64_metrics_table.json",
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
    y_label="PBE ω_avg (THz)",
    metric_axis_labels={
        OMEGA_MAX_METRIC_ID: (
            "Predicted ω_max (THz)",
            "PBE ω_max (THz)",
        ),
        FREE_ENERGY_0K_METRIC_ID: (
            "Predicted F at 0 K (eV/atom)",
            "PBE F at 0 K (eV/atom)",
        ),
        FREE_ENERGY_2000K_METRIC_ID: (
            "Predicted F at 2000 K (eV/atom)",
            "PBE F at 2000 K (eV/atom)",
        ),
    },
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
    dataset: dict[str, Any] = {"metrics": METRIC_ID_TO_LABEL, "models": {}}

    for model_name, model_data in ti64_stats.items():
        metrics_data: dict[str, Any] = {}
        for metric_id, points_for_metric in model_data["frequency_points"].items():
            if points_for_metric:
                metrics_data[metric_id] = {
                    "points": points_for_metric,
                    "mae": model_data["metrics"][metric_id],
                }
        for metric_id, points_for_metric in model_data["free_energy_points"].items():
            if points_for_metric:
                metrics_data[metric_id] = {
                    "points": points_for_metric,
                    "mae": model_data["metrics"][metric_id],
                }
        if not metrics_data:
            continue
        dataset["models"][model_name] = {
            "metrics": metrics_data,
        }

    return dataset


def test_phonons_ti64_analysis(
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
    for model_data in interactive_dataset["models"].values():
        assert OMEGA_AVG_METRIC_ID in model_data["figures"]
        assert OMEGA_MAX_METRIC_ID in model_data["figures"]

    table_path = OUT_PATH / "phonons_ti64_metrics_table.json"
    assert table_path.exists()
    assert SCATTER_FILENAME.exists()
