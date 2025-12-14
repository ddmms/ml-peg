"""Analyse phonon dispersion benchmark."""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pickle
from typing import Any

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest
from sklearn.metrics import f1_score
from tqdm import tqdm

from ml_peg.analysis.utils.decorators import build_table, table_scatter_png
from ml_peg.analysis.utils.utils import load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

LOGGER = logging.getLogger(__name__)

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "bulk_crystal" / "phonons" / "outputs"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "phonons"
ASSETS_PATH = OUT_PATH.parent.parent / "assets" / "bulk_crystal" / "phonons"
SCATTER_FILENAME = OUT_PATH / "phonon_interactive.json"

METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)

METRIC_LABELS = {
    "max_freq": "ω_max [THz]",
    "avg_freq": "ω_avg [THz]",
    "min_freq": "ω_min [THz]",
    "S": "S [J/mol·K]",
    "F": "F [kJ/mol]",
    "C_V": "C_V [J/mol·K]",
}
BZ_COLUMN = "Avg BZ MAE [THz]"
STABILITY_COLUMN = "Stability Classification (F1)"
STABILITY_THRESHOLD = -0.05
T_300K_INDEX = 3  # Index for 300K in thermal properties (0, 75, 150, 300, 600)


def _load_band_structure(file_path: Path) -> dict[str, Any] | None:
    """
    Load a serialized band-structure payload.

    Parameters
    ----------
    file_path
        Absolute path to the ``npz`` file containing the band structure.

    Returns
    -------
    dict[str, Any] | None
        Parsed dictionary describing distances/frequencies, or ``None`` if the
        file cannot be read.
    """
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        LOGGER.warning(f"Failed to load band structure from {file_path}: {e}")
        return None


def _load_dos(file_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Load frequencies and densities of states.

    Parameters
    ----------
    file_path
        Location of the ``npz`` file storing DOS data.

    Returns
    -------
    tuple[np.ndarray, np.ndarray] | None
        Pair of ``(frequency_points, total_dos)`` arrays, or ``None`` if the
        payload could not be opened.
    """
    try:
        with open(file_path, "rb") as f:
            dos_dict = pickle.load(f)
        return dos_dict["frequency_points"], dos_dict["total_dos"]
    except Exception as e:
        LOGGER.warning(f"Failed to load DOS from {file_path}: {e}")
        return None


def _load_thermal_properties(file_path: Path) -> dict[str, Any] | None:
    """
    Load vibrational thermodynamic properties.

    Parameters
    ----------
    file_path
        Path to the JSON file containing entropy, free energy, and heat
        capacity arrays.

    Returns
    -------
    dict[str, Any] | None
        Mapping of thermal quantities keyed by temperature, or ``None`` if the
        JSON file is missing.
    """
    try:
        with open(file_path, encoding="utf8") as f:
            return json.load(f)
    except Exception as e:
        LOGGER.warning(f"Failed to load thermal properties from {file_path}: {e}")
        return None


def _get_mp_ids() -> list[str]:
    """
    Return all Materials Project IDs with reference data.

    Returns
    -------
    list[str]
        Sorted collection of MP identifiers discovered in the reference DFT
        directory.
    """
    ref_dir = CALC_PATH / "DFT"
    if not ref_dir.exists():
        print(f"ERROR: Reference directory not found: {ref_dir}")
        print(f"Expected location: {ref_dir.absolute()}")
        LOGGER.error(f"Reference directory not found: {ref_dir}")
        return []

    mp_ids = set()
    for file_path in ref_dir.glob("mp-*_thermal_properties.json"):
        mp_id = file_path.stem.replace("_thermal_properties", "")
        mp_ids.add(mp_id)

    return sorted(mp_ids)


def _prettify_chemical_formula(formula: str) -> str:
    """
    Convert a raw chemical formula to a LaTeX-friendly string.

    Parameters
    ----------
    formula
        Plain-text chemical formula (e.g. ``Al2O3``).

    Returns
    -------
    str
        String with subscripts added so Matplotlib renders chemical species
        correctly.
    """
    import re

    parts = re.findall(r"([A-Z][a-z]*)(\d*)", formula)
    pretty = ""
    for element, count in parts:
        if count == "":
            pretty += element
        else:
            pretty += f"{element}$_{{{count}}}$"
    return pretty


def _build_xticks(
    distances: list, labels: list, connections: list
) -> tuple[list, list]:
    """
    Build x-axis ticks for the band-structure plot.

    Parameters
    ----------
    distances
        Distances along the Brillouin-path segments.
    labels
        High-symmetry labels for each vertex.
    connections
        Boolean mask describing segment connectivity.

    Returns
    -------
    tuple[list, list]
        Tick positions and tick labels for Matplotlib.
    """
    xticks, xticklabels = [], []
    cumulative_dist, i = 0.0, 0
    connections = [True] + connections

    for seg_dist, connected in zip(distances, connections, strict=False):
        start, end = labels[i], labels[i + 1]
        pos_start = cumulative_dist
        pos_end = cumulative_dist + (seg_dist[-1] - seg_dist[0])
        xticks.append(pos_start)
        xticklabels.append(f"{start}|{end}" if not connected else start)
        i += 2 if not connected else 1
        cumulative_dist = pos_end

    xticks.append(cumulative_dist)
    xticklabels.append(labels[-1])
    return xticks, xticklabels


def _classify_stability(ref_val: float, pred_val: float) -> str:
    """
    Classify stability based on minimum frequency predictions.

    Parameters
    ----------
    ref_val
        Reference minimum frequency (THz).
    pred_val
        Predicted minimum frequency (THz).

    Returns
    -------
    str
        ``"TP"``, ``"TN"``, ``"FP"``, or ``"FN"`` according to the
        STABILITY_THRESHOLD.
    """
    ref_stable = ref_val > STABILITY_THRESHOLD
    pred_stable = pred_val > STABILITY_THRESHOLD
    if ref_stable and pred_stable:
        return "TN"
    if (not ref_stable) and (not pred_stable):
        return "TP"
    if (not ref_stable) and pred_stable:
        return "FN"
    return "FP"


def _stability_statistics(
    points: list[dict[str, Any]],
) -> tuple[float | None, list[list[int]]]:
    """
    Calculate F1 score and confusion matrix for stability classification.

    Parameters
    ----------
    points
        Sequence of dictionaries with ``ref`` and ``pred`` ω_min values.

    Returns
    -------
    tuple[float | None, list[list[int]]]
        ``(f1_score, confusion_matrix)`` where the confusion matrix is
        formatted as ``[[TP, FN], [FP, TN]]``.
    """
    if not points:
        return None, [[0, 0], [0, 0]]

    y_true = np.array(
        [entry["ref"] > STABILITY_THRESHOLD for entry in points], dtype=bool
    )
    y_pred = np.array(
        [entry["pred"] > STABILITY_THRESHOLD for entry in points], dtype=bool
    )

    f1_value = f1_score(y_true, y_pred, zero_division=0)

    tp = int(np.logical_and(y_true, y_pred).sum())
    fn = int(np.logical_and(y_true, ~y_pred).sum())
    fp = int(np.logical_and(~y_true, y_pred).sum())
    tn = int(np.logical_and(~y_true, ~y_pred).sum())
    matrix = [[tp, fn], [fp, tn]]

    return f1_value, matrix


@pytest.fixture
def phonon_stats() -> dict[str, dict[str, Any]]:
    """
    Aggregate phonon benchmark statistics per model.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of model display name to its metrics, BZ errors, and stability
        summaries.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    ASSETS_PATH.mkdir(parents=True, exist_ok=True)

    mp_ids = _get_mp_ids()
    if not mp_ids:
        print("ERROR: No reference data found!")
        print(f"Expected DFT reference files in: {CALC_PATH / 'DFT'}")
        LOGGER.error("No reference data found!")
        return {}

    print(f"✓ Found {len(mp_ids)} systems with reference data")
    LOGGER.info(f"Found {len(mp_ids)} systems with reference data")

    # OPTIMIZATION 1: Pre-load all reference data once (not per-model!)
    print("→ Pre-loading reference data...")
    ref_cache: dict[str, dict[str, Any]] = {}
    for mp_id in tqdm(mp_ids, desc="Loading reference data", leave=False):
        ref_band_path = CALC_PATH / "DFT" / f"{mp_id}_band_structure.npz"
        ref_dos_path = CALC_PATH / "DFT" / f"{mp_id}_dos.npz"
        ref_thermal_path = CALC_PATH / "DFT" / f"{mp_id}_thermal_properties.json"
        ref_labels_path = CALC_PATH / "DFT" / f"{mp_id}_labels.json"
        ref_connections_path = CALC_PATH / "DFT" / f"{mp_id}_connections.json"

        ref_band = _load_band_structure(ref_band_path)
        ref_dos = _load_dos(ref_dos_path)
        ref_thermal = _load_thermal_properties(ref_thermal_path)
        # Reuse JSON loader for band metadata
        ref_labels = _load_thermal_properties(ref_labels_path)
        ref_connections = _load_thermal_properties(ref_connections_path)

        if all([ref_band, ref_dos, ref_thermal]):
            # Add labels and connections to band structure dict
            if ref_labels and ref_connections:
                ref_band["labels"] = ref_labels
                ref_band["path_connections"] = ref_connections

            ref_cache[mp_id] = {
                "band": ref_band,
                "dos": ref_dos,
                "thermal": ref_thermal,
                "band_path": ref_band_path,
                "dos_path": ref_dos_path,
            }

    print(f"✓ Cached {len(ref_cache)} reference systems\n")

    stats: dict[str, dict[str, Any]] = {}

    for model_name in tqdm(MODELS, desc="Processing models"):
        print(f"\n→ Processing model: {model_name}")
        LOGGER.info(f"Processing model: {model_name}")
        model_dir = CALC_PATH / model_name

        if not model_dir.exists():
            print(f"  ⚠ Model directory not found: {model_dir}")
            LOGGER.warning(f"Model directory not found: {model_dir}")
            continue

        metrics_data: dict[str, dict[str, Any]] = {
            key: {"points": [], "ref": [], "pred": [], "mae": None}
            for key in METRIC_LABELS
        }
        band_errors: dict[str, list[float]] = {}
        stability_points: list[dict[str, Any]] = []

        # Mean-of-means calculation: store mean per system, then average those means
        system_mean_errors: list[float] = []

        processed_count = 0
        skipped_count = 0

        for mp_id in tqdm(ref_cache.keys(), desc=f"  {model_name[:20]}", leave=False):
            # OPTIMIZATION 1: Use pre-loaded reference data
            ref_data = ref_cache[mp_id]
            ref_band = ref_data["band"]
            ref_dos = ref_data["dos"]
            ref_thermal = ref_data["thermal"]
            ref_band_path = ref_data["band_path"]
            ref_dos_path = ref_data["dos_path"]

            # Load predicted data
            pred_band_path = model_dir / f"{mp_id}_band_structure.npz"
            pred_dos_path = model_dir / f"{mp_id}_dos.npz"
            pred_thermal_path = model_dir / f"{mp_id}_thermal_properties.json"

            pred_band = _load_band_structure(pred_band_path)
            pred_dos = _load_dos(pred_dos_path)
            pred_thermal = _load_thermal_properties(pred_thermal_path)

            if not all([pred_band, pred_dos, pred_thermal]):
                skipped_count += 1
                continue

            processed_count += 1

            # Calculate metrics
            ref_freqs = np.concatenate(ref_band["frequencies"])
            pred_freqs = np.concatenate(pred_band["frequencies"])

            max_freq_ref = float(np.max(ref_freqs))
            max_freq_pred = float(np.max(pred_freqs))
            avg_freq_ref = float(np.mean(ref_freqs))
            avg_freq_pred = float(np.mean(pred_freqs))
            min_freq_ref = float(np.min(ref_freqs))
            min_freq_pred = float(np.min(pred_freqs))

            s_ref = ref_thermal["entropy"][T_300K_INDEX]
            s_pred = pred_thermal["entropy"][T_300K_INDEX]
            f_ref = ref_thermal["free_energy"][T_300K_INDEX]
            f_pred = pred_thermal["free_energy"][T_300K_INDEX]
            cv_ref = ref_thermal["heat_capacity"][T_300K_INDEX]
            cv_pred = pred_thermal["heat_capacity"][T_300K_INDEX]

            # Store data paths for on-demand plot generation
            # This replaces pre-generated PNGs and dramatically reduces runtime
            data_paths = {
                "ref_band": str(ref_band_path.relative_to(CALC_PATH.parent)),
                "ref_dos": str(ref_dos_path.relative_to(CALC_PATH.parent)),
                "pred_band": str(pred_band_path.relative_to(CALC_PATH.parent)),
                "pred_dos": str(pred_dos_path.relative_to(CALC_PATH.parent)),
            }

            # Store metric points
            metric_values = {
                "max_freq": (max_freq_ref, max_freq_pred),
                "avg_freq": (avg_freq_ref, avg_freq_pred),
                "min_freq": (min_freq_ref, min_freq_pred),
                "S": (s_ref, s_pred),
                "F": (f_ref, f_pred),
                "C_V": (cv_ref, cv_pred),
            }

            for metric_key, (ref_val, pred_val) in metric_values.items():
                metrics_data[metric_key]["ref"].append(ref_val)
                metrics_data[metric_key]["pred"].append(pred_val)
                metrics_data[metric_key]["points"].append(
                    {
                        "id": mp_id,
                        "label": mp_id,
                        "ref": ref_val,
                        "pred": pred_val,
                        "data_paths": data_paths,
                    }
                )

            # Calculate band errors - skip entire system if band shapes don't match
            # This matches the old method which treats each material equally
            band_abs_diffs = []
            skip_system = False

            for p, r in zip(
                pred_band["frequencies"], ref_band["frequencies"], strict=False
            ):
                len_p = len(p)
                len_r = len(r)
                min_len = min(len_p, len_r)

                p_arr = np.array(p[:min_len])
                r_arr = np.array(r[:min_len])

                try:
                    # Skip entire system if band shapes don't match
                    # This matches old behavior of treating each material equally
                    if (
                        p_arr.ndim == 2
                        and r_arr.ndim == 2
                        and p_arr.shape[1] != r_arr.shape[1]
                    ):
                        print(
                            f"  Skipping {mp_id} due to band mismatch: "
                            f"{p_arr.shape} vs {r_arr.shape}"
                        )
                        skip_system = True
                        break

                    abs_diff = np.abs(p_arr - r_arr)
                    band_abs_diffs.append(abs_diff)

                except ValueError as e:
                    # Skip entire system on ValueError (matching old behavior)
                    print(f"  Skipping {mp_id} due to ValueError: {e}")
                    skip_system = True
                    break

            # If system was skipped, don't include it in BZ MAE calculation
            if skip_system:
                LOGGER.warning(f"No valid band differences for {mp_id}/{model_name}")
                band_errors[mp_id] = float("nan")
                continue  # Skip to next system without adding to system_mean_errors

            # Calculate mean for this system and store for mean-of-means
            if band_abs_diffs:
                stacked = np.concatenate([arr.ravel() for arr in band_abs_diffs])
                valid = stacked[np.isfinite(stacked)]
                if valid.size:
                    system_mean = float(np.nanmean(np.abs(valid)))
                    band_errors[mp_id] = system_mean
                    system_mean_errors.append(system_mean)  # For mean-of-means
                else:
                    band_errors[mp_id] = float("nan")
                del stacked, band_abs_diffs  # Free memory immediately
            else:
                LOGGER.warning(f"No valid band differences for {mp_id}/{model_name}")
                band_errors[mp_id] = float("nan")

            # Stability classification
            stability_points.append(
                {
                    "id": mp_id,
                    "label": mp_id,
                    "ref": min_freq_ref,
                    "pred": min_freq_pred,
                    "class": _classify_stability(min_freq_ref, min_freq_pred),
                    "data_paths": data_paths,
                }
            )

        # Calculate MAEs
        for metric_key in METRIC_LABELS:
            ref_vals = metrics_data[metric_key]["ref"]
            pred_vals = metrics_data[metric_key]["pred"]
            metrics_data[metric_key]["mae"] = (
                mae(ref_vals, pred_vals) if ref_vals and pred_vals else None
            )

        # BZ mean error - mean-of-means (treats each material equally)
        # This matches old method: mean(mean_per_system)
        if system_mean_errors:
            bz_mean = float(np.mean(system_mean_errors))
        else:
            bz_mean = None

        # Stability statistics
        stability_f1, confusion = _stability_statistics(stability_points)

        stats[model_name] = {
            "model": model_name,
            "metrics": metrics_data,
            "band_errors": band_errors,
            "bz_mean": bz_mean,
            "stability": {
                "points": stability_points,
                "f1": stability_f1,
                "confusion": confusion,
            },
        }

        print(f"Completed {model_name}: {processed_count} processed")
        LOGGER.info(
            f"Completed {model_name}: {len(metrics_data['max_freq']['ref'])} systems"
        )

    return stats


@pytest.fixture
@build_table(
    filename=OUT_PATH / "phonon_metrics_table.json",
    thresholds=DEFAULT_THRESHOLDS,
    metric_tooltips=DEFAULT_TOOLTIPS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    phonon_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """
    Build phonon benchmark metrics table.

    Parameters
    ----------
    phonon_stats
        Aggregated statistics per model from ``phonon_stats``.

    Returns
    -------
    dict[str, dict[str, float | None]]
        Mapping of metric label to model-value pairs consumed by Dash tables.
    """

    def _metric_value(model: str, metric_key: str) -> float | None:
        """
        Return MAE for a given model/metric combination.

        Parameters
        ----------
        model
            Display name of the model.
        metric_key
            Internal metric key (e.g. ``"max_freq"``).

        Returns
        -------
        float | None
            MAE value or ``None`` when data is missing.
        """
        model_data = phonon_stats.get(model)
        if not model_data:
            return None
        return model_data["metrics"][metric_key]["mae"]

    table_data: dict[str, dict[str, float | None]] = {}
    for metric_key, label in METRIC_LABELS.items():
        table_data[label] = {
            model: _metric_value(model, metric_key) for model in MODELS
        }

    table_data[BZ_COLUMN] = {
        model: phonon_stats.get(model, {}).get("bz_mean") for model in MODELS
    }
    table_data[STABILITY_COLUMN] = {
        model: phonon_stats.get(model, {}).get("stability", {}).get("f1")
        for model in MODELS
    }
    return table_data


@pytest.fixture
@table_scatter_png(filename=SCATTER_FILENAME)
def interactive_payload(phonon_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Serialise scatter metadata for the Dash app.

    Parameters
    ----------
    phonon_stats
        Aggregated statistics per model from ``phonon_stats``.

    Returns
    -------
    dict[str, Any]
        JSON-serialisable payload containing scatter points, MAEs, and
        stability metadata.
    """
    payload = {
        "metrics": METRIC_LABELS,
        "bz_column": BZ_COLUMN,
        "stability_column": STABILITY_COLUMN,
        "stability_threshold": STABILITY_THRESHOLD,
        "models": {},
    }

    for model_name, model_data in phonon_stats.items():
        payload["models"][model_name] = {
            "metrics": {},
            "band_errors": model_data["band_errors"],
            "bz_mean": model_data["bz_mean"],
            "stability": model_data["stability"],
        }
        for metric_key in METRIC_LABELS:
            payload["models"][model_name]["metrics"][metric_key] = {
                "points": model_data["metrics"][metric_key]["points"],
                "mae": model_data["metrics"][metric_key]["mae"],
            }

    return payload


def test_phonons(metrics, interactive_payload) -> None:
    """
    Exercise the phonon fixtures to ensure they build without errors.

    Parameters
    ----------
    metrics
        Metrics dictionary produced by the ``metrics`` fixture.
    interactive_payload
        Scatter metadata produced by the ``interactive_payload`` fixture.
    """
    return
