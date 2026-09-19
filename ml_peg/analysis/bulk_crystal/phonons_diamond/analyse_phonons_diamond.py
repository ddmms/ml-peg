"""Analyse diamond phonon benchmark (band structure + thermal properties)."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil
from typing import Any

import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "bulk_crystal" / "phonons_diamond" / "outputs"
REF_PATH = CALC_PATH / "DFT"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "phonons_diamond"

SCATTER_FILENAME = OUT_PATH / "phonons_diamond_bands_interactive.json"

METRIC_KEY_MAE = "band_mae"
METRIC_KEY_GAMMA = "gamma"
METRIC_KEY_THETA_D = "theta_d"
METRIC_KEY_KAPPA = "kappa"

METRIC_LABEL_MAE = "Band MAE"
METRIC_LABEL_GAMMA = "Δγ"
METRIC_LABEL_THETA_D = "Δθ_D (K)"
METRIC_LABEL_KAPPA = "Δκ_L (W/m/K)"

METRICS_YML = Path(__file__).with_name("metrics.yml")
THRESHOLDS, METRIC_TOOLTIPS, WEIGHTS = load_metrics_config(METRICS_YML)

INFO = get_struct_info(
    calc_path=CALC_PATH,
    glob_pattern="*.xyz",
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


@pytest.fixture
def diamond_stats() -> dict[str, dict[str, Any]]:
    """
    Aggregate diamond benchmark statistics per model.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of model name to band MAE, parity points, data paths, and
        thermal property errors.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)

    ref_band_path = REF_PATH / "diamond_band_structure.npz"
    ref_band = _load_pickle(ref_band_path)
    if ref_band is None:
        print(f"ERROR: DFT reference not found at {ref_band_path}")
        return {}
    # Reference and phonopy bands are both frequency-sorted per q-point, so
    # modes can be compared by position without branch-labelling ambiguity.
    ref_freqs = np.vstack([np.asarray(seg) for seg in ref_band["frequencies"]])
    ref_flat = ref_freqs.reshape(-1)

    ref_thermal = _load_json(REF_PATH / "diamond_thermal.json")

    # Copy the DFT structure for the app's structure viewer.
    ref_struct_src = REF_PATH / "diamond.xyz"
    if ref_struct_src.exists():
        (OUT_PATH / "DFT").mkdir(parents=True, exist_ok=True)
        shutil.copy2(ref_struct_src, OUT_PATH / "DFT" / "diamond.xyz")

    ref_dos_path = REF_PATH / "diamond_dos.npz"

    stats: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        pred_band_path = model_dir / "diamond_band_structure.npz"
        pred_dos_path = model_dir / "diamond_dos.npz"

        # The app renders band + DOS on the fly from these files via the
        # shared phonon helpers, as in the Ti64 benchmark.
        data_paths = {
            "ref_band": str(ref_band_path.relative_to(CALC_PATH.parent)),
            "ref_dos": str(ref_dos_path.relative_to(CALC_PATH.parent)),
            "pred_band": str(pred_band_path.relative_to(CALC_PATH.parent)),
            "pred_dos": str(pred_dos_path.relative_to(CALC_PATH.parent)),
        }
        pred_band = _load_pickle(pred_band_path)
        pred_segments = (
            [np.asarray(seg) for seg in pred_band["frequencies"]]
            if pred_band is not None
            else None
        )
        pred_freqs = np.vstack(pred_segments) if pred_segments is not None else None

        band_mae: float | None = None
        points: list[dict[str, Any]] = []
        structure_paths = None

        if pred_freqs is not None and pred_freqs.shape == ref_freqs.shape:
            pred_flat = pred_freqs.reshape(-1)
            if np.isfinite(pred_flat).all():
                band_mae = mae(ref_flat, pred_flat)

                pred_struct_src = model_dir / "diamond.xyz"
                if pred_struct_src.exists() and ref_struct_src.exists():
                    (OUT_PATH / model_name).mkdir(parents=True, exist_ok=True)
                    shutil.copy2(pred_struct_src, OUT_PATH / model_name / "diamond.xyz")
                    structure_paths = {
                        "ref": "/assets/bulk_crystal/phonons_diamond/DFT/diamond.xyz",
                        "pred": (
                            "/assets/bulk_crystal/phonons_diamond/"
                            f"{model_name}/diamond.xyz"
                        ),
                    }
                path_labels = [
                    str(label).replace("$", "").replace("\\Gamma", "Γ")
                    for label in ref_band.get("labels", [])
                ]
                for segment_idx, (ref_segment, pred_segment) in enumerate(
                    zip(ref_band["frequencies"], pred_segments, strict=True)
                ):
                    segment_name = (
                        f"{path_labels[segment_idx]} → {path_labels[segment_idx + 1]}"
                        if segment_idx + 1 < len(path_labels)
                        else f"Segment {segment_idx + 1}"
                    )
                    for q_idx, (ref_modes, pred_modes) in enumerate(
                        zip(ref_segment, pred_segment, strict=True)
                    ):
                        for branch_idx, (ref_val, pred_val) in enumerate(
                            zip(ref_modes, pred_modes, strict=True)
                        ):
                            points.append(
                                {
                                    "id": (
                                        f"{segment_name}, q-point {q_idx + 1}/"
                                        f"{len(ref_segment)}, branch {branch_idx + 1}"
                                    ),
                                    "ref": float(ref_val),
                                    "pred": float(pred_val),
                                }
                            )
        elif pred_freqs is not None:
            print(
                f"{model_name}: band shape mismatch "
                f"{pred_freqs.shape} vs {ref_freqs.shape}, skipping."
            )

        thermal_comparisons: dict[str, dict[str, float]] = {}
        pred_thermal = _load_json(model_dir / "diamond_thermal.json")
        if ref_thermal is not None and pred_thermal is not None:
            thermal_fields = {
                METRIC_KEY_GAMMA: "mean_gamma",
                METRIC_KEY_THETA_D: "debye_temperature_K",
                METRIC_KEY_KAPPA: "kappa_W_per_mK",
            }
            for metric_key, field in thermal_fields.items():
                ref_value = float(ref_thermal[field])
                pred_value = float(pred_thermal[field])
                if np.isfinite([ref_value, pred_value]).all():
                    thermal_comparisons[metric_key] = {
                        "ref": ref_value,
                        "pred": pred_value,
                        "error": abs(pred_value - ref_value),
                    }

        stats[model_name] = {
            "band_mae": band_mae,
            "thermal_comparisons": thermal_comparisons,
            "points": points,
            "data_paths": data_paths,
            "structure_paths": structure_paths,
        }

    return stats


@pytest.fixture
@build_table(
    filename=OUT_PATH / "phonons_diamond_bands_table.json",
    thresholds=THRESHOLDS,
    metric_tooltips=METRIC_TOOLTIPS,
    weights=WEIGHTS,
)
def metrics(
    diamond_stats: dict[str, dict[str, Any]],
) -> dict[str, dict[str, float | None]]:
    """
    Build the metrics table mapping for the Dash table.

    Parameters
    ----------
    diamond_stats
        Per-model statistics from :func:`diamond_stats`.

    Returns
    -------
    dict[str, dict[str, float | None]]
        Mapping from visible metric label to per-model values.
    """

    def _thermal_value(model: str, key: str) -> float | None:
        """
        Return one error value for a model, or None when unavailable.

        Parameters
        ----------
        model
            Model name.
        key
            Thermal error key.

        Returns
        -------
        float | None
            Error value or ``None`` when data is missing.
        """
        model_data = diamond_stats.get(model)
        if not model_data:
            return None
        comparison = model_data["thermal_comparisons"].get(key)
        return comparison["error"] if comparison else None

    return {
        METRIC_LABEL_MAE: {m: diamond_stats.get(m, {}).get("band_mae") for m in MODELS},
        METRIC_LABEL_GAMMA: {m: _thermal_value(m, "gamma") for m in MODELS},
        METRIC_LABEL_THETA_D: {m: _thermal_value(m, "theta_d") for m in MODELS},
        METRIC_LABEL_KAPPA: {m: _thermal_value(m, "kappa") for m in MODELS},
    }


@pytest.fixture
@cell_to_scatter(
    filename=SCATTER_FILENAME,
    x_label="Predicted frequency (THz)",
    y_label="RSCAN frequency (THz)",
)
def interactive_dataset(diamond_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Build the interactive scatter dataset for the diamond phonon Dash app.

    Parameters
    ----------
    diamond_stats
        Per-model statistics from :func:`diamond_stats`.

    Returns
    -------
    dict[str, Any]
        Interactive dataset payload written to JSON by the decorator.
    """
    dataset: dict[str, Any] = {
        "metrics": {
            METRIC_KEY_MAE: METRIC_LABEL_MAE,
            METRIC_KEY_GAMMA: METRIC_LABEL_GAMMA,
            METRIC_KEY_THETA_D: METRIC_LABEL_THETA_D,
            METRIC_KEY_KAPPA: METRIC_LABEL_KAPPA,
        },
        "models": {},
    }

    for model_name, model_data in diamond_stats.items():
        metrics_data: dict[str, Any] = {}
        if model_data["points"]:
            metrics_data[METRIC_KEY_MAE] = {
                "points": model_data["points"],
                "mae": model_data["band_mae"],
            }
        scalar_metrics = model_data["thermal_comparisons"]
        if not metrics_data and not scalar_metrics:
            continue
        dataset["models"][model_name] = {
            "metrics": metrics_data,
            "scalar_metrics": scalar_metrics,
            "data_paths": model_data["data_paths"],
            "structure_paths": model_data["structure_paths"],
        }

    return dataset


def test_phonons_diamond_analysis(
    metrics: dict[str, Any],
    interactive_dataset: dict[str, Any],
) -> None:
    """
    Generate JSON artifacts for the diamond phonons benchmark.

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
        assert set(model_data["figures"]) <= {METRIC_KEY_MAE}
        assert set(model_data["scalar_metrics"]) <= {
            METRIC_KEY_GAMMA,
            METRIC_KEY_THETA_D,
            METRIC_KEY_KAPPA,
        }

    table_path = OUT_PATH / "phonons_diamond_bands_table.json"
    assert table_path.exists()
    assert SCATTER_FILENAME.exists()
