"""Analyse diamond phonon benchmark (band structure + thermal properties)."""

from __future__ import annotations

import json
from pathlib import Path
import pickle
import shutil
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import get_struct_info, load_metrics_config, mae, rmse
from ml_peg.app import APP_ROOT
from ml_peg.app.bulk_crystal.phonons.interactive_helpers import _build_xticks
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

MODELS = get_model_names(current_models)

CALC_PATH = CALCS_ROOT / "bulk_crystal" / "diamond_phonons" / "outputs"
REF_PATH = CALC_PATH / "DFT"
OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "diamond_phonons"

SCATTER_FILENAME = OUT_PATH / "diamond_phonons_bands_interactive.json"

METRIC_KEY_MAE = "band_mae"
METRIC_KEY_RMSE = "band_rmse"

METRIC_LABEL_MAE = "Band MAE"
METRIC_LABEL_RMSE = "Band RMSE"
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


def _plot_dispersion(
    ref_band: dict[str, Any], pred_band: dict[str, Any], model_name: str, out_png: Path
) -> None:
    """
    Render the reference vs predicted dispersion comparison to a PNG.

    Parameters
    ----------
    ref_band
        Reference band-structure dict (with labels and path connections).
    pred_band
        Model band-structure dict.
    model_name
        Model name used for the legend.
    out_png
        Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for band, colour, style, label in (
        (pred_band, "red", "--", model_name),
        (ref_band, "blue", "-", "RSCAN"),
    ):
        for i, (dist, freqs) in enumerate(
            zip(band["distances"], band["frequencies"], strict=True)
        ):
            ax.plot(
                dist,
                np.asarray(freqs),
                lw=1,
                linestyle=style,
                color=colour,
                label=label if i == 0 else None,
            )

    xticks, xticklabels = _build_xticks(
        ref_band["distances"], ref_band["labels"], ref_band["path_connections"]
    )
    for x_val in xticks:
        ax.axvline(x=x_val, color="k", linewidth=1)
    ax.set_xticks(xticks, xticklabels)
    ax.set_xlim(xticks[0], xticks[-1])
    ax.axhline(0, color="k", linewidth=1)
    ax.set_ylabel("Frequency (THz)", fontsize=14)
    ax.set_xlabel("Wave Vector", fontsize=14)
    ax.grid(True, linestyle=":", linewidth=0.5)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles, strict=False))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


@pytest.fixture
def diamond_stats() -> dict[str, dict[str, Any]]:
    """
    Aggregate diamond benchmark statistics per model.

    Returns
    -------
    dict[str, dict[str, Any]]
        Mapping of model name to band errors, parity points, data paths, and
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

    stats: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        model_dir = CALC_PATH / model_name
        pred_band_path = model_dir / "diamond_band_structure.npz"
        pred_band = _load_pickle(pred_band_path)
        pred_freqs = (
            np.vstack([np.asarray(seg) for seg in pred_band["frequencies"]])
            if pred_band is not None
            else None
        )

        band_errors: dict[str, float | None] = {"mae": None, "rmse": None}
        points: list[dict[str, Any]] = []

        if pred_freqs is not None and pred_freqs.shape == ref_freqs.shape:
            pred_flat = pred_freqs.reshape(-1)
            if np.isfinite(pred_flat).all():
                band_errors["mae"] = mae(ref_flat, pred_flat)
                band_errors["rmse"] = rmse(ref_flat, pred_flat)

                # Pre-render the dispersion preview so the app only loads it.
                (OUT_PATH / model_name).mkdir(parents=True, exist_ok=True)
                png_path = OUT_PATH / model_name / "diamond_dispersion.png"
                _plot_dispersion(ref_band, pred_band, model_name, png_path)
                structure_paths = None
                pred_struct_src = model_dir / "diamond.xyz"
                if pred_struct_src.exists() and ref_struct_src.exists():
                    shutil.copy2(pred_struct_src, OUT_PATH / model_name / "diamond.xyz")
                    structure_paths = {
                        "ref": "/assets/bulk_crystal/diamond_phonons/DFT/diamond.xyz",
                        "pred": (
                            "/assets/bulk_crystal/diamond_phonons/"
                            f"{model_name}/diamond.xyz"
                        ),
                    }
                points = [
                    {
                        "id": "diamond",
                        "label": "diamond",
                        "ref": float(ref_val),
                        "pred": float(pred_val),
                    }
                    for pred_val, ref_val in zip(pred_flat, ref_flat, strict=True)
                ]
                # All points belong to the same system; the app resolves a
                # click by id to the first matching point, so the (identical)
                # asset paths only need to be stored once.
                points[0]["image"] = (
                    f"assets/bulk_crystal/diamond_phonons/{model_name}/"
                    "diamond_dispersion.png"
                )
                points[0]["structure_paths"] = structure_paths
        elif pred_freqs is not None:
            print(
                f"{model_name}: band shape mismatch "
                f"{pred_freqs.shape} vs {ref_freqs.shape}, skipping."
            )

        thermal_errors: dict[str, float | None] = {
            "gamma": None,
            "theta_d": None,
            "kappa": None,
        }
        pred_thermal = _load_json(model_dir / "diamond_thermal.json")
        if ref_thermal is not None and pred_thermal is not None:
            thermal_errors = {
                "gamma": abs(pred_thermal["mean_gamma"] - ref_thermal["mean_gamma"]),
                "theta_d": abs(
                    pred_thermal["debye_temperature_K"]
                    - ref_thermal["debye_temperature_K"]
                ),
                "kappa": abs(
                    pred_thermal["kappa_W_per_mK"] - ref_thermal["kappa_W_per_mK"]
                ),
            }

        stats[model_name] = {
            "band_errors": band_errors,
            "thermal_errors": thermal_errors,
            "points": points,
        }

    return stats


@pytest.fixture
@build_table(
    filename=OUT_PATH / "diamond_phonons_bands_table.json",
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

    def _value(model: str, group: str, key: str) -> float | None:
        """
        Return one error value for a model, or None when unavailable.

        Parameters
        ----------
        model
            Model name.
        group
            Error group key (``"band_errors"`` or ``"thermal_errors"``).
        key
            Error key within the group.

        Returns
        -------
        float | None
            Error value or ``None`` when data is missing.
        """
        model_data = diamond_stats.get(model)
        if not model_data:
            return None
        return model_data[group].get(key)

    return {
        METRIC_LABEL_MAE: {m: _value(m, "band_errors", "mae") for m in MODELS},
        METRIC_LABEL_RMSE: {m: _value(m, "band_errors", "rmse") for m in MODELS},
        METRIC_LABEL_GAMMA: {m: _value(m, "thermal_errors", "gamma") for m in MODELS},
        METRIC_LABEL_THETA_D: {
            m: _value(m, "thermal_errors", "theta_d") for m in MODELS
        },
        METRIC_LABEL_KAPPA: {m: _value(m, "thermal_errors", "kappa") for m in MODELS},
    }


@pytest.fixture
@cell_to_scatter(
    filename=SCATTER_FILENAME,
    x_label="Predicted frequency (THz)",
    y_label="DFT frequency (THz)",
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
            METRIC_KEY_RMSE: METRIC_LABEL_RMSE,
        },
        "models": {},
    }

    for model_name, model_data in diamond_stats.items():
        if not model_data["points"]:
            continue
        dataset["models"][model_name] = {
            "metrics": {
                METRIC_KEY_MAE: {
                    "points": model_data["points"],
                    "mae": model_data["band_errors"]["mae"],
                },
                METRIC_KEY_RMSE: {
                    "points": model_data["points"],
                    "rmse": model_data["band_errors"]["rmse"],
                },
            },
        }

    return dataset


def test_diamond_phonons_analysis(
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

    table_path = OUT_PATH / "diamond_phonons_bands_table.json"
    assert table_path.exists()
    assert SCATTER_FILENAME.exists()
