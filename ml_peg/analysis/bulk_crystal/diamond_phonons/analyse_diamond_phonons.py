"""
Analyse diamond phonon dispersion benchmark (bands only).

Notes
-----
This analysis produces two JSON artifacts used by the diamond phonon Dash app:

- ``diamond_phonons_bands_table.json``: metrics table for the Dash table component.
- ``diamond_phonons_bands_interactive.json``: interactive scatter dataset consumed by
  the PhononApp callbacks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml  # type: ignore

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)

CATEGORY = "bulk_crystal"
BENCH = "diamond_phonons"

CALC_PATH = CALCS_ROOT / CATEGORY / BENCH / "outputs"
CALC_DATA = CALCS_ROOT / CATEGORY / BENCH / "data"
OUT_PATH = APP_ROOT / "data" / CATEGORY / BENCH

THZ_TO_CM1 = 33.35640951981521
NBANDS = 6

SCATTER_FILENAME = OUT_PATH / "diamond_phonons_bands_interactive.json"

# Internal keys used in the interactive dataset (not shown to user)
METRIC_KEY_MAE = "band_mae"
METRIC_KEY_RMSE = "band_rmse"

# Visible labels (must match the table column headers exactly)
METRIC_LABEL_MAE = "Band MAE"
METRIC_LABEL_RMSE = "Band RMSE"

METRICS_YML = Path(__file__).with_name("metrics.yml")
with METRICS_YML.open("r", encoding="utf8") as f:
    _cfg = yaml.safe_load(f) or {}

_metric_specs = _cfg.get("metrics", {})
if not isinstance(_metric_specs, dict) or not _metric_specs:
    raise ValueError(f"{METRICS_YML} must contain a 'metrics:' mapping.")

THRESHOLDS = {
    name: {
        "good": spec.get("good"),
        "bad": spec.get("bad"),
        "unit": spec.get("unit", "cm-1"),
    }
    for name, spec in _metric_specs.items()
}


def _load_reference_npz(path: Path) -> dict[str, Any]:
    """
    Load DFT reference bands from an NPZ file.

    Parameters
    ----------
    path
        Path to ``dft_band.npz``. Must contain a ``frequencies`` array in cm-1 with
        shape ``(Nq, NBANDS)`` or ``(1, Nq, NBANDS)``.

    Returns
    -------
    dict[str, Any]
        Mapping with keys:

        - ``freqs``: array of shape ``(Nq, NBANDS)`` in cm-1
        - ``units``: ``"cm-1"``
        - ``path``: input path
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing DFT reference: {path}")

    d = np.load(path, allow_pickle=False)
    if "frequencies" not in d.files:
        raise KeyError(f"{path}: missing 'frequencies'. Found keys: {list(d.files)}")

    freqs_cm1 = np.asarray(d["frequencies"], dtype=float)

    # Allow legacy (1, Nq, NBANDS) storage
    if freqs_cm1.ndim == 3 and freqs_cm1.shape[0] == 1:
        freqs_cm1 = freqs_cm1[0]

    if freqs_cm1.ndim != 2 or freqs_cm1.shape[1] != NBANDS:
        msg = f"{path}: expected (Nq, {NBANDS}) frequencies, got {freqs_cm1.shape}"
        raise ValueError(msg)

    if not np.isfinite(freqs_cm1).all():
        raise ValueError(f"{path}: contains non-finite reference frequencies.")

    return {"freqs": freqs_cm1, "units": "cm-1", "path": path}


def _load_band_yaml(path: Path) -> dict[str, Any]:
    """
    Load phonopy ``band.yaml`` and return frequencies in THz.

    Parameters
    ----------
    path
        Path to a phonopy ``band.yaml``.

    Returns
    -------
    dict[str, Any]
        Mapping with keys ``freqs`` (``(Nq, NBANDS)``), ``units`` (``"THz"``),
        and ``path``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing predicted band.yaml: {path}")

    with path.open("r", encoding="utf8") as f:
        y = yaml.safe_load(f)

    phonon = y.get("phonon")
    if not isinstance(phonon, list) or not phonon:
        raise ValueError(
            f"{path} does not look like a phonopy band.yaml (missing 'phonon' list)."
        )

    freqs = np.array(
        [[b.get("frequency", np.nan) for b in p.get("band", [])] for p in phonon],
        dtype=float,
    )

    if freqs.ndim != 2 or freqs.shape[1] != NBANDS:
        raise ValueError(f"{path}: expected (Nq, {NBANDS}) freqs, got {freqs.shape}")

    if not np.isfinite(freqs).all():
        raise ValueError(f"{path}: contains non-finite predicted frequencies.")

    return {"freqs": freqs, "units": "THz", "path": path}


def _convert_units(freqs: np.ndarray, src: str, dst: str) -> np.ndarray:
    """
    Convert between THz and cm-1.

    Parameters
    ----------
    freqs
        Frequency array.
    src
        Source unit (``"THz"`` or ``"cm-1"``).
    dst
        Destination unit (``"THz"`` or ``"cm-1"``).

    Returns
    -------
    numpy.ndarray
        Converted frequencies.
    """
    if src == dst:
        return freqs
    if src == "THz" and dst == "cm-1":
        return freqs * THZ_TO_CM1
    if src == "cm-1" and dst == "THz":
        return freqs / THZ_TO_CM1
    raise ValueError(f"Unsupported unit conversion {src} -> {dst}")


def _sorted_flat(freqs: np.ndarray) -> np.ndarray:
    """
    Sort each q-point's bands then flatten.

    Parameters
    ----------
    freqs
        Array of shape ``(Nq, NBANDS)``.

    Returns
    -------
    numpy.ndarray
        Flattened array of shape ``(Nq * NBANDS,)``.
    """
    if freqs.ndim != 2:
        raise ValueError(f"Expected (Nq, nb). Got {freqs.shape}")
    if freqs.shape[1] != NBANDS:
        raise ValueError(f"Expected {NBANDS} bands, got {freqs.shape[1]}")
    return np.sort(freqs, axis=1).reshape(-1)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute mean absolute error.

    Parameters
    ----------
    a
        Predicted values.
    b
        Reference values.

    Returns
    -------
    float
        Mean absolute error.
    """
    return float(np.mean(np.abs(a - b)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute root mean squared error.

    Parameters
    ----------
    a
        Predicted values.
    b
        Reference values.

    Returns
    -------
    float
        Root mean squared error.
    """
    d = a - b
    return float(np.sqrt(np.mean(d * d)))


@pytest.fixture
def reference() -> dict[str, Any]:
    """
    Load the DFT reference and ensure output directory exists.

    Returns
    -------
    dict[str, Any]
        Reference data mapping returned by :func:`_load_reference_npz`.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    return _load_reference_npz(CALC_DATA / "dft_band.npz")


def _model_flat(model_name: str, ref_units: str) -> np.ndarray:
    """
    Load, unit-convert, and flatten one model's predicted bands.

    Parameters
    ----------
    model_name
        Model name under ``outputs/``.
    ref_units
        Reference unit string.

    Returns
    -------
    numpy.ndarray
        Flattened predicted frequencies.
    """
    pred = _load_band_yaml(CALC_PATH / model_name / "band.yaml")
    freqs = _convert_units(pred["freqs"], pred["units"], ref_units)
    return _sorted_flat(freqs)


@pytest.fixture
def flat_bands(reference: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load and cache flattened reference and predicted bands.

    Parameters
    ----------
    reference
        Reference mapping from the ``reference`` fixture.

    Returns
    -------
    tuple
        ``(ref_flat, pred_flats)`` where ``ref_flat`` is a 1D array and
        ``pred_flats`` maps model name to a 1D array.
    """
    ref_flat = _sorted_flat(reference["freqs"])
    ref_units = reference["units"]

    pred_flats: dict[str, np.ndarray] = {}
    for model_name in MODELS:
        pred_flat = _model_flat(model_name, ref_units)
        if pred_flat.shape != ref_flat.shape:
            raise ValueError(
                f"{model_name}: prediction and reference flattened shapes differ "
                f"{pred_flat.shape} vs {ref_flat.shape}."
            )
        pred_flats[model_name] = pred_flat

    return ref_flat, pred_flats


@pytest.fixture
def band_errors(
    flat_bands: tuple[np.ndarray, dict[str, np.ndarray]],
) -> dict[str, dict[str, float]]:
    """
    Compute MAE and RMSE for each model.

    Parameters
    ----------
    flat_bands
        Output of the ``flat_bands`` fixture.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping ``model -> {"mae": ..., "rmse": ...}``.
    """
    ref_flat, pred_flats = flat_bands

    out: dict[str, dict[str, float]] = {}
    for model_name in MODELS:
        pred_flat = pred_flats[model_name]
        out[model_name] = {
            "mae": _mae(pred_flat, ref_flat),
            "rmse": _rmse(pred_flat, ref_flat),
        }
    return out


@pytest.fixture
@build_table(
    filename=OUT_PATH / "diamond_phonons_bands_table.json",
    thresholds=THRESHOLDS,
    metric_tooltips={
        "Band MAE": "Mean absolute error over all q-points and phonon branches",
        "Band RMSE": "Root mean squared error over all q-points and phonon branches",
    },
)
def metrics(band_errors: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Build the metrics table mapping for the Dash table.

    Parameters
    ----------
    band_errors
        Output of the ``band_errors`` fixture.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from metric label to per-model values.
    """
    return {
        METRIC_LABEL_MAE: {m: band_errors[m]["mae"] for m in MODELS},
        METRIC_LABEL_RMSE: {m: band_errors[m]["rmse"] for m in MODELS},
    }


@pytest.fixture
def band_stats(
    flat_bands: tuple[np.ndarray, dict[str, np.ndarray]],
    band_errors: dict[str, dict[str, float]],
) -> dict[str, dict[str, Any]]:
    """
    Build per-model structures consumed by ``cell_to_scatter``.

    Parameters
    ----------
    flat_bands
        Output of the ``flat_bands`` fixture.
    band_errors
        Output of the ``band_errors`` fixture.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-model mapping containing scatter points and scalar metrics.
    """
    ref_flat, pred_flats = flat_bands

    stats: dict[str, dict[str, Any]] = {}
    for model_name in MODELS:
        pred_flat = pred_flats[model_name]

        points = [
            {
                "id": f"diamond-{i}",
                "label": "diamond",
                "ref": float(ref_val),
                "pred": float(pred_val),
            }
            for i, (pred_val, ref_val) in enumerate(
                zip(pred_flat, ref_flat, strict=True)
            )
        ]

        stats[model_name] = {
            "model": model_name,
            "metrics": {
                METRIC_KEY_MAE: {
                    "points": points,
                    "mae": float(band_errors[model_name]["mae"]),
                },
                METRIC_KEY_RMSE: {
                    "points": points,
                    "rmse": float(band_errors[model_name]["rmse"]),
                },
            },
        }

    return stats


@pytest.fixture
@cell_to_scatter(
    filename=SCATTER_FILENAME,
    x_label="Predicted frequency (cm-1)",
    y_label="DFT frequency (cm-1)",
)
def interactive_dataset(band_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Build the interactive scatter dataset for the phonon Dash app.

    Parameters
    ----------
    band_stats
        Output of the ``band_stats`` fixture.

    Returns
    -------
    dict[str, Any]
        Interactive dataset written to JSON by the decorator.
    """
    dataset: dict[str, Any] = {
        "metrics": {
            METRIC_KEY_MAE: METRIC_LABEL_MAE,
            METRIC_KEY_RMSE: METRIC_LABEL_RMSE,
        },
        "models": {},
    }

    for model_name, model_data in band_stats.items():
        dataset["models"][model_name] = {"metrics": {}}

        dataset["models"][model_name]["metrics"][METRIC_KEY_MAE] = {
            "points": model_data["metrics"][METRIC_KEY_MAE]["points"],
            "mae": model_data["metrics"][METRIC_KEY_MAE]["mae"],
        }

        dataset["models"][model_name]["metrics"][METRIC_KEY_RMSE] = {
            "points": model_data["metrics"][METRIC_KEY_RMSE]["points"],
            "rmse": model_data["metrics"][METRIC_KEY_RMSE]["rmse"],
        }

    return dataset


def test_diamond_phonons_analysis(metrics, interactive_dataset) -> None:
    """
    Generate JSON artifacts for the diamond phonons benchmark.

    Parameters
    ----------
    metrics
        Table metrics fixture output (decorator writes JSON).
    interactive_dataset
        Scatter dataset fixture output (decorator writes JSON).

    Returns
    -------
    None
        This test passes if fixtures execute successfully.
    """
    _ = metrics
    _ = interactive_dataset
