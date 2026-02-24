"""Analyse diamond phonon dispersion benchmark (bands only)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml  # type: ignore

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.utils import download_github_data
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

GITHUB_BASE = "https://raw.githubusercontent.com/7radians/ml-peg-data/main"

EXTRACTED_ROOT = Path(
    download_github_data(
        filename="diamond_data/data.zip",
        github_uri=GITHUB_BASE,
    )
)

CALC_DATA = EXTRACTED_ROOT / "data"


MODELS = get_model_names(current_models)

CATEGORY = "bulk_crystal"
BENCH = "diamond_phonons"

CALC_PATH = CALCS_ROOT / CATEGORY / BENCH / "outputs"

# CALC_DATA = CALCS_ROOT / CATEGORY / BENCH / "data"


OUT_PATH = APP_ROOT / "data" / CATEGORY / BENCH

# cm^-1 per THz (to convert the DFT reference to THz)
THZ_TO_CM1 = 33.35640951981521
NBANDS = 6

SCATTER_FILENAME = OUT_PATH / "diamond_phonons_bands_interactive.json"

METRIC_KEY_MAE = "band_mae"
METRIC_KEY_RMSE = "band_rmse"

METRIC_LABEL_MAE = "Band MAE"
METRIC_LABEL_RMSE = "Band RMSE"


METRICS_YML = Path(__file__).with_name("metrics.yml")
THRESHOLDS, METRIC_TOOLTIPS, WEIGHTS = load_metrics_config(METRICS_YML)

_expected_metric_labels = {METRIC_LABEL_MAE, METRIC_LABEL_RMSE}
_yaml_metric_labels = set(THRESHOLDS.keys())
missing = _expected_metric_labels - _yaml_metric_labels
if missing:
    raise ValueError(
        f"{METRICS_YML}: missing metrics for labels {sorted(missing)}. "
        f"Found: {sorted(_yaml_metric_labels)}"
    )


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

        - ``freqs``: array of shape ``(Nq, NBANDS)`` in THz
        - ``units``: ``"THz"``
        - ``path``: input path
    """
    if not path.exists():
        raise FileNotFoundError(f"Missing DFT reference: {path}")

    d = np.load(path, allow_pickle=False)
    if "frequencies" not in d.files:
        raise KeyError(f"{path}: missing 'frequencies'. Found keys: {list(d.files)}")

    freqs_cm1 = np.asarray(d["frequencies"], dtype=float)

    if freqs_cm1.ndim == 3 and freqs_cm1.shape[0] == 1:
        freqs_cm1 = freqs_cm1[0]

    if freqs_cm1.ndim != 2 or freqs_cm1.shape[1] != NBANDS:
        msg = f"{path}: expected (Nq, {NBANDS}) frequencies, got {freqs_cm1.shape}"
        raise ValueError(msg)

    if not np.isfinite(freqs_cm1).all():
        raise ValueError(f"{path}: contains non-finite reference frequencies.")

    freqs_thz = freqs_cm1 / THZ_TO_CM1

    return {"freqs": freqs_thz, "units": "THz", "path": path}


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
    Compute mean absolute error (THz).

    Parameters
    ----------
    a
        Predicted values (THz), same shape as ``b``.
    b
        Reference values (THz), same shape as ``a``.

    Returns
    -------
    float
        Mean absolute error in THz.
    """
    return float(np.mean(np.abs(a - b)))


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute root mean squared error (THz).

    Parameters
    ----------
    a
        Predicted values (THz), same shape as ``b``.
    b
        Reference values (THz), same shape as ``a``.

    Returns
    -------
    float
        Root mean squared error in THz.
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
        Reference mapping as returned by :func:`_load_reference_npz`.
    """
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    return _load_reference_npz(CALC_DATA / "dft_band.npz")


def _model_flat(model_name: str) -> np.ndarray:
    """
    Load and flatten one model's predicted bands (THz).

    Parameters
    ----------
    model_name
        Model identifier used to locate ``{CALC_PATH}/{model_name}/band.yaml``.

    Returns
    -------
    numpy.ndarray
        Flattened frequencies of shape ``(Nq * NBANDS,)`` in THz.
    """
    pred = _load_band_yaml(CALC_PATH / model_name / "band.yaml")
    return _sorted_flat(np.asarray(pred["freqs"], dtype=float))


@pytest.fixture
def flat_bands(reference: dict[str, Any]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Load and cache flattened reference and predicted bands.

    Parameters
    ----------
    reference
        Reference mapping as returned by :func:`reference`.

    Returns
    -------
    tuple[numpy.ndarray, dict[str, numpy.ndarray]]
        ``(ref_flat, pred_flats)`` where ``ref_flat`` has shape ``(Nq * NBANDS,)``
        and ``pred_flats`` maps model name to an array of the same shape.
    """
    ref_flat = _sorted_flat(np.asarray(reference["freqs"], dtype=float))

    pred_flats: dict[str, np.ndarray] = {}
    for model_name in MODELS:
        pred_flat = _model_flat(model_name)
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
    Compute MAE and RMSE for each model (THz).

    Parameters
    ----------
    flat_bands
        Tuple ``(ref_flat, pred_flats)`` as returned by :func:`flat_bands`.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping ``model_name -> {"mae": float, "rmse": float}`` in THz.
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
    metric_tooltips=METRIC_TOOLTIPS,
    weights=WEIGHTS,
)
def metrics(band_errors: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """
    Build the metrics table mapping for the Dash table.

    Parameters
    ----------
    band_errors
        Per-model MAE/RMSE mapping as returned by :func:`band_errors`.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from visible metric label to per-model values.
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
        Tuple ``(ref_flat, pred_flats)`` as returned by :func:`flat_bands`.
    band_errors
        Per-model MAE/RMSE mapping as returned by :func:`band_errors`.

    Returns
    -------
    dict[str, dict[str, Any]]
        Per-model structures containing points and metric values used to build the
        interactive scatter dataset.
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
    x_label="Predicted frequency (THz)",
    y_label="DFT frequency (THz)",
)
def interactive_dataset(band_stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """
    Build the interactive scatter dataset for the phonon Dash app.

    Parameters
    ----------
    band_stats
        Per-model point/metric structures as returned by :func:`band_stats`.

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


def test_diamond_phonons_analysis(
    metrics: dict[str, Any], interactive_dataset: dict[str, Any]
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

    table_payload = json.loads(table_path.read_text(encoding="utf8"))
    rows = table_payload.get("data", [])
    ids = {row.get("id") for row in rows if isinstance(row, dict)}
    missing_rows = [m for m in MODELS if m not in ids]
    assert not missing_rows, f"Table missing model rows: {missing_rows}"

    assert SCATTER_FILENAME.exists()
    scatter_payload = json.loads(SCATTER_FILENAME.read_text(encoding="utf8"))
    models = scatter_payload.get("models", {})
    missing_models = [m for m in MODELS if m not in models]
    assert not missing_models, f"Interactive dataset missing models: {missing_models}"

    # Ensure each model has both metrics and some points.
    for model_name in MODELS:
        model_metrics = (models.get(model_name) or {}).get("metrics", {})
        for key in (METRIC_KEY_MAE, METRIC_KEY_RMSE):
            assert key in model_metrics, f"{model_name}: missing metric '{key}'"
            points = model_metrics[key].get("points", [])
            assert points, f"{model_name}: empty points for '{key}'"
