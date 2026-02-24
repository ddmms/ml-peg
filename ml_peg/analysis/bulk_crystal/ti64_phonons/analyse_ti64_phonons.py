"""Analyse Ti64 phonons benchmark."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ml_peg.analysis.utils.decorators import build_table, cell_to_scatter
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import load_models
from ml_peg.models.models import current_models

THIS_DIR = Path(__file__).resolve().parent

CALC_OUT_PATH = CALCS_ROOT / "bulk_crystal" / "ti64_phonons" / "outputs"
APP_OUT_PATH = APP_ROOT / "data" / "bulk_crystal" / "ti64_phonons"

METRICS_YAML_PATH = THIS_DIR / "metrics.yml"
SCATTER_FILENAME = APP_OUT_PATH / "ti64_phonons_interactive.json"

TP_ON: set[str] = {
    "hcp_Ti6AlV",
    "hex_Ti8AlV",
    "hcp_Ti6Al2",
    "hcp_Ti6V2",
    "hcp_Ti7V",
    "hex_Ti10Al2",
    "hex_Ti10V2",
}

CASES: list[str] = [
    "hcp_Ti6AlV",
    "bcc_Ti6AlV",
    "hex_Ti8AlV",
    "hcp_Ti6Al2",
    "hcp_Ti6V2",
    "hcp_Ti7V",
    "bcc_Ti6Al2",
    "bcc_Ti6V2",
    "hex_Ti10Al2",
    "hex_Ti10V2",
]


THRESHOLDS, METRIC_TOOLTIPS, WEIGHTS = load_metrics_config(METRICS_YAML_PATH)

METRIC_ID_TO_LABEL: dict[str, str] = {
    "dispersion_rmse_thz_avg": "Dispersion RMSE (mean)",
    "dispersion_rmse_thz_max": "Dispersion RMSE (max)",
    "deltaF_0K_eV_per_atom_avg": "ΔF (0 K) mean",
    "deltaF_2000K_eV_per_atom_avg": "ΔF (2000 K) mean",
    "omega_avg_thz_mae": "ω_avg MAE",
}

TABLE_METRIC_LABELS: list[str] = list(METRIC_ID_TO_LABEL.values())
METRIC_LABELS: dict[str, str] = dict(METRIC_ID_TO_LABEL)  # id -> label


MODELS = load_models(current_models)
MODEL_ITEMS = list(MODELS.items())
MODEL_IDS: list[str] = [name for name, _ in MODEL_ITEMS]


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute root mean squared error.

    Parameters
    ----------
    a
        First array of values.
    b
        Second array of values. Must be broadcast-compatible with ``a``.

    Returns
    -------
    float
        Root mean squared error.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def resample_dft_to_ml_grid(
    dft_x: np.ndarray, dft_freqs: np.ndarray, n_ml: int
) -> np.ndarray:
    """
    Resample DFT frequencies onto an inferred ML grid spanning the same path.

    Parameters
    ----------
    dft_x
        DFT path coordinate array of shape ``(n_dft,)``.
    dft_freqs
        DFT frequencies array of shape ``(n_dft, n_branches)``.
    n_ml
        Number of ML q-points.

    Returns
    -------
    numpy.ndarray
        DFT frequencies interpolated onto the ML grid, shape
        ``(n_ml, n_branches)``.
    """
    dft_x = np.asarray(dft_x, dtype=float)
    dft_freqs = np.asarray(dft_freqs, dtype=float)

    ml_x = np.linspace(dft_x[0], dft_x[-1], n_ml, dtype=float)

    out = np.empty((n_ml, dft_freqs.shape[1]), dtype=float)
    for j in range(dft_freqs.shape[1]):
        out[:, j] = np.interp(ml_x, dft_x, dft_freqs[:, j])
    return out


def write_json(path: Path, obj: dict[str, Any]) -> None:
    """
    Write a JSON object to disk.

    Parameters
    ----------
    path
        Output file path.
    obj
        JSON-serialisable mapping.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf8")


k_b = 8.617333262e-5  # eV/K
hbar = 6.582119569e-16  # eV*s


def zp_energy(weights_flat: np.ndarray, freqs_flat_thz: np.ndarray) -> float:
    """
    Compute zero-point energy from phonon frequencies.

    Parameters
    ----------
    weights_flat
        Flattened q-point weights (tiled over branches), shape ``(n_modes,)``.
    freqs_flat_thz
        Flattened frequencies in THz, shape ``(n_modes,)``.

    Returns
    -------
    float
        Zero-point energy contribution (eV).
    """
    w = np.asarray(weights_flat, dtype=float).reshape(-1)
    f_thz = np.asarray(freqs_flat_thz, dtype=float).reshape(-1)

    omega = f_thz * 2.0 * np.pi * 1e12  # rad/s
    zpe = w * (hbar * omega)
    zpe = zpe[np.isfinite(zpe)]
    zpe[zpe == float("-inf")] = 0.0
    zpe[zpe == float("+inf")] = 0.0
    return float(0.5 * np.sum(zpe))


def helmholtz_free_energy_og(
    weights_flat: np.ndarray, freqs_flat_thz: np.ndarray, t: float
) -> float:
    """
    Compute Helmholtz free energy.

    Parameters
    ----------
    weights_flat
        Flattened q-point weights (tiled over branches), shape ``(n_modes,)``.
    freqs_flat_thz
        Flattened frequencies in THz, shape ``(n_modes,)``.
    t
        Temperature (K).

    Returns
    -------
    float
        Helmholtz free-energy thermal contribution (eV).
    """
    w = np.asarray(weights_flat, dtype=float).reshape(-1)
    f_thz = np.asarray(freqs_flat_thz, dtype=float).reshape(-1)

    if t <= 1e-12:
        t = 1e-12

    omega = f_thz * 2.0 * np.pi * 1e12  # rad/s
    arg = -(hbar * omega) / (k_b * t)

    integrand = w * np.log(1.0 - np.exp(arg))
    integrand = integrand[np.isfinite(integrand)]
    integrand[integrand == float("-inf")] = 0.0
    integrand[integrand == float("+inf")] = 0.0
    return float(np.sum(integrand) * k_b * t)


def analyse_one_model(model_id: str) -> None:
    """
    Analyse a single Ti64 phonon model and write per-model metrics.

    Parameters
    ----------
    model_id
        Identifier of the model under
        ``ml_peg/calcs/bulk_crystal/ti64_phonons/outputs``.

    Notes
    -----
    Writes per-model metrics to:

    - ``ml_peg/app/data/bulk_crystal/ti64_phonons/<model_id>/metrics.json``

    The JSON file contains aggregated metrics and per-case values.
    """
    model_calc_dir = CALC_OUT_PATH / model_id
    assert model_calc_dir.exists(), (
        f"No calc outputs found for model '{model_id}'. Expected:\n"
        f"  {model_calc_dir}\n\n"
        "Calc stage writes to:\n"
        "  ml_peg/calcs/bulk_crystal/ti64_phonons/outputs/<model>/...\n"
    )

    model_app_dir = APP_OUT_PATH / model_id
    model_app_dir.mkdir(parents=True, exist_ok=True)

    rmse_by_case: dict[str, float] = {}
    df0_by_case: dict[str, float] = {}
    df2000_by_case: dict[str, float] = {}

    omega_avg_ref_thz_by_case: dict[str, float] = {}
    omega_avg_pred_thz_by_case: dict[str, float] = {}

    for case in CASES:
        npz_path = model_calc_dir / f"{case}.npz"
        assert npz_path.exists(), f"Missing {npz_path}"

        data = np.load(npz_path, allow_pickle=True)

        dft_x = np.asarray(data["dft_x"], dtype=float)
        dft_freq = np.asarray(data["dft_frequencies"], dtype=float)
        ml_freq = np.asarray(data["ml_frequencies"], dtype=float)

        dft_on_ml = resample_dft_to_ml_grid(dft_x, dft_freq, n_ml=ml_freq.shape[0])

        omega_avg_ref = float(np.mean(dft_on_ml))
        omega_avg_pred = float(np.mean(ml_freq))
        omega_avg_ref_thz_by_case[case] = omega_avg_ref
        omega_avg_pred_thz_by_case[case] = omega_avg_pred

        rmse_by_case[case] = rmse(dft_on_ml, ml_freq)

        if case in TP_ON:
            assert "tp_temperatures" in data and "tp_free_energy" in data, (
                f"TP expected for {case} but tp_* arrays missing in {npz_path}"
            )

            required = ["q_weights", "q_frequencies_dft", "n_atoms"]
            missing = [k for k in required if k not in data.files]
            assert not missing, (
                f"{case}: missing required thermo keys in {npz_path}: {missing}"
            )

            q_w = np.asarray(data["q_weights"], dtype=float)
            q_f = np.asarray(data["q_frequencies_dft"], dtype=float)

            weights_tile = np.tile(q_w[:, None], (1, q_f.shape[1])).reshape(
                -1, order="F"
            )
            freqs_flat = q_f.reshape(-1, order="F")  # THz

            n_atoms = int(np.asarray(data["n_atoms"]).item())

            ml_t = np.asarray(data["tp_temperatures"], dtype=float)
            ml_f = np.asarray(data["tp_free_energy"], dtype=float)

            # Legacy unit fix (keep existing behavior)
            if np.nanmax(np.abs(ml_f)) > 100.0:
                ml_f = ml_f / 96.32

            t_dense = np.linspace(0.0, 2000.0, 2000, dtype=float)
            zpe_const = zp_energy(weights_tile, freqs_flat)
            dft_f_dense = np.array(
                [
                    helmholtz_free_energy_og(weights_tile, freqs_flat, tt) + zpe_const
                    for tt in t_dense
                ],
                dtype=float,
            )
            dft_f_on_mlt = np.interp(ml_t, t_dense, dft_f_dense)

            df0_by_case[case] = float(np.abs(dft_f_on_mlt[0] - ml_f[0]) / n_atoms)
            df2000_by_case[case] = float(np.abs(dft_f_on_mlt[-1] - ml_f[-1]) / n_atoms)

    rmse_vals = np.asarray(list(rmse_by_case.values()), dtype=float)

    omega_avg_mae = (
        float(
            np.mean(
                [
                    abs(omega_avg_pred_thz_by_case[c] - omega_avg_ref_thz_by_case[c])
                    for c in omega_avg_ref_thz_by_case
                ]
            )
        )
        if omega_avg_ref_thz_by_case
        else None
    )

    write_json(
        model_app_dir / "metrics.json",
        {
            "model": model_id,
            "n_cases": len(CASES),
            "metrics": {
                "dispersion_rmse_thz_avg": float(np.mean(rmse_vals)),
                "dispersion_rmse_thz_max": float(np.max(rmse_vals)),
                "deltaF_0K_eV_per_atom_avg": float(np.mean(list(df0_by_case.values())))
                if df0_by_case
                else None,
                "deltaF_2000K_eV_per_atom_avg": float(
                    np.mean(list(df2000_by_case.values()))
                )
                if df2000_by_case
                else None,
                "omega_avg_thz_mae": omega_avg_mae,
            },
            "by_case": {
                "rmse_thz": rmse_by_case,
                "deltaF_0K_eV_per_atom": df0_by_case,
                "deltaF_2000K_eV_per_atom": df2000_by_case,
                "omega_avg_ref_thz": omega_avg_ref_thz_by_case,
                "omega_avg_pred_thz": omega_avg_pred_thz_by_case,
            },
        },
    )

    assert len(rmse_by_case) == len(CASES)


@pytest.fixture(scope="session")
def run_all_models() -> None:
    """
    Generate per-model ``metrics.json`` for all configured models.

    Returns
    -------
    None
        This fixture exists for its side effects (writing per-model metrics).
    """
    for model_id in MODEL_IDS:
        analyse_one_model(model_id)


@pytest.fixture(scope="session")
@build_table(
    filename=APP_OUT_PATH / "ti64_phonons_metrics_table.json",
    thresholds=THRESHOLDS,
    metric_tooltips=METRIC_TOOLTIPS,
    weights=WEIGHTS,
)
def metrics_table(run_all_models: None) -> dict[str, dict[str, float | None]]:
    """
    Build the Ti64 metrics table for the Dash app.

    Parameters
    ----------
    run_all_models
        Session-scoped fixture ensuring per-model metrics are generated.

    Returns
    -------
    dict[str, dict[str, float | None]]
        Mapping of metric label to per-model values.
    """
    _ = run_all_models

    table: dict[str, dict[str, float | None]] = {
        label: {} for label in TABLE_METRIC_LABELS
    }

    for model_id in MODEL_IDS:
        mpath = APP_OUT_PATH / model_id / "metrics.json"
        if not mpath.exists():
            continue

        m = json.loads(mpath.read_text(encoding="utf8"))
        metrics = m.get("metrics", {})

        for metric_id, label in METRIC_ID_TO_LABEL.items():
            table[label][model_id] = metrics.get(metric_id)

    return table


@pytest.fixture(scope="session")
@cell_to_scatter(
    filename=SCATTER_FILENAME,
    x_label="Predicted",
    y_label="Reference",
)
def interactive_dataset(run_all_models: None) -> dict[str, Any]:
    """
    Build the interactive scatter dataset for the Ti64 phonons Dash app.

    Parameters
    ----------
    run_all_models
        Session-scoped fixture ensuring per-model metrics are generated.

    Returns
    -------
    dict[str, Any]
        Interactive dataset written to JSON by the decorator.
    """
    _ = run_all_models

    dataset: dict[str, Any] = {
        "metrics": METRIC_LABELS,  # id -> label
        "models": {},
    }

    metric_id = "omega_avg_thz_mae"

    for model_id in MODEL_IDS:
        metrics_path = APP_OUT_PATH / model_id / "metrics.json"
        if not metrics_path.exists():
            continue

        m = json.loads(metrics_path.read_text(encoding="utf8"))
        by_case = m.get("by_case") or {}
        ref_map = by_case.get("omega_avg_ref_thz", {}) or {}
        pred_map = by_case.get("omega_avg_pred_thz", {}) or {}

        points: list[dict[str, Any]] = []
        for case in CASES:
            if case not in ref_map or case not in pred_map:
                continue

            data_paths = {
                "npz": str(
                    (CALC_OUT_PATH / model_id / f"{case}.npz").relative_to(
                        CALC_OUT_PATH.parent
                    )
                ),
                "meta": str(
                    (CALC_OUT_PATH / model_id / f"{case}.json").relative_to(
                        CALC_OUT_PATH.parent
                    )
                ),
            }

            points.append(
                {
                    "id": case,
                    "label": case,
                    "ref": ref_map[case],
                    "pred": pred_map[case],
                    "data_paths": data_paths,
                }
            )

        dataset["models"][model_id] = {
            "model": model_id,
            "metrics": {
                metric_id: {
                    "points": points,
                    "mae": (m.get("metrics") or {}).get(metric_id),
                }
            },
        }

    return dataset


def test_all_models_metrics_written(run_all_models: None) -> None:
    """
    Check per-model ``metrics.json`` exists for every configured model.

    Parameters
    ----------
    run_all_models
        Session-scoped fixture ensuring per-model metrics are generated.
    """
    _ = run_all_models

    missing: list[str] = []
    for model_id in MODEL_IDS:
        if not (APP_OUT_PATH / model_id / "metrics.json").exists():
            missing.append(model_id)

    assert not missing, f"Missing metrics.json for models: {missing}"


def test_write_metrics_table(metrics_table: dict[str, Any]) -> None:
    """
    Check the table JSON artifact is produced and includes all models.

    Parameters
    ----------
    metrics_table
        Fixture providing the metrics table mapping (and/or triggering JSON writing).
    """
    assert isinstance(metrics_table, dict)

    table_path = APP_OUT_PATH / "ti64_phonons_metrics_table.json"
    assert table_path.exists()

    payload = json.loads(table_path.read_text(encoding="utf8"))
    rows = payload.get("data", [])
    ids = {row.get("id") for row in rows if isinstance(row, dict)}
    missing = [m for m in MODEL_IDS if m not in ids]
    assert not missing, f"Table missing model rows: {missing}"


def test_write_interactive_json(interactive_dataset: dict[str, Any]) -> None:
    """
    Check the interactive JSON artifact is produced and includes all models.

    Parameters
    ----------
    interactive_dataset
        Fixture providing the interactive dataset (and/or triggering JSON writing).
    """
    assert isinstance(interactive_dataset, dict)
    assert SCATTER_FILENAME.exists()

    payload = json.loads(SCATTER_FILENAME.read_text(encoding="utf8"))
    models = payload.get("models", {})
    missing = [m for m in MODEL_IDS if m not in models]
    assert not missing, f"Interactive dataset missing models: {missing}"
