"""Analyse diatomics benchmark."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table, periodic_curve_gallery
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "physicality" / "diatomics" / "outputs"
OUT_PATH = APP_ROOT / "data" / "physicality" / "diatomics"
CURVE_PATH = OUT_PATH / "curves"


METRICS_CONFIG_PATH = Path(__file__).with_name("metrics.yml")
DEFAULT_THRESHOLDS, DEFAULT_TOOLTIPS, DEFAULT_WEIGHTS = load_metrics_config(
    METRICS_CONFIG_PATH
)


def load_model_data(model_name: str) -> pd.DataFrame:
    """
    Load diatomic curve data for a model.

    Parameters
    ----------
    model_name
        Model identifier.

    Returns
    -------
    pd.DataFrame
        Dataframe containing pair, distance, energy, and projected force columns.
    """
    csv_path = CALC_PATH / model_name / "diatomics.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    return pd.read_csv(csv_path)


def prepare_pair_series(
    pair_dataframe: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort and align energy/force series for a diatomic pair.

    Parameters
    ----------
    pair_dataframe
        Pair-specific dataframe.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Distances, shifted energies, and projected forces sorted by decreasing
        distance.
    """
    df_sorted = pair_dataframe.sort_values("distance").drop_duplicates("distance")
    series = df_sorted[["distance", "energy", "force_parallel"]].to_numpy()
    if len(series) < 3:
        return np.array([]), np.array([]), np.array([])

    distances = series[:, 0]
    energies = series[:, 1]
    forces = series[:, 2]

    descending = np.argsort(distances)[::-1]
    distances = distances[descending]
    energies = energies[descending]
    forces = forces[descending]

    shifted_energies = energies - energies[0]
    return distances, shifted_energies, forces


def count_sign_changes(array: np.ndarray, tol: float) -> int:
    """
    Count sign changes in a sequence while ignoring small magnitudes.

    Parameters
    ----------
    array
        Input values.
    tol
        Absolute tolerance below which values are treated as zero.

    Returns
    -------
    int
        Number of sign changes exceeding the specified tolerance.
    """
    if array.size < 3:
        return 0
    clipped = array.copy()
    clipped = clipped[np.abs(clipped) > tol]
    signs = np.sign(clipped)
    sign_flips = signs[:-1] != signs[1:]
    return int(np.sum(sign_flips))


def compute_pair_metrics(
    df_pair: pd.DataFrame,
) -> dict[str, float] | None:
    """
    Compute diagnostics for a single diatomic pair.

    Parameters
    ----------
    df_pair
        Pair-specific dataframe.

    Returns
    -------
    dict[str, float] | None
        Dictionary of metrics, or None if insufficient data.
    """
    distances, shifted_energies, projected_forces = prepare_pair_series(df_pair)
    if distances.size < 3:
        return None

    energy_gradient = np.gradient(shifted_energies, distances)
    energy_curvature = np.gradient(energy_gradient, distances)

    minima = 0
    if shifted_energies.size >= 3:
        second_diff = np.diff(np.sign(np.diff(shifted_energies)))
        minima = int(np.sum(second_diff > 0))

    inflections = count_sign_changes(energy_curvature, tol=0.5)

    # Force flip calculation: count sign changes in projected forces
    # Use tolerance of 0.01 eV/Å to ignore numerical noise
    force_flip_count = count_sign_changes(projected_forces, tol=1e-2)

    spearman_repulsion = None
    spearman_attraction = None

    try:
        from scipy import stats

        well_index = int(np.argmin(shifted_energies))
        if distances[well_index:].size > 1:
            spearman_repulsion = float(
                stats.spearmanr(
                    distances[well_index:], shifted_energies[well_index:]
                ).statistic
            )
        if distances[:well_index].size > 1:
            spearman_attraction = float(
                stats.spearmanr(
                    distances[:well_index], shifted_energies[:well_index]
                ).statistic
            )
    except Exception:
        pass

    return {
        "Force flips": float(force_flip_count),
        "Energy minima": float(minima),
        "Energy inflections": float(inflections),
        "ρ(E, repulsion)": float(spearman_repulsion),
        "ρ(E, attraction)": float(spearman_attraction),
    }


def aggregate_model_metrics(
    model_dataframe: pd.DataFrame,
) -> dict[str, float]:
    """
    Aggregate metrics across all homo/heteronuclear diatomic pairs.

    Parameters
    ----------
    model_dataframe
        Per-model diatomic dataset.

    Returns
    -------
    dict[str, float]
        Aggregated model metrics (averaged across all pairs).
    """
    if model_dataframe.empty:
        return {}

    pair_metrics: list[dict[str, float]] = []

    for _pair, pair_dataframe in model_dataframe.groupby("pair"):
        metrics = compute_pair_metrics(pair_dataframe)
        if metrics is None:
            continue
        pair_metrics.append(metrics)

    if not pair_metrics:
        return {}

    return {
        key: float(np.nanmean([metrics.get(key, np.nan) for metrics in pair_metrics]))
        for key in DEFAULT_THRESHOLDS.keys()
    }


# helper to load per-model pair data -----------------------------------------


def _load_pair_data() -> dict[str, pd.DataFrame]:
    """
    Load per-model diatomic curve dataframes from calculator outputs.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of model name to curve dataframe.
    """
    pair_data: dict[str, pd.DataFrame] = {}
    for model_name in MODELS:
        model_dataframe = load_model_data(model_name)
        if not model_dataframe.empty:
            pair_data[model_name] = model_dataframe
    return pair_data


persist_diatomics_pair_data = periodic_curve_gallery(
    curve_dir=CURVE_PATH,
    periodic_dir=None,
    overview_title=None,
    overview_formats=(),
    focus_title_template=None,
    focus_formats=(),
    series_columns={"force_parallel": "force_parallel"},
    x_ticks=(0.0, 2.0, 4.0, 6.0),
    y_ticks=(-20.0, -10.0, 0.0, 10.0, 20.0),
    x_range=(0.0, 6.0),
    y_range=(-20.0, 20.0),
)(_load_pair_data)


@pytest.fixture
def diatomics_pair_data_fixture() -> dict[str, pd.DataFrame]:
    """
    Load curve data and persist gallery assets for pytest use.

    Returns
    -------
    dict[str, pd.DataFrame]
        Mapping of model name to per-pair curve data.
    """
    return persist_diatomics_pair_data()


def collect_metrics(
    pair_data: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame:
    """
    Gather metrics for all models.

    Metrics are averaged across all diatomic pairs (both homonuclear and heteronuclear).

    Parameters
    ----------
    pair_data
        Optional mapping of model names to curve dataframes. When ``None``,
        the data is loaded via ``persist_diatomics_pair_data``.

    Returns
    -------
    pd.DataFrame
        Aggregated metrics table (all pairs).
    """
    metrics_rows: list[dict[str, float | str]] = []

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    data = pair_data if pair_data is not None else persist_diatomics_pair_data()

    for model_name, model_dataframe in data.items():
        metrics = aggregate_model_metrics(model_dataframe)
        row = {"Model": model_name} | metrics
        metrics_rows.append(row)

    columns = ["Model"] + list(DEFAULT_THRESHOLDS.keys())

    return pd.DataFrame(metrics_rows).reindex(columns=columns)


@pytest.fixture
def diatomics_collection(
    diatomics_pair_data_fixture: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Collect diatomics metrics across all models.

    Parameters
    ----------
    diatomics_pair_data_fixture
        Mapping of model names to curve dataframes generated by the fixture.

    Returns
    -------
    pd.DataFrame
        Aggregated metrics dataframe.
    """
    return collect_metrics(diatomics_pair_data_fixture)


@pytest.fixture
def diatomics_metrics_dataframe(
    diatomics_collection: pd.DataFrame,
) -> pd.DataFrame:
    """
    Provide the aggregated diatomics metrics dataframe.

    Parameters
    ----------
    diatomics_collection
        Metrics dataframe produced by ``collect_metrics``.

    Returns
    -------
    pd.DataFrame
        Aggregated diatomics metrics indexed by model.
    """
    return diatomics_collection


@pytest.fixture
@build_table(
    filename=OUT_PATH / "diatomics_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    diatomics_metrics_dataframe: pd.DataFrame,
) -> dict[str, dict]:
    """
    Compute diatomics metrics for all models.

    Parameters
    ----------
    diatomics_metrics_dataframe
        Aggregated per-model metrics produced by ``collect_metrics``.

    Returns
    -------
    dict[str, dict]
        Mapping of metric names to per-model results.
    """
    metrics_df = diatomics_metrics_dataframe
    metrics_dict: dict[str, dict[str, float | None]] = {}
    for column in metrics_df.columns:
        if column == "Model":
            continue
        values = [
            value if pd.notna(value) else None for value in metrics_df[column].tolist()
        ]
        metrics_dict[column] = dict(zip(metrics_df["Model"], values, strict=False))
    return metrics_dict


def test_diatomics(metrics: dict[str, dict]) -> None:
    """
    Run diatomics analysis.

    Parameters
    ----------
    metrics
        Benchmark metrics generated by fixtures.
    """
    return
