"""Analyse diatomics benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import (
    build_table,
    periodic_curve_gallery,
)
from ml_peg.analysis.utils.utils import load_metrics_config
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "molecular" / "Diatomics" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "Diatomics"
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


def prepare_pair_series(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sort and align energy/force series for a diatomic pair.

    Parameters
    ----------
    df
        Pair-specific dataframe.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        Distances, shifted energies, and projected forces sorted by decreasing
        distance.
    """
    df_sorted = df.sort_values("distance").drop_duplicates("distance")
    values = df_sorted[["distance", "energy", "force_parallel"]].to_numpy()
    if len(values) < 3:
        return np.array([]), np.array([]), np.array([])

    rs = values[:, 0]
    es = values[:, 1]
    fs = values[:, 2]

    order = np.argsort(rs)[::-1]
    rs = rs[order]
    es = es[order]
    fs = fs[order]

    es_shifted = es - es[0]
    return rs, es_shifted, fs


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
    clipped[np.abs(clipped) < tol] = 0.0
    signs = np.sign(clipped)
    mask = signs[:-1] != signs[1:]
    valid = (np.abs(clipped[:-1]) > tol) & (np.abs(clipped[1:]) > tol)
    return int(np.count_nonzero(mask & valid))


def compute_pair_metrics(
    df_pair: pd.DataFrame,
) -> tuple[dict[str, float], float] | tuple[None, None]:
    """
    Compute diagnostics for a single diatomic pair.

    Parameters
    ----------
    df_pair
        Pair-specific dataframe.

    Returns
    -------
    tuple[dict[str, float], float] | tuple[None, None]
        Dictionary of metrics and the homonuclear well depth (if applicable).
    """
    rs, es, fs = prepare_pair_series(df_pair)
    if rs.size < 3:
        return None, None

    de_dr = np.gradient(es, rs)
    d2e_dr2 = np.gradient(de_dr, rs)

    minima = 0
    if es.size >= 3:
        minima = int(np.sum(np.diff(np.sign(np.diff(es))) > 0))

    inflections = count_sign_changes(d2e_dr2, tol=0.5)

    rounded_fs = fs.copy()
    rounded_fs[np.abs(rounded_fs) < 1e-2] = 0.0
    fs_sign = np.sign(rounded_fs)
    mask = fs_sign != 0
    f_flip = int(np.sum(np.diff(fs_sign[mask]) != 0)) if mask.any() else 0

    well_depth = float(es.min())

    spearman_repulsion = np.nan
    spearman_attraction = np.nan

    try:
        from scipy import stats

        imine = int(np.argmin(es))
        if rs[imine:].size > 1:
            spearman_repulsion = float(
                stats.spearmanr(rs[imine:], es[imine:]).statistic
            )
        if rs[:imine].size > 1:
            spearman_attraction = float(
                stats.spearmanr(rs[:imine], es[:imine]).statistic
            )
    except Exception:
        pass

    metrics = {
        "Force flips": float(f_flip),
        "Energy minima": float(minima),
        "Energy inflections": float(inflections),
        "ρ(E, repulsion)": float(spearman_repulsion),
        "ρ(E, attraction)": float(spearman_attraction),
    }
    return metrics, well_depth


def aggregate_model_metrics(
    df: pd.DataFrame,
) -> tuple[dict[str, float], dict[str, float]]:
    """
    Aggregate metrics for a model across all pairs.

    Parameters
    ----------
    df
        Model dataframe.

    Returns
    -------
    tuple[dict[str, float], dict[str, float]]
        Aggregated model metrics and homonuclear well depths.
    """
    if df.empty:
        return {}, {}

    pair_metrics: list[dict[str, float]] = []
    well_depths: dict[str, float] = {}

    for pair, df_pair in df.groupby("pair"):
        metrics, well_depth = compute_pair_metrics(df_pair)
        if metrics is None:
            continue
        pair_metrics.append(metrics)

        symbols = pair.split("-")
        if len(symbols) == 2 and symbols[0] == symbols[1]:
            element = symbols[0]
            well_depths[element] = well_depth

    if not pair_metrics:
        return {}, well_depths

    aggregated = {
        key: float(np.nanmean([metrics.get(key, np.nan) for metrics in pair_metrics]))
        for key in DEFAULT_THRESHOLDS.keys()
    }
    return aggregated, well_depths


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
        df = load_model_data(model_name)
        if not df.empty:
            pair_data[model_name] = df
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
def diatomics_pair_data() -> dict[str, pd.DataFrame]:
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
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Gather metrics and well depths for all models.

    Parameters
    ----------
    pair_data
        Optional mapping of model names to curve dataframes. When ``None``,
        the data is loaded via ``persist_diatomics_pair_data``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[str, float]]]
        Aggregated metrics table and per-model homonuclear well depths.
    """
    rows: list[dict[str, float | str]] = []
    model_well_depths: dict[str, dict[str, float]] = {}

    OUT_PATH.mkdir(parents=True, exist_ok=True)

    data = pair_data if pair_data is not None else persist_diatomics_pair_data()

    for model_name, df in data.items():
        metrics, well_depths = aggregate_model_metrics(df)

        row = {"Model": model_name} | metrics
        rows.append(row)

        model_well_depths[model_name] = well_depths

    columns = ["Model"] + list(DEFAULT_THRESHOLDS.keys())

    metrics_df = pd.DataFrame(rows).reindex(columns=columns)
    return metrics_df, model_well_depths


@pytest.fixture
def diatomics_collection(
    diatomics_pair_data: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Collect diatomics metrics and well depths across all models.

    Parameters
    ----------
    diatomics_pair_data
        Mapping of model names to curve dataframes generated by the fixture.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[str, float]]]
        Aggregated metrics dataframe and mapping of homonuclear well depths.
    """
    return collect_metrics(diatomics_pair_data)


@pytest.fixture
def diatomics_metrics_dataframe(
    diatomics_collection: tuple[pd.DataFrame, dict[str, dict[str, float]]],
) -> pd.DataFrame:
    """
    Provide the aggregated diatomics metrics dataframe.

    Parameters
    ----------
    diatomics_collection
        Tuple of metrics dataframe and well depths produced by ``collect_metrics``.

    Returns
    -------
    pd.DataFrame
        Aggregated diatomics metrics indexed by model.
    """
    metrics_df, _ = diatomics_collection
    return metrics_df


@pytest.fixture
def diatomics_well_depths(
    diatomics_collection: tuple[pd.DataFrame, dict[str, dict[str, float]]],
) -> dict[str, dict[str, float]]:
    """
    Provide homonuclear well-depth metadata and persist it for downstream use.

    Parameters
    ----------
    diatomics_collection
        Tuple produced by ``collect_metrics`` containing metrics and well depths.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping of model name to per-element well depths.
    """
    _, well_depths = diatomics_collection
    if well_depths:
        with (OUT_PATH / "well_depths.json").open("w", encoding="utf8") as fh:
            json.dump(well_depths, fh, indent=2)
    return well_depths


@pytest.fixture
@build_table(
    filename=OUT_PATH / "diatomics_metrics_table.json",
    metric_tooltips=DEFAULT_TOOLTIPS,
    thresholds=DEFAULT_THRESHOLDS,
    weights=DEFAULT_WEIGHTS,
)
def metrics(
    diatomics_metrics_dataframe: pd.DataFrame,
    diatomics_well_depths: dict[str, dict[str, float]],
) -> dict[str, dict]:
    """
    Compute diatomics metrics for all models.

    Parameters
    ----------
    diatomics_metrics_dataframe
        Aggregated per-model metrics produced by ``collect_metrics``.
    diatomics_well_depths
        Mapping of homonuclear well depths persisted for cross-analysis usage.

    Returns
    -------
    dict[str, dict]
        Mapping of metric names to per-model results.
    """
    metrics_df = diatomics_metrics_dataframe

    # _ = diatomics_well_depths

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
    Run Diatomics analysis.

    Parameters
    ----------
    metrics
        Benchmark metrics generated by fixtures.
    """
    return
