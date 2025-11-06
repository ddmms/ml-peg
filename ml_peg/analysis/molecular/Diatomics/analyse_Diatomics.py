"""Analyse diatomics benchmark."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import build_table, plot_periodic_table
from ml_peg.app import APP_ROOT
from ml_peg.calcs import CALCS_ROOT
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

MODELS = get_model_names(current_models)
CALC_PATH = CALCS_ROOT / "molecular" / "Diatomics" / "outputs"
OUT_PATH = APP_ROOT / "data" / "molecular" / "Diatomics"
CURVE_PATH = OUT_PATH / "curves"
PERIODIC_TABLE_PATH = OUT_PATH / "periodic_tables"

DIATOMICS_THRESHOLDS = {
    "Force flips": (1.0, 5.0),
    "Tortuosity": (1.0, 5.0),
    "Energy minima": (1.0, 5.0),
    "Energy inflections": (1.0, 5.0),
    "Spearman's coefficient (E: repulsion)": (-1.0, 1.0),
    "Spearman's coefficient (F: descending)": (-1.0, 1.0),
    "Spearman's coefficient (E: attraction)": (1.0, -1.0),
    "Spearman's coefficient (F: ascending)": (1.0, -1.0),
}


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

    fdiff = np.diff(fs)
    fdiff[np.abs(fdiff) < 1e-3] = 0.0
    fdiff_sign = np.sign(fdiff)
    mask = fdiff_sign != 0
    fjump = 0.0
    if mask.any():
        diff = fdiff[mask]
        diff_sign = fdiff_sign[mask]
        flips = np.diff(diff_sign) != 0
        if flips.any():
            fjump = float(
                np.abs(diff[:-1][flips]).sum() + np.abs(diff[1:][flips]).sum()
            )

    ediff = np.diff(es)
    ediff[np.abs(ediff) < 1e-3] = 0.0
    ediff_sign = np.sign(ediff)
    mask = ediff_sign != 0
    ejump = 0.0
    ediff_flip_times = 0
    if mask.any():
        diff = ediff[mask]
        diff_sign = ediff_sign[mask]
        flips = np.diff(diff_sign) != 0
        ediff_flip_times = int(np.sum(flips))
        if flips.any():
            ejump = float(
                np.abs(diff[:-1][flips]).sum() + np.abs(diff[1:][flips]).sum()
            )

    conservation_deviation = float(np.mean(np.abs(fs + de_dr)))
    energy_total_variation = float(np.sum(np.abs(np.diff(es))))

    well_depth = float(es.min())

    spearman_repulsion = np.nan
    spearman_attraction = np.nan
    spearman_force_desc = np.nan
    spearman_force_asc = np.nan

    try:
        from scipy import stats

        imine = int(np.argmin(es))
        iminf = int(np.argmin(fs))
        if rs[imine:].size > 1:
            spearman_repulsion = float(
                stats.spearmanr(rs[imine:], es[imine:]).statistic
            )
        if rs[:imine].size > 1:
            spearman_attraction = float(
                stats.spearmanr(rs[:imine], es[:imine]).statistic
            )
        if rs[iminf:].size > 1:
            spearman_force_desc = float(
                stats.spearmanr(rs[iminf:], fs[iminf:]).statistic
            )
        if rs[:iminf].size > 1:
            spearman_force_asc = float(
                stats.spearmanr(rs[:iminf], fs[:iminf]).statistic
            )
    except Exception:
        pass

    tortuosity = 0.0
    denominator = abs(es[0] - es.min()) + (es[-1] - es.min())
    if denominator > 0:
        tortuosity = float(energy_total_variation / denominator)

    metrics = {
        "Force flips": float(f_flip),
        "Tortuosity": tortuosity,
        "Energy minima": float(minima),
        "Energy inflections": float(inflections),
        "Spearman's coefficient (E: repulsion)": float(spearman_repulsion),
        "Spearman's coefficient (F: descending)": float(spearman_force_desc),
        "Spearman's coefficient (E: attraction)": float(spearman_attraction),
        "Spearman's coefficient (F: ascending)": float(spearman_force_asc),
        "Energy diff flips": float(ediff_flip_times),
        "Energy jump": float(ejump),
        "Force jump": float(fjump),
        "Conservation deviation": float(conservation_deviation),
        "Energy total variation": float(energy_total_variation),
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
        for key in DIATOMICS_THRESHOLDS
    }
    return aggregated, well_depths


def score_diatomics(
    metrics_df: pd.DataFrame, normalise_to_model: str | None
) -> pd.DataFrame:
    """
    Compute aggregate score based on physical targets.

    Parameters
    ----------
    metrics_df
        Dataframe containing per-model metrics.
    normalise_to_model
        Optional model to normalise scores against.

    Returns
    -------
    pd.DataFrame
        Dataframe with ``Score`` column appended.
    """
    ideal_targets = {
        "Force flips": 1.0,
        "Tortuosity": 1.0,
        "Energy minima": 1.0,
        "Energy inflections": 1.0,
        "Spearman's coefficient (E: repulsion)": -1.0,
        "Spearman's coefficient (F: descending)": -1.0,
        "Spearman's coefficient (E: attraction)": 1.0,
        "Spearman's coefficient (F: ascending)": 1.0,
    }

    metrics_df["Score"] = 0.0
    for model in metrics_df["Model"]:
        for column, target in ideal_targets.items():
            value = metrics_df.loc[metrics_df["Model"] == model, column].iloc[0]
            if pd.isna(value):
                continue
            metrics_df.loc[metrics_df["Model"] == model, "Score"] += abs(value - target)

    if normalise_to_model in metrics_df["Model"].values:
        baseline = metrics_df.loc[
            metrics_df["Model"] == normalise_to_model, "Score"
        ].iloc[0]
        if baseline != 0:
            metrics_df["Score"] /= baseline

    metrics_df["Rank"] = (
        metrics_df["Score"].rank(ascending=True, method="min").astype(int)
    )
    return metrics_df


def write_curve_data(model_name: str, df: pd.DataFrame) -> None:
    """
    Persist per-pair curve data for application callbacks.

    Parameters
    ----------
    model_name
        Name of the model being processed.
    df
        Dataframe containing curve samples for the model.
    """
    model_dir = CURVE_PATH / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    for pair, df_pair in df.groupby("pair"):
        payload = {
            "pair": pair,
            "element_1": df_pair["element_1"].iloc[0],
            "element_2": df_pair["element_2"].iloc[0],
            "distance": df_pair["distance"].tolist(),
            "energy": df_pair["energy"].tolist(),
            "force_parallel": df_pair["force_parallel"].tolist(),
        }
        with (model_dir / f"{pair}.json").open("w", encoding="utf8") as fh:
            json.dump(payload, fh)


def write_periodic_table_figures(
    model_name: str, well_depths: dict[str, float]
) -> None:
    """
    Create periodic-table figure JSON for a model.

    Parameters
    ----------
    model_name
        Name of the model being processed.
    well_depths
        Mapping of element symbol to homonuclear well depth.
    """

    @plot_periodic_table(
        title=f"{model_name} homonuclear well depths",
        colorbar_title="Well depth / eV",
        filename=str(PERIODIC_TABLE_PATH / f"{model_name}.json"),
        colorscale="Viridis",
    )
    def generate_plot(values: dict[str, float]) -> dict[str, float]:
        """
        Identity helper to leverage the periodic-table decorator.

        Parameters
        ----------
        values
            Mapping of element symbol to well depth.

        Returns
        -------
        dict[str, float]
            The unchanged mapping of element well depths.
        """
        return values

    generate_plot(well_depths)


def collect_metrics(
    normalise_to_model: str | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Gather metrics and well depths for all models.

    Parameters
    ----------
    normalise_to_model
        Optional reference model for score normalisation.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[str, float]]]
        Aggregated metrics table and per-model homonuclear well depths.
    """
    rows = []
    model_well_depths: dict[str, dict[str, float]] = {}

    OUT_PATH.mkdir(parents=True, exist_ok=True)
    CURVE_PATH.mkdir(parents=True, exist_ok=True)
    PERIODIC_TABLE_PATH.mkdir(parents=True, exist_ok=True)

    for model_name in MODELS:
        df = load_model_data(model_name)
        if df.empty:
            continue

        metrics, well_depths = aggregate_model_metrics(df)
        if not metrics:
            continue

        row = {"Model": model_name} | metrics
        rows.append(row)

        write_curve_data(model_name, df)
        write_periodic_table_figures(model_name, well_depths)
        model_well_depths[model_name] = well_depths

    if not rows:
        return pd.DataFrame(), model_well_depths

    metrics_df = pd.DataFrame(rows)
    metrics_df = score_diatomics(metrics_df, normalise_to_model)
    metrics_df = metrics_df.reindex(
        columns=["Model"] + list(DIATOMICS_THRESHOLDS.keys()) + ["Score", "Rank"]
    )
    return metrics_df, model_well_depths


@pytest.fixture
def diatomics_collection(
    request: pytest.FixtureRequest,
) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
    """
    Collect diatomics metrics and well depths across all models.

    Parameters
    ----------
    request
        Pytest fixture request used to access optional parametrisation, namely
        the ``normalise_to_model`` override.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, dict[str, float]]]
        Aggregated metrics dataframe and mapping of homonuclear well depths.
    """
    normalise_to_model: str | None = None
    if hasattr(request, "param"):
        param = request.param
        if isinstance(param, dict):
            normalise_to_model = param.get("normalise_to_model")
        else:
            normalise_to_model = param
    return collect_metrics(normalise_to_model)


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
    metric_tooltips={
        "Model": "Name of the model",
        "Force flips": "Mean count of force-direction changes per pair (ideal 1)",
        "Tortuosity": "Energy curve tortuosity (lower is smoother)",
        "Energy minima": "Average number of energy minima per pair",
        "Energy inflections": "Average number of energy inflection points per pair",
        "Spearman's coefficient (E: repulsion)": (
            "Spearman correlation for energy in repulsive regime (ideal -1)"
        ),
        "Spearman's coefficient (F: descending)": (
            "Spearman correlation for force in descending regime (ideal -1)"
        ),
        "Spearman's coefficient (E: attraction)": (
            "Spearman correlation for energy in attractive regime (ideal +1)"
        ),
        "Spearman's coefficient (F: ascending)": (
            "Spearman correlation for force in ascending regime (ideal +1)"
        ),
        "Score": "Aggregate deviation from physical targets (lower is better)",
        "Rank": "Model ranking based on score (lower is better)",
    },
    thresholds=DIATOMICS_THRESHOLDS,
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

    metrics_dict: dict[str, dict[str, float]] = {}
    for column in metrics_df.columns:
        if column == "Model":
            continue
        metrics_dict[column] = dict(
            zip(metrics_df["Model"], metrics_df[column], strict=False)
        )
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
