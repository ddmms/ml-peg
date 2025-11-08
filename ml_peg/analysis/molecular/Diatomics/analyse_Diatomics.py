"""Analyse diatomics benchmark."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from ml_peg.analysis.utils.decorators import (
    PERIODIC_TABLE_COLS,
    PERIODIC_TABLE_POSITIONS,
    PERIODIC_TABLE_ROWS,
    build_table,
    render_periodic_table_grid,
)
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
    "Energy minima": (1.0, 5.0),
    "Energy inflections": (1.0, 5.0),
    "ρ(E, repulsion)": (-1.0, 1.0),
    "ρ(E, attraction)": (1.0, -1.0),
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
        "Energy minima": 1.0,
        "Energy inflections": 1.0,
        "ρ(E, repulsion)": -1.0,
        "ρ(E, attraction)": 1.0,
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


def write_periodic_table_assets(
    model_name: str,
    df: pd.DataFrame,
    well_depths: dict[str, float],
) -> None:
    """
    Create periodic-table overview and element-focused plots for app consumption.

    Parameters
    ----------
    model_name
        Name of the model being processed.
    df
        Dataframe containing curve samples for the model.
    well_depths
        Mapping of element symbol to homonuclear well depth.
    """
    if df.empty:
        return

    model_dir = PERIODIC_TABLE_PATH / model_name
    elements_dir = model_dir / "elements"
    model_dir.mkdir(parents=True, exist_ok=True)
    elements_dir.mkdir(parents=True, exist_ok=True)

    def _plot_overview(ax, element: str) -> bool:
        """
        Render the homonuclear curve for a single element into ``ax``.

        Parameters
        ----------
        ax
            Matplotlib axes to draw on.
        element
            Chemical symbol identifying the homonuclear pair.

        Returns
        -------
        bool
            ``True`` if the element had data and was plotted, else ``False``.
        """
        pair_label = f"{element}-{element}"
        pair_df = (
            df[df["pair"] == pair_label]
            .sort_values("distance")
            .drop_duplicates("distance")
        )
        if pair_df.empty:
            return False

        x = pair_df["distance"].to_numpy()
        y = pair_df["energy"].to_numpy()
        y_shifted = y - y[-1]

        ax.plot(x, y_shifted, linewidth=1, color="tab:blue", zorder=1)
        ax.axhline(0, color="lightgray", linewidth=0.6, zorder=0)
        ax.set_facecolor("white")
        ax.set_xlim(0.0, 6.0)
        ax.set_ylim(-20.0, 20.0)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_yticks([-20, -10, 0, 10, 20])
        ax.tick_params(labelsize=7, length=2, pad=1)

        depth = well_depths.get(element)
        label = f"{element}\n{depth:.2f} eV" if depth is not None else element
        ax.text(
            0.02,
            0.95,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
        )
        return True

    render_periodic_table_grid(
        title=f"Homonuclear diatomic curves: {model_name}",
        filename_stem=model_dir / "overview",
        plot_cell=_plot_overview,
        figsize=(36, 20),
        formats=("svg",),
        suptitle_kwargs={"fontsize": 28, "fontweight": "bold"},
    )

    manifest: dict[str, dict[str, str] | str] = {
        "overview": "overview.svg",
        "elements": {},
    }

    available_elements = sorted(
        e
        for e in (set(df["element_1"].tolist()) | set(df["element_2"].tolist()))
        if isinstance(e, str) and e
    )
    for element in available_elements:
        rel_path = f"elements/{element}.png"
        output_path = elements_dir / f"{element}.png"
        if _render_element_focus(df, element, output_path):
            manifest["elements"][element] = rel_path

    manifest_path = model_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))


def _render_element_focus(df: pd.DataFrame, selected_element: str, output_path) -> bool:
    """
    Render heteronuclear overview for a selected element.

    Parameters
    ----------
    df
        Dataframe containing pair data.
    selected_element
        Element to highlight in the periodic table.
    output_path
        File path to save the PNG figure.

    Returns
    -------
    bool
        ``True`` if any data was rendered for the element.
    """
    pair_groups: dict[str, pd.DataFrame] = {}
    for pair, df_pair in df.groupby("pair"):
        try:
            element1, element2 = pair.split("-")
        except ValueError:
            continue
        if selected_element not in {element1, element2}:
            continue
        other = element2 if element1 == selected_element else element1
        pair_groups[other] = df_pair.sort_values("distance").drop_duplicates("distance")

    if not pair_groups:
        return False

    fig, axes = plt.subplots(
        PERIODIC_TABLE_ROWS,
        PERIODIC_TABLE_COLS,
        figsize=(30, 15),
        constrained_layout=True,
    )
    axes = axes.reshape(PERIODIC_TABLE_ROWS, PERIODIC_TABLE_COLS)
    for ax in axes.ravel():
        ax.axis("off")

    has_data = False
    for element, (row, col) in PERIODIC_TABLE_POSITIONS.items():
        pair_df = pair_groups.get(element)
        if pair_df is None:
            continue
        x = pair_df["distance"].to_numpy()
        y = pair_df["energy"].to_numpy()
        shifted = y - y[-1]

        ax = axes[row, col]
        ax.axis("on")
        ax.set_facecolor("white")
        ax.plot(x, shifted, linewidth=1, color="tab:blue", zorder=1)
        ax.axhline(0, color="lightgray", linewidth=0.6, zorder=0)
        ax.set_xlim(0.0, 6.0)
        ax.set_ylim(-20.0, 20.0)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_yticks([-20, -10, 0, 10, 20])
        ax.tick_params(labelsize=7, length=2, pad=1)
        ax.set_title(
            f"{selected_element}-{element}, shift: {float(y[-1]):.4f}",
            fontsize=8,
        )
        if element == selected_element:
            for spine in ax.spines.values():
                spine.set_edgecolor("crimson")
                spine.set_linewidth(2)
        has_data = True

    if not has_data:
        plt.close(fig)
        return False

    fig.suptitle(
        f"Diatomics involving {selected_element}",
        fontsize=22,
        fontweight="bold",
    )
    fig.savefig(output_path, format="png", dpi=200)
    plt.close(fig)
    return True


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
        write_periodic_table_assets(model_name, df, well_depths)
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
        "Energy minima": "Average number of energy minima per pair",
        "Energy inflections": "Average number of energy inflection points per pair",
        "ρ(E, repulsion)": (
            "Spearman correlation for energy in repulsive regime (ideal -1)"
        ),
        "ρ(E, attraction)": (
            "Spearman correlation for energy in attractive regime (ideal +1)"
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
