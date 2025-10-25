"""Utility functions for analysis."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from matplotlib import cm
from matplotlib.colors import Colormap
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error, mean_squared_error


def mae(ref: list, prediction: list) -> float:
    """
    Get mean absolute error.

    Parameters
    ----------
    ref
        Reference data.
    prediction
        Predicted data.

    Returns
    -------
    float
        Mean absolute error.
    """
    return mean_absolute_error(ref, prediction)


def rmse(ref: list, prediction: list) -> float:
    """
    Get root mean squared error.

    Parameters
    ----------
    ref
        Reference data.
    prediction
        Predicted data.

    Returns
    -------
    float
        Root mean squared error.
    """
    return mean_squared_error(ref, prediction)


def calc_scores(
    metrics_data: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    thresholds: dict[str, tuple[float, float]] | None = None,
    normalizer: Callable[[float, float, float], float] | None = None,
) -> list[dict]:
    """
    Calculate (normalised) score for each model and add to table data.

    If `thresholds` is not None, `normalizer` will be used to normalise the score.

    Parameters
    ----------
    metrics_data
        Rows data containing model name and metric values.
    weights
        Weight for each metric. Default is 1.0 for each metric.
    thresholds
        Normalisation thresholds keyed by metric name, where each value is
        a (good_threshold, bad_threshold) tuple.
    normalizer
        Optional function to map (value, good, bad) -> normalised score.
        If None, and thresholds are specified, uses `normalize_metric`.

    Returns
    -------
    list[dict]
        Rows of data with combined score for each model added.
    """
    weights = weights if weights else {}
    if thresholds:
        normalizer = normalizer if normalizer is not None else normalize_metric

    for row in metrics_data:
        scores = []
        weights_list = []
        for key, value in row.items():
            # Value may be ``None`` if missing for a benchmark
            if key not in {"MLIP", "Score", "Rank", "id"} and value is not None:
                # If thresholds given, use to normalise
                if thresholds is not None and key in thresholds:
                    good_threshold, bad_threshold = thresholds[key]
                    scores.append(normalizer(value, good_threshold, bad_threshold))
                else:
                    scores.append(value)
                weights_list.append(weights.get(key, 1.0))

        # Ensure at least one score is being averaged
        if scores:
            try:
                row["Score"] = np.average(scores, weights=weights_list)
            except ZeroDivisionError:
                row["Score"] = np.mean(scores)
        else:
            row["Score"] = None

    return metrics_data


def calc_ranks(metrics_data: list[dict]) -> list[dict]:
    """
    Calculate rank for each model and add to table data.

    Parameters
    ----------
    metrics_data
        Rows data containing model name, metric values, and Score.
        The "Score" column is used to calculate the rank, with the highest score ranked
        1.

    Returns
    -------
    list[dict]
        Rows of data with rank for each model added.
    """
    # If a score is None, set to NaN for ranking purposes, but do not rank
    ranked_scores = rankdata(
        [x["Score"] if x.get("Score") is not None else np.nan for x in metrics_data],
        nan_policy="omit",
        method="max",
    )
    for i, row in enumerate(metrics_data):
        if np.isnan(ranked_scores[i]):
            row["Rank"] = None
        else:
            row["Rank"] = len(ranked_scores) - int(ranked_scores[i]) + 1
    return metrics_data


def get_table_style(
    data: list[dict],
    all_cols: bool = True,
    col_names: list[str] | str | None = None,
) -> list[dict[str, Any]]:
    """
    Viridis-style colormap for Dash DataTable.

    Parameters
    ----------
    data
        Data from Dash table to be coloured.
    all_cols
        Whether to colour all numerical columns.
    col_names
        Column name or list of names to be coloured.

    Returns
    -------
    list[dict[str, Any]]
        Conditional style data to apply to table.
    """
    cmap = cm.get_cmap("viridis_r")

    def rgba_from_val(val: float, vmin: float, vmax: float, cmap: Colormap) -> str:
        """
        Get RGB values for a cell.

        Parameters
        ----------
        val
            Value to colour.
        vmin
            Minimum value in column.
        vmax
            Maximum value in column.
        cmap
            Colour map for cell colours.

        Returns
        -------
        str
            RGB colours in backgroundColor format.
        """
        norm = (val - vmin) / (vmax - vmin) if vmax != vmin else 0
        rgba = cmap(norm)
        r, g, b = [int(255 * x) for x in rgba[:3]]
        return f"rgb({r}, {g}, {b})"

    style_data_conditional = []

    if all_cols:
        cols = data[0].keys() - {"MLIP", "id"}
    elif col_names:
        if isinstance(col_names, str):
            cols = [col_names]
    else:
        raise ValueError("Specify either all_cols=True or provide col_name.")

    for col in cols:
        if col not in data[0]:
            raise ValueError(f"Column '{col}' not found in data.")

    for col in cols:
        numeric_entries: list[tuple[object, float]] = []
        for row in data:
            if col not in row:
                continue
            raw_value = row[col]
            if raw_value is None:
                continue
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue
            numeric_entries.append((raw_value, numeric_value))

        if not numeric_entries:
            continue

        numeric_values = [numeric for _, numeric in numeric_entries]
        min_value = min(numeric_values)
        max_value = max(numeric_values)

        for raw_value, numeric_value in numeric_entries:
            style_data_conditional.append(
                {
                    "if": {
                        "filter_query": f"{{{col}}} = {raw_value}",
                        "column_id": col,
                    },
                    "backgroundColor": rgba_from_val(
                        numeric_value, min_value, max_value, cmap
                    ),
                    "color": "white"
                    if numeric_value > (min_value + max_value) / 2
                    else "black",
                }
            )

    return style_data_conditional


def normalize_metric(
    value: float, good_threshold: float, bad_threshold: float
) -> float | None:
    """
    Normalize a metric value to [0.0, 1.0].

    `good_threshold` is mapped to 1.0 and `bad_threshold` mapped to 0.0. Values beyond
    these thresholds are clipped to 0.0 and 1.0.

    Works regardless of whether `good_threshold` > `bad_threshold` or
    `good_threshold` < `bad_threshold`.

    Parameters
    ----------
    value
        The metric value to normalize.
    good_threshold
        Threshold that maps to score 1.0.
    bad_threshold
        Threshold that maps to score 0.0.

    Returns
    -------
    float | None
        Normalized score between 0 and 1, or `None` if normalization process
        raises an error.
    """
    if value is None or good_threshold is None or bad_threshold is None:
        return None

    try:
        # Handle NaNs robustly
        if np.isnan([value, good_threshold, bad_threshold]).any():
            return None
    except TypeError:
        return None

    if good_threshold == bad_threshold:
        return 1.0 if value == good_threshold else 0.0

    # Linear map: Y -> 0, X -> 1
    t = (value - bad_threshold) / (good_threshold - bad_threshold)

    # Clip to [0, 1]
    return max(min(1.0, float(t)), 0.0)


def calc_normalized_scores(
    metrics_data: list[dict],
    thresholds: dict[str, tuple[float, float]],
    weights: dict[str, float] | None = None,
) -> list[dict]:
    """
    Calculate normalized scores for each model and add to table data.

    Each metric is normalized to 0-1 scale then averaged to get final score.

    Parameters
    ----------
    metrics_data
        Rows data containing model name and metric values.
    thresholds
        Dictionary mapping metric names to (X, Y) threshold tuples.
    weights
        Optional mapping of metric names to weighting factors.

    Returns
    -------
    list[dict]
        Rows of data with normalized score for each model added.
    """
    for row in metrics_data:
        normalized_scores = []
        weights_list = []

        for key, value in row.items():
            if key not in ("MLIP", "Score", "Rank", "id") and key in thresholds:
                try:
                    good_threshold, bad_threshold = thresholds[key]
                    good_threshold = float(good_threshold)
                    bad_threshold = float(bad_threshold)
                    metric_value = float(value)
                except (TypeError, ValueError):
                    continue

                normalized_score = normalize_metric(
                    metric_value, good_threshold, bad_threshold
                )
                normalized_scores.append(normalized_score)
                if weights and key in weights:
                    try:
                        weight_value = float(weights.get(key, 1.0))
                    except (TypeError, ValueError):
                        weight_value = 1.0
                else:
                    weight_value = 1.0
                weights_list.append(weight_value)

        # Average the normalized scores (higher is better)
        if normalized_scores:
            try:
                row["Score"] = float(
                    np.average(normalized_scores, weights=weights_list)
                )
            except (TypeError, ValueError):
                row["Score"] = float(np.mean(normalized_scores))
        else:
            row["Score"] = 0.0

    return metrics_data
