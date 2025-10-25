"""Utility functions for analysis."""

from __future__ import annotations

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
    metrics_data: list[dict], weights: dict[str, float] | None = None
) -> list[dict]:
    """
    Calculate score for each model and add to table data.

    Parameters
    ----------
    metrics_data
        Rows data containing model name and metric values.
    weights
        Weight for each metric. Default is 1.0 for each metric.

    Returns
    -------
    list[dict]
        Rows of data with combined score for each model added.
    """
    weights = weights if weights else {}

    for row in metrics_data:
        scores = []
        weights_list = []
        for key, value in row.items():
            # Value may be ``None`` if missing for a benchmark
            if key in {"MLIP", "Score", "Rank", "id"}:
                continue
            if value is None:
                continue
            scores.append(value)
            weights_list.append(weights.get(key, 1.0))

        # Ensure at least one score is being averaged
        if scores:
            try:
                row["Score"] = float(np.average(scores, weights=weights_list))
            except ZeroDivisionError:
                row["Score"] = float(np.mean(scores))
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

    Returns
    -------
    list[dict]
        Rows of data with rank for each model added.
    """
    # If a score is None, set to NaN for ranking purposes, but do not rank
    ranked_scores = rankdata(
        [x["Score"] if x.get("Score") is not None else np.nan for x in metrics_data],
        nan_policy="omit",
    )
    for i, row in enumerate(metrics_data):
        if np.isnan(ranked_scores[i]):
            row["Rank"] = None
        else:
            row["Rank"] = int(ranked_scores[i])
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


def normalize_metric(value: float, x_threshold: float, y_threshold: float) -> float:
    """
    Normalize a metric value to [0, 1] with X mapped to 1 and Y mapped to 0.

    Works regardless of whether X > Y or X < Y. Values beyond the thresholds are
    clipped (≤ Y => 0, ≥ X => 1 after orientation is accounted for).

    Parameters
    ----------
    value
        The metric value to normalize.
    x_threshold
        Threshold that maps to score 1.0.
    y_threshold
        Threshold that maps to score 0.0.

    Returns
    -------
    float
        Normalized score between 0 and 1.
    """
    if value is None or x_threshold is None or y_threshold is None:
        return 0.0

    try:
        # Handle NaNs robustly
        if np.isnan([value, x_threshold, y_threshold]).any():
            return 0.0
    except Exception:
        pass

    if x_threshold == y_threshold:
        return 1.0 if value == x_threshold else 0.0

    # Linear map: Y -> 0, X -> 1
    t = (value - y_threshold) / (x_threshold - y_threshold)
    # Clip to [0, 1]
    if t < 0.0:
        return 0.0
    if t > 1.0:
        return 1.0
    return float(t)


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
                    x_threshold, y_threshold = thresholds[key]
                    x_threshold = float(x_threshold)
                    y_threshold = float(y_threshold)
                    metric_value = float(value)
                except (TypeError, ValueError):
                    continue

                normalized_score = normalize_metric(
                    metric_value, x_threshold, y_threshold
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
