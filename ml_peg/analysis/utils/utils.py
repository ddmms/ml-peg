"""Utility functions for analysis."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from matplotlib import cm
from matplotlib.colors import Colormap
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import mean_absolute_error, mean_squared_error
from yaml import safe_load

from ml_peg.app.utils.utils import clean_weights


def load_metrics_config(config_path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    """
    Load metric thresholds and tooltips from a YAML configuration file.

    Parameters
    ----------
    config_path
        Path to the YAML file containing metric definitions.

    Returns
    -------
    tuple[dict[str, Any], dict[str, str]]
        Mapping of metric thresholds and mapping of metric tooltips.
    """
    if not config_path.exists():
        msg = f"Metrics configuration file not found: {config_path}"
        raise FileNotFoundError(msg)

    with config_path.open(encoding="utf8") as stream:
        data = safe_load(stream) or {}

    metrics_data = data.get("metrics", {}) or {}
    thresholds: dict[str, Any] = {}
    tooltips: dict[str, str] = {}

    for metric_name, metric_config in metrics_data.items():
        if not isinstance(metric_config, dict):
            msg = f"Metric configuration for '{metric_name}' must be a mapping."
            raise TypeError(msg)

        required_threshold_keys = {"good", "bad", "unit"}
        missing_threshold_keys = required_threshold_keys - metric_config.keys()
        if missing_threshold_keys:
            missing_keys = ", ".join(sorted(missing_threshold_keys))
            msg = (
                f"Metric '{metric_name}' in '{config_path}' is missing required "
                f"threshold entries: {missing_keys}. Include 'unit' even when its "
                "value should be None."
            )
            raise KeyError(msg)

        threshold_unit = metric_config["unit"]
        level_of_theory = metric_config.get("level_of_theory")
        metric_threshold = {
            "good": metric_config["good"],
            "bad": metric_config["bad"],
            "unit": threshold_unit,
            "level_of_theory": level_of_theory,
        }

        thresholds[metric_name] = metric_threshold

        tooltip = metric_config.get("tooltip")
        if tooltip is None or tooltip == "":
            msg = (
                f"Metric '{metric_name}' in '{config_path}' must define a non-empty "
                "'tooltip' entry."
            )
            raise ValueError(msg)

        tooltip_text = tooltip.strip()
        if level_of_theory:
            tooltip_text = f"{tooltip_text}\nLevel of theory: {level_of_theory}"
        markdown_value = tooltip_text.replace("\n", "  \n")
        tooltips[metric_name] = {
            "value": markdown_value,
            "type": "markdown",
        }

    return thresholds, tooltips


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


def calc_metric_scores(
    metrics_data: list[dict[str, Any]],
    thresholds: dict[str, dict[str, Any]] | None = None,
    normalizer: Callable[[float, float, float], float] | None = None,
) -> list[dict[str, float]]:
    """
    Calculate all normalised scores.

    Parameters
    ----------
    metrics_data
        Rows data containing model name and metric values.
    thresholds
        Normalisation thresholds keyed by metric name. Each value must be a mapping
        containing ``good`` and ``bad`` entries (optionally including a ``unit``).
    normalizer
        Optional function to map (value, good, bad) -> normalised score.
        If `None`, and thresholds are specified, uses `normalize_metric`.

    Returns
    -------
    list[dict[str, float]]
        Rows data with metric scores in place of values.
    """
    normalizer = normalizer if normalizer is not None else normalize_metric

    metrics_scores = [row.copy() for row in metrics_data]
    for row in metrics_scores:
        for key, value in row.items():
            # Value may be ``None`` if missing for a benchmark
            if key not in {"MLIP", "Score", "Rank", "id"} and value is not None:
                if thresholds is None or key not in thresholds:
                    row[key] = value
                    continue

                entry = thresholds[key]
                if not isinstance(entry, dict):
                    row[key] = value
                    continue
                try:
                    good_threshold = float(entry["good"])
                    bad_threshold = float(entry["bad"])
                except (KeyError, TypeError, ValueError):
                    row[key] = value
                    continue

                row[key] = normalizer(value, good_threshold, bad_threshold)

    return metrics_scores


def calc_table_scores(
    metrics_data: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    thresholds: dict[str, dict[str, Any]] | None = None,
    normalizer: Callable[[float, float, float], float] | None = None,
) -> list[dict]:
    """
    Calculate (normalised) score for each model and add to table data.

    If `thresholds` is not `None`, `normalizer` will be used to normalise the score.

    Parameters
    ----------
    metrics_data
        Rows data containing model name and metric values.
    weights
        Weight for each metric. Default is 1.0 for each metric.
    thresholds
        Normalisation thresholds keyed by metric name. Each value must be a mapping
        with ``good`` and ``bad`` entries (optionally ``unit``).
    normalizer
        Optional function to map (value, good, bad) -> normalised score.
        If `None`, and thresholds are specified, uses `normalize_metric`.

    Returns
    -------
    list[dict]
        Rows of data with combined score for each model added.
    """
    weights = weights if weights else {}

    metrics_scores = calc_metric_scores(metrics_data, thresholds, normalizer)

    for metrics_row, scores_row in zip(metrics_data, metrics_scores, strict=True):
        scores_list = []
        weights_list = []
        for key, value in metrics_row.items():
            # Value may be ``None`` if missing for a benchmark
            if key not in {"MLIP", "Score", "Rank", "id"} and value is not None:
                scores_list.append(scores_row[key])
                weights_list.append(weights.get(key, 1.0))

        # Ensure at least one score is being averaged
        if scores_list:
            try:
                metrics_row["Score"] = np.average(scores_list, weights=weights_list)
            except ZeroDivisionError:
                metrics_row["Score"] = np.mean(scores_list)
        else:
            metrics_row["Score"] = None

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
    *,
    scored_data: list[dict] | None = None,
    normalized: bool = True,
    all_cols: bool = True,
    col_names: list[str] | str | None = None,
) -> list[dict[str, Any]]:
    """
    Viridis-style colormap for Dash DataTable.

    Parameters
    ----------
    data
        Data from Dash table to be coloured.
    scored_data
        Data with metric values replaced with scores.
    normalized
        Whether metric/score columns have been normalized to between 0 and 1. Default is
        `True`.
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

    # All columns other than MLIP and ID (not displayed) should be coloured
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
        numeric_entries: list[tuple[Any, float]] = []
        for i, row in enumerate(data):
            if col not in row:
                continue
            raw_value = row[col]
            # Skip if unable to convert to float (e.g. `None`)
            try:
                numeric_value = float(raw_value)
            except (TypeError, ValueError):
                continue

            # Get scored value, if is exists
            try:
                scored_value = float(scored_data[i][col])
            except (TypeError, ValueError, IndexError):
                scored_value = raw_value

            numeric_entries.append((raw_value, numeric_value, scored_value))

        if not numeric_entries:
            continue

        numeric_values = [numeric for _, numeric, _ in numeric_entries]

        # Use thresholds
        if normalized:
            if col != "Rank":
                min_value, max_value = 1, 0
            else:
                min_value, max_value = 1, len(numeric_values)
        else:
            min_value = min(numeric_values)
            max_value = max(numeric_values)

        for raw_value, _, scored_value in numeric_entries:
            # Determine direction of values
            mid = (min_value + max_value) / 2
            increasing = max_value >= min_value

            style_data_conditional.append(
                {
                    "if": {
                        "filter_query": f"{{{col}}} = {raw_value}",
                        "column_id": col,
                    },
                    "backgroundColor": rgba_from_val(
                        scored_value, min_value, max_value, cmap
                    ),
                    "color": "white"
                    if (scored_value > mid if increasing else scored_value < mid)
                    else "black",
                }
            )

    return style_data_conditional


def update_score_rank_style(
    data: list[dict[str, Any]],
    weights: dict[str, float] | None = None,
    thresholds: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, float] | None]:
    """
    Update table scores, ranks, and table styles.

    Parameters
    ----------
    data
        Rows data containing model name and metric values.
    weights
        Weight for each metric. Default is `None`.
    thresholds
        Normalisation thresholds keyed by metric name. Each value may be a numeric
        pair or a mapping with ``good``/``bad`` entries. Default is `None`.

    Returns
    -------
    tuple[list[dict[str, Any]], dict[str, float] | None]
        Updated table rows and style.
    """
    weights = clean_weights(weights)
    data = calc_table_scores(data, weights, thresholds)
    data = calc_ranks(data)
    scored_data = calc_metric_scores(data, thresholds)
    style = get_table_style(data, scored_data=scored_data)
    return data, style


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
        The metric value to normalise.
    good_threshold
        Threshold that maps to score 1.0.
    bad_threshold
        Threshold that maps to score 0.0.

    Returns
    -------
    float | None
        Normalized score between 0 and 1, or `None` if normalisation process
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
