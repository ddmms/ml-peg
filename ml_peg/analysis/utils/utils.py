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
            # Value may be `None` if missing for a benchmark
            if key not in ("MLIP", "Score", "Rank", "id") and value:
                scores.append(value)
                weights_list.append(weights.get(key, 1.0))

        # Ensure at least one score is being averaged
        if scores:
            row["Score"] = np.average(scores, weights=weights_list)
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
        [x["Score"] if x["Score"] else np.nan for x in metrics_data], nan_policy="omit"
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
        # Ensure that model is present in the row, and the score is not None
        min_value = min([row[col] for row in data if col in row and row[col]])
        max_value = max([row[col] for row in data if col in row and row[col]])

        for val in [row[col] for row in data if col in row and row[col]]:
            style_data_conditional.append(
                {
                    "if": {"filter_query": f"{{{col}}} = {val}", "column_id": col},
                    "backgroundColor": rgba_from_val(val, min_value, max_value, cmap),
                    "color": "white" if val > (min_value + max_value) / 2 else "black",
                }
            )

    return style_data_conditional
