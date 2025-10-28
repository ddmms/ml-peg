"""General utility functions for Dash application."""

from __future__ import annotations

import dash.dash_table.Format as TableFormat


def calculate_column_widths(
    columns: list[str],
    widths: dict[str, float] | None = None,
    *,
    char_width: int = 9,
    padding: int = 40,
    min_metric_width: int = 140,
) -> dict[str, int]:
    """
    Calculate column widths based on column titles with minimum width enforcement.

    Parameters
    ----------
    columns
        List of column names from DataTable.
    widths
        Dictionary of column widths. Default is {}.
    char_width
        Approximate pixel width per character.
    padding
        Extra padding to add to calculated width.
    min_metric_width
        Minimum width for metric columns in pixels.

    Returns
    -------
    dict[str, int]
        Mapping of column IDs to pixel widths.
    """
    widths = widths if widths else {}
    # Fixed widths for static columns
    widths.setdefault("MLIP", 150)
    widths.setdefault("Score", 100)
    widths.setdefault("Rank", 100)

    for col in columns:
        if col not in ("MLIP", "Score", "Rank"):
            # Calculate width based on column title length
            calculated_width = len(col) * char_width + padding
            # Enforce minimum width
            widths.setdefault(col, max(calculated_width, min_metric_width))

    return widths


def is_numeric_column(rows: list[dict], column_id: str) -> bool:
    """
    Determine whether a column contains numeric values.

    Parameters
    ----------
    rows
        Table rows to inspect.
    column_id
        Column identifier to check.

    Returns
    -------
    bool
        ``True`` when any non-null entry is numeric, otherwise ``False``.
    """
    for row in rows:
        value = row.get(column_id)
        if value is None:
            continue
        if isinstance(value, int | float):
            return True
        try:
            float(value)
        except (TypeError, ValueError):
            return False
        else:
            return True
    return False


def sig_fig_format() -> TableFormat.Format:
    """
    Build a formatter that displays three significant figures.

    Returns
    -------
    TableFormat.Format
        Dash table format configured for three significant figures.
    """
    return TableFormat.Format(precision=3).scheme(
        TableFormat.Scheme.decimal_or_exponent
    )


def rank_format() -> TableFormat.Format:
    """
    Build a formatter that displays integer ranks.

    Returns
    -------
    TableFormat.Format
        Dash table format configured for integer values.
    """
    return TableFormat.Format().scheme(TableFormat.Scheme.decimal_integer)


def clean_thresholds(
    raw_thresholds: dict[str, dict[str, float]],
) -> dict[str, tuple[float, float]] | None:
    """
    Convert a raw normalization mapping into ``(good, bad)`` float tuples.

    Parameters
    ----------
    raw_thresholds
        Raw normalization structure read from json file.

    Returns
    -------
    dict[str, tuple[float, float]] | None
        Cleaned mapping or ``None`` when conversion fails.
    """
    if not isinstance(raw_thresholds, dict):
        return None

    thresholds = {}

    for metric, bounds in raw_thresholds.items():
        try:
            if isinstance(bounds, dict):
                good_val = float(bounds["good"])
                bad_val = float(bounds["bad"])
            elif isinstance(bounds, list | tuple) and len(bounds) == 2:
                good_val = float(bounds[0])
                bad_val = float(bounds[1])
            else:
                continue
        except (KeyError, TypeError, ValueError):
            continue

        thresholds[metric] = (good_val, bad_val)
    return thresholds


def clean_weights(raw_weights: dict[str, float] | None) -> dict[str, float]:
    """
    Convert potentially non-numeric weight values into floats.

    Parameters
    ----------
    raw_weights
        Mapping from metric name to supplied weight.

    Returns
    -------
    dict[str, float]
        Dictionary containing only numeric weight values.
    """
    if not raw_weights:
        return {}

    weights: dict[str, float] = {}
    for metric, value in raw_weights.items():
        try:
            weights[metric] = float(value)
        except (TypeError, ValueError):
            continue
    return weights


def get_scores(
    raw_rows: list[dict],
    scored_rows: list[dict],
    threshold_pairs: dict[str, tuple[float, float]] | None,
    toggle_value: list[str] | None,
) -> list[dict]:
    """
    Build table rows, with either unitful or unitless scores as requested.

    Parameters
    ----------
    raw_rows
        Unitful metric values.
    scored_rows
        Rows with calculated unitless scores.
    threshold_pairs
        Normalisation thresholds, or ``None`` for raw metrics.
    toggle_value
        Current state of the “Show normalized values” toggle.

    Returns
    -------
    list[dict]
        Rows to render in the DataTable.
    """
    show_normalized = bool(toggle_value) and toggle_value[0] == "norm"
    if not (show_normalized and threshold_pairs):
        return raw_rows

    return scored_rows
