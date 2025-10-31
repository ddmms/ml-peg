"""General utility functions for Dash application."""

from __future__ import annotations

import re
from typing import Any

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
    raw_thresholds: dict[str, Any] | None,
) -> dict[str, dict[str, float | str | None]] | None:
    """
    Convert raw normalization mappings into ``good``/``bad`` bounds with units.

    Parameters
    ----------
    raw_thresholds
        Raw normalization structure read from json file.

    Returns
    -------
    dict[str, dict[str, float | str | None]] | None
        Mapping containing float thresholds and optional unit strings, or ``None`` when
        conversion fails.
    """
    if not isinstance(raw_thresholds, dict):
        return None

    thresholds = {}

    for metric, bounds in raw_thresholds.items():
        good_val: float | None
        bad_val: float | None
        unit_val: str | None = None
        try:
            if isinstance(bounds, dict):
                good_val = float(bounds["good"])
                bad_val = float(bounds["bad"])
                raw_unit = bounds.get("unit")
                unit_val = str(raw_unit) if raw_unit not in (None, "") else None
            elif isinstance(bounds, list | tuple) and len(bounds) == 2:
                good_val = float(bounds[0])
                bad_val = float(bounds[1])
            elif bounds is None:
                continue
            else:
                continue
        except (KeyError, TypeError, ValueError):
            continue

        thresholds[metric] = {"good": good_val, "bad": bad_val, "unit": unit_val}
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
    threshold_pairs: dict[str, dict[str, float | str | None]] | None,
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


def _base_column_label(column: dict[str, Any]) -> str:
    """
    Extract the base metric label from a column definition.

    Parameters
    ----------
    column
        Dash DataTable column definition.

    Returns
    -------
    str
        Base label without unit suffix. Falls back to column ID when the name is
        unavailable.
    """
    name = column.get("name")
    if isinstance(name, str):
        return name.split(" [", 1)[0]

    column_id = column.get("id")
    if isinstance(column_id, str):
        return column_id

    if name is not None:
        return str(name)
    if column_id is not None:
        return str(column_id)
    return ""


def format_metric_columns(
    columns: list[dict[str, Any]] | None,
    thresholds: dict[str, dict[str, float | str | None]] | None,
    show_normalized: bool,
) -> list[dict[str, Any]] | None:
    """
    Generate updated column labels based on unit metadata and toggle state.

    Parameters
    ----------
    columns
        Current DataTable columns configuration.
    thresholds
        Normalisation thresholds keyed by metric name including optional units.
    show_normalized
        Whether the table is displaying normalized (unitless) values.

    Returns
    -------
    list[dict[str, Any]] | None
        Updated column definitions with unit-aware labels. Returns `None` when
        `columns` is `None`.
    """
    if columns is None:
        return None

    thresholds = thresholds or {}
    reserved = {"MLIP", "Score", "Rank", "id"}
    updated_columns: list[dict[str, Any]] = []

    for column in columns:
        if not isinstance(column, dict):
            updated_columns.append(column)
            continue

        column_copy = column.copy()
        column_id = column_copy.get("id")

        if (
            not isinstance(column_id, str)
            or column_id in reserved
            or column_id not in thresholds
        ):
            updated_columns.append(column_copy)
            continue

        base_label = _base_column_label(column_copy)
        unit = thresholds[column_id].get("unit")

        if unit:
            if show_normalized:
                column_copy["name"] = f"{base_label} [-]"
            else:
                column_copy["name"] = f"{base_label} [{unit}]"

        updated_columns.append(column_copy)

    return updated_columns


def _swap_tooltip_unit(text: str, unit: str, replacement: str) -> str:
    """
    Replace a unit substring within tooltip text with a new label.

    Parameters
    ----------
    text
        Original tooltip text.
    unit
        Unit string to replace.
    replacement
        Replacement string (e.g. "[-]").

    Returns
    -------
    str
        Tooltip string with updated unit descriptor.
    """
    if not text or not unit:
        return text

    pattern = re.compile(rf"\s*[\(\[]\s*{re.escape(unit)}\s*[\)\]]")
    if pattern.search(text):
        return pattern.sub(f" [{replacement}]", text, count=1)

    # Fallback: append replacement in brackets
    return f"{text} [{replacement}]"


def format_tooltip_headers(
    tooltip_header: dict[str, str] | None,
    thresholds: dict[str, dict[str, float | str | None]] | None,
    show_normalized: bool,
) -> dict[str, str] | None:
    """
    Update tooltip headers to reflect unitless normalised values.

    Parameters
    ----------
    tooltip_header
        Original tooltip header mapping.
    thresholds
        Normalisation thresholds containing unit metadata.
    show_normalized
        Whether normalised values are currently displayed.

    Returns
    -------
    dict[str, str] | None
        Updated tooltip header mapping.
    """
    if tooltip_header is None:
        return None

    thresholds = thresholds or {}
    reserved = {"MLIP", "Score", "Rank", "id"}

    updated: dict[str, str] = {}
    for key, text in tooltip_header.items():
        if not isinstance(text, str) or key in reserved:
            updated[key] = text
            continue

        unit = thresholds.get(key, {}).get("unit")
        if not unit or unit in ("", "-"):
            updated[key] = text
            continue

        base_text = _swap_tooltip_unit(text, unit, unit)
        if show_normalized:
            updated[key] = base_text.replace(f"[{unit}]", "[-]", 1)
        else:
            updated[key] = base_text

    return updated
