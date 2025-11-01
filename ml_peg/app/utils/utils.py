"""General utility functions for Dash application."""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping, Sequence
from functools import lru_cache
import json
from typing import Any, TypedDict

import dash.dash_table.Format as TableFormat
import yaml

from ml_peg.models import MODELS_ROOT


class ThresholdEntry(TypedDict):
    """Structure describing the normalization thresholds for a metric."""

    good: float
    bad: float
    unit: str


Thresholds = dict[str, ThresholdEntry]


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

    for col in columns:
        if col not in ("MLIP", "Score"):
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


def clean_thresholds(
    raw_thresholds: Mapping[str, Mapping[str, object]] | None,
) -> Thresholds:
    """
    Convert raw normalization mappings into ``good``/``bad`` bounds with units.

    Parameters
    ----------
    raw_thresholds
        Raw normalization structure read from json file.

    Returns
    -------
    Thresholds
        Mapping containing float thresholds and unit strings.
    """
    if raw_thresholds is None:
        raise TypeError("Threshold metadata must be provided as a dictionary.")

    thresholds: Thresholds = {}

    for metric, bounds in raw_thresholds.items():
        good_val: float | None
        bad_val: float | None
        unit_val: str | None = None
        level_val: str | None = None
        try:
            if isinstance(bounds, dict):
                good_val = float(bounds["good"])
                bad_val = float(bounds["bad"])
                raw_unit = bounds.get("unit")
                unit_val = str(raw_unit) if raw_unit not in (None, "") else None
                raw_level = bounds.get("level_of_theory", bounds.get("level"))
                level_val = str(raw_level) if raw_level not in (None, "") else None
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Threshold entries must include 'good', 'bad', and 'unit': {bounds}"
            ) from exc
        if not unit_val:
            raise ValueError(f"Unit must be a non-empty string for metric '{metric}'.")

        entry: dict[str, float | str | None] = {
            "good": good_val,
            "bad": bad_val,
            "unit": unit_val,
        }

        if level_val:
            entry["level_of_theory"] = level_val

        thresholds[metric] = entry

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
    thresholds: Thresholds | None,
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
    thresholds
        Normalisation thresholds, or ``None`` for raw metrics.
    toggle_value
        Current state of the “Show normalized values” toggle.

    Returns
    -------
    list[dict]
        Rows to render in the DataTable.
    """
    show_normalized = bool(toggle_value) and toggle_value[0] == "norm"
    if not (show_normalized and thresholds):
        return raw_rows

    return scored_rows


WARNING_ICON_URL = (
    'url("data:image/svg+xml;utf8,'
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
    "<path fill='%23f0ad4e' d='M8 1.1 15.2 14H0.8Z'/>"
    "<path fill='%23212121' d='M7.25 11.9h1.5V13h-1.5zM7.36 5.2h1.28l.3 5.4h-1.88z'/>"
    '</svg>")'
)


def build_level_of_theory_warnings(
    rows: list[dict[str, Any]] | None,
    model_levels: Mapping[str, str | None] | None,
    metric_levels: Mapping[str, str | None] | None,
    model_configs: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generate inline styles and tooltips for MLIP column metadata.

    Parameters
    ----------
    rows
        Table rows currently displayed in the DataTable.
    model_levels
        Mapping of model name to configured level of theory.
    metric_levels
        Mapping of metric name to benchmark level of theory.
    model_configs
        Mapping of model name to stored configuration details.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        Conditional style data and tooltip definitions to apply per row.
    """
    if not rows:
        return [], []

    model_levels = model_levels or {}
    metric_levels = {
        metric: level for metric, level in (metric_levels or {}).items() if level
    }
    model_configs = model_configs or {}

    warning_styles: list[dict[str, Any]] = []
    tooltip_rows: list[dict[str, Any]] = [{} for _ in rows]

    def _stringify(value: Any) -> str:
        """
        Return a readable string for tooltip values.

        Parameters
        ----------
        value
            Arbitrary value to stringify.

        Returns
        -------
        str
            JSON-formatted output for containers, otherwise ``str(value)``.
        """
        if isinstance(value, dict | list | tuple):
            try:
                return json.dumps(value, sort_keys=True)
            except (TypeError, ValueError):
                return str(value)
        return str(value)

    def _section(title: str, lines: list[str]) -> list[str]:
        """
        Build a markdown section with a heading and bullet lines.

        Parameters
        ----------
        title
            Section heading to display in bold.
        lines
            Sequence of line items to include beneath the heading.

        Returns
        -------
        list[str]
            Formatted lines including the heading and a trailing blank line.
        """
        cleaned = [line for line in lines if line]
        if not cleaned:
            return []
        section: list[str] = [f"**{title}**"]
        section.extend(cleaned)
        section.append("")
        return section

    for idx, row in enumerate(rows):
        mlip = row.get("MLIP")
        if not isinstance(mlip, str):
            continue

        model_level = model_levels.get(mlip)
        config = model_configs.get(mlip) or {}
        if not isinstance(config, Mapping):
            config = {}
        if not config:
            fallback_cfg = load_model_registry_configs().get(mlip) or {}
            if isinstance(fallback_cfg, Mapping):
                config = fallback_cfg
        if model_level is None and isinstance(config, Mapping):
            model_level = config.get("level_of_theory")

        module = config.get("module")
        class_name = config.get("class_name")
        device = config.get("device")
        args = config.get("args")
        kwargs = config.get("kwargs")
        other_keys = {
            key: value
            for key, value in config.items()
            if key
            not in {
                "module",
                "class_name",
                "device",
                "level_of_theory",
                "args",
                "kwargs",
            }
        }

        level_display = (
            _stringify(model_level) if model_level not in (None, "") else "n/a"
        )
        overview_lines = [
            f"- **Model:** `{mlip}`",
            f"- **Level of theory:** `{level_display}`",
        ]
        if module:
            overview_lines.append(f"- **Module:** `{module}`")
        if class_name:
            overview_lines.append(f"- **Class:** `{class_name}`")
        if device:
            overview_lines.append(f"- **Device:** `{device}`")

        if isinstance(args, list | tuple):
            if args:
                args_lines = [f"- `{_stringify(arg)}`" for arg in args]
            else:
                args_lines = ["- (none)"]
        elif args is None:
            args_lines = ["- (none)"]
        else:
            args_lines = [f"- `{_stringify(args)}`"]

        if isinstance(kwargs, Mapping) and kwargs:
            kwargs_lines = [
                f"- `{key}`: `{_stringify(value)}`"
                for key, value in sorted(kwargs.items())
            ]
        elif kwargs in (None, {}):
            kwargs_lines = ["- (none)"]
        else:
            kwargs_lines = [f"- `{_stringify(kwargs)}`"]

        other_lines: list[str] = []
        if other_keys:
            other_lines = [
                f"- `{key}`: `{_stringify(value)}`"
                for key, value in sorted(other_keys.items())
            ]

        mismatch_metrics: list[tuple[str, str]] = []
        for metric_name, metric_level in metric_levels.items():
            if model_level is None or model_level != metric_level:
                mismatch_metrics.append((metric_name, metric_level))

        if mismatch_metrics:
            align_lines = [
                "- [!] Mismatch detected between model and benchmark levels.",
                f"- Model level: `{level_display}`",
                "- Benchmark metrics:",
            ]
            for metric_name, metric_level in mismatch_metrics:
                level_repr = _stringify(metric_level) if metric_level else "n/a"
                align_lines.append(f"  - {metric_name}: `{level_repr}`")
        else:
            if metric_levels:
                align_lines = ["- All benchmark metrics match the model level."]
            else:
                align_lines = ["- No benchmark level metadata available."]

        tooltip_sections: list[str] = []
        tooltip_sections.extend(_section("Model Overview", overview_lines))
        tooltip_sections.extend(_section("Arguments", args_lines))
        tooltip_sections.extend(_section("Keyword Arguments", kwargs_lines))
        if other_lines:
            tooltip_sections.extend(_section("Additional Settings", other_lines))
        tooltip_sections.extend(_section("Benchmark Alignment", align_lines))

        while tooltip_sections and tooltip_sections[-1] == "":
            tooltip_sections.pop()

        tooltip_rows[idx]["MLIP"] = {
            "type": "markdown",
            "value": "\n".join(tooltip_sections),
        }

        if mismatch_metrics:
            row_id = row.get("id", mlip)
            filter_query = "{id} = " + json.dumps(str(row_id))
            warning_styles.append(
                {
                    "if": {"column_id": "MLIP", "filter_query": filter_query},
                    "backgroundImage": WARNING_ICON_URL,
                    "backgroundRepeat": "no-repeat",
                    "backgroundPosition": "8px center",
                    "backgroundSize": "14px 14px",
                    "paddingLeft": "28px",
                }
            )

    return warning_styles, tooltip_rows


def base_column_label(column: Mapping[str, object]) -> str:
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

    raise TypeError("Column definitions must include a string 'name' or 'id' value.")


def format_metric_columns(
    columns: Sequence[Mapping[str, object]] | None,
    thresholds: Thresholds | None,
    show_normalized: bool,
) -> list[dict[str, object]] | None:
    """
    Generate updated column labels based on unit metadata and toggle state.

    Parameters
    ----------
    columns
        Current DataTable columns configuration.
    thresholds
        Normalisation thresholds keys by metric name.
    show_normalized
        Whether the table is displaying normalized (unitless) values.

    Returns
    -------
    list[dict[str, object]] | None
        Updated column definitions with unit-aware labels. Returns `None` when
        `columns` is `None`.
    """
    if columns is None:
        return None

    thresholds = thresholds or {}
    reserved = {"MLIP", "Score", "id"}
    updated_columns: list[dict[str, object]] = []

    for column in columns:
        column_copy: MutableMapping[str, object] = dict(column)
        column_id = column_copy.get("id")

        if (
            not isinstance(column_id, str)
            or column_id in reserved
            or column_id not in thresholds
        ):
            updated_columns.append(column_copy)
            continue

        base_label = base_column_label(column)
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

    # find pattern like " [unit]" at end of string
    base = text.split(" [", 1)[0]
    return f"{base} [{replacement}]"


def format_tooltip_headers(
    tooltip_header: dict[str, str] | None,
    thresholds: Thresholds | None,
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
    reserved = {"MLIP", "Score", "id"}

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


@lru_cache(maxsize=1)
def load_model_registry_configs() -> dict[str, Any]:
    """
    Load model configurations from the models registry YAML.

    Returns
    -------
    dict[str, Any]
        Mapping of model name to configuration dictionary.
    """
    try:
        with open(MODELS_ROOT / "models.yml", encoding="utf8") as handle:
            data = yaml.safe_load(handle) or {}
            if isinstance(data, dict):
                return data
    except FileNotFoundError:
        pass
    return {}
