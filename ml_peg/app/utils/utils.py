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
        try:
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


WARNING_ICON_URLS = {
    "dft": (
        'url("data:image/svg+xml;utf8,'
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
        "<path fill='%23d9534f' d='M8 1.1 15.2 14H0.8Z'/>"
        "<path fill='%23212121' "
        "d='M7.25 11.9h1.5V13h-1.5zM7.36 5.2h1.28l.3 5.4h-1.88z'/>"
        '</svg>")'
    ),
    "high_level": (
        'url("data:image/svg+xml;utf8,'
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
        "<path fill='%23f0ad4e' d='M8 1.1 15.2 14H0.8Z'/>"
        "<path fill='%23212121' "
        "d='M7.25 11.9h1.5V13h-1.5zM7.36 5.2h1.28l.3 5.4h-1.88z'/>"
        '</svg>")'
    ),
    "experimental": (
        'url("data:image/svg+xml;utf8,'
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16'>"
        "<path fill='%235cb85c' d='M8 1.1 15.2 14H0.8Z'/>"
        "<path fill='%23212121' "
        "d='M7.25 11.9h1.5V13h-1.5zM7.36 5.2h1.28l.3 5.4h-1.88z'/>"
        '</svg>")'
    ),
}
WARNING_CATEGORY_PRIORITY = {"dft": 3, "high_level": 2, "experimental": 1}
WARNING_CATEGORY_DESCRIPTIONS = {
    "dft": "DFT functional mismatch",
    "high_level": "High-level theory mismatch",
    "experimental": "Experimental reference mismatch",
}
DEFAULT_WARNING_CATEGORY = "dft"

# Tokens grouped by category keeps the classification simple.
CATEGORY_TOKENS = {
    "experimental": ("experimental",),
    "high_level": ("ccsd(t)", "mp2", "dlpno-ccsd(t)/cbs", "dmc"),
    "dft": ("pbe", "r2scan"),
}


def _categorize_benchmark_level(level: str | None) -> str | None:
    """
    Infer the warning icon category for a benchmark level.

    Parameters
    ----------
    level
        Raw benchmark level-of-theory string.

    Returns
    -------
    str | None
        Canonical category key (``"dft"``, ``"high_level"``, or
        ``"experimental"``). Returns ``DEFAULT_WARNING_CATEGORY`` when the level
        is missing or unknown.
    """
    if not level:
        return DEFAULT_WARNING_CATEGORY

    normalized = str(level).strip().lower()

    for category, tokens in CATEGORY_TOKENS.items():
        if any(token in normalized for token in tokens):
            return category

    return DEFAULT_WARNING_CATEGORY


def build_level_of_theory_warnings(
    rows: list[dict[str, Any]] | None,
    model_theory_levels: Mapping[str, str | None] | None,
    metric_levels: Mapping[str, str | None] | None,
    model_configs: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Generate inline styles and tooltips for MLIP column metadata.

    Parameters
    ----------
    rows
        Table rows currently displayed in the DataTable.
    model_theory_levels
        Mapping of model name to configured level of theory.
    metric_levels
        Mapping of metric name to benchmark level of theory.
    model_configs
        Mapping of model name to stored configuration details.

    Returns
    -------
    tuple[list[dict[str, Any]], list[dict[str, Any]]]
        A tuple of (warning_styles, tooltip_rows):
        - warning_styles: List of conditional style dicts for warning icons
        - tooltip_rows: List of tooltip data dicts, one per row
    """
    if not rows:
        return [], []

    model_theory_levels = model_theory_levels or {}
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

    def _get_model_config(mlip: str) -> tuple[Mapping[str, Any], str | None]:
        """
        Retrieve the stored configuration and theory level for a model.

        Parameters
        ----------
        mlip
            Model identifier used in the table.

        Returns
        -------
        tuple[Mapping[str, Any], str | None]
            Tuple of (config, level_of_theory) where config falls back to the
            registry entry when missing.
        """
        model_theory_level = model_theory_levels.get(mlip)
        config = model_configs.get(mlip) or {}
        if not isinstance(config, Mapping):
            config = {}
        if not config:
            fallback_cfg = load_model_registry_configs().get(mlip) or {}
            if isinstance(fallback_cfg, Mapping):
                config = fallback_cfg
        if model_theory_level is None and isinstance(config, Mapping):
            model_theory_level = config.get("level_of_theory")
        return config, model_theory_level

    def _build_overview_lines(
        mlip: str, config: Mapping[str, Any], model_theory_level: str | None
    ) -> list[str]:
        """
        Create the overview section describing model metadata.

        Parameters
        ----------
        mlip
            Model identifier to display.
        config
            Configuration mapping for the model.
        model_theory_level
            Stored level-of-theory metadata for the model.

        Returns
        -------
        list[str]
            Render-ready markdown lines describing the model overview.
        """
        level_display = (
            _stringify(model_theory_level)
            if model_theory_level not in (None, "")
            else "n/a"
        )
        overview_lines = [
            f"- **Model:** `{mlip}`",
            f"- **Level of theory:** `{level_display}`",
        ]

        for key in ("module", "class_name", "device"):
            value = config.get(key)
            if value:
                label = {"module": "Module", "class_name": "Class", "device": "Device"}[
                    key
                ]
                overview_lines.append(f"- **{label}:** `{value}`")

        kwargs = config.get("kwargs")
        kwarg_items: list[tuple[str, Any]] = []
        if isinstance(kwargs, Mapping):
            kwarg_items = [
                (key, value) for key, value in kwargs.items() if value not in (None, "")
            ]
        elif kwargs not in (None, {}):
            kwarg_items = [("value", kwargs)]

        if kwarg_items:
            overview_lines.append("- **Kwargs:**")
            for key, value in sorted(kwarg_items):
                overview_lines.append(f"  - `{key}`: `{_stringify(value)}`")
        else:
            overview_lines.append("- **Kwargs:** (none)")

        return overview_lines

    def _build_other_settings(config: Mapping[str, Any]) -> list[str]:
        """
        Build additional settings lines from configuration values.

        Parameters
        ----------
        config
            Model configuration dictionary.

        Returns
        -------
        list[str]
            Markdown formatted lines for non-core configuration entries.
        """
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
        if other_keys:
            return [
                f"- `{key}`: `{_stringify(value)}`"
                for key, value in sorted(other_keys.items())
            ]
        return []

    def _find_level_mismatches(
        model_theory_level: str | None,
    ) -> tuple[list[tuple[str, str]], str | None]:
        """
        Detect benchmark metrics whose levels differ from the model metadata.

        Parameters
        ----------
        model_theory_level
            Level-of-theory configured for the model.

        Returns
        -------
        tuple[list[tuple[str, str]], str | None]
            Pair containing the mismatched metrics and the warning category
            with the highest priority.
        """
        mismatch_metrics: list[tuple[str, str]] = []
        row_warning_category = None
        row_priority = 0

        for metric_name, metric_level in metric_levels.items():
            if model_theory_level is None or model_theory_level != metric_level:
                mismatch_metrics.append((metric_name, metric_level))
                category = _categorize_benchmark_level(metric_level)
                priority = WARNING_CATEGORY_PRIORITY.get(
                    category, WARNING_CATEGORY_PRIORITY[DEFAULT_WARNING_CATEGORY]
                )
                if priority > row_priority:
                    row_priority = priority
                    row_warning_category = category

        return mismatch_metrics, row_warning_category

    def _build_alignment_lines(
        model_theory_level: str | None, mismatch_metrics: list[tuple[str, str]]
    ) -> list[str]:
        """
        Build markdown rows describing benchmark alignment status.

        Parameters
        ----------
        model_theory_level
            Stored level-of-theory metadata for the model.
        mismatch_metrics
            Sequence of (metric, level) entries that do not match the model.

        Returns
        -------
        list[str]
            Markdown formatted lines for the benchmark alignment section.
        """
        if not metric_levels:
            return []

        level_display = (
            _stringify(model_theory_level)
            if model_theory_level not in (None, "")
            else "n/a"
        )

        if mismatch_metrics:
            align_lines = [
                f"- Model level: `{level_display}`",
                "- Benchmark metrics:",
            ]
            for metric_name, metric_level in mismatch_metrics:
                level_repr = _stringify(metric_level) if metric_level else "n/a"
                category = _categorize_benchmark_level(metric_level)
                category_label = WARNING_CATEGORY_DESCRIPTIONS.get(
                    category, "Level mismatch"
                )
                align_lines.append(
                    f"  - {metric_name}: `{level_repr}` (**{category_label}**)"
                )
            return align_lines
        return [
            f"- Model level: `{level_display}`",
            "- All benchmark metrics match the model level.",
        ]

    # Process each row to build tooltips and warning styles
    for idx, row in enumerate(rows):
        mlip = row.get("MLIP")
        if not isinstance(mlip, str):
            continue

        # Get model configuration and level of theory
        config, model_theory_level = _get_model_config(mlip)

        # Build tooltip sections
        overview_lines = _build_overview_lines(mlip, config, model_theory_level)
        other_lines = _build_other_settings(config)
        mismatch_metrics, row_warning_category = _find_level_mismatches(
            model_theory_level
        )
        align_lines = _build_alignment_lines(model_theory_level, mismatch_metrics)

        # Assemble tooltip
        tooltip_sections: list[str] = []
        tooltip_sections.extend(_section("Model Overview", overview_lines))
        if other_lines:
            tooltip_sections.extend(_section("Additional Settings", other_lines))
        if align_lines:
            tooltip_sections.extend(_section("Benchmark Alignment", align_lines))

        while tooltip_sections and tooltip_sections[-1] == "":
            tooltip_sections.pop()

        tooltip_rows[idx]["MLIP"] = {
            "type": "markdown",
            "value": "\n".join(tooltip_sections),
        }

        # Add warning icon if mismatches exist
        if mismatch_metrics:
            row_id = row.get("id", mlip)
            filter_query = "{id} = " + json.dumps(str(row_id))
            icon_key = row_warning_category or DEFAULT_WARNING_CATEGORY
            icon_url = WARNING_ICON_URLS.get(
                icon_key, WARNING_ICON_URLS[DEFAULT_WARNING_CATEGORY]
            )
            warning_styles.append(
                {
                    "if": {"column_id": "MLIP", "filter_query": filter_query},
                    "backgroundImage": icon_url,
                    "backgroundRepeat": "no-repeat",
                    "backgroundPosition": "8px center",
                    "backgroundSize": "14px 14px",
                    "paddingLeft": "38px",
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

    pattern = f" [{unit}]"
    if pattern not in text:
        return text
    return text.replace(pattern, f" [{replacement}]", 1)


def format_tooltip_headers(
    tooltip_header: dict[str, str] | None,
    thresholds: Thresholds | None,
    show_normalized: bool,
) -> dict[str, Any] | None:
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

    updated: dict[str, Any] = {}
    for key, entry in tooltip_header.items():
        if key in reserved:
            updated[key] = entry
            continue

        text_val: str | None = None

        if isinstance(entry, dict):
            raw_value = entry.get("value")
            if isinstance(raw_value, str):
                text_val = raw_value
        elif isinstance(entry, str):
            text_val = entry

        if text_val is None:
            updated[key] = entry
            continue

        unit = thresholds.get(key, {}).get("unit")
        if unit and unit not in ("", "-"):
            base_text = _swap_tooltip_unit(text_val, unit, unit)
            if show_normalized:
                text_val = base_text.replace(f"[{unit}]", "[-]", 1)
            else:
                text_val = base_text

        if isinstance(entry, dict):
            new_entry = entry.copy()
            new_entry["value"] = text_val
            updated[key] = new_entry
        else:
            updated[key] = text_val

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
