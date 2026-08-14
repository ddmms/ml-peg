"""Utility functions for building app components."""

from __future__ import annotations

from collections.abc import Sequence
from importlib import metadata
from pathlib import Path
import time

from dash import html
from dash.dash_table import DataTable
from dash.dcc import Checklist, Download, Dropdown, Loading, Store
from dash.dcc import Input as DCC_Input
from dash.development.base_component import Component
from dash.html import H2, H3, Br, Button, Details, Div, Label, Summary
import yaml

from ml_peg.analysis.utils.utils import Thresholds, calc_table_scores, get_table_style
from ml_peg.app.utils.register_callbacks import (
    register_category_table_callbacks,
    register_download_callbacks,
    register_normalization_callbacks,
    register_summary_table_callbacks,
    register_weight_callbacks,
)
from ml_peg.app.utils.utils import (
    build_level_of_theory_warnings,
    build_threshold_input_style,
    calculate_column_widths,
    get_framework_config,
    get_mlip_column_width,
    get_threshold_colours,
    load_model_registry_configs,
    sig_fig_format,
    weight_input_style,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Width (px) of the docs-link column of the summary table (see build_app.py).
# kept so the weights row can be translated to align with cols
LINK_COLUMN_WIDTH = 36

# Max width of the interactive zone below a table (plots, structure viewers, ...).
# Bounds it so it does not stretch to the largest table width by default.
INTERACTIVE_ZONE_MAX_WIDTH = "1500px"

# Models to include as summary-table rows
MODELS = get_model_names(current_models)


def _format_summary_column_header(column_id: str) -> str:
    """
    Format summary-table headers for more compact wrapping.

    Non-static summary columns always place ``Score`` on its own line. Longer
    multi-word titles are split across two title lines first, yielding a compact
    three-line header.

    Parameters
    ----------
    column_id
        Summary-table column identifier.

    Returns
    -------
    str
        Header label with explicit newline breaks.
    """
    if column_id in {"MLIP", "Score"} or not column_id.endswith(" Score"):
        return column_id

    title = column_id.removesuffix(" Score")
    if len(title) > 14 and " " in title:
        words = title.split()
        split_index = min(
            range(1, len(words)),
            key=lambda index: abs(
                len(" ".join(words[:index])) - len(" ".join(words[index:]))
            ),
        )
        title = "\n".join(
            [" ".join(words[:split_index]), " ".join(words[split_index:])]
        )
    return f"{title}\nScore"


def build_summary_table(
    tables: dict[str, DataTable],
    table_id: str = "summary-table",
    description: str | None = None,
    weights: dict[str, float] | None = None,
    header_labels: dict[str, str] | None = None,
) -> DataTable:
    """
    Build summary table from a set of tables.

    Parameters
    ----------
    tables
        Dictionary of tables to be summarised.
    table_id
        ID of table being built. Default is 'summary-table'.
    description
        Description of summary table. Default is None.
    weights
        Weights for each column. Default is `None`, which sets all weights to 1.
    header_labels
        Optional mapping of table key to display header text. Column ids are
        unchanged, so only the rendered header differs. Default is None.

    Returns
    -------
    DataTable
        Summary table with scores from tables being summarised.
    """
    summary_data = {}
    category_columns = []  # Track all category columns
    for category_name, table in tables.items():
        # Prepare rows for all current models
        if not summary_data:
            summary_data = {model: {} for model in MODELS}

        category_col = f"{category_name} Score"
        category_columns.append(category_col)

        table_name_map = getattr(table, "model_name_map", {}) or {}
        for row in table.data:
            # Category tables may include models not to be included
            # Table headings are of the form "[category] Score"
            # ``original_name`` refers to the original model identifier
            # (no display suffix)
            original_name = table_name_map.get(row["MLIP"], row["MLIP"])
            if original_name in summary_data:
                summary_data[original_name][category_col] = row["Score"]

    # Ensure all models have entries for all category columns (None if missing)
    data = []
    for mlip in summary_data:
        row = {"MLIP": mlip}
        for category_col in category_columns:
            row[category_col] = summary_data[mlip].get(category_col, None)
        data.append(row)

    data = calc_table_scores(data, weights=weights)

    columns_headers = ("MLIP", "Score") + tuple(key + " Score" for key in tables)

    # Map each column id to its visible header text (labels + line wrapping)
    display_headers = {}
    for column_id in columns_headers:
        header = column_id
        if (
            header not in {"MLIP", "Score"}
            and header.endswith(" Score")
            and header_labels
        ):
            key = header.removesuffix(" Score")
            if key in header_labels:
                header = header_labels[key] + " Score"
        if table_id != "summary-table":
            display_headers[column_id] = _format_summary_column_header(header)
        elif header in {"MLIP", "Score"} or not header.endswith(" Score"):
            display_headers[column_id] = header
        else:
            display_headers[column_id] = "\n".join(
                [*header.removesuffix(" Score").split(), "Score"]
            )

    columns = [
        {"name": display_headers[header], "id": header} for header in columns_headers
    ]
    tooltip_header = {
        header + " Score": table.description for header, table in tables.items()
    }

    for column in columns:
        column_id = column["id"]
        if column_id != "MLIP":
            column["type"] = "numeric"
            column["format"] = sig_fig_format()

    style = get_table_style(data)
    registry_configs = load_model_registry_configs()
    row_models: list[str] = []
    for row in data:
        mlip = row.get("MLIP")
        if isinstance(mlip, str) and mlip not in row_models:
            row_models.append(mlip)
    model_configs = {mlip: (registry_configs.get(mlip) or {}) for mlip in row_models}
    model_levels = {
        mlip: (model_configs[mlip].get("level_of_theory")) for mlip in row_models
    }
    warning_styles, tooltip_rows = build_level_of_theory_warnings(
        data,
        model_levels,
        {},
        model_configs,
    )
    style_with_warnings = style + warning_styles

    summary_header_padding = 12 if table_id == "summary-table" else 24
    header_cell_padding = "4px" if table_id == "summary-table" else "8px"
    column_widths = {"MLIP": get_mlip_column_width(), "Score": 100}
    for column_id in columns_headers:
        if column_id in {"MLIP", "Score"}:
            continue
        longest_line = max(
            len(line) for line in display_headers[column_id].splitlines()
        )
        column_widths[column_id] = min(
            max(longest_line * 9 + summary_header_padding, 100), 150
        )

    style_cell_conditional = []
    for column_id, width in column_widths.items():
        col_width = f"{width}px"
        alignment = "left" if column_id == "MLIP" else "center"
        style_cell_conditional.append(
            {
                "if": {"column_id": column_id},
                "width": col_width,
                "minWidth": col_width,
                "maxWidth": col_width,
                "textAlign": alignment,
            }
        )

    tooltip_header["Score"] = "Weighted average of scores (higher is better)"

    # Per-model docs link, on the overall summary table only, rendered as an
    # icon just after the model name. Its styling lives in
    # ml_peg/app/data/utils/link_column.css (auto-loaded as a Dash asset);
    # NaN/level-of-theory greying is kept off for the link column.
    if table_id == "summary-table":
        models_url = "https://ddmms.github.io/ml-peg/user_guide/models.html"
        for row in data:
            anchor = row.get("MLIP")
            row["link"] = f"[🔗]({models_url}#{anchor})" if anchor else ""
        columns.insert(1, {"id": "link", "name": "", "presentation": "markdown"})
        style_cell_conditional.append(
            {
                "if": {"column_id": "link"},
                "width": f"{LINK_COLUMN_WIDTH}px",
                "minWidth": f"{LINK_COLUMN_WIDTH}px",
                "maxWidth": f"{LINK_COLUMN_WIDTH}px",
                "textAlign": "left",
                "padding": "0",
                "borderLeft": "none",
            }
        )
        style_cell_conditional.append(
            {"if": {"column_id": "MLIP"}, "borderRight": "none"}
        )
        style_with_warnings = style_with_warnings + [
            {
                "if": {"column_id": "link"},
                "backgroundColor": "white",
                "backgroundImage": "none",
            }
        ]

    table = DataTable(
        data=data,
        columns=columns,
        id=table_id,
        markdown_options={"link_target": "_blank"},
        sort_action="native",
        style_data_conditional=style_with_warnings,
        style_cell_conditional=style_cell_conditional,
        style_header={
            "whiteSpace": "pre-line",
            "height": "auto",
            "minHeight": "70px",
            "textAlign": "center",
            "verticalAlign": "middle",
            "lineHeight": "1.4",
            "padding": header_cell_padding,
        },
        style_header_conditional=[
            {
                "if": {"column_id": "MLIP"},
                "textAlign": "left",
            }
        ],
        tooltip_data=tooltip_rows,
        tooltip_delay=100,
        tooltip_duration=None,
        tooltip_header=tooltip_header,
        editable=False,
        fill_width=False,
    )
    table.column_widths = column_widths
    table.description = description
    table.model_levels_of_theory = model_levels
    table.metric_levels_of_theory = {}
    table.model_configs = model_configs
    table.weights = weights
    return table


def grid_template_from_widths(
    widths: dict[str, int],
    column_order: list[str],
) -> str:
    """
    Compose a CSS grid template string from column widths.

    Parameters
    ----------
    widths
        Mapping of column names to pixel widths.
    column_order
        Ordered metric columns rendered after the ``MLIP`` and ``Score`` columns.

    Returns
    -------
    str
        CSS grid template definition using fixed pixel tracks.
    """
    tracks: list[tuple[str, int]] = [
        ("MLIP", widths["MLIP"]),
        ("Score", widths["Score"]),
    ]
    tracks.extend((col, widths[col]) for col in column_order)

    return " ".join(f"{max(width, 40)}px" for _, width in tracks)


def build_weight_input(
    input_id: str,
    default_value: float | None,
    show_label: bool = False,
) -> Div:
    """
    Build numeric input for a metric weight.

    Parameters
    ----------
    input_id
        ID for text box input component.
    default_value
        Default value for the text box input.
    show_label
        Whether to show the "Weight:" label (only for first column).

    Returns
    -------
    Div
        Div wrapping the input box.
    """
    wrapper_style: dict[str, str] = {
        "display": "flex",
        "justifyContent": "center",
        "alignItems": "center",
        "boxSizing": "border-box",
        "border": "1px solid transparent",
        "position": "relative",
    }
    wrapper_style.update(
        {
            "width": "100%",
            "minWidth": "0",
            "maxWidth": "100%",
            "height": "100%",
        }
    )

    children = []
    if show_label:
        children.append(
            Label(
                "Weight:",
                style={
                    "fontSize": "13px",
                    "color": "#6c757d",
                    "textAlign": "right",
                    "position": "absolute",
                    "right": "calc(50% + 38px)",
                },
            )
        )

    children.append(
        DCC_Input(
            id=input_id,
            type="number",
            value=default_value,
            step=0.01,
            debounce=True,
            style=weight_input_style(),
        )
    )

    return Div(children, style=wrapper_style)


def build_weight_components(
    header: str,
    table: DataTable,
    *,
    use_thresholds: bool = False,
    include_download_controls: bool = True,
    column_widths: dict[str, int] | None = None,
    thresholds: Thresholds | None = None,
) -> Div:
    """
    Build weight sliders, text boxes and reset button.

    Parameters
    ----------
    header
        Header for above sliders.
    table
        DataTable to build weight components for.
    use_thresholds
        Whether this table also exposes normalization thresholds. When True,
        weight callbacks will reuse the raw-data store and normalization store to
        recompute Scores consistently.
    include_download_controls
        Whether to render download controls in the Score column slot.
    column_widths
        Optional mapping of table column IDs to pixel widths used to align the
        inputs with the rendered table.
    thresholds
        Threshold metadata used when ``use_thresholds`` is ``True``.

    Returns
    -------
    Div
        Div containing header, weight sliders, text boxes and reset button.
    """
    if use_thresholds and thresholds is None:
        raise ValueError(
            "Threshold metadata must be provided when use_thresholds=True."
        )
    # Identify metric columns (exclude reserved columns)
    reserved = {"MLIP", "Score", "id", "link"}
    columns = [col["id"] for col in table.columns if col.get("id") not in reserved]

    if not columns:
        return Div()

    input_ids = [f"{table.id}-{col}" for col in columns]

    # Set default weights
    weights = table.weights if table.weights else {}
    for column in columns:
        weights.setdefault(column, 1.0)

    widths = calculate_column_widths(columns, column_widths)
    grid_template = grid_template_from_widths(widths, columns)

    # The overall summary table has a "link" column inserted after MLIP. Reserve
    # a matching spacer track (and grid cell) so the weight boxes stay aligned.
    has_link_column = any(col.get("id") == "link" for col in table.columns)
    link_spacer: list[Div] = []
    if has_link_column:
        tracks = grid_template.split(" ")
        tracks.insert(1, f"{LINK_COLUMN_WIDTH}px")
        grid_template = " ".join(tracks)
        link_spacer = [Div(style={"border": "1px solid transparent"})]

    weight_inputs = [
        build_weight_input(
            input_id=f"{input_id}-input",
            default_value=weights[column],
            show_label=(i == 0),
        )
        for i, (column, input_id) in enumerate(zip(columns, input_ids, strict=True))
    ]

    container = Div(
        [
            Div(
                [
                    Div(
                        header,
                        style={
                            "fontWeight": "bold",
                            "fontSize": "13px",
                            "padding": "2px 4px",
                            "color": "#212529",
                            "whiteSpace": "nowrap",
                            "boxSizing": "border-box",
                            "border": "1px solid transparent",
                        },
                    ),
                    Button(
                        "Reset",
                        id=f"{table.id}-reset-button",
                        n_clicks=0,
                        style={
                            "fontSize": "11px",
                            "padding": "4px 8px",
                            "marginTop": "0px",
                            "marginLeft": "4px",
                            "backgroundColor": "#6c757d",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "3px",
                            "width": "fit-content",
                            "cursor": "pointer",
                        },
                    ),
                    Div(
                        "Press Enter or click away to apply new weights or thresholds",
                        style={
                            "fontSize": "11px",
                            "color": "#6c757d",
                            "fontStyle": "italic",
                            "marginTop": "2px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "flex-start",
                    "gap": "4px",
                    "boxSizing": "border-box",
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "border": "1px solid transparent",  # #dee2e6 or transparent
                },
            ),
            *link_spacer,
            build_download_controls(table.id)
            if include_download_controls
            else Div(
                "",
                style={
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "boxSizing": "border-box",
                    "border": "1px solid transparent",
                },
            ),
            *weight_inputs,
        ],
        style={
            "display": "grid",
            "gridTemplateColumns": grid_template,
            "alignItems": "start",
            "columnGap": "0px",
            "rowGap": "4px",
            "marginTop": "-5px",
            "padding": "2px 0px",
            "backgroundColor": "#f8f9fa",
            "border": "1px solid transparent"
            if header == "Weights"
            else "1px solid #dee2e6",
            "borderRadius": "6px",
            "width": "100%",
            "minWidth": "0",
            "boxSizing": "border-box",
        },
    )

    layout = [container]

    model_levels = getattr(table, "model_levels_of_theory", None)
    metric_levels = getattr(table, "metric_levels_of_theory", None)
    model_configs = getattr(table, "model_configs", None)

    # Callbacks to update table scores when table weight dicts change
    if table.id == "summary-table":
        register_summary_table_callbacks(
            initial_rows=table.data,
            model_levels=model_levels,
            metric_levels=metric_levels,
            model_configs=model_configs,
            prefix="summary-table",
        )
    elif table.id == "framework-summary-table":
        register_summary_table_callbacks(
            initial_rows=table.data,
            model_levels=model_levels,
            metric_levels=metric_levels,
            model_configs=model_configs,
            prefix="framework-summary-table",
        )
    elif table.id.endswith("-framework-summary-table"):
        register_category_table_callbacks(
            table_id=table.id,
            use_thresholds=use_thresholds,
            model_levels=model_levels,
            metric_levels=metric_levels,
            model_configs=model_configs,
            scores_store_id="framework-summary-table-scores-store",
            summary_suffix="-framework-summary-table",
        )
    else:
        register_category_table_callbacks(
            table_id=table.id,
            use_thresholds=use_thresholds,
            model_levels=model_levels,
            metric_levels=metric_levels,
            model_configs=model_configs,
        )

    # Callbacks to sync sliders, text boxes, and stored table weights
    for column, input_id in zip(columns, input_ids, strict=True):
        register_weight_callbacks(
            input_id=input_id,
            table_id=table.id,
            column=column,
            default_weights=getattr(table, "weights", None),
        )

    register_download_callbacks(table.id)

    return Div(layout)


def build_download_controls(table_id: str, *, row: bool = False) -> Div:
    """
    Build minimal table download controls.

    Parameters
    ----------
    table_id
        ID of the table to export.
    row
        When True, arrange the dropdown and button horizontally.

    Returns
    -------
    Div
        Download controls and target components.
    """
    if row:
        container_style = {
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "flex-end",
            "gap": "8px",
            "flexShrink": "0",
            "marginBottom": "16px",
        }
    else:
        container_style = {
            "display": "flex",
            "flexDirection": "column",
            "alignItems": "center",
            "gap": "6px",
            "width": "100%",
            "padding": "2px 0px",
        }

    return Div(
        [
            Dropdown(
                id=f"{table_id}-download-format",
                className="download-format table-download-format",
                options=[
                    {"label": "CSV", "value": "csv"},
                    {"label": "PNG", "value": "png"},
                    {"label": "SVG", "value": "svg"},
                ],
                value="csv",
                clearable=False,
                searchable=False,
                style={
                    "width": "76px",
                    "height": "30px",
                    "fontSize": "12px",
                },
            ),
            Button(
                "Download table",
                id=f"{table_id}-download-button",
                className="download-button table-download-button",
                n_clicks=0,
                style={
                    "width": "118px",
                },
            ),
            Download(id=f"{table_id}-download"),
            Store(id=f"{table_id}-download-request", storage_type="memory"),
        ],
        style=container_style,
    )


def build_page_loading_spinner() -> Div:
    """
    Build the initial page-load spinner overlay.

    Returns
    -------
    Div
        Page-wide loading overlay with spinner and status text.
    """
    return Div(
        [
            Div(
                style={
                    "width": "52px",
                    "height": "52px",
                    "border": "5px solid #d0ebff",
                    "borderTopColor": "#119DFF",
                    "borderRadius": "50%",
                    "animation": "ml-peg-spin 0.8s linear infinite",
                    "boxSizing": "border-box",
                },
            ),
            Div(
                "Loading page...",
                style={
                    "fontSize": "16px",
                    "fontWeight": "600",
                    "color": "#212529",
                },
            ),
        ],
        style={
            "position": "absolute",
            "top": "0",
            "right": "0",
            "bottom": "0",
            "left": "0",
            "minHeight": "420px",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "flex-start",
            "flexDirection": "column",
            "gap": "14px",
            "paddingTop": "96px",
            "boxSizing": "border-box",
            "backgroundColor": "rgba(255, 255, 255, 0.78)",
            "zIndex": "1400",
            "pointerEvents": "auto",
        },
    )


def build_table_loading_spinner() -> Div:
    """
    Build a compact table loading spinner.

    Returns
    -------
    Div
        Table-sized loading overlay with the same ring style as page loading.
    """
    return Div(
        Div(
            style={
                "width": "34px",
                "height": "34px",
                "border": "4px solid #d0ebff",
                "borderTopColor": "#119DFF",
                "borderRadius": "50%",
                "animation": "ml-peg-spin 0.8s linear infinite",
                "boxSizing": "border-box",
            },
        ),
        style={
            "position": "absolute",
            "top": "0",
            "right": "0",
            "bottom": "0",
            "left": "0",
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "backgroundColor": "rgba(255, 255, 255, 0.65)",
            "zIndex": "1200",
            "pointerEvents": "auto",
        },
    )


def build_filter_overlay(table_id: str, child, delay_hide: int = 250) -> Loading:
    """
    Wrap a table in a filter-aware loading overlay.

    The overlay id uses the ``{"type": "filter-overlay", "index": table_id}``
    pattern so a single callback can drive every table's spinner. ``display``
    starts as ``"auto"`` (driven by ``target_components`` for weight/threshold
    and post-recompute renders); the filter-loading callback flips it to
    ``"show"`` for the duration of an "Apply" recompute, then back to ``"auto"``.

    Parameters
    ----------
    table_id
        Id of the wrapped DataTable, used for both the overlay pattern index and
        the loading target component.
    child
        Component tree to render under the overlay.
    delay_hide
        Milliseconds to keep the spinner up after loading ends. A larger value
        bridges the gaps between the several quick updates that a summary table
        receives as scores propagate, so the user sees one spinner not several.

    Returns
    -------
    Loading
        Loading wrapper scoped to the table.
    """
    return Loading(
        child,
        id={"type": "filter-overlay", "index": table_id},
        fullscreen=False,
        custom_spinner=build_table_loading_spinner(),
        target_components={table_id: ["data", "style_data_conditional"]},
        show_initially=False,
        delay_hide=delay_hide,
        overlay_style={"visibility": "visible", "opacity": 1},
        parent_style={"position": "relative", "width": "fit-content"},
    )


def build_loading_summary_table(table: DataTable) -> Loading:
    """
    Wrap a summary DataTable with a local loading spinner.

    Parameters
    ----------
    table
        Summary table to show as loading while Dash applies filters and updates
        its rendered data.

    Returns
    -------
    Loading
        Loading wrapper scoped to applied filter changes and table updates.
    """
    # Longer delay_hide than the default: a summary table gets several quick
    # updates in a row as category scores propagate, so bridge the gaps into one
    # spinner instead of several flashes.
    return build_filter_overlay(table.id, Div(table), delay_hide=600)


def build_plot_download_controls(graph_id: str) -> Div:
    """
    Build plot download controls for CSV, PNG, SVG, and HTML exports.

    Parameters
    ----------
    graph_id
        ID of the Plotly graph to export.

    Returns
    -------
    Div
        Download controls and target components.
    """
    return Div(
        [
            Dropdown(
                id={"type": "plot-download-format", "index": graph_id},
                className="download-format plot-download-format",
                options=[
                    {"label": "CSV", "value": "csv"},
                    {"label": "PNG", "value": "png"},
                    {"label": "SVG", "value": "svg"},
                    {"label": "HTML", "value": "html"},
                ],
                value="csv",
                clearable=False,
                searchable=False,
                style={
                    "width": "76px",
                    "height": "30px",
                    "fontSize": "12px",
                },
            ),
            Button(
                "Download plot",
                id={"type": "plot-download-button", "index": graph_id},
                className="download-button plot-download-button",
                n_clicks=0,
                style={
                    "width": "112px",
                },
            ),
            Download(id={"type": "plot-download", "index": graph_id}),
            Store(
                id={"type": "plot-download-target", "index": graph_id},
                storage_type="memory",
                data=graph_id,
            ),
        ],
        style={
            "display": "flex",
            "flexDirection": "row",
            "alignItems": "flex-end",
            "gap": "8px",
            "flexShrink": "0",
            "marginTop": "12px",
            "marginBottom": "0px",
        },
    )


def _plot_setting_id(control: str, graph_id: str) -> dict[str, str]:
    """
    Build a pattern-matching ID for one graph's plot setting control.

    Parameters
    ----------
    control
        Name of the plot setting control.
    graph_id
        String ID of the graph controlled by the menu.

    Returns
    -------
    dict[str, str]
        Dash pattern-matching component ID.
    """
    return {"type": f"plot-settings-{control}", "index": graph_id}


def build_plot_settings_controls(graph_id: str) -> Details:
    """
    Build minimal axis controls for a Plotly graph.

    Parameters
    ----------
    graph_id
        String ID of the graph controlled by the menu.

    Returns
    -------
    Details
        Collapsible X/Y scale and range controls.
    """

    def axis_controls(axis: str) -> Div:
        """
        Build the scale and range inputs for one axis.

        Parameters
        ----------
        axis
            Axis identifier, either ``"x"`` or ``"y"``.

        Returns
        -------
        Div
            Row containing the axis label, scale selector, and range inputs.
        """
        label = axis.upper()
        return Div(
            [
                Label(f"{label} axis", className="plot-settings-axis-label"),
                Dropdown(
                    id=_plot_setting_id(f"{axis}-scale", graph_id),
                    options=[
                        {"label": "Linear", "value": "linear"},
                        {"label": "Log", "value": "log"},
                    ],
                    value="linear",
                    clearable=False,
                    searchable=False,
                    className="plot-settings-scale",
                ),
                DCC_Input(
                    id=_plot_setting_id(f"{axis}-min", graph_id),
                    type="number",
                    placeholder="Min",
                    debounce=True,
                    className="plot-settings-number",
                ),
                DCC_Input(
                    id=_plot_setting_id(f"{axis}-max", graph_id),
                    type="number",
                    placeholder="Max",
                    debounce=True,
                    className="plot-settings-number",
                ),
            ],
            className="plot-settings-axis-row",
        )

    return Details(
        [
            Summary("Plot settings", className="plot-settings-summary"),
            Div(
                [
                    axis_controls("x"),
                    axis_controls("y"),
                    Div(
                        "For explicit limits, enter both minimum and maximum. "
                        "Log axes hide non-positive values.",
                        className="plot-settings-help",
                    ),
                    Div(
                        [
                            Button(
                                "Apply",
                                id=_plot_setting_id("apply", graph_id),
                                n_clicks=0,
                                className="plot-settings-apply",
                            ),
                            Button(
                                "Reset axes",
                                id=_plot_setting_id("reset", graph_id),
                                n_clicks=0,
                                className="plot-settings-reset",
                            ),
                        ],
                        className="plot-settings-actions",
                    ),
                    Div(
                        id=_plot_setting_id("message", graph_id),
                        className="plot-settings-message",
                        role="alert",
                    ),
                    Store(id=_plot_setting_id("result", graph_id)),
                    Store(
                        id=_plot_setting_id("graph-id", graph_id),
                        data=graph_id,
                    ),
                ],
                className="plot-settings-body",
            ),
        ],
        className="plot-settings",
    )


def build_faqs() -> Div:
    """
    Build FAQ section with collapsible dropdowns from YAML file.

    Returns
    -------
    Div
        Styled FAQ section with questions as dropdown titles and answers inside.
    """
    # Load FAQs from YAML file
    faqs_path = Path(__file__).parent / "faqs.yml"

    try:
        with open(faqs_path, encoding="utf8") as f:
            faqs_data = yaml.safe_load(f)
    except FileNotFoundError:
        return Div(
            "FAQs file not found",
            style={
                "color": "#dc3545",
                "padding": "10px",
                "fontStyle": "italic",
            },
        )

    if not faqs_data or not isinstance(faqs_data, list):
        return Div("No FAQs available")

    # Build FAQ dropdowns
    faq_components = []

    for faq in faqs_data:
        question = faq.get("question", "")
        answer = faq.get("answer", "")
        docs_url = faq.get("docs_url")

        if not question or not answer:
            continue

        # Build answer content with optional docs link
        answer_content = [answer]
        if docs_url:
            answer_content.append(
                Div(
                    html.A(
                        "View documentation →",
                        href=docs_url,
                        target="_blank",
                        style={
                            "color": "#0d6efd",
                            "textDecoration": "none",
                            "fontWeight": "600",
                            "fontSize": "13px",
                        },
                    ),
                    style={"marginTop": "8px"},
                )
            )

        faq_components.append(
            Details(
                [
                    Summary(
                        question,
                        style={
                            "cursor": "pointer",
                            "fontWeight": "bold",
                            "padding": "10px",
                            "backgroundColor": "#f8f9fa",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "4px",
                            "marginBottom": "8px",
                        },
                    ),
                    Div(
                        answer_content,
                        style={
                            "padding": "10px 15px",
                            "backgroundColor": "#ffffff",
                            "border": "1px solid #dee2e6",
                            "borderTop": "none",
                            "borderRadius": "0 0 4px 4px",
                            "marginTop": "-8px",
                            "marginBottom": "8px",
                        },
                    ),
                ],
                style={
                    "marginBottom": "8px",
                },
            )
        )

    return Div(
        [
            H2(
                "Frequently Asked Questions",
                style={"color": "black", "marginTop": "30px"},
            ),
            Div(faq_components),
        ]
    )


def build_footer() -> html.Footer:
    """
    Build shared footer with author, copyright, and repository link.

    Returns
    -------
    html.Footer
        Styled footer component rendered at the bottom of each tab.
    """
    copyright_first_year = "2025"
    current_year = str(time.localtime().tm_year)
    copyright_owners = metadata.metadata("ml-peg")["author"]

    copyright_year_string = (
        current_year
        if current_year == copyright_first_year
        else f"{copyright_first_year}-{current_year}"
    )
    copyright_txt = (
        f"Copyright © {copyright_year_string}, {copyright_owners}. All rights reserved"
    )

    return html.Footer(
        [
            Div(
                [
                    html.Span(copyright_txt, style={"fontWeight": "800"}),
                ],
                style={
                    "display": "flex",
                    "justifyContent": "center",
                    "flexWrap": "wrap",
                    "gap": "6px",
                },
            ),
            Div(
                html.A(
                    "GPL-3.0 license",
                    href="https://github.com/ddmms/ml-peg/blob/main/LICENSE",
                    target="_blank",
                ),
                style={"marginTop": "4px", "fontWeight": "800"},
            ),
            Div(
                html.A(
                    [
                        html.Img(
                            src=(
                                "https://github.githubassets.com/images/modules/logos_page/"
                                "GitHub-Mark.png"
                            ),
                            alt="GitHub",
                            width="16",
                            height="16",
                            style={
                                "marginRight": "6px",
                                "verticalAlign": "middle",
                                "display": "block",
                            },
                        ),
                        html.Span("View on GitHub"),
                    ],
                    href="https://github.com/ddmms/ml-peg",
                    target="_blank",
                    style={
                        "color": "#0d6efd",
                        "textDecoration": "none",
                        "fontWeight": "800",
                        "display": "inline-flex",
                        "alignItems": "center",
                    },
                ),
                style={"marginTop": "6px"},
            ),
        ],
        style={
            "marginTop": "24px",
            "padding": "14px 12px",
            "color": "#343a40",
            "fontSize": "12px",
            "textAlign": "center",
            "borderTop": "1px solid #dee2e6",
            "background": "#f8f9fa",
            "borderRadius": "6px",
        },
    )


def build_framework_badge(framework_id: str) -> Component:
    """
    Build a visual framework attribution badge.

    Parameters
    ----------
    framework_id
        Framework identifier for the benchmark.

    Returns
    -------
    Component
        Styled badge, wrapped as a link when framework docs URL is configured.
    """
    config = get_framework_config(framework_id)
    label = config["label"]
    color = config["color"]
    text_color = config["text_color"]
    logo = config.get("logo")
    icon = config.get("icon")
    tooltip = config.get("tooltip")
    url = config.get("url")

    badge_style = {
        "display": "inline-flex",
        "alignItems": "center",
        "padding": "2px 8px",
        "borderRadius": "999px",
        "fontSize": "11px",
        "fontWeight": "600",
        "letterSpacing": "0.02em",
        "textTransform": "uppercase",
        "backgroundColor": color,
        "color": text_color,
        "lineHeight": "1.8",
    }

    badge_children: list[Component] = []
    if logo:
        badge_children.append(
            html.Img(
                src=logo,
                alt=f"{label} logo",
                style={
                    "width": "14px",
                    "height": "14px",
                    "borderRadius": "50%",
                    "objectFit": "cover",
                },
            )
        )
    if icon:
        badge_children.append(html.Span(icon, **{"aria-hidden": "true"}))
    badge_children.append(html.Span(label))
    badge = html.Span(
        badge_children,
        style={
            **badge_style,
            "gap": "6px",
        },
    )
    if url:
        return html.A(
            badge,
            href=url,
            target="_blank",
            style={"textDecoration": "none"},
            title=tooltip or f"Open {label} website",
        )
    if tooltip:
        badge.title = tooltip
    return badge


def build_test_layout(
    name: str,
    description: str,
    framework_ids: Sequence[str],
    table: DataTable,
    thresholds: Thresholds,
    extra_components: list[Component] | None = None,
    docs_url: str | None = None,
    column_widths: dict[str, int] | None = None,
) -> Div:
    """
    Build app layout for a test.

    Parameters
    ----------
    name
        Name of test.
    description
        Description of test.
    framework_ids
        Framework identifiers used to render attribution badges.
    table
        Dash Table with metric results. Can include a `weights` attribute to be used by
        `build_weight_components`.
    thresholds
        Normalization metadata (metric -> (good, bad, unit)) supplied via the
        analysis pipeline. Inline threshold controls are rendered automatically.
    extra_components
        List of Dash Components to include after the metrics table.
    docs_url
        URL to online documentation. Default is None.
    column_widths
        Optional column-width mapping inferred from analysis output. Used to align
        threshold controls beneath the table columns when available.

    Returns
    -------
    Div
        Layout for test layout.
    """
    layout_contents = [
        Div(
            [
                H2(name, style={"color": "black", "margin": "0"}),
                *[
                    build_framework_badge(framework_id)
                    for framework_id in framework_ids
                ],
            ],
            style={
                "display": "flex",
                "alignItems": "center",
                "flexWrap": "wrap",
                "gap": "10px",
            },
        ),
        H3(description),
    ]

    layout_contents.extend(
        [
            Details(
                [
                    Summary(
                        "Click for more information",
                        style={
                            "cursor": "pointer",
                            "fontWeight": "bold",
                            "padding": "5px",
                        },
                    ),
                    Label(
                        [html.A("Online documentation", href=docs_url, target="_blank")]
                    ),
                ],
                style={
                    # "border": "1px solid #ddd",
                    "padding": "10px",
                    # "borderRadius": "5px",
                },
            ),
            Div(style={"height": "4px"}),
        ]
    )

    reserved = {"MLIP", "Score", "id", "link"}
    metric_columns = [
        col["id"] for col in table.columns if col.get("id") not in reserved
    ]

    layout_contents.append(
        Store(
            id=f"{table.id}-raw-tooltip-store",
            storage_type="session",
            data=table.tooltip_header,
        )
    )

    threshold_controls = build_threshold_inputs(
        table_columns=metric_columns,
        thresholds=thresholds,
        table_id=table.id,
        column_widths=column_widths,
    )

    # Add metric-weight controls for every benchmark table
    metric_weights = build_weight_components(
        header="Weights",
        table=table,
        use_thresholds=thresholds is not None,
        include_download_controls=False,
        column_widths=column_widths,
        thresholds=thresholds,
    )

    # Build the controls element before the table wrapper so both can go into the
    # same fit-content div. The controls use width:100% of that wrapper, which
    # equals the table width, keeping the columns aligned.
    controls_visual = Div(
        [
            Div(threshold_controls, style={"marginBottom": "0px"}),
            Div(metric_weights, style={"marginTop": "0"}),
        ],
        style={
            "backgroundColor": "#f8f9fa",
            "border": "1px solid #dee2e6",
            "borderRadius": "6px",
            "padding": "0px 0px 0px 0px",  # top right bottom left
            "marginTop": "-5px",
            "boxSizing": "border-box",
            "width": "100%",
        },
    )

    table_section = [
        build_download_controls(table.id, row=True),
        build_filter_overlay(table.id, Div(table)),
        Br(),
        controls_visual,
    ]
    layout_contents.append(Div(table_section, style={"width": "fit-content"}))

    if extra_components:
        layout_contents.append(
            Div(extra_components, style={"maxWidth": INTERACTIVE_ZONE_MAX_WIDTH})
        )

    return Div(layout_contents)


def build_threshold_inputs(
    table_columns: list[str],
    thresholds: Thresholds,
    table_id: str,
    column_widths: dict[str, int] | None = None,
) -> Div:
    """
    Build inline Good/Bad threshold inputs aligned to the table columns.

    Parameters
    ----------
    table_columns : list[str]
        Ordered metric column names in the table.
    thresholds : Thresholds
        Default (good, bad) threshold ranges keyed by metric column.
    table_id : str
        Identifier prefix for threshold inputs and related controls.
    column_widths : dict[str, int] or None
        Optional pixel widths used to align the grid with the table columns.

    Returns
    -------
    Div
        Container with threshold inputs and associated controls.
    """
    widths = calculate_column_widths(table_columns, column_widths)
    grid_template = grid_template_from_widths(widths, table_columns)

    container_style = {
        "display": "grid",
        "gridTemplateColumns": grid_template,
        "alignItems": "start",
        "justifyItems": "center",
        "columnGap": "0px",
        "rowGap": "0px",
        "marginTop": "0px",
        "padding": "2px 0px",
        "backgroundColor": "#f8f9fa",
        "border": "1px solid transparent",
        "borderRadius": "5px",
        "width": "100%",
        "minWidth": "0",
        "boxSizing": "border-box",
    }

    cells: list[Div] = []
    default_thresholds: Thresholds = {}
    threshold_colours = get_threshold_colours()

    cells.append(
        Div(
            [
                Div(
                    "Thresholds",
                    style={
                        "fontWeight": "bold",
                        "fontSize": "13px",
                        "padding": "2px 4px",
                        "whiteSpace": "nowrap",
                        "boxSizing": "border-box",
                        "color": "#212529",
                        "border": "1px solid transparent",
                    },
                ),
                Button(
                    "Reset",
                    id=f"{table_id}-reset-thresholds-button",
                    n_clicks=0,
                    style={
                        "fontSize": "11px",
                        "padding": "4px 8px",
                        "marginTop": "0px",
                        "marginLeft": "4px",
                        "backgroundColor": "#6c757d",
                        "color": "white",
                        "border": "none",
                        "borderRadius": "3px",
                        "width": "fit-content",
                        "cursor": "pointer",
                    },
                ),
                # Toggle to view normalized metric values in the table
                Checklist(
                    id=f"{table_id}-normalized-toggle",
                    options=[{"label": "Show normalised scores", "value": "norm"}],
                    value=[],
                    style={"marginTop": "6px", "fontSize": "11px"},
                    inputStyle={"marginRight": "6px"},
                    labelStyle={"display": "inline-flex", "alignItems": "center"},
                ),
            ],
            style={
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "flex-start",
                "justifyContent": "center",
                "padding": "1px 2px",
                "justifySelf": "start",
                "width": "100%",
                "minWidth": "0",
                "maxWidth": "100%",
                "boxSizing": "border-box",
                "border": "1px solid transparent",
            },
        )
    )
    cells.append(
        Div(
            "",
            style={
                "width": "100%",
                "minWidth": "0",
                "maxWidth": "100%",
                "boxSizing": "border-box",
                "border": "1px solid transparent",
            },
        )
    )

    for metric in table_columns:
        bounds = thresholds.get(metric)
        good_val, bad_val, unit_label = bounds["good"], bounds["bad"], bounds["unit"]
        default_thresholds[metric] = {
            "good": good_val,
            "bad": bad_val,
            "unit": unit_label,
        }

        first_metric = metric == table_columns[0]

        good_children = [
            Label(
                "Good:" if first_metric else "",
                style={
                    "fontSize": "13px",
                    "color": "#6c757d",
                    "textAlign": "right",
                    "position": "absolute",
                    "right": "calc(50% + 34px)",
                },
            ),
            DCC_Input(
                id=f"{table_id}-{metric}-good-threshold",
                type="number",
                value=good_val,
                step=0.0001,
                debounce=True,
                style=build_threshold_input_style(threshold_colours["good"]),
            ),
        ]
        if unit_label:
            good_children.append(
                html.Span(
                    f"[{unit_label}]",
                    style={
                        "fontSize": "12px",
                        "color": "#6c757d",
                        "position": "absolute",
                        "left": "calc(50% + 34px)",
                        "top": "50%",
                        "transform": "translateY(-50%)",
                        "whiteSpace": "nowrap",
                    },
                )
            )

        bad_children = [
            Label(
                "Bad:" if first_metric else "",
                style={
                    "fontSize": "13px",
                    "color": "#6c757d",
                    "textAlign": "right",
                    "position": "absolute",
                    "right": "calc(50% + 34px)",
                    "top": "50%",
                    "transform": "translateY(-50%)",
                    "whiteSpace": "nowrap",
                },
            ),
            DCC_Input(
                id=f"{table_id}-{metric}-bad-threshold",
                type="number",
                value=bad_val,
                step=0.0001,
                debounce=True,
                style=build_threshold_input_style(threshold_colours["bad"]),
            ),
        ]
        if unit_label:
            bad_children.append(
                html.Span(
                    f"[{unit_label}]",
                    style={
                        "fontSize": "12px",
                        "color": "#6c757d",
                        "position": "absolute",
                        "left": "calc(50% + 34px)",
                        "top": "50%",
                        "transform": "translateY(-50%)",
                        "whiteSpace": "nowrap",
                    },
                )
            )

        cells.append(
            Div(
                [
                    Div(
                        good_children,
                        style={
                            "position": "relative",
                            "width": "100%",
                            "padding": "4px 32px",
                            "boxSizing": "border-box",
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                            "marginBottom": "2px",
                        },
                    ),
                    Div(
                        bad_children,
                        style={
                            "position": "relative",
                            "width": "100%",
                            "padding": "4px 32px",
                            "boxSizing": "border-box",
                            "display": "flex",
                            "justifyContent": "center",
                            "alignItems": "center",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "flexDirection": "column",
                    "alignItems": "center",
                    "justifyContent": "center",
                    "padding": "2px 0",
                    "width": "100%",
                    "minWidth": "0",
                    "maxWidth": "100%",
                    "height": "100%",
                    "boxSizing": "border-box",
                    "border": "1px solid transparent",
                },
            )
        )

    # Register callbacks for these metrics, pass default_thresholds for reset
    register_normalization_callbacks(
        table_id,
        table_columns,
        default_thresholds,
        register_toggle=False,
    )

    return Div([Div(cells, id=f"{table_id}-threshold-grid", style=container_style)])
