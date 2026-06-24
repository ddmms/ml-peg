"""Build main Dash application."""

from __future__ import annotations

from importlib import import_module
import warnings

from dash import (
    Dash,
    Input,
    Output,
    callback,
    clientside_callback,
    ctx,
    no_update,
)
from dash.dash_table import DataTable
from dash.dcc import Dropdown, Interval, Link, Loading, Location, Store
from dash.exceptions import PreventUpdate
from dash.html import H1, H3, A, Br, Button, Details, Div, Img, Span, Summary
from yaml import safe_load

from ml_peg import __version__
from ml_peg.analysis.utils.utils import calc_table_scores, get_table_style
from ml_peg.app import APP_ROOT
from ml_peg.app.filters import (
    get_element_filter,
    get_model_filter,
    register_element_filter_callbacks,
)
from ml_peg.app.utils.build_components import (
    LINK_COLUMN_WIDTH,
    build_download_controls,
    build_faqs,
    build_footer,
    build_loading_summary_table,
    build_page_loading_spinner,
    build_weight_components,
)
from ml_peg.app.utils.onboarding import (
    build_onboarding_modal,
    build_tutorial_button,
    register_onboarding_callbacks,
)
from ml_peg.app.utils.register_callbacks import (
    register_benchmark_to_category_callback,
    register_filter_loading_callback,
    register_filter_tables_callback,
)
from ml_peg.app.utils.utils import (
    build_level_of_theory_warnings,
    get_framework_config,
    get_mlip_column_width,
    load_model_registry_configs,
    sig_fig_format,
)
from ml_peg.app.utils.weight_presets import (
    build_weight_preset_selector,
    register_weight_preset_callbacks,
)
from ml_peg.models import current_models
from ml_peg.models.get_models import get_model_names

# Get all models
MODELS = get_model_names(current_models)


def _nav_link_style(is_active: bool) -> dict[str, str]:
    """
    Return sidebar link style.

    Parameters
    ----------
    is_active
        Whether the link is active.

    Returns
    -------
    dict[str, str]
        Style dictionary for the link.
    """
    return {
        "display": "block",
        "padding": "6px 10px",
        "borderRadius": "4px",
        "textDecoration": "none",
        "color": "#119DFF" if is_active else "#495057",
        "fontWeight": "600" if is_active else "normal",
        "backgroundColor": "#e8f4ff" if is_active else "transparent",
        "borderLeft": ("3px solid #119DFF" if is_active else "3px solid transparent"),
    }


def _category_to_path(category_name: str) -> str:
    """
    Convert a category name to a stable URL path.

    Parameters
    ----------
    category_name
        Name of category to convert.

    Returns
    -------
    str
        URL path corresponding to category.
    """
    slug = "".join(
        character.lower() if character.isalnum() else "-" for character in category_name
    )
    slug = "-".join(part for part in slug.split("-") if part)
    if not slug:
        raise ValueError(f"Unable to construct path for {category_name}")
    return f"/category/{slug}"


def _framework_to_path(framework_id: str) -> str:
    """
    Convert a framework identifier to a stable URL path.

    Parameters
    ----------
    framework_id
        Framework identifier to convert.

    Returns
    -------
    str
        URL path corresponding to framework.
    """
    slug = "".join(
        character.lower() if character.isalnum() else "-" for character in framework_id
    )
    slug = "-".join(part for part in slug.split("-") if part)
    if not slug:
        raise ValueError(f"Unable to construct path for framework {framework_id}")
    return f"/framework/{slug}"


def _default_weight_store_data(table: DataTable) -> dict[str, float]:
    """
    Build initial weight-store data for globally mounted summary tables.

    Parameters
    ----------
    table
        Category-summary or overall-summary table whose configurable columns need
        explicit stored weights when the page controls are rendered elsewhere.

    Returns
    -------
    dict[str, float]
        Weight mapping containing one entry for every non-reserved table column
        (i.e. not Score etc).
        This is used when the category and overall summary weight stores are
        kept at the top level of the app, rather than inside an individual
        page, so updates made from framework pages can still propagate even if
        the corresponding category page is not open. For example, if a user
        changes benchmark weights from the MLIP Arena page, the category
        summary still needs a complete set of weights available so it can be
        recomputed immediately. Missing values are filled with ``1.0`` so reset
        and input-sync callbacks always see a complete dictionary in the
        backing ``dcc.Store``.
    """
    reserved = {"MLIP", "Score", "id", "link"}
    weights = dict(getattr(table, "weights", None) or {})
    for column in table.columns:
        column_id = column.get("id")
        if column_id not in reserved:
            weights.setdefault(column_id, 1.0)
    return weights


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


def _framework_sidebar_label(framework_id: str, label: str) -> Div:
    """
    Build a framework sidebar label with an optional logo.

    Parameters
    ----------
    framework_id
        Framework identifier used to look up logo metadata.
    label
        Visible framework label shown in the sidebar.

    Returns
    -------
    Div
        Sidebar label content with optional logo and text.
    """
    config = get_framework_config(framework_id)
    logo = config.get("logo")
    icon = config.get("icon")
    children = []
    if logo:
        children.append(
            Img(
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
        children.append(Span(icon, **{"aria-hidden": "true"}))
    children.append(Span(label))
    return Div(
        children,
        style={"display": "flex", "alignItems": "center", "gap": "8px"},
    )


def build_sidebar(
    pathname: str | None,
    category_paths: dict[str, str],
    framework_paths: dict[str, str] | None = None,
    framework_labels: dict[str, str] | None = None,
) -> list[Details]:
    """
    Build sidebar navigation children with active-link highlighting.

    Parameters
    ----------
    pathname
        Current URL pathname.
    category_paths
        Mapping of category name to its URL path.
    framework_paths
        Optional mapping of framework ID to its URL path.
    framework_labels
        Optional mapping of framework ID to display label.

    Returns
    -------
    list[Details]
        Sidebar section elements.
    """
    current_path = pathname or "/"
    summary_active = current_path in ("", "/", "/summary")
    sidebar_sections = [
        Details(
            [
                Summary(
                    "Overview",
                    style={
                        "fontWeight": "600",
                        "fontSize": "11px",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.07em",
                        "color": "#6c757d",
                        "cursor": "pointer",
                    },
                ),
                Div(
                    [
                        Link(
                            "Summary",
                            href="/",
                            style=_nav_link_style(summary_active),
                            className="sidebar-link",
                        )
                    ]
                ),
            ],
            open=True,
        ),
        Details(
            [
                Summary(
                    "Categories",
                    style={
                        "fontWeight": "600",
                        "fontSize": "11px",
                        "textTransform": "uppercase",
                        "letterSpacing": "0.07em",
                        "color": "#6c757d",
                        "cursor": "pointer",
                    },
                ),
                Div(
                    [
                        Link(
                            category_name,
                            href=category_path,
                            style=_nav_link_style(current_path == category_path),
                            className="sidebar-link",
                        )
                        for category_name, category_path in category_paths.items()
                    ]
                ),
            ],
            open=True,
        ),
    ]

    if framework_paths and framework_labels:
        sidebar_sections.append(
            Details(
                [
                    Summary(
                        "Frameworks",
                        style={
                            "fontWeight": "600",
                            "fontSize": "11px",
                            "textTransform": "uppercase",
                            "letterSpacing": "0.07em",
                            "color": "#6c757d",
                            "cursor": "pointer",
                        },
                    ),
                    Div(
                        [
                            Link(
                                _framework_sidebar_label(
                                    framework_id, framework_labels[framework_id]
                                ),
                                href=framework_path,
                                style=_nav_link_style(current_path == framework_path),
                                className="sidebar-link",
                            )
                            for framework_id, framework_path in framework_paths.items()
                        ]
                    ),
                ],
                open=True,
            )
        )

    return sidebar_sections


def get_all_tests(
    category: str = "*",
    test: str = "*",
) -> tuple[
    dict[str, dict[str, Dash]],
    dict[str, dict[str, list[Div]]],
    dict[str, dict[str, DataTable]],
    dict[str, dict[str, str]],
]:
    """
    Get layout and register callbacks for all categories.

    Parameters
    ----------
    category
        Name of category directory to search for tests. Default is '*'.
    test
        Name of test directory to search for. Default is '*'.

    Returns
    -------
    tuple
        Apps by test name, and layouts, tables, and framework IDs for all categories.
    """
    # Find Python files e.g. app_OC157.py in mlip_tesing.app module.
    # We will get the category from the parent's parent directory
    # E.g. ml_peg/app/surfaces/OC157/app_OC157.py -> surfaces
    tests = APP_ROOT.glob(f"{category}/{test}/app*.py")
    apps = {}
    layouts = {}
    tables = {}
    frameworks = {}

    # Build all layouts, and register all callbacks to main app.
    for test in tests:
        try:
            # Import test layout/callbacks
            test_name = test.parent.name
            category_name = test.parent.parent.name
            test_module = import_module(
                f"ml_peg.app.{category_name}.{test_name}.app_{test_name}"
            )
            test_app = test_module.get_app()
            apps[test_name] = test_app

            # Get layouts and tables for each category/test
            if category_name not in layouts:
                layouts[category_name] = {}
                tables[category_name] = {}
                frameworks[category_name] = {}

            layouts[category_name][test_app.name] = test_app.layout
            tables[category_name][test_app.name] = test_app.table
            frameworks[category_name][test_app.name] = test_app.framework_ids

        except FileNotFoundError as err:
            warnings.warn(
                f"Unable to load layout for {test_name} in {category_name} category. "
                f"Full error:\n{err}",
                stacklevel=2,
            )
            continue

        # Register test callbacks
        try:
            test_app.register_callbacks()
        except FileNotFoundError as err:
            warnings.warn(
                f"Unable to register callbacks for {test_name} in {category_name} "
                f"category. Full error:\n{err}",
                stacklevel=2,
            )
            continue

    return apps, layouts, tables, frameworks


def build_category(
    all_layouts: dict[str, dict[str, list[Div]]],
    all_tables: dict[str, dict[str, DataTable]],
    all_frameworks: dict[str, dict[str, str]],
) -> tuple[
    dict[str, dict[str, object]],
    dict[str, DataTable],
    dict[str, float],
    set[str],
]:
    """
    Build category layouts and summary tables.

    Parameters
    ----------
    all_layouts
        Layouts of all tests, grouped by category.
    all_tables
        Tables for all tests, grouped by category.
    all_frameworks
        Framework IDs for all tests, grouped by category.

    Returns
    -------
    tuple
        Category view metadata, category summary tables, category weights, and all
        discovered framework IDs.
    """
    category_views = {}
    category_tables = {}
    category_weights = {}
    category_to_title = {}
    framework_ids: set[str] = set()

    # `category` corresponds to the category's directory name
    # We will use the loaded `category_title` for IDs/dictionary keys returned
    for category in all_layouts:
        # Get category name and description
        try:
            with open(APP_ROOT / category / f"{category}.yml") as file:
                category_info = safe_load(file)
                category_title = category_info.get("title", category)
                category_descrip = category_info.get("description", "")
                category_weight = category_info.get("weight", 1)
                benchmark_weights = category_info.get("benchmark_weights", {})
        except FileNotFoundError:
            category_title = category
            category_descrip = ""
            category_weight = 1
            benchmark_weights = {}

        category_to_title[category] = category_title

        # Build category summary table
        summary_table = build_summary_table(
            dict(sorted(all_tables[category].items())),
            table_id=f"{category_title}-summary-table",
            description=category_descrip,
            weights={f"{key} Score": value for key, value in benchmark_weights.items()},
        )

        # Store category weight for overall summary
        category_weights[f"{category_title} Score"] = category_weight

        category_tables[category_title] = summary_table

        # Build weight components for category summary table
        weight_components = build_weight_components(
            header="Weights",
            table=summary_table,
            include_download_controls=False,
            column_widths=getattr(summary_table, "column_widths", None),
        )

        test_entries = []
        for test_name in sorted(all_layouts[category]):
            test_framework_ids = all_frameworks[category][test_name]
            framework_ids.update(test_framework_ids)
            test_entries.append(
                {
                    "name": test_name,
                    "framework_ids": test_framework_ids,
                    "layout": all_layouts[category][test_name],
                }
            )

        category_views[category_title] = {
            "title": category_title,
            "description": category_descrip,
            "summary_table": summary_table,
            "weight_components": weight_components,
            "tests": test_entries,
        }

    # Register callback for all benchmark tables -> category table
    # Category summary table columns add "Score" to name for clarity
    register_benchmark_to_category_callback(all_tables, category_to_title)

    return category_views, category_tables, category_weights, framework_ids


def build_category_page_layout(
    category_view: dict[str, object],
) -> Div:
    """
    Build a category page layout.

    Parameters
    ----------
    category_view
        Category metadata including summary table, controls, and benchmark layouts.

    Returns
    -------
    Div
        Category page layout.
    """
    category_title = category_view["title"]
    category_description = category_view["description"]
    summary_table = category_view["summary_table"]
    weight_components = category_view["weight_components"]
    tests = category_view["tests"]
    benchmark_section = Div(
        [test["layout"] for test in tests],
        style={"display": "grid", "gap": "24px"},
    )

    return Div(
        [
            H1(category_title),
            H3(category_description),
            Div(
                [
                    build_download_controls(summary_table.id, row=True),
                    build_loading_summary_table(summary_table),
                    Br(),
                    weight_components,
                ],
                style={"width": "fit-content"},
            ),
            Div(
                [
                    Div(
                        style={
                            "width": "100%",
                            "height": "1px",
                            "backgroundColor": "#a7adb3",
                        }
                    ),
                ],
                style={"margin": "32px 0 24px"},
            ),
            benchmark_section,
        ]
    )


def build_framework_views(
    category_views: dict[str, dict[str, object]],
    framework_ids: set[str],
) -> dict[str, dict[str, object]]:
    """
    Build extra framework-focused page metadata for non-default frameworks.

    Parameters
    ----------
    category_views
        Category metadata including benchmark layout components.
    framework_ids
        All framework IDs discovered from benchmark apps.

    Returns
    -------
    dict[str, dict[str, object]]
        Mapping of framework ID to grouped benchmark layouts by category.
    """
    framework_views: dict[str, dict[str, object]] = {}
    for framework_id in sorted(framework_ids):
        if framework_id == "ml_peg":
            continue

        category_groups = []
        for category_name, category_view in category_views.items():
            tests = [
                test["layout"]
                for test in category_view["tests"]
                if framework_id in test["framework_ids"]
            ]
            if tests:
                category_groups.append({"category": category_name, "tests": tests})

        if category_groups:
            framework_views[framework_id] = {
                "framework_id": framework_id,
                "label": get_framework_config(framework_id)["label"],
                "category_groups": category_groups,
            }

    return framework_views


def build_framework_page_layout(framework_view: dict[str, object]) -> Div:
    """
    Build a framework-focused page containing benchmark sections only.

    Parameters
    ----------
    framework_view
        Framework page metadata with grouped benchmark layouts by category.

    Returns
    -------
    Div
        Framework page layout.
    """
    framework_label = framework_view["label"]
    category_groups = framework_view["category_groups"]

    sections = []
    for group in category_groups:
        sections.append(H3(group["category"], style={"marginTop": "26px"}))
        sections.append(Div(group["tests"], style={"display": "grid", "gap": "24px"}))

    return Div(
        [
            H1(f"{framework_label} Benchmarks"),
            Div(
                (
                    "These benchmarks also remain in their category pages. "
                    "Framework pages omit the category summary table and reuse the "
                    "same benchmark controls, so weight and threshold edits stay in "
                    "sync across both views."
                ),
                style={
                    "fontSize": "13px",
                    "fontStyle": "italic",
                    "color": "#64748b",
                    "marginTop": "8px",
                    "marginBottom": "8px",
                },
            ),
            *sections,
        ]
    )


def build_summary_table(
    tables: dict[str, DataTable],
    table_id: str = "summary-table",
    description: str | None = None,
    weights: dict[str, float] | None = None,
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
    if table_id == "summary-table":
        display_headers = {
            header: (
                header
                if header in {"MLIP", "Score"} or not header.endswith(" Score")
                else "\n".join([*header.removesuffix(" Score").split(), "Score"])
            )
            for header in columns_headers
        }
    else:
        display_headers = {
            header: _format_summary_column_header(header) for header in columns_headers
        }

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


def build_nav(
    full_app: Dash,
    category_views: dict[str, dict[str, object]],
    framework_views: dict[str, dict[str, object]],
    summary_table: DataTable,
    weight_components: Div,
    all_apps: dict[str, Dash],
) -> None:
    """
    Build page layouts and sidebar navigation.

    Parameters
    ----------
    full_app
        Full application with all sub-apps.
    category_views
        Category metadata required to render page content.
    framework_views
        Framework page metadata for extra grouped benchmark pages.
    summary_table
        Summary table with score from each category.
    weight_components
        Weight sliders, text boxes and reset button.
    all_apps
        Dictionary of all test apps.
    """
    category_paths = {
        category_name: _category_to_path(category_name)
        for category_name in sorted(category_views)
    }
    framework_order = sorted(
        framework_views,
        key=lambda framework_id: framework_views[framework_id]["label"],
    )
    framework_paths = {
        framework_id: _framework_to_path(framework_id)
        for framework_id in framework_order
    }
    framework_labels = {
        framework_id: framework_views[framework_id]["label"]
        for framework_id in framework_order
    }

    _summary_label_style = {
        "cursor": "pointer",
        "fontWeight": "600",
        "fontSize": "11px",
        "textTransform": "uppercase",
        "letterSpacing": "0.07em",
        "color": "#6c757d",
        "padding": "5px",
    }
    cmap_selector = Details(
        [
            Summary("Colour scheme", style=_summary_label_style),
            Div(
                Dropdown(
                    id="cmap-dropdown",
                    options=[
                        {"label": "Viridis (colourblind safe)", "value": "viridis_r"},
                        {"label": "Blue-Red (colourblind safe)", "value": "coolwarm"},
                        {
                            "label": "Green-Red",
                            "value": "RdYlGn_r",
                        },
                    ],
                    value="viridis_r",
                    clearable=False,
                    persistence=True,
                    persistence_type="local",
                    persisted_props=["value"],
                    style={"fontSize": "13px"},
                ),
                style={"padding": "8px 12px"},
            ),
        ],
        style={"marginBottom": "8px", "fontSize": "13px"},
    )

    weight_preset_selector = build_weight_preset_selector(_summary_label_style)

    sidebar = Div(
        id="sidebar-nav",
        children=build_sidebar("/", category_paths, framework_paths, framework_labels),
        style={
            "width": "220px",
            "overflowY": "auto",
            "borderRight": "1px solid #dee2e6",
            "padding": "12px",
            "flexShrink": "0",
            "backgroundColor": "#f8f9fa",
        },
    )

    path_to_category = {path: category for category, path in category_paths.items()}
    path_to_framework = {
        path: framework_id for framework_id, path in framework_paths.items()
    }
    category_state_stores = []
    for category_view in category_views.values():
        summary_table_component = category_view["summary_table"]
        category_state_stores.extend(
            [
                Store(
                    id=f"{summary_table_component.id}-computed-store",
                    storage_type="session",
                    data=summary_table_component.data,
                ),
                Store(
                    id=f"{summary_table_component.id}-weight-store",
                    storage_type="session",
                    data=_default_weight_store_data(summary_table_component),
                ),
            ]
        )

    test_state_stores = []
    for app in all_apps.values():
        test_state_stores.extend(app.stores)

    global_state_stores = [
        Store(
            id="summary-table-weight-store",
            storage_type="session",
            data=_default_weight_store_data(summary_table),
        ),
        Store(id="cmap-store", storage_type="local", data="viridis_r"),
        *category_state_stores,
        *test_state_stores,
    ]

    full_layout = [
        # Start-up mask: covers the page and tutorial with a spinner until the
        # page is interactive, so the tutorial isn't shown on a still-rendering
        # page where it feels frozen. Hidden by the callback below.
        Div(
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
                    "Loading ML-PEG…",
                    style={
                        "fontSize": "16px",
                        "fontWeight": "600",
                        "color": "#212529",
                    },
                ),
            ],
            id="startup-mask",
            style={
                "position": "fixed",
                "top": "0",
                "right": "0",
                "bottom": "0",
                "left": "0",
                "display": "flex",
                "flexDirection": "column",
                "alignItems": "center",
                "justifyContent": "center",
                "gap": "14px",
                "backgroundColor": "#ffffff",
                "zIndex": "2100",  # Above the onboarding modal (2000).
            },
        ),
        Interval(id="startup-mask-poll", interval=250, n_intervals=0),
        build_onboarding_modal(),
        # Fixed header controls (top-right): clear-cache button next to the
        # tutorial button. The hidden Divs are the clear-storage and
        # version-check clientside-callback targets.
        Div(
            [
                Button(
                    "Clear cache",
                    id="clear-storage-button",
                    n_clicks=0,
                    title=(
                        "Clear browser-stored app state (weights, thresholds, "
                        "tutorial progress) and reload. Use after an update if "
                        "the app shows stale data."
                    ),
                    style={
                        "padding": "8px 16px",
                        "borderRadius": "6px",
                        "border": "1px solid #cbd5e1",
                        "background": "white",
                        "color": "#475569",
                        "cursor": "pointer",
                        "fontWeight": 600,
                        "fontSize": "14px",
                        "boxShadow": "0 2px 8px rgba(0, 0, 0, 0.1)",
                        "transition": "all 0.2s ease",
                    },
                ),
                build_tutorial_button(),
                Div(id="clear-storage-dummy", style={"display": "none"}),
                Div(id="storage-version-dummy", style={"display": "none"}),
            ],
            style={
                "position": "fixed",
                "top": "20px",
                "right": "20px",
                "display": "flex",
                "alignItems": "center",
                "gap": "10px",
                "zIndex": "1600",  # Above loading overlays (1200/1400).
            },
        ),
        Location(id="app-location", refresh=False),
        Store(
            id="summary-table-scores-store",
            storage_type="session",
        ),
        Div(global_state_stores, style={"display": "none"}),
        Div(
            [
                H1(
                    [
                        Span(
                            "ML-PEG",
                            style={
                                "display": "block",
                                "fontSize": "1.0em",
                                "fontWeight": "700",
                                "letterSpacing": "-0.03em",
                            },
                        ),
                        Span(
                            "Machine Learning Performance and Extrapolation Guide",
                            style={
                                "display": "block",
                                "marginTop": "4px",
                                "fontSize": "0.54em",
                                "fontWeight": "500",
                                "letterSpacing": "0.01em",
                                "color": "#6c757d",
                            },
                        ),
                        A(
                            "📖 Read the documentation →",
                            href="https://ddmms.github.io/ml-peg/",
                            target="_blank",
                            rel="noopener noreferrer",
                            style={
                                "display": "block",
                                "marginTop": "6px",
                                "fontSize": "0.5em",
                                "fontWeight": "600",
                                "color": "#119DFF",
                                "textDecoration": "none",
                            },
                        ),
                    ],
                    style={
                        "padding": "12px 16px 16px",
                        "margin": "0",
                        "borderBottom": "1px solid #dee2e6",
                        "color": "#212529",
                        "lineHeight": "1.05",
                    },
                ),
                Div(
                    [
                        sidebar,
                        Div(
                            [
                                get_model_filter(MODELS),
                                cmap_selector,
                                weight_preset_selector,
                                get_element_filter(),
                                Store(
                                    id="selected-models-store",
                                    storage_type="session",
                                    data=MODELS,
                                ),
                                Store(
                                    id="summary-table-computed-store",
                                    storage_type="session",
                                    data=summary_table.data,
                                ),
                                Store(
                                    id="filter-recompute-done",
                                    storage_type="memory",
                                ),
                                Loading(
                                    Div(id="page-content"),
                                    fullscreen=False,
                                    custom_spinner=build_page_loading_spinner(),
                                    target_components={"page-content": "children"},
                                    show_initially=False,
                                    delay_hide=300,
                                    overlay_style={
                                        "visibility": "visible",
                                        "opacity": 1,
                                    },
                                    parent_style={
                                        "position": "relative",
                                        "minHeight": "60vh",
                                    },
                                ),
                            ],
                            style={"flex": "1", "padding": "16px 16px"},
                        ),
                    ],
                    style={"display": "flex", "minHeight": "0", "flex": "1"},
                ),
            ],
            style={
                "flex": "1",
                "marginBottom": "40px",
                "display": "flex",
                "flexDirection": "column",
            },
        ),
        build_footer(),
    ]

    full_app.layout = Div(
        full_layout,
        style={"display": "flex", "flexDirection": "column", "minHeight": "100vh"},
    )

    # Hide the start-up mask once the page has rendered, or after a timeout as
    # a safety net, then stop polling. Clientside, so it adds no server load.
    clientside_callback(
        """
        function(n) {
            var nu = window.dash_clientside.no_update;
            var ready = document.querySelector('#page-content table tbody tr');
            if (ready || n > 40) {
                return [{'display': 'none'}, true];
            }
            return [nu, nu];
        }
        """,
        Output("startup-mask", "style"),
        Output("startup-mask-poll", "disabled"),
        Input("startup-mask-poll", "n_intervals"),
    )

    # Clear all browser-persisted dcc.Store data (session + local) and reload, so
    # stale cached state after an update can be wiped from the footer button.
    clientside_callback(
        """
        function (n_clicks) {
            if (n_clicks && window.confirm(
                "Clear cached app data and reload? Saved weights and thresholds"
                + " will be reset."
            )) {
                window.localStorage.clear();
                window.sessionStorage.clear();
                window.location.reload();
            }
            return "";
        }
        """,
        Output("clear-storage-dummy", "children"),
        Input("clear-storage-button", "n_clicks"),
        prevent_initial_call=True,
    )

    # Auto-clear browser-persisted stores when the released version changes, so a
    # new release drops stale cached state automatically. The
    # version is recorded in localStorage; a real change (not a first visit) clears
    # both storages and reloads so the dcc.Stores reinitialise from fresh defaults.
    clientside_callback(
        f"""
        function (pathname) {{
            const current = "{__version__}";
            const stored = window.localStorage.getItem("ml-peg-store-version");
            if (stored !== current) {{
                window.localStorage.clear();
                window.sessionStorage.clear();
                window.localStorage.setItem("ml-peg-store-version", current);
                if (stored !== null) {{
                    window.location.reload();
                }}
            }}
            return "";
        }}
        """,
        Output("storage-version-dummy", "children"),
        Input("app-location", "pathname"),
        prevent_initial_call=False,
    )

    @callback(
        Output("model-filter-checklist", "value"),
        Output("selected-models-store", "data"),
        Input("model-filter-checklist", "value"),
        Input("selected-models-store", "data"),
        prevent_initial_call=False,
    )
    def sync_model_filter(
        checklist_value: list[str] | None,
        stored_selection: list[str] | None,
    ) -> tuple[list[str], list[str] | object]:
        """
        Keep the model selector checklist and backing store synchronised.

        Parameters
        ----------
        checklist_value
            Current selection from the model filter control.
        stored_selection
            Previously persisted selection from ``selected-models-store``.

        Returns
        -------
        tuple[list[str], list[str] | object]
            Updated checklist value and store payload. The second element may be
            ``dash.no_update`` when only syncing from store to UI.
        """
        trigger_id = ctx.triggered_id

        if trigger_id in (None, "selected-models-store"):
            stored = stored_selection if stored_selection is not None else MODELS
            return stored, no_update
        if trigger_id == "model-filter-checklist":
            selected = checklist_value or []
            return selected, selected
        raise PreventUpdate

    @callback(
        Output("cmap-dropdown", "value"),
        Output("cmap-store", "data"),
        Input("cmap-dropdown", "value"),
        Input("cmap-store", "data"),
        prevent_initial_call=False,
    )
    def sync_cmap(
        cmap_name: str | None, stored_cmap: str | None
    ) -> tuple[str, str | object]:
        """
        Keep the colour scheme dropdown and backing store synchronised.

        Parameters
        ----------
        cmap_name
            Matplotlib colormap name selected from the dropdown control.
        stored_cmap
            Previously persisted colormap name from ``cmap-store``.

        Returns
        -------
        tuple[str, str | object]
            Dropdown value and store payload, or ``dash.no_update`` when only
            the dropdown needs syncing from the stored value.
        """
        trigger_id = ctx.triggered_id

        if trigger_id in (None, "cmap-store"):
            selected = stored_cmap or "viridis_r"
            return selected, no_update
        if trigger_id == "cmap-dropdown":
            selected = cmap_name or "viridis_r"
            return selected, selected
        raise PreventUpdate

    register_weight_preset_callbacks(
        summary_table, _default_weight_store_data(summary_table)
    )

    @callback(
        Output("model-filter-details", "open"),
        Input("app-location", "pathname"),
        prevent_initial_call=False,
    )
    def toggle_filter_panel(pathname: str | None) -> bool:
        """
        Expand the visible-models panel on the summary page only.

        Parameters
        ----------
        pathname
            Current URL pathname.

        Returns
        -------
        bool
            ``True`` when the summary page is active, otherwise ``False``.
        """
        return pathname in (None, "", "/", "/summary")

    @callback(
        Output("page-content", "children"),
        Output("sidebar-nav", "children"),
        Input("app-location", "pathname"),
    )
    def select_page(
        pathname: str | None,
    ) -> tuple[Div, list[Details]]:
        """
        Select page contents to be displayed.

        Parameters
        ----------
        pathname
            Current URL pathname.

        Returns
        -------
        Div
            Summary or category contents to be displayed.
        """
        sidebar_children = build_sidebar(
            pathname, category_paths, framework_paths, framework_labels
        )

        if pathname in (None, "", "/", "/summary"):
            summary_counts = (
                f"{len(category_views)} categories · {len(all_apps)} benchmarks"
            )
            return Div(
                [
                    H1("Categories Summary"),
                    Div(
                        summary_counts,
                        style={
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": "#212529",
                            "backgroundColor": "#f1f3f5",
                            "border": "1px solid #dee2e6",
                            "borderRadius": "6px",
                            "padding": "8px 14px",
                            "marginBottom": "12px",
                            "width": "fit-content",
                        },
                    ),
                    Div(
                        "Scores range from 0 (worst) to 1 (best).",
                        style={
                            "fontSize": "14px",
                            "fontWeight": "600",
                            "color": "#212529",
                            "backgroundColor": "#e8f4fd",
                            "border": "1px solid #bee3f8",
                            "borderRadius": "6px",
                            "padding": "8px 14px",
                            "marginBottom": "16px",
                            "width": "fit-content",
                        },
                    ),
                    Div(
                        [
                            build_download_controls(summary_table.id, row=True),
                            build_loading_summary_table(summary_table),
                            Br(),
                            weight_components,
                        ],
                        style={"width": "fit-content"},
                    ),
                    build_faqs(),
                ]
            ), sidebar_children

        selected_framework = path_to_framework.get(pathname)
        if selected_framework is not None:
            return (
                Div([build_framework_page_layout(framework_views[selected_framework])]),
                sidebar_children,
            )

        selected_category = path_to_category.get(pathname)
        if selected_category is None:
            return Div([H3("Page not found")]), sidebar_children
        return (
            Div([build_category_page_layout(category_views[selected_category])]),
            sidebar_children,
        )


def build_full_app(full_app: Dash, category: str = "*", test: str = "*") -> None:
    """
    Build full app layout and register callbacks.

    Parameters
    ----------
    full_app
        Full application with all sub-apps.
    category
        Category to build app for. Default is `*`, corresponding to all categories.
    test
        Test to build app for. Default is `*`, corresponding to all tests.
    """
    # Get layouts and tables for each test, grouped by categories
    all_apps, all_layouts, all_tables, all_frameworks = get_all_tests(
        category=category, test=test
    )

    if not all_layouts:
        raise ValueError("No tests were built successfully")

    register_filter_tables_callback(all_apps)
    register_element_filter_callbacks()
    register_filter_loading_callback()

    # Combine tests into categories and create category summary
    cat_views, cat_tables, cat_weights, framework_ids = build_category(
        all_layouts, all_tables, all_frameworks
    )
    framework_views = build_framework_views(cat_views, framework_ids)
    # Build overall summary table
    summary_table = build_summary_table(
        dict(sorted(cat_tables.items())), weights=cat_weights
    )
    weight_components = build_weight_components(
        header="Weights",
        table=summary_table,
        include_download_controls=False,
        column_widths=summary_table.column_widths,
    )
    # Build summary and category pages and navigation
    build_nav(
        full_app,
        cat_views,
        framework_views,
        summary_table,
        weight_components,
        all_apps,
    )
    register_onboarding_callbacks()
