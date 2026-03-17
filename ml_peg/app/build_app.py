"""Build main Dash application."""

from __future__ import annotations

from importlib import import_module
import warnings

from dash import Dash, Input, Output, callback, ctx, no_update
from dash.dash_table import DataTable
from dash.dcc import Dropdown, Link, Loading, Location, Store
from dash.exceptions import PreventUpdate
from dash.html import H1, H3, Button, Details, Div, Span, Summary
from yaml import safe_load

from ml_peg.analysis.utils.utils import calc_table_scores, get_table_style
from ml_peg.app import APP_ROOT
from ml_peg.app.utils.build_components import (
    build_faqs,
    build_footer,
    build_weight_components,
)
from ml_peg.app.utils.onboarding import (
    build_onboarding_modal,
    build_tutorial_button,
    register_onboarding_callbacks,
)
from ml_peg.app.utils.register_callbacks import register_benchmark_to_category_callback
from ml_peg.app.utils.utils import (
    build_level_of_theory_warnings,
    calculate_column_widths,
    get_framework_config,
    load_model_registry_configs,
    sig_fig_format,
)
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

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


def build_sidebar(
    pathname: str | None, category_paths: dict[str, str]
) -> list[Details]:
    """
    Build sidebar navigation children with active-link highlighting.

    Parameters
    ----------
    pathname
        Current URL pathname.
    category_paths
        Mapping of category name to its URL path.

    Returns
    -------
    list[Details]
        Sidebar section elements.
    """
    current_path = pathname or "/"
    summary_active = current_path in ("", "/", "/summary")
    return [
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
                Div([Link("Summary", href="/", style=_nav_link_style(summary_active))]),
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
                        )
                        for category_name, category_path in category_paths.items()
                    ]
                ),
            ],
            open=True,
        ),
    ]


def get_all_tests(
    category: str = "*",
) -> tuple[
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

    Returns
    -------
    tuple
        Layouts, tables, and framework IDs for all categories.
    """
    # Find Python files e.g. app_OC157.py in mlip_tesing.app module.
    # We will get the category from the parent's parent directory
    # E.g. ml_peg/app/surfaces/OC157/app_OC157.py -> surfaces
    tests = APP_ROOT.glob(f"{category}/*/app*.py")
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

            # Get layouts and tables for each category/test
            if category_name not in layouts:
                layouts[category_name] = {}
                tables[category_name] = {}
                frameworks[category_name] = {}
            layouts[category_name][test_app.name] = test_app.layout
            tables[category_name][test_app.name] = test_app.table
            frameworks[category_name][test_app.name] = test_app.framework_id
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

    return layouts, tables, frameworks


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

        # Build category summary table
        summary_table = build_summary_table(
            all_tables[category],
            table_id=f"{category_title}-summary-table",
            description=category_descrip,
            weights={f"{key} Score": value for key, value in benchmark_weights.items()},
        )

        # Store category weight for overall summary
        category_weights[f"{category_title} Score"] = category_weight

        category_tables[category_title] = summary_table

        # Build weight components for category summary table
        weight_components = build_weight_components(
            header="Benchmark weights",
            table=summary_table,
            column_widths=getattr(summary_table, "column_widths", None),
        )

        test_entries = []
        for test_name in all_layouts[category]:
            framework_id = all_frameworks[category][test_name]
            framework_ids.add(framework_id)
            test_entries.append(
                {
                    "name": test_name,
                    "framework_id": framework_id,
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

        # Register benchmark table -> category table callbacks
        # Category summary table columns add "Score" to name for clarity
        for test_name, benchmark_table in all_tables[category].items():
            register_benchmark_to_category_callback(
                benchmark_table_id=benchmark_table.id,
                category_table_id=f"{category_title}-summary-table",
                benchmark_column=test_name + " Score",
                model_name_map=getattr(benchmark_table, "model_name_map", None),
            )

    return category_views, category_tables, category_weights, framework_ids


def build_category_tab_layout(
    category_view: dict[str, object],
    selected_frameworks: list[str],
) -> Div:
    """
    Build category tab layout, optionally filtering benchmark sections by framework.

    Parameters
    ----------
    category_view
        Category metadata including summary table, controls, and benchmark layouts.
    selected_frameworks
        Framework IDs to display in benchmark sections.

    Returns
    -------
    Div
        Category tab layout.
    """
    category_title = category_view["title"]
    category_description = category_view["description"]
    summary_table = category_view["summary_table"]
    weight_components = category_view["weight_components"]
    tests = category_view["tests"]

    selected_frameworks = selected_frameworks or []
    selected_framework_set = set(selected_frameworks)
    selected_tests = [
        test for test in tests if test["framework_id"] in selected_framework_set
    ]

    if not selected_frameworks:
        filter_notice = Div(
            "No frameworks selected. Choose one or more frameworks above.",
            style={
                "fontSize": "13px",
                "fontStyle": "italic",
                "color": "#64748b",
                "marginTop": "12px",
            },
        )
    else:
        if len(selected_tests) == len(tests):
            filter_notice = None
        else:
            selected_in_category = {
                test["framework_id"]
                for test in tests
                if test["framework_id"] in selected_framework_set
            }
            framework_labels = [
                get_framework_config(framework_id)["label"]
                for framework_id in sorted(selected_in_category)
            ]
            filter_notice = Div(
                f"Showing benchmarks from: {', '.join(framework_labels)}.",
                style={
                    "fontSize": "13px",
                    "fontStyle": "italic",
                    "color": "#64748b",
                    "marginTop": "12px",
                },
            )

    if selected_tests:
        benchmark_section = Div([test["layout"] for test in selected_tests])
    else:
        benchmark_section = Div(
            "No benchmarks are available for the selected framework filters.",
            style={
                "padding": "12px",
                "border": "1px dashed #94a3b8",
                "borderRadius": "6px",
                "color": "#64748b",
                "fontStyle": "italic",
            },
        )

    return Div(
        [
            H1(category_title),
            H3(category_description),
            summary_table,
            Store(
                id=f"{category_title}-summary-table-computed-store",
                storage_type="session",
                data=summary_table.data,
            ),
            weight_components,
            filter_notice,
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

    columns = [{"name": headers, "id": headers} for headers in columns_headers]
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

    # Calculate column widths based on column names
    calculated_widths = calculate_column_widths(columns_headers)
    # Limit max width to 150px for better wrapping on long column names
    column_widths = {
        col_id: min(width, 150) for col_id, width in calculated_widths.items()
    }

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

    table = DataTable(
        data=data,
        columns=columns,
        id=table_id,
        sort_action="native",
        style_data_conditional=style_with_warnings,
        style_cell_conditional=style_cell_conditional,
        style_header={
            "whiteSpace": "normal",
            "height": "auto",
            "minHeight": "70px",
            "textAlign": "center",
            "verticalAlign": "middle",
            "lineHeight": "1.4",
            "padding": "8px",
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
        persistence=True,
        persistence_type="session",
        persisted_props=["data"],
        tooltip_header=tooltip_header,
        editable=False,
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
    summary_table: DataTable,
    weight_components: Div,
    framework_options: list[dict[str, str]],
) -> None:
    """
    Build page layouts and sidebar navigation.

    Parameters
    ----------
    full_app
        Full application with all sub-apps.
    category_views
        Category metadata required to render page content.
    summary_table
        Summary table with score from each category.
    weight_components
        Weight sliders, text boxes and reset button.
    framework_options
        Dropdown options for global framework filter.
    """
    category_paths = {
        category_name: _category_to_path(category_name)
        for category_name in category_views
    }
    framework_filter = Details(
        [
            Summary(
                "Benchmark framework",
                style={"cursor": "pointer", "fontWeight": "bold", "padding": "5px"},
            ),
            Div(
                [
                    Div(
                        [
                            Button(
                                "Select all",
                                id="framework-filter-select-all",
                                n_clicks=0,
                                style={
                                    "padding": "4px 10px",
                                    "fontSize": "12px",
                                    "borderRadius": "4px",
                                    "border": "1px solid #cbd5e1",
                                    "backgroundColor": "#f8fafc",
                                    "cursor": "pointer",
                                },
                            ),
                            Button(
                                "Deselect all",
                                id="framework-filter-deselect-all",
                                n_clicks=0,
                                style={
                                    "padding": "4px 10px",
                                    "fontSize": "12px",
                                    "borderRadius": "4px",
                                    "border": "1px solid #cbd5e1",
                                    "backgroundColor": "#ffffff",
                                    "cursor": "pointer",
                                },
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "8px",
                            "marginBottom": "8px",
                        },
                    ),
                    Dropdown(
                        id="framework-filter",
                        options=framework_options,
                        value=[option["value"] for option in framework_options],
                        multi=True,
                        closeOnSelect=False,
                        placeholder="Select visible frameworks",
                        style={"fontSize": "13px"},
                    ),
                ],
                style={"padding": "8px 12px"},
            ),
        ],
        id="framework-filter-details",
        open=True,
        style={"marginBottom": "8px", "fontSize": "13px"},
    )

    model_options = [{"label": m, "value": m} for m in MODELS]

    model_filter = Details(
        [
            Summary(
                "Visible models",
                style={
                    "cursor": "pointer",
                    "fontWeight": "600",
                    "fontSize": "11px",
                    "textTransform": "uppercase",
                    "letterSpacing": "0.07em",
                    "color": "#6c757d",
                    "padding": "5px",
                },
            ),
            Div(
                [
                    Dropdown(
                        id="model-filter-checklist",
                        options=model_options,
                        value=MODELS,
                        multi=True,
                        placeholder="Select visible models",
                        closeOnSelect=False,
                        style={"fontSize": "13px"},
                    ),
                ],
                style={"padding": "8px 12px"},
            ),
        ],
        id="model-filter-details",
        open=True,
        style={"marginBottom": "8px", "fontSize": "13px"},
    )

    sidebar = Div(
        id="sidebar-nav",
        children=build_sidebar("/", category_paths),
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

    full_layout = [
        build_onboarding_modal(),
        build_tutorial_button(),
        Location(id="app-location", refresh=False),
        Store(
            id="summary-table-scores-store",
            storage_type="session",
        ),
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
                                framework_filter,
                                model_filter,
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
                                Loading(
                                    Div(id="page-content"),
                                    type="circle",
                                    color="#119DFF",
                                    fullscreen=False,
                                    target_components={"page-content": "children"},
                                    style={
                                        "position": "fixed",
                                        "top": "300px",
                                        "left": "50%",
                                        "transform": "translateX(-50%)",
                                        "zIndex": "1100",
                                    },
                                    parent_style={"position": "relative"},
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
        Output("framework-filter-details", "open"),
        Input("app-location", "pathname"),
        prevent_initial_call=False,
    )
    def toggle_framework_filter_panel(pathname: str | None) -> bool:
        """
        Expand the framework filter panel on category pages only.

        Parameters
        ----------
        pathname
            Current URL pathname.

        Returns
        -------
        bool
            ``True`` when a category page is active, otherwise ``False``.
        """
        return pathname not in (None, "", "/", "/summary")

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
        Input("framework-filter", "value"),
    )
    def select_page(
        pathname: str | None,
        framework_filter: list[str] | None,
    ) -> tuple[Div, list[Details]]:
        """
        Select page contents to be displayed.

        Parameters
        ----------
        pathname
            Current URL pathname.
        framework_filter
            Selected framework IDs from the filter dropdown.

        Returns
        -------
        Div
            Summary or category contents to be displayed.
        """
        sidebar_children = build_sidebar(pathname, category_paths)
        all_framework_values = [option["value"] for option in framework_options]
        all_framework_set = set(all_framework_values)
        active_frameworks = framework_filter or []
        active_framework_set = set(active_frameworks)

        if pathname in (None, "", "/", "/summary"):
            filter_notice = None
            if not active_framework_set:
                filter_notice = Div(
                    (
                        "No frameworks selected for category tabs. "
                        "Summary scores still include all benchmarks."
                    ),
                    style={
                        "marginTop": "12px",
                        "padding": "8px 10px",
                        "fontSize": "13px",
                        "color": "#475569",
                        "backgroundColor": "#f8fafc",
                        "border": "1px solid #cbd5e1",
                        "borderRadius": "6px",
                    },
                )
            elif active_framework_set != all_framework_set:
                framework_labels = [
                    get_framework_config(framework_id)["label"]
                    for framework_id in sorted(active_framework_set)
                ]
                filter_notice = Div(
                    (
                        f"Framework filter ({', '.join(framework_labels)}) only "
                        "changes which "
                        "benchmark sections are shown in category tabs. Summary scores "
                        "still include all benchmarks."
                    ),
                    style={
                        "marginTop": "12px",
                        "padding": "8px 10px",
                        "fontSize": "13px",
                        "color": "#475569",
                        "backgroundColor": "#f8fafc",
                        "border": "1px solid #cbd5e1",
                        "borderRadius": "6px",
                    },
                )
            return Div(
                [
                    H1("Categories Summary"),
                    summary_table,
                    weight_components,
                    filter_notice,
                    build_faqs(),
                ]
            ), sidebar_children

        selected_category = path_to_category.get(pathname)
        if selected_category is None:
            return Div([H3("Page not found")]), sidebar_children
        return (
            Div(
                [
                    build_category_tab_layout(
                        category_views[selected_category],
                        active_frameworks,
                    )
                ]
            ),
            sidebar_children,
        )

    @callback(
        Output("framework-filter", "value"),
        Input("framework-filter-select-all", "n_clicks"),
        Input("framework-filter-deselect-all", "n_clicks"),
        prevent_initial_call=True,
    )
    def update_framework_filter_selection(
        select_all_clicks: int,
        deselect_all_clicks: int,
    ) -> list[str]:
        """
        Select or clear framework selections from quick-action buttons.

        Parameters
        ----------
        select_all_clicks
            Number of clicks on the select-all button.
        deselect_all_clicks
            Number of clicks on the deselect-all button.

        Returns
        -------
        list[str]
            Updated selected framework IDs.
        """
        trigger_id = ctx.triggered_id
        if trigger_id == "framework-filter-select-all":
            return [option["value"] for option in framework_options]
        return []


def build_full_app(full_app: Dash, category: str = "*") -> None:
    """
    Build full app layout and register callbacks.

    Parameters
    ----------
    full_app
        Full application with all sub-apps.
    category
        Category to build app for. Default is `*`, corresponding to all categories.
    """
    # Get layouts and tables for each test, grouped by categories
    all_layouts, all_tables, all_frameworks = get_all_tests(category=category)

    if not all_layouts:
        raise ValueError("No tests were built successfully")

    # Combine tests into categories and create category summary
    cat_views, cat_tables, cat_weights, framework_ids = build_category(
        all_layouts, all_tables, all_frameworks
    )
    # Build overall summary table
    summary_table = build_summary_table(cat_tables, weights=cat_weights)
    weight_components = build_weight_components(
        header="Category weights",
        table=summary_table,
        column_widths=summary_table.column_widths,
    )
    framework_options = []
    for framework_id in sorted(framework_ids):
        framework_label = get_framework_config(framework_id)["label"]
        framework_options.append({"label": framework_label, "value": framework_id})
    # Build summary and category pages and navigation
    build_nav(
        full_app,
        cat_views,
        summary_table,
        weight_components,
        framework_options,
    )
    register_onboarding_callbacks()
