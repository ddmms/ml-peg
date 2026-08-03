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
from dash.html import H1, H3, A, Br, Details, Div, Img, Span, Summary
from yaml import safe_load

from ml_peg.app import APP_ROOT
from ml_peg.app.filters import (
    get_element_filter,
    get_model_filter,
    register_element_filter_callbacks,
)
from ml_peg.app.utils.build_components import (
    build_cost_panel,
    build_download_controls,
    build_faqs,
    build_footer,
    build_loading_summary_table,
    build_page_loading_spinner,
    build_summary_table,
    build_weight_components,
)
from ml_peg.app.utils.build_frameworks import (
    build_framework_page_layout,
    build_framework_summary_tables,
    build_framework_views,
)
from ml_peg.app.utils.onboarding import (
    build_onboarding_modal,
    register_onboarding_callbacks,
)
from ml_peg.app.utils.register_callbacks import (
    register_benchmark_to_group_callback,
    register_filter_loading_callback,
    register_filter_tables_callback,
)
from ml_peg.app.utils.storage import (
    build_header_controls,
    register_storage_callbacks,
)
from ml_peg.app.utils.utils import (
    framework_sort_key,
    get_framework_config,
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
            test_app.table.benchmark_key = f"{category_name}/{test_name}"

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


def _collect_benchmark_speeds(
    all_tables: dict[str, dict[str, DataTable]],
) -> dict[str, str | None]:
    """
    Collect speeds by stable, directory-based benchmark identifier.

    Parameters
    ----------
    all_tables
        Tables grouped by category and benchmark.

    Returns
    -------
    dict[str, str | None]
        Speed level for each ``<category>/<benchmark>`` identifier.
    """
    return {
        table.benchmark_key: getattr(table, "speed", None)
        for tests in all_tables.values()
        for table in tests.values()
    }


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
            table = all_tables[category][test_name]
            test_entries.append(
                {
                    "name": test_name,
                    "framework_ids": test_framework_ids,
                    "layout": all_layouts[category][test_name],
                    "key": table.benchmark_key,
                    "speed": getattr(table, "speed", None),
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
    register_benchmark_to_group_callback(
        all_tables, category_to_title, table_id_suffix="-summary-table"
    )

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


def build_nav(
    full_app: Dash,
    category_views: dict[str, dict[str, object]],
    framework_views: dict[str, dict[str, object]],
    summary_table: DataTable,
    weight_components: Div,
    all_apps: dict[str, Dash],
    benchmark_speeds: dict[str, str | None],
    combined_framework_table: DataTable | None = None,
    framework_weight_components: Div | None = None,
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
    benchmark_speeds
        Speed level of each benchmark, used for the summary cost panel.
    combined_framework_table
        Frameworks summary table shown on the home page, or None when there are
        no external frameworks.
    framework_weight_components
        Weight controls for the combined framework summary table, or None.
    """
    category_paths = {
        category_name: _category_to_path(category_name)
        for category_name in sorted(category_views)
    }
    # Frameworks first, then papers, alphabetical by label within each
    framework_order = sorted(framework_views, key=framework_sort_key)
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

    # Computed + weight stores for each per-framework summary table
    framework_state_stores = []
    for framework_view in framework_views.values():
        fw_table = framework_view.get("summary_table")
        if fw_table is None:
            continue
        framework_state_stores.extend(
            [
                Store(
                    id=f"{fw_table.id}-computed-store",
                    storage_type="session",
                    data=fw_table.data,
                ),
                Store(
                    id=f"{fw_table.id}-weight-store",
                    storage_type="session",
                    data=_default_weight_store_data(fw_table),
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
        *framework_state_stores,
        *test_state_stores,
    ]
    if combined_framework_table is not None:
        global_state_stores.append(
            Store(
                id="framework-summary-table-weight-store",
                storage_type="session",
                data=_default_weight_store_data(combined_framework_table),
            )
        )

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
                # Time-based progress bar: fills continuously over ~10s, easing
                # toward ~95% (keyframe in loading.css; same single-element bar
                # as the pre-hydration loader in dash_loading.css). It vanishes
                # with the mask when ready, so it never reaches a fake 100%.
                Div(
                    style={
                        "width": "200px",
                        "height": "6px",
                        "borderRadius": "3px",
                        "background": (
                            "linear-gradient(#119DFF, #119DFF) left center "
                            "/ 5% 100% no-repeat, #d0ebff"
                        ),
                        "animation": "ml-peg-bar-fill 10s ease-out forwards",
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
        build_header_controls(),
        Location(id="app-location", refresh=False),
        Store(
            id="summary-table-scores-store",
            storage_type="session",
        ),
        *(
            [Store(id="framework-summary-table-scores-store", storage_type="session")]
            if combined_framework_table is not None
            else []
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
                                *(
                                    [
                                        Store(
                                            id="framework-summary-table-computed-store",
                                            storage_type="session",
                                            data=combined_framework_table.data,
                                        )
                                    ]
                                    if combined_framework_table is not None
                                    else []
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
    # (The progress bar fills via a CSS animation, not this callback.)
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

    register_storage_callbacks()

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
                f" · {len(framework_views)} frameworks"
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
                    *(
                        [
                            H1(
                                "Frameworks Summary",
                                style={"marginTop": "32px"},
                            ),
                            Div(
                                [
                                    build_download_controls(
                                        combined_framework_table.id, row=True
                                    ),
                                    build_loading_summary_table(
                                        combined_framework_table
                                    ),
                                    Br(),
                                    framework_weight_components,
                                ],
                                style={"width": "fit-content"},
                            ),
                        ]
                        if combined_framework_table is not None
                        else []
                    ),
                    build_cost_panel(benchmark_speeds),
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
    # Build per-framework summary tables and the combined framework summary
    framework_tables, framework_grouping = build_framework_summary_tables(
        all_tables, all_frameworks, framework_views
    )
    combined_framework_table = None
    framework_weight_components = None
    if framework_tables:
        combined_framework_table = build_summary_table(
            dict(
                sorted(
                    framework_tables.items(),
                    key=lambda item: framework_sort_key(item[0]),
                )
            ),
            table_id="framework-summary-table",
            header_labels={
                fid: str(framework_views[fid]["label"]) for fid in framework_tables
            },
        )
        framework_weight_components = build_weight_components(
            header="Weights",
            table=combined_framework_table,
            include_download_controls=False,
            column_widths=combined_framework_table.column_widths,
        )
        register_benchmark_to_group_callback(
            framework_grouping,
            {fid: fid for fid in framework_grouping},
            table_id_suffix="-framework-summary-table",
        )
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
    benchmark_speeds = _collect_benchmark_speeds(all_tables)

    build_nav(
        full_app,
        cat_views,
        framework_views,
        summary_table,
        weight_components,
        all_apps,
        benchmark_speeds,
        combined_framework_table,
        framework_weight_components,
    )
    register_onboarding_callbacks()
