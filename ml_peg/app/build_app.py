"""Build main Dash application."""

from __future__ import annotations

from importlib import import_module
import warnings

from dash import Dash, Input, Output, callback
from dash.dash_table import DataTable
from dash.dcc import Store, Tab, Tabs
from dash.html import H1, H3, Div
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
    load_model_registry_configs,
    sig_fig_format,
)
from ml_peg.models.get_models import get_model_names
from ml_peg.models.models import current_models

# Get all models
MODELS = get_model_names(current_models)


def get_all_tests(
    category: str = "*",
) -> tuple[dict[str, dict[str, list[Div]]], dict[str, dict[str, DataTable]]]:
    """
    Get layout and register callbacks for all categories.

    Parameters
    ----------
    category
        Name of category directory to search for tests. Default is '*'.

    Returns
    -------
    tuple[dict[str, dict[str, list[Div]]], dict[str, dict[str, DataTable]]]
        Layouts and tables for all categories.
    """
    # Find Python files e.g. app_OC157.py in mlip_tesing.app module.
    # We will get the category from the parent's parent directory
    # E.g. ml_peg/app/surfaces/OC157/app_OC157.py -> surfaces
    tests = APP_ROOT.glob(f"{category}/*/app*.py")
    layouts = {}
    tables = {}

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
            layouts[category_name][test_app.name] = test_app.layout
            tables[category_name][test_app.name] = test_app.table
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

    return layouts, tables


def build_category(
    all_layouts: dict[str, dict[str, list[Div]]],
    all_tables: dict[str, dict[str, DataTable]],
) -> tuple[dict[str, list[Div]], dict[str, DataTable]]:
    """
    Build category layouts and summary tables.

    Parameters
    ----------
    all_layouts
        Layouts of all tests, grouped by category.
    all_tables
        Tables for all tests, grouped by category.

    Returns
    -------
    tuple[dict[str, list[Div]], dict[str, DataTable]]
        Dictionary of category layouts, and dictionary of category summary tables.
    """
    # Take all tables in category, build new table, and set layout
    category_layouts = {}
    category_tables = {}
    category_weights = {}

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

        # Build full layout with summary table, weight controls, and test layouts
        category_layouts[category_title] = Div(
            [
                H1(category_title),
                H3(category_descrip),
                summary_table,
                Store(
                    id=f"{category_title}-summary-table-computed-store",
                    storage_type="session",
                    data=summary_table.data,
                ),
                weight_components,
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
                Div([all_layouts[category][test] for test in all_layouts[category]]),
            ]
        )

        # Register benchmark table -> category table callbacks
        # Category summary table columns add "Score" to name for clarity
        for test_name, benchmark_table in all_tables[category].items():
            register_benchmark_to_category_callback(
                benchmark_table_id=benchmark_table.id,
                category_table_id=f"{category_title}-summary-table",
                benchmark_column=test_name + " Score",
                model_name_map=getattr(benchmark_table, "model_name_map", None),
            )

    return category_layouts, category_tables, category_weights


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

    data = calc_table_scores(data)

    columns_headers = ("MLIP",) + tuple(key + " Score" for key in tables) + ("Score",)

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


def build_tabs(
    full_app: Dash,
    layouts: dict[str, list[Div]],
    summary_table: DataTable,
    weight_components: Div,
) -> None:
    """
    Build tab layouts and summary tab.

    Parameters
    ----------
    full_app
        Full application with all sub-apps.
    layouts
        Layouts for all tabs.
    summary_table
        Summary table with score from each category.
    weight_components
        Weight sliders, text boxes and reset button.
    """
    all_tabs = [Tab(label="Summary", value="summary-tab", id="summary-tab")] + [
        Tab(label=category_name, value=category_name) for category_name in layouts
    ]

    tabs_layout = [
        build_onboarding_modal(),
        build_tutorial_button(),
        Div(
            [
                H1("ML-PEG"),
                Tabs(id="all-tabs", value="summary-tab", children=all_tabs),
                Div(id="tabs-content"),
            ],
            style={"flex": "1", "marginBottom": "40px"},
        ),
        build_footer(),
    ]

    full_app.layout = Div(
        tabs_layout,
        style={"display": "flex", "flexDirection": "column", "minHeight": "100vh"},
    )

    @callback(Output("tabs-content", "children"), Input("all-tabs", "value"))
    def select_tab(tab) -> Div:
        """
        Select tab contents to be displayed.

        Parameters
        ----------
        tab
            Name of tab selected.

        Returns
        -------
        Div
            Summary or tab contents to be displayed.
        """
        if tab == "summary-tab":
            return Div(
                [
                    H1("Benchmarks Summary"),
                    summary_table,
                    weight_components,
                    Store(
                        id="summary-table-scores-store",
                        storage_type="session",
                    ),
                    build_faqs(),
                ]
            )
        return Div([layouts[tab]])


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
    all_layouts, all_tables = get_all_tests(category=category)

    if not all_layouts:
        raise ValueError("No tests were built successfully")

    # Combine tests into categories and create category summary
    cat_layouts, cat_tables, cat_weights = build_category(all_layouts, all_tables)
    # Build overall summary table
    summary_table = build_summary_table(cat_tables, weights=cat_weights)
    weight_components = build_weight_components(
        header="Category weights",
        table=summary_table,
        column_widths=summary_table.column_widths,
    )
    # Build summary and category tabs
    build_tabs(full_app, cat_layouts, summary_table, weight_components)
    register_onboarding_callbacks()
