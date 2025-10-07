"""Build main Dash application."""

from __future__ import annotations

from importlib import import_module
import warnings

from dash import Dash, Input, Output, callback
from dash.dash_table import DataTable
from dash.dcc import Store, Tab, Tabs
from dash.html import H1, H3, Div
from yaml import safe_load

from ml_peg.analysis.utils.utils import calc_ranks, calc_scores, get_table_style
from ml_peg.app import APP_ROOT
from ml_peg.app.utils.build_components import build_weight_components
from ml_peg.app.utils.register_callbacks import (
    register_benchmark_to_category_callback,
)


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

    # `category` corresponds to the category's directory name
    # We will use the loaded `category_title` for IDs/dictionary keys returned
    for category in all_layouts:
        # Get category name and description
        try:
            with open(APP_ROOT / category / f"{category}.yml") as file:
                category_info = safe_load(file)
                category_title = category_info.get("title", category)
                category_descrip = category_info.get("description", "")
        except FileNotFoundError:
            category_title = category
            category_descrip = ""

        # Build category summary table
        summary_table = build_summary_table(
            all_tables[category], table_id=f"{category_title}-summary-table"
        )
        category_tables[category_title] = summary_table

        # Build weight components for category summary table
        weight_components = build_weight_components(
            header="Benchmark weights",
            table=summary_table,
        )

        # Build full layout with summary table, weight controls, and test layouts
        category_layouts[category_title] = Div(
            [
                H1(category_title),
                H3(category_descrip),
                summary_table,
                weight_components,
                Div([all_layouts[category][test] for test in all_layouts[category]]),
            ]
        )

        # Register benchmark table -> category table callbacks
        for test_name, benchmark_table in all_tables[category].items():
            register_benchmark_to_category_callback(
                benchmark_table_id=benchmark_table.id,
                category_table_id=f"{category_title}-summary-table",
                benchmark_column=test_name,
            )

    return category_layouts, category_tables


def build_summary_table(
    tables: dict[str, DataTable], table_id: str = "summary-table"
) -> DataTable:
    """
    Build summary table from a set of tables.

    Parameters
    ----------
    tables
        Dictionary of tables to be summarised.
    table_id
        ID of table being built. Default is 'summary-table'.

    Returns
    -------
    DataTable
        Summary table with score from table being summarised.
    """
    summary_data = {}
    for category_name, table in tables.items():
        if not summary_data:
            summary_data = {row["MLIP"]: {} for row in table.data}
        for row in table.data:
            summary_data[row["MLIP"]][category_name] = row["Score"]

    data = []
    for mlip in summary_data:
        data.append({"MLIP": mlip} | summary_data[mlip])

    data = calc_scores(data)
    data = calc_ranks(data)

    columns_headers = ("MLIP",) + tuple(tables.keys()) + ("Score", "Rank")
    columns = [{"name": headers, "id": headers} for headers in columns_headers]

    style = get_table_style(data)

    return DataTable(
        data=data,
        columns=columns,
        id=table_id,
        sort_action="native",
        style_data_conditional=style,
    )


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
        H1("MLIP benchmarking"),
        Tabs(id="all-tabs", value="summary-tab", children=all_tabs),
        Div(id="tabs-content"),
    ]

    full_app.layout = Div(tabs_layout)

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
                ]
            )
        return Div([layouts[tab]])


def build_full_app(full_app: Dash) -> None:
    """
    Build full app layout and register callbacks.

    Parameters
    ----------
    full_app
        Full application with all sub-apps.
    """
    # Get layouts and tables for each test, grouped by categories
    all_layouts, all_tables = get_all_tests()

    if not all_layouts:
        raise ValueError("No tests were built successfully")

    # Combine tests into categories and create category summary
    category_layouts, category_tables = build_category(all_layouts, all_tables)
    # Build overall summary table
    summary_table = build_summary_table(category_tables)
    weight_components = build_weight_components(
        header="Benchmark weights",
        table=summary_table,
    )
    # Build summary and category tabs
    build_tabs(full_app, category_layouts, summary_table, weight_components)
