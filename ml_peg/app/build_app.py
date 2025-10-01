"""Build main Dash application."""

from __future__ import annotations

from importlib import import_module
import warnings

from dash import Dash, Input, Output, State, callback
from dash.dash_table import DataTable
from dash.dcc import Store, Tab, Tabs
from dash.exceptions import PreventUpdate
from dash.html import H1, H3, Div
from yaml import safe_load

from ml_peg.analysis.utils.utils import calc_ranks, calc_scores, get_table_style
from ml_peg.app import APP_ROOT
from ml_peg.app.utils.build_components import build_weight_components


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

        # Build summary table
        summary_table = build_summary_table(
            all_tables[category], table_id=f"{category}-summary-table"
        )
        category_tables[category_title] = summary_table

        # Build weight components for summary table
        weight_components = build_weight_components(
            header="Benchmark weights",
            columns=list(all_tables[category].keys()),
            input_ids=list(all_tables[category].keys()),
            table_id=f"{category}-summary-table",
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

        # Bridge A: when any benchmark table under this category updates its Score,
        # push the new values into this category summary table column, then recompute
        # the category summary Score/Rank using current category weights.
        cat_table_id = f"{category}-summary-table"
        for test_name, test_table in all_tables[category].items():
            test_table_id = test_table.id

            @callback(
                Output(cat_table_id, "data", allow_duplicate=True),
                Output(cat_table_id, "style_data_conditional", allow_duplicate=True),
                Input(test_table_id, "data_timestamp"),
                State(f"{cat_table_id}-weight-store", "data"),
                State(cat_table_id, "data"),
                State(test_table_id, "data"),
                prevent_initial_call=True,
            )
            def _sync_category_from_benchmark(
                _ts,
                cat_weights,
                cat_data,
                bench_data,
                _test_name=test_name,
            ):
                if not cat_data or not bench_data:
                    return cat_data, get_table_style(cat_data or [])
                # Map MLIP -> Score from benchmark
                bench_scores = {
                    r.get("MLIP"): r.get("Score") for r in bench_data if "MLIP" in r
                }
                # Update the specific column for this benchmark
                for row in cat_data:
                    mlip = row.get("MLIP")
                    if mlip in bench_scores:
                        row[_test_name] = bench_scores[mlip]
                # Recompute category Score/Rank using current category weights
                try:
                    updated = calc_scores(cat_data, cat_weights or {})
                except Exception:
                    updated = calc_scores(cat_data, {})
                updated = calc_ranks(updated)
                style = get_table_style(updated)
                return updated, style

        # Bridge B: also listen to the global summary scores store. This ensures
        # that tables which push their Scores via the shared store (e.g. normalized
        # tables like OC157) propagate reliably to the category table without having
        # to rely solely on their DataTable data_timestamp updates.
        tests_in_category = tuple(all_tables[category].keys())

        @callback(
            Output(cat_table_id, "data", allow_duplicate=True),
            Output(cat_table_id, "style_data_conditional", allow_duplicate=True),
            Input("summary-table-scores-store", "data"),
            State(f"{cat_table_id}-weight-store", "data"),
            State(cat_table_id, "data"),
            prevent_initial_call=True,
        )
        def _sync_category_from_scores_store(
            scores_data: dict | None,
            cat_weights: dict[str, float] | None,
            cat_data: list[dict] | None,
            _tests_in_category=tests_in_category,
        ):
            if not scores_data or not cat_data:
                raise PreventUpdate

            print(f"BRIDGE B DEBUG [{category}]: Received scores_data: {scores_data}")
            print(
                f"BRIDGE B DEBUG [{category}]: Tests in category: {_tests_in_category}"
            )

            # Build a prospective update; only proceed if any benchmark column value changes
            changed = False
            new_data = []
            for row in cat_data:
                mlip = row.get("MLIP")
                new_row = row.copy()
                if mlip:
                    for bn in _tests_in_category:
                        bn_scores = scores_data.get(bn) or {}
                        if mlip in bn_scores:
                            new_val = bn_scores[mlip]
                            if new_row.get(bn) != new_val:
                                print(
                                    f"BRIDGE B DEBUG [{category}]: Updating {mlip} {bn}: {new_row.get(bn)} -> {new_val}"
                                )
                                new_row[bn] = new_val
                                changed = True
                new_data.append(new_row)

            if not changed:
                print(
                    f"BRIDGE B DEBUG [{category}]: No changes detected, preventing update"
                )
                raise PreventUpdate

            print(
                f"BRIDGE B DEBUG [{category}]: Changes detected, updating category table"
            )

            # Recompute category Score/Rank using current category weights
            try:
                updated = calc_scores(new_data, cat_weights or {})
            except Exception:
                updated = calc_scores(new_data, {})
            updated = calc_ranks(updated)
            style = get_table_style(updated)
            return updated, style

        # Bridge C: push this category's scores into the global summary-store
        # using the display title as the store key so the Summary table columns
        # (which use titles) are updated correctly.
        @callback(
            Output("summary-table-scores-store", "data", allow_duplicate=True),
            Input(cat_table_id, "data_timestamp"),
            Input(
                f"{cat_table_id}-weight-store", "data"
            ),  # Also listen for weight changes
            State(cat_table_id, "data"),
            State("summary-table-scores-store", "data"),
            prevent_initial_call="initial_duplicate",
        )
        def _push_category_scores_by_title(
            _ts: int | None,
            _weight_store: dict[str, float] | None,
            cat_data: list[dict] | None,
            scores_data: dict[str, dict[str, float]] | None,
            _category_title=category_title,
        ):
            print(
                f"BRIDGE C DEBUG [{_category_title}]: Callback triggered - _ts: {_ts}, _weight_store keys: {list(_weight_store.keys()) if _weight_store else None}"
            )
            if not cat_data:
                print(
                    f"BRIDGE C DEBUG [{_category_title}]: No cat_data, preventing update"
                )
                raise PreventUpdate
            mapping = {r.get("MLIP"): r.get("Score") for r in cat_data if "MLIP" in r}

            print(
                f"BRIDGE C DEBUG [{_category_title}]: Current category scores: {mapping}"
            )
            print(
                f"BRIDGE C DEBUG [{_category_title}]: Existing scores_data: {scores_data}"
            )

            # Always return a fresh object so Store change is detected
            base = dict(scores_data or {})
            base[_category_title] = mapping
            print(
                f"BRIDGE C DEBUG [{_category_title}]: Pushing updated scores_data: {base}"
            )
            return base

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
        # Global store for benchmark/category scores available across all tabs
        Store(id="summary-table-scores-store", storage_type="session"),
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
        columns=list(category_tables.keys()),
        input_ids=list(category_tables.keys()),
        table_id="summary-table",
    )
    # Build summary and category tabs
    build_tabs(full_app, category_layouts, summary_table, weight_components)
