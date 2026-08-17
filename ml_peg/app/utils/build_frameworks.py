"""Framework page and summary-table builders for the ML-PEG app."""

from __future__ import annotations

from typing import TypedDict

from dash.dash_table import DataTable
from dash.html import H1, H3, A, Br, Div, Img, Span

from ml_peg.app.utils.build_components import (
    build_download_controls,
    build_loading_summary_table,
    build_summary_table,
    build_weight_components,
)
from ml_peg.app.utils.utils import get_framework_config

# GitHub mark from Bootstrap Icons `github.svg` (MIT), fill set to #181717:
# https://github.com/twbs/icons/blob/main/icons/github.svg
GITHUB_ICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNiAxNiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ij48cGF0aCBmaWxsPSIjMTgxNzE3IiBkPSJNOCAwQzMuNTggMCAwIDMuNTggMCA4YzAgMy41NCAyLjI5IDYuNTMgNS40NyA3LjU5LjQuMDcuNTUtLjE3LjU1LS4zOCAwLS4xOS0uMDEtLjgyLS4wMS0xLjQ5LTIuMDEuMzctMi41My0uNDktMi42OS0uOTQtLjA5LS4yMy0uNDgtLjk0LS44Mi0xLjEzLS4yOC0uMTUtLjY4LS41Mi0uMDEtLjUzLjYzLS4wMSAxLjA4LjU4IDEuMjMuODIuNzIgMS4yMSAxLjg3Ljg3IDIuMzMuNjYuMDctLjUyLjI4LS44Ny41MS0xLjA3LTEuNzgtLjItMy42NC0uODktMy42NC0zLjk1IDAtLjg3LjMxLTEuNTkuODItMi4xNS0uMDgtLjItLjM2LTEuMDIuMDgtMi4xMiAwIDAgLjY3LS4yMSAyLjIuODIuNjQtLjE4IDEuMzItLjI3IDItLjI3LjY4IDAgMS4zNi4wOSAyIC4yNyAxLjUzLTEuMDQgMi4yLS44MiAyLjItLjgyLjQ0IDEuMS4xNiAxLjkyLjA4IDIuMTIuNTEuNTYuODIgMS4yNy44MiAyLjE1IDAgMy4wNy0xLjg3IDMuNzUtMy42NSAzLjk1LjI5LjI1LjU0LjczLjU0IDEuNDggMCAxLjA3LS4wMSAxLjkzLS4wMSAyLjIgMCAuMjEuMTUuNDYuNTUuMzhBOC4wMTMgOC4wMTMgMCAwIDAgMTYgOGMwLTQuNDItMy41OC04LTgtOHoiLz48L3N2Zz4="  # noqa: E501


class CategoryTest(TypedDict):
    """One benchmark's entry within a category view."""

    name: str
    framework_ids: list[str]
    layout: Div


class CategoryView(TypedDict):
    """Data needed to build a single category page."""

    title: str
    description: str
    summary_table: DataTable
    weight_components: Div
    tests: list[CategoryTest]


class _FrameworkViewRequired(TypedDict):
    """Keys always present in a framework page view."""

    framework_id: str
    label: str
    benchmarks_by_category: dict[str, list[Div]]


class FrameworkView(_FrameworkViewRequired, total=False):
    """
    Data needed to build a single framework page.

    Inherits the required keys and adds the summary table and weight controls,
    which are only set once the framework's summary table has been built.
    """

    summary_table: DataTable
    weight_components: Div


def build_framework_views(
    category_views: dict[str, CategoryView],
    framework_ids: set[str],
) -> dict[str, FrameworkView]:
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
    dict[str, FrameworkView]
        Mapping of framework ID to grouped benchmark layouts by category.
    """
    framework_views: dict[str, FrameworkView] = {}
    for framework_id in sorted(framework_ids):
        if framework_id == "ml_peg":
            continue

        benchmarks_by_category: dict[str, list[Div]] = {}
        for category_name, category_view in category_views.items():
            tests = [
                test["layout"]
                for test in category_view["tests"]
                if framework_id in test["framework_ids"]
            ]
            if tests:
                benchmarks_by_category[category_name] = tests

        if benchmarks_by_category:
            framework_views[framework_id] = {
                "framework_id": framework_id,
                "label": get_framework_config(framework_id)["label"],
                "benchmarks_by_category": benchmarks_by_category,
            }

    return framework_views


def build_framework_summary_tables(
    all_tables: dict[str, dict[str, DataTable]],
    all_frameworks: dict[str, dict[str, str]],
    framework_views: dict[str, FrameworkView],
) -> tuple[dict[str, DataTable], dict[str, dict[str, DataTable]]]:
    """
    Build a per-framework summary table for each framework page.

    Parameters
    ----------
    all_tables
        Benchmark tables grouped by category.
    all_frameworks
        Framework ids for each benchmark, grouped by category.
    framework_views
        Framework page metadata, keyed by the frameworks to build tables for.
        ``ml_peg`` and empty frameworks are already excluded.

    Returns
    -------
    tuple[dict[str, DataTable], dict[str, dict[str, DataTable]]]
        Per-framework summary tables keyed by framework id, and the
        {framework_id: {benchmark_name: benchmark_table}} grouping.
    """
    framework_tables: dict[str, DataTable] = {}
    framework_grouping: dict[str, dict[str, DataTable]] = {}

    for framework_id in framework_views:
        # Gather benchmark tables tagged with this framework (deterministic order)
        benchmarks: dict[str, DataTable] = {}
        for category in sorted(all_tables):
            for test_name in sorted(all_tables[category]):
                if framework_id in all_frameworks[category][test_name]:
                    benchmarks[test_name] = all_tables[category][test_name]
        if not benchmarks:
            continue

        summary_table = build_summary_table(
            benchmarks,
            table_id=f"{framework_id}-framework-summary-table",
            description=str(framework_views[framework_id]["label"]),
        )
        weight_components = build_weight_components(
            header="Weights",
            table=summary_table,
            include_download_controls=False,
            column_widths=getattr(summary_table, "column_widths", None),
        )
        framework_views[framework_id]["summary_table"] = summary_table
        framework_views[framework_id]["weight_components"] = weight_components
        framework_tables[framework_id] = summary_table
        framework_grouping[framework_id] = benchmarks

    return framework_tables, framework_grouping


def build_framework_page_layout(framework_view: FrameworkView) -> Div:
    """
    Build a framework-focused page with its summary table and benchmark sections.

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
    benchmarks_by_category = framework_view["benchmarks_by_category"]
    config = get_framework_config(framework_view["framework_id"])
    summary_table = framework_view.get("summary_table")
    weight_components = framework_view.get("weight_components")

    link_style = {
        "display": "inline-flex",
        "alignItems": "center",
        "gap": "6px",
        "padding": "6px 12px",
        "borderRadius": "8px",
        "backgroundColor": "#ffffff",
        "border": "1px solid var(--mlpeg-border)",
        "color": "#334155",
        "fontSize": "13px",
        "fontWeight": "500",
        "textDecoration": "none",
    }
    link_specs = []
    if config.get("paper_url"):
        link_specs.append((Span("📄"), "Paper", config["paper_url"]))
    if config.get("project_url"):
        link_specs.append((Span("🌐"), "Website", config["project_url"]))
    if config.get("github"):
        github_icon = Img(src=GITHUB_ICON, style={"height": "15px", "display": "block"})
        link_specs.append((github_icon, "GitHub", config["github"]))
    links = [
        A([icon, label], href=href, target="_blank", style=link_style)
        for icon, label, href in link_specs
    ]

    description_box_children = []
    if config.get("description"):
        description_box_children.append(
            Div(
                config["description"],
                style={
                    "fontSize": "15px",
                    "lineHeight": "1.6",
                    "color": "#475569",
                    "maxWidth": "760px",
                },
            )
        )
    if links:
        description_box_children.append(
            Div(
                links,
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "10px",
                    "marginTop": "12px" if description_box_children else "0",
                },
            )
        )

    description_box_block = []
    if description_box_children:
        description_box_block.append(
            Div(
                description_box_children,
                style={
                    "backgroundColor": "#f8fafc",
                    "border": "1px solid var(--mlpeg-border)",
                    "borderRadius": "12px",
                    "padding": "16px 20px",
                    "marginTop": "10px",
                    "marginBottom": "8px",
                    "width": "fit-content",
                    "maxWidth": "820px",
                },
            )
        )

    sections = []
    for category_name, tests in benchmarks_by_category.items():
        sections.append(H3(category_name, style={"marginTop": "26px"}))
        # Same grid as category pages: an implicit `auto` track sizes to
        # max-content, which would let a wide benchmark table stretch the page.
        sections.append(Div(tests, className="mlpeg-benchmark-grid"))

    summary_block = []
    if summary_table is not None:
        # Wrap in the same scroll-card as category pages so a wide framework
        # summary (e.g. MACE-POLAR-1's many benchmark columns) scrolls inside
        # its card instead of spanning the whole page.
        summary_block = [
            Div(
                Div(
                    [
                        build_download_controls(summary_table.id, row=True),
                        build_loading_summary_table(summary_table),
                        Br(),
                        weight_components,
                    ],
                    style={"width": "fit-content"},
                ),
                className="mlpeg-table-scroll",
            ),
        ]

    return Div(
        [
            H1(f"{framework_label} Benchmarks"),
            *description_box_block,
            Div(
                (
                    "These benchmarks also appear on their category pages and "
                    "share the same benchmark controls, so weight and threshold "
                    "edits stay in sync across both views."
                ),
                style={
                    "fontSize": "13px",
                    "fontStyle": "italic",
                    "color": "#64748b",
                    "marginTop": "8px",
                    "marginBottom": "8px",
                },
            ),
            *summary_block,
            *sections,
        ]
    )
