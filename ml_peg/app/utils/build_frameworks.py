"""Framework page and summary-table builders for the ML-PEG app."""

from __future__ import annotations

from typing import NotRequired, TypedDict

from dash.dash_table import DataTable
from dash.html import H1, H3, Br, Div

from ml_peg.app.utils.build_components import (
    build_download_controls,
    build_loading_summary_table,
    build_weight_components,
)
from ml_peg.app.utils.utils import get_framework_config


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


class CategoryGroup(TypedDict):
    """Benchmarks of one category, grouped for display on a framework page."""

    category: str
    tests: list[Div]


class FrameworkView(TypedDict):
    """Data needed to build a single framework page."""

    framework_id: str
    label: str
    category_groups: list[CategoryGroup]
    summary_table: NotRequired[DataTable]
    weight_components: NotRequired[Div]


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
        Framework page metadata (keys are the frameworks to build tables for;
        ``ml_peg`` and empty frameworks are already excluded).

    Returns
    -------
    tuple[dict[str, DataTable], dict[str, dict[str, DataTable]]]
        Per-framework summary tables keyed by framework id, and the
        {framework_id: {benchmark_name: benchmark_table}} grouping.
    """
    # Local import avoids a circular import (build_app imports this module).
    from ml_peg.app.build_app import build_summary_table

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
    category_groups = framework_view["category_groups"]
    summary_table = framework_view.get("summary_table")
    weight_components = framework_view.get("weight_components")

    sections = []
    for group in category_groups:
        sections.append(H3(group["category"], style={"marginTop": "26px"}))
        sections.append(Div(group["tests"], style={"display": "grid", "gap": "24px"}))

    summary_block = []
    if summary_table is not None:
        summary_block = [
            Div(
                [
                    build_download_controls(summary_table.id, row=True),
                    build_loading_summary_table(summary_table),
                    Br(),
                    weight_components,
                ],
                style={"width": "fit-content"},
            ),
        ]

    return Div(
        [
            H1(f"{framework_label} Benchmarks"),
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
