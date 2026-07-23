"""Framework-focused page building (landing pages and their metadata)."""

from __future__ import annotations

from dash.html import H1, H2, A, Div, Img, Span

from ml_peg.app.utils.utils import get_framework_config

# GitHub "mark" (Octocat) logo as a self-contained inline SVG data URI.
GITHUB_ICON = "data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNiAxNiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ij48cGF0aCBmaWxsPSIjMTgxNzE3IiBkPSJNOCAwQzMuNTggMCAwIDMuNTggMCA4YzAgMy41NCAyLjI5IDYuNTMgNS40NyA3LjU5LjQuMDcuNTUtLjE3LjU1LS4zOCAwLS4xOS0uMDEtLjgyLS4wMS0xLjQ5LTIuMDEuMzctMi41My0uNDktMi42OS0uOTQtLjA5LS4yMy0uNDgtLjk0LS44Mi0xLjEzLS4yOC0uMTUtLjY4LS41Mi0uMDEtLjUzLjYzLS4wMSAxLjA4LjU4IDEuMjMuODIuNzIgMS4yMSAxLjg3Ljg3IDIuMzMuNjYuMDctLjUyLjI4LS44Ny41MS0xLjA3LTEuNzgtLjItMy42NC0uODktMy42NC0zLjk1IDAtLjg3LjMxLTEuNTkuODItMi4xNS0uMDgtLjItLjM2LTEuMDIuMDgtMi4xMiAwIDAgLjY3LS4yMSAyLjIuODIuNjQtLjE4IDEuMzItLjI3IDItLjI3LjY4IDAgMS4zNi4wOSAyIC4yNyAxLjUzLTEuMDQgMi4yLS44MiAyLjItLjgyLjQ0IDEuMS4xNiAxLjkyLjA4IDIuMTIuNTEuNTYuODIgMS4yNy44MiAyLjE1IDAgMy4wNy0xLjg3IDMuNzUtMy42NSAzLjk1LjI5LjI1LjU0LjczLjU0IDEuNDggMCAxLjA3LS4wMSAxLjkzLS4wMSAyLjIgMCAuMjEuMTUuNDYuNTUuMzhBOC4wMTMgOC4wMTMgMCAwIDAgMTYgOGMwLTQuNDItMy41OC04LTgtOHoiLz48L3N2Zz4="  # noqa: E501


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
    config = get_framework_config(framework_view["framework_id"])

    chip_style = {
        "display": "inline-flex",
        "alignItems": "center",
        "gap": "6px",
        "padding": "6px 12px",
        "borderRadius": "8px",
        "backgroundColor": "#ffffff",
        "border": "1px solid #e2e8f0",
        "color": "#334155",
        "fontSize": "13px",
        "fontWeight": "500",
        "textDecoration": "none",
    }
    chip_specs = []
    if config.get("paper_url"):
        chip_specs.append((Span("📄"), "Paper", config["paper_url"]))
    if config.get("project_url"):
        chip_specs.append((Span("🌐"), "Website", config["project_url"]))
    if config.get("github"):
        github_icon = Img(src=GITHUB_ICON, style={"height": "15px", "display": "block"})
        chip_specs.append((github_icon, "GitHub", config["github"]))
    chips = [
        A([icon, label], href=href, target="_blank", style=chip_style)
        for icon, label, href in chip_specs
    ]

    card_children = []
    if config.get("description"):
        card_children.append(
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
    if chips:
        card_children.append(
            Div(
                chips,
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "10px",
                    "marginTop": "12px" if card_children else "0",
                },
            )
        )

    header = [H1(f"{framework_label} Benchmarks")]
    if card_children:
        header.append(
            Div(
                card_children,
                style={
                    "backgroundColor": "#f8fafc",
                    "border": "1px solid #e2e8f0",
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
    for group in category_groups:
        sections.append(
            H2(
                group["category"],
                style={
                    "marginTop": "32px",
                    "marginBottom": "8px",
                    "fontSize": "28px",
                    "fontWeight": "700",
                    "borderBottom": "2px solid #e2e8f0",
                    "paddingBottom": "6px",
                },
            )
        )
        sections.append(Div(group["tests"], style={"display": "grid", "gap": "24px"}))

    return Div(
        [
            *header,
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
