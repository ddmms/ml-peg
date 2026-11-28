"""Browser-driven tests for the ML-PEG Model Explorer page."""

from __future__ import annotations

from collections.abc import Callable

from conftest import goto_explorer
from playwright.sync_api import Locator, Page, expect

from ml_peg.app.utils.explorer_ids import (
    COVERAGE_GRAPH,
    FINDER_GRAPH,
    H2H_DETAIL_GRAPH,
    H2H_GRAPH,
    REPORT_PROFILE,
    SIMILARITY_GRAPH,
    STRUCT_FAILS_GRAPH,
)
from ml_peg.app.utils.explorer_layout import _EXPLORER_JUMP_NAV, SCORING_GUIDE_URL

TIMEOUT = 60_000

# Plotly graph selectors (the ``.js-plotly-plot`` element carries ``data``/``layout``),
# built from the page's own ids so a rename cannot turn a check into a no-op.
FINDER = f"#{FINDER_GRAPH} .js-plotly-plot"
REPORT = f"#{REPORT_PROFILE} .js-plotly-plot"
H2H = f"#{H2H_GRAPH} .js-plotly-plot"
H2H_DETAIL = f"#{H2H_DETAIL_GRAPH} .js-plotly-plot"
FAILS = f"#{STRUCT_FAILS_GRAPH} .js-plotly-plot"
ROSE = f"#{COVERAGE_GRAPH} .js-plotly-plot"
HEATMAP = f"#{SIMILARITY_GRAPH} .js-plotly-plot"


def _goto_explorer(page: Page) -> None:
    """Open the Explorer and await its report-card plot."""
    goto_explorer(page)
    expect(page.locator(REPORT)).to_be_visible(timeout=TIMEOUT)


def _plot_value(page: Page, selector: str, expression: str) -> object:
    """
    Read a value off a rendered Plotly graph.

    Parameters
    ----------
    page
        The Playwright page.
    selector
        Selector of the ``.js-plotly-plot`` element.
    expression
        JavaScript over ``gd`` (the graph div), e.g. ``gd.data[0].y.length``.

    Returns
    -------
    object
        The evaluated value.
    """
    page.wait_for_function(
        "(sel) => { const gd = document.querySelector(sel); "
        "return !!(gd && gd.data && gd.data.length); }",
        arg=selector,
        timeout=TIMEOUT,
    )
    return page.evaluate(
        f"(sel) => {{ const gd = document.querySelector(sel); return {expression}; }}",
        selector,
    )


def _wait_plot(page: Page, selector: str, predicate: str) -> None:
    """
    Wait until a JavaScript predicate over a rendered graph holds.

    Parameters
    ----------
    page
        The Playwright page.
    selector
        Selector of the ``.js-plotly-plot`` element.
    predicate
        JavaScript boolean expression over ``gd`` (the graph div).
    """
    page.wait_for_function(
        "(sel) => { const gd = document.querySelector(sel); "
        f"return !!(gd && gd.data && gd.data.length) && ({predicate}); }}",
        arg=selector,
        timeout=TIMEOUT,
    )


def _emit_click(page: Page, selector: str, point: dict) -> None:
    """
    Fire a Plotly click on a graph once its data has rendered.

    Scattergl points cannot be clicked reliably by canvas coordinates, so the event
    is emitted directly; Dash re-derives the point's ``customdata`` from
    ``gd.data[curveNumber].customdata[pointNumber]``.

    Parameters
    ----------
    page
        The Playwright page.
    selector
        Selector of the ``.js-plotly-plot`` element.
    point
        The ``points[0]`` payload (``curveNumber``, ``pointNumber`` and, for bar
        charts keyed on the y axis, ``y``).
    """
    _wait_plot(page, selector, "true")
    page.evaluate(
        """(arg) => {
            const gd = document.querySelector(arg.sel);
            const p = Object.assign({}, arg.point);
            if (p.y === "__first__") p.y = gd.data[0].y[0];
            gd.emit('plotly_click', {points: [p]});
        }""",
        {"sel": selector, "point": point},
    )


def _dropdown_options(page: Page) -> Locator:
    """
    Locate the options of the currently open dropdown menu.

    Radio groups are ``role="listbox"`` too, so the dropdown's menu is told apart by
    not being one of them.

    Parameters
    ----------
    page
        The Playwright page.

    Returns
    -------
    Locator
        The visible dropdown options.
    """
    return page.locator(
        '[role="listbox"]:not(.dash-radioitems):visible [role="option"]'
    )


def _wait_focus(page: Page, class_name: str) -> None:
    """
    Wait until the focused element carries a class.

    Parameters
    ----------
    page
        The Playwright page.
    class_name
        A class the active element must have.
    """
    page.wait_for_function(
        "(cls) => !!document.activeElement && "
        "document.activeElement.classList.contains(cls)",
        arg=class_name,
        timeout=TIMEOUT,
    )


def _pick_option(page: Page, settle: Callable[[Locator], None], text: str) -> None:
    """
    Choose an option from the currently open dropdown menu.

    Parameters
    ----------
    page
        The Playwright page.
    settle
        The ``settle`` fixture (waits out post-load reflow).
    text
        Option text.
    """
    option = _dropdown_options(page).filter(has_text=text).first
    settle(option)
    # force past the sticky topbar, which can overlap the option once Playwright
    # scrolls it up under the header.
    option.click(force=True)


def test_explorer_panels_render(ready_page: Page) -> None:
    """Every panel's chart renders with data."""
    _goto_explorer(ready_page)
    for selector in (FINDER, REPORT, H2H, FAILS, ROSE):
        expect(ready_page.locator(selector)).to_be_visible(timeout=TIMEOUT)
        # Bar charts key their points on y; the polar rose on theta.
        assert _plot_value(
            ready_page, selector, "(gd.data[0].y || gd.data[0].theta).length"
        )
    heatmap = ready_page.locator(HEATMAP)
    expect(heatmap).to_be_visible(timeout=TIMEOUT)


def test_explorer_report_card(ready_page: Page) -> None:
    """The report card shows the summary, profile, legend and per-benchmark table."""
    _goto_explorer(ready_page)
    summary = ready_page.locator("#explorer-report-summary")
    for label in ("Trained at", "In-domain", "Extrapolated", "Physicality"):
        expect(summary).to_contain_text(label, timeout=TIMEOUT)
    expect(ready_page.locator("#explorer-report-table table")).to_be_visible(
        timeout=TIMEOUT
    )
    # The colour key names every test type the bars can take.
    legend = ready_page.locator("#explorer-sec-report .mlpeg-explorer-legend")
    expect(legend).to_contain_text("in-domain")
    expect(legend).to_contain_text("functional mismatch")


def test_explorer_report_card_follows_model_picker(
    ready_page: Page, settle: Callable[[Locator], None]
) -> None:
    """Choosing another model re-renders the report card for it."""
    _goto_explorer(ready_page)
    table = ready_page.locator("#explorer-report-table")
    before = table.inner_text()
    ready_page.locator("#explorer-report-model").click()
    # The second option: the first is the current model, so nothing would change.
    option = _dropdown_options(ready_page).nth(1)
    settle(option)
    model = option.inner_text().split("·")[1].strip()
    option.click(force=True)
    expect(ready_page.locator("#explorer-report-model")).to_contain_text(
        model, timeout=TIMEOUT
    )
    expect(table).not_to_have_text(before, timeout=TIMEOUT)


def test_explorer_head_to_head(ready_page: Page) -> None:
    """The head-to-head renders its verdict bars and headline; drilldown is closed."""
    _goto_explorer(ready_page)
    expect(ready_page.locator(H2H)).to_be_visible(timeout=TIMEOUT)
    assert _plot_value(ready_page, H2H, "gd.data[0].y.length") > 0
    expect(ready_page.locator("#explorer-h2h-detail-graph")).to_be_hidden()


def test_explorer_head_to_head_drilldown_by_click(ready_page: Page) -> None:
    """Clicking a verdict bar opens that category's structure scatter and picker."""
    _goto_explorer(ready_page)
    expect(ready_page.locator("#explorer-h2h-detail-graph")).to_be_hidden()
    # Bars are keyed by category on the y axis; the callback reads it off the point.
    _emit_click(ready_page, H2H, {"curveNumber": 0, "pointNumber": 0, "y": "__first__"})
    expect(ready_page.locator(H2H_DETAIL)).to_be_visible(timeout=TIMEOUT)
    category = _plot_value(ready_page, H2H, "gd.data[0].y[0]")
    _wait_plot(ready_page, H2H_DETAIL, f"gd.layout.title.text.startsWith({category!r})")
    # The area picker follows the click, so the two controls never disagree.
    expect(ready_page.locator("#explorer-h2h-detail-category")).to_contain_text(
        str(category), timeout=TIMEOUT
    )


def test_explorer_head_to_head_drilldown_by_keyboard(ready_page: Page) -> None:
    """The area picker opens the drilldown without a pointer (keyboard route)."""
    _goto_explorer(ready_page)
    expect(ready_page.locator("#explorer-h2h-detail-graph")).to_be_hidden()
    picker = ready_page.locator("#explorer-h2h-detail-category")
    picker.scroll_into_view_if_needed()
    # Open the menu, arrow past its search box onto the first area and choose it
    # with Enter -- what a keyboard user does instead of clicking a bar.
    picker.click()
    search = ready_page.locator("input.dash-dropdown-search:visible")
    expect(search).to_be_visible(timeout=TIMEOUT)
    # Confirm each step before the next key, so a slow render cannot drop one
    search.focus()
    _wait_focus(ready_page, "dash-dropdown-search")
    # Narrow to the pair's top verdict-bar area (a shared one) by typing it
    area = _plot_value(ready_page, H2H, "gd.data[0].y[0]")
    search.fill(area)
    expect(_dropdown_options(ready_page)).to_have_count(1, timeout=TIMEOUT)
    ready_page.keyboard.press("ArrowDown")
    _wait_focus(ready_page, "dash-options-list-option-checkbox")
    ready_page.keyboard.press("Enter")
    expect(picker).to_contain_text(area, timeout=TIMEOUT)
    expect(ready_page.locator(H2H_DETAIL)).to_be_visible(timeout=TIMEOUT)
    _wait_plot(ready_page, H2H_DETAIL, "gd.layout.title.text.includes('wins')")


def test_explorer_structure_click_shows_detail(ready_page: Page) -> None:
    """Clicking a head-to-head detail point loads its error table and 3D viewer."""
    _goto_explorer(ready_page)
    _emit_click(ready_page, H2H, {"curveNumber": 0, "pointNumber": 0, "y": "__first__"})
    expect(ready_page.locator(H2H_DETAIL)).to_be_visible(timeout=TIMEOUT)
    _emit_click(ready_page, H2H_DETAIL, {"curveNumber": 0, "pointNumber": 0})
    expect(ready_page.locator("#explorer-struct-table table")).to_be_visible(
        timeout=TIMEOUT
    )
    # The viewer shows the WEAS iframe when a geometry ships, or the fallback note.
    viewer = ready_page.locator("#explorer-struct-viewer")
    geometry = viewer.locator("iframe")
    fallback = viewer.get_by_text("No 3D geometry", exact=False)
    expect(geometry.or_(fallback).first).to_be_visible(timeout=TIMEOUT)


def test_explorer_finder_experiment_preset(ready_page: Page) -> None:
    """Choosing an experiment preset fills the category filter and re-ranks."""
    _goto_explorer(ready_page)
    preset = ready_page.locator('#explorer-finder-preset input[value="bulk"]')
    preset.check()
    # The preset writes its category bundle into the multi-select ...
    categories = ready_page.locator("#explorer-finder-categories")
    # (only areas with data are offered: here bulk_crystal alone)
    expect(categories).to_have_text("bulk_crystal", timeout=TIMEOUT)
    # ... without that write clearing the preset it came from ...
    expect(preset).to_be_checked(timeout=TIMEOUT)
    # ... and re-ranks on it: no model has the three bulk benchmarks the finder
    # needs (the synthetic models have one), so it says so.
    ready_page.wait_for_function(
        """(sel) => {
            const ann = document.querySelector(sel)?.layout?.annotations || [];
            return ann.some((a) => a.text.includes('enough evidence'));
        }""",
        arg=FINDER,
        timeout=TIMEOUT,
    )
    ready_page.locator('#explorer-finder-preset input[value="any"]').check()
    _wait_plot(ready_page, FINDER, "gd.data[0].y.length > 0")


def test_explorer_finder_tolerance_reports_thin_evidence(ready_page: Page) -> None:
    """With only in-domain evidence admitted, the fixture ranks nothing and says so."""
    _goto_explorer(ready_page)
    ready_page.locator('#explorer-finder-tolerance input[value="in_domain"]').check()
    ready_page.wait_for_function(
        """(sel) => {
            const gd = document.querySelector(sel);
            const ann = gd && gd.layout && gd.layout.annotations;
            return !!(ann && ann.length && ann[0].text.includes('enough evidence'));
        }""",
        arg=FINDER,
        timeout=TIMEOUT,
    )
    # Widening it back restores the ranking.
    ready_page.locator('#explorer-finder-tolerance input[value="any"]').check()
    _wait_plot(ready_page, FINDER, "gd.data[0].y.length > 0")


def test_explorer_finder_pick_opens_report_card(ready_page: Page) -> None:
    """The top-pick card is a button that loads its model into the report card."""
    _goto_explorer(ready_page)
    # The synthetic tree's second-best pick (fits 0.6 > 0.5 > 0.45 > 0.25)
    card = ready_page.locator("#explorer-finder-picks button", has_text="alpha-1")
    card = card.filter(has_not_text="alpha-1-D3")
    expect(card).to_be_visible(timeout=TIMEOUT)
    model = "alpha-1"
    card.focus()
    ready_page.keyboard.press("Enter")
    expect(ready_page.locator("#explorer-report-model")).to_contain_text(
        model, timeout=TIMEOUT
    )


def test_explorer_coverage_rose(
    ready_page: Page, settle: Callable[[Locator], None]
) -> None:
    """Switching the rose to per-benchmark grouping draws one wedge per benchmark."""
    _goto_explorer(ready_page)
    per_category = _plot_value(ready_page, ROSE, "gd.data[0].theta.length")
    ready_page.locator("#explorer-coverage-granularity").click()
    _pick_option(ready_page, settle, "By benchmark")
    _wait_plot(ready_page, ROSE, f"gd.data[0].theta.length > {per_category}")


def test_explorer_structure_fails_panel(ready_page: Page) -> None:
    """Switching the failure ranking to "most contested" re-ranks the bars."""
    _goto_explorer(ready_page)
    _wait_plot(
        ready_page, FAILS, "gd.layout.xaxis.title.text.includes('share of models')"
    )
    ready_page.locator('#explorer-fails-sort input[value="contested"]').check()
    _wait_plot(ready_page, FAILS, "gd.layout.xaxis.title.text.includes('disagreement')")
    assert _plot_value(ready_page, FAILS, "gd.data[0].y.length") > 0


def test_explorer_structure_fails_click_shows_detail(ready_page: Page) -> None:
    """Clicking a failure bar loads that structure's shared 3D detail panel."""
    _goto_explorer(ready_page)
    _emit_click(ready_page, FAILS, {"curveNumber": 0, "pointNumber": 0})
    expect(ready_page.locator("#explorer-struct-table table")).to_be_visible(
        timeout=TIMEOUT
    )


def test_explorer_structure_list_is_keyboard_route(ready_page: Page) -> None:
    """The structure list under the failure bars opens the 3D panel by keyboard."""
    _goto_explorer(ready_page)
    listing = ready_page.locator("#explorer-fails-list")
    listing.locator("summary").click()
    # One of the synthetic tree's two structures every model fails
    sid = "atomisation_3"
    button = listing.locator("button", has_text=sid)
    expect(button).to_be_visible(timeout=TIMEOUT)
    button.focus()
    ready_page.keyboard.press("Enter")
    expect(ready_page.locator("#explorer-struct-table")).to_contain_text(
        sid, timeout=TIMEOUT
    )
    expect(ready_page.locator("#explorer-struct-table table")).to_be_visible(
        timeout=TIMEOUT
    )


def test_explorer_similarity_heatmap(ready_page: Page) -> None:
    """The who-fails-alike heatmap renders as a square model x model matrix."""
    _goto_explorer(ready_page)
    heatmap = HEATMAP
    expect(ready_page.locator(heatmap)).to_be_visible(timeout=TIMEOUT)
    n_x = _plot_value(ready_page, heatmap, "gd.data[0].x.length")
    n_y = _plot_value(ready_page, heatmap, "gd.data[0].y.length")
    assert n_x == n_y > 1


def test_explorer_references_expand(
    ready_page: Page, settle: Callable[[Locator], None]
) -> None:
    """The methods panel reveals the scoring explainer and cited sources when opened."""
    _goto_explorer(ready_page)
    summary = ready_page.get_by_text("How the scores work", exact=False).first
    settle(summary)
    summary.click(force=True)
    # A known, verified citation link and the scoring-guide link become visible.
    citation = ready_page.get_by_text("Liang et al. (2022), HELM", exact=False)
    expect(citation.first).to_be_visible(timeout=TIMEOUT)
    guide = ready_page.get_by_text("scoring & normalisation guide", exact=False)
    expect(guide.first).to_have_attribute("href", SCORING_GUIDE_URL)


def test_explorer_jump_nav(ready_page: Page) -> None:
    """The intro jump nav exposes an anchor to every section, and each resolves."""
    _goto_explorer(ready_page)
    links = ready_page.locator("a.mlpeg-explorer-jump")
    expect(links.first).to_be_visible(timeout=TIMEOUT)
    hrefs = [links.nth(i).get_attribute("href") for i in range(links.count())]
    assert len(hrefs) == len(_EXPLORER_JUMP_NAV)
    for href in hrefs:
        expect(ready_page.locator(href)).to_have_count(1)
