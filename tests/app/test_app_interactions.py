"""Deeper interactivity characterization for the ML-PEG app.

These pin the click-driven behaviours a user relies on — normalised toggle,
threshold edits, colour scheme, CSV export and cross-page navigation — so the
frontend overhaul cannot quietly break them.
"""

from __future__ import annotations

from playwright.sync_api import Locator, Page, expect

TIMEOUT = 60_000


def _settle(locator: Locator, timeout: int = TIMEOUT) -> None:
    """Wait until a locator's bounding box stops moving.

    The benchmark page keeps reflowing for a moment after load while its Plotly
    figures autosize and re-theme; on a slow CI runner that shift can outlast
    Playwright's own actionability wait, so hovers/clicks time out with "element
    is not stable". Poll the box until it is unchanged across two ~120ms samples
    before interacting.

    Parameters
    ----------
    locator
        Element to wait on.
    timeout
        Maximum time to wait, in milliseconds.
    """
    page = locator.page
    locator.wait_for(state="visible", timeout=timeout)
    deadline = timeout
    prev = locator.bounding_box()
    while deadline > 0:
        page.wait_for_timeout(120)
        deadline -= 120
        curr = locator.bounding_box()
        if prev is not None and curr == prev:
            return
        prev = curr


def _goto_ionpi19(page: Page) -> None:
    """Open the non-covalent-interactions category and wait for the benchmark table."""
    page.locator('#sidebar-nav a[href^="/category/"]').first.click()
    expect(page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)


def _expand_scoring(page: Page) -> None:
    """Expand the collapsed 'Adjust scoring' panel to reveal its controls."""
    page.get_by_text("Adjust scoring", exact=False).first.click()
    expect(page.locator("#IONPI19-table-normalized-toggle")).to_be_visible(
        timeout=TIMEOUT
    )


def test_normalized_toggle_changes_values(ready_page: Page) -> None:
    """Toggling normalised scores changes the displayed metric values."""
    _goto_ionpi19(ready_page)
    _expand_scoring(ready_page)
    mae = ready_page.locator('#IONPI19-table td[data-dash-column="MAE"]').first
    expect(mae).to_be_visible(timeout=TIMEOUT)
    before = mae.inner_text()

    ready_page.locator("#IONPI19-table-normalized-toggle input").first.check()

    expect(mae).not_to_have_text(before, timeout=TIMEOUT)


def test_threshold_edit_updates_score(ready_page: Page) -> None:
    """Editing a metric threshold rescores the benchmark table."""
    _goto_ionpi19(ready_page)
    _expand_scoring(ready_page)
    score = ready_page.locator('#IONPI19-table td[data-dash-column="Score"]').first
    expect(score).to_be_visible(timeout=TIMEOUT)
    before = score.inner_text()

    good = ready_page.locator("#IONPI19-table-MAE-good-threshold")
    good.fill("15")
    good.press("Enter")

    expect(score).not_to_have_text(before, timeout=TIMEOUT)


def test_csv_download(ready_page: Page) -> None:
    """The CSV export control downloads a .csv file for the benchmark table."""
    _goto_ionpi19(ready_page)
    ready_page.locator("#IONPI19-table-download-format").click()
    csv_option = ready_page.locator('[role="option"]', has_text="CSV").first
    _settle(csv_option)
    # force past the sticky topbar (z-index 1600), which can overlap the option
    # once Playwright scrolls it up under the header.
    csv_option.click(force=True)

    with ready_page.expect_download(timeout=TIMEOUT) as download_info:
        ready_page.locator("#IONPI19-table-download-button").click()

    assert download_info.value.suggested_filename.endswith(".csv")


def test_colormap_change_keeps_table(ready_page: Page) -> None:
    """Switching the colour scheme recolours cells without dropping any rows."""
    rows = ready_page.locator("#summary-table tbody tr")
    expect(rows.first).to_be_visible(timeout=TIMEOUT)
    count = rows.count()

    # The colour-scheme control lives in the header settings popover; open it.
    ready_page.locator(".mlpeg-settings-summary").click()
    ready_page.locator("#cmap-dropdown").click()
    ready_page.locator('[role="option"]', has_text="Green-Red").first.click()

    expect(rows).to_have_count(count, timeout=TIMEOUT)
    expect(rows.first).to_be_visible()


def test_navigation_roundtrip(ready_page: Page) -> None:
    """Summary → category → summary navigation restores each page."""
    _goto_ionpi19(ready_page)
    ready_page.locator('#sidebar-nav a[href="/"]').first.click()
    expect(ready_page.locator("#summary-table")).to_be_visible(timeout=TIMEOUT)


def test_scoring_panel_collapsed_by_default(ready_page: Page) -> None:
    """The heavy weight/threshold controls stay collapsed until expanded."""
    _goto_ionpi19(ready_page)
    expect(ready_page.locator("#IONPI19-table-normalized-toggle")).to_be_hidden()
    _expand_scoring(ready_page)
    expect(ready_page.locator("#IONPI19-table-normalized-toggle")).to_be_visible()


def test_no_horizontal_page_overflow(ready_page: Page) -> None:
    """A heavy category page must not overflow horizontally (tables scroll)."""
    _goto_ionpi19(ready_page)
    overflow = ready_page.evaluate(
        "document.documentElement.scrollWidth - document.documentElement.clientWidth"
    )
    assert overflow <= 1, f"page overflows horizontally by {overflow}px"


def test_benchmark_card_collapse_expand(ready_page: Page) -> None:
    """Collapsing a benchmark card unmounts its table; expanding remounts it."""
    _goto_ionpi19(ready_page)
    expect(ready_page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)

    header = ready_page.locator(".mlpeg-bench-header").first
    header.click()  # collapse -> body unmounts
    expect(ready_page.locator("#IONPI19-table")).to_have_count(0, timeout=TIMEOUT)

    header.click()  # expand -> body remounts (lazy)
    expect(ready_page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)


def test_summary_weights_align_with_columns(ready_page: Page) -> None:
    """Weight boxes stay centred under their table columns on a category page.

    Regression guard for the fixed-width summary table vs stretching weight-grid
    misalignment: on a few-column page with a wide viewport the ``width:100%``
    weight grid used to stretch past the (non-stretching) table and drift the
    boxes to the right (reported on Molecular Crystals).
    """
    # A wide viewport maximises the surplus width the bug fed on.
    ready_page.set_viewport_size({"width": 1600, "height": 1000})
    _goto_ionpi19(ready_page)
    card = ready_page.locator(".mlpeg-summary-card")
    expect(card.locator(".mlpeg-scoring-grid input").first).to_be_visible(
        timeout=TIMEOUT
    )

    # Compare the centre-x of every weight input to its metric column header. The
    # grid tracks and the table columns share a template, so the Nth metric input
    # must sit under the Nth metric header (skip MLIP / link / Score — no weights).
    result = ready_page.evaluate(
        """() => {
            const card = document.querySelector('.mlpeg-summary-card');
            const grid = card && card.querySelector('.mlpeg-scoring-grid');
            const table = card && card.querySelector('table');
            if (!grid || !table) return {error: 'grid or table not found'};
            const headers = [];
            table.querySelectorAll('th[data-dash-column]').forEach((th) => {
                const col = th.getAttribute('data-dash-column');
                if (col === 'MLIP' || col === 'link' || col === 'Score') return;
                const r = th.getBoundingClientRect();
                if (r.width > 0) headers.push(r.left + r.width / 2);
            });
            const inputs = [];
            grid.querySelectorAll('input').forEach((inp) => {
                const r = inp.getBoundingClientRect();
                if (r.width > 0) inputs.push(r.left + r.width / 2);
            });
            if (!inputs.length) return {error: 'no weight inputs found'};
            if (headers.length !== inputs.length) {
                return {error: 'header/input count mismatch: '
                        + headers.length + ' vs ' + inputs.length};
            }
            let max = 0;
            for (let i = 0; i < inputs.length; i++) {
                max = Math.max(max, Math.abs(inputs[i] - headers[i]));
            }
            return {max: max, count: inputs.length};
        }"""
    )
    assert "error" not in result, result.get("error")
    # A few px covers sub-pixel rounding and borders; the bug drifts tens of px.
    assert result["max"] <= 6, (
        f"weight inputs misaligned with columns by up to {result['max']}px"
    )


def test_cell_click_detail_not_clipped(ready_page: Page) -> None:
    """Clicking a cell shows detail that the benchmark card no longer clips.

    Regression guard for ``.mlpeg-bench-card { overflow: hidden }`` clipping the
    injected detail plot / structure viewer at the card edge.
    """
    _goto_ionpi19(ready_page)
    # IONPI19 plots the scatter into its figure placeholder on a MAE-cell click.
    cell = ready_page.locator('#IONPI19-table td[data-dash-column="MAE"]').first
    expect(cell).to_be_visible(timeout=TIMEOUT)
    cell.click()

    ready_page.wait_for_function(
        "document.getElementById('IONPI19-figure-placeholder')"
        "?.querySelector('.js-plotly-plot') !== null",
        timeout=TIMEOUT,
    )

    result = ready_page.evaluate(
        """() => {
            const ph = document.getElementById('IONPI19-figure-placeholder');
            const card = ph && ph.closest('.mlpeg-bench-card');
            if (!ph || !card) return {error: 'placeholder or card not found'};
            return {
                height: ph.offsetHeight,
                cardOverflow: getComputedStyle(card).overflow,
            };
        }"""
    )
    assert "error" not in result, result.get("error")
    assert result["height"] > 0, "detail plot rendered with zero height"
    assert result["cardOverflow"] != "hidden", (
        "benchmark card still clips detail (overflow: hidden)"
    )


def test_hover_tooltip_shows_fully_in_viewport(ready_page: Page) -> None:
    """DataTable hover tooltips are re-presented at body level, unclipped.

    The native Dash tooltip is trapped inside the horizontally-scrolling table
    wrapper (and hidden outright near cell edges); tooltip_portal.js mirrors it
    into a viewport-fixed card. Every table's Score header carries a description
    tooltip, so hovering it must surface that card fully within the viewport.
    """
    _goto_ionpi19(ready_page)
    # The asset marks <html> once it has wired up.
    ready_page.wait_for_function(
        "document.documentElement.classList.contains('mlpeg-tooltip-js')",
        timeout=TIMEOUT,
    )
    header = ready_page.locator('#IONPI19-table th[data-dash-column="Score"]')
    expect(header).to_be_visible(timeout=TIMEOUT)
    header.hover()

    portal = ready_page.locator(".mlpeg-tooltip-portal.is-visible")
    expect(portal).to_be_visible(timeout=TIMEOUT)
    assert (portal.inner_text() or "").strip(), "tooltip portal rendered empty"

    # Fully inside the viewport (1px tolerance for sub-pixel rounding).
    box = portal.bounding_box()
    assert box is not None, "tooltip portal has no bounding box"
    viewport = ready_page.evaluate(
        "() => ({ w: window.innerWidth, h: window.innerHeight })"
    )
    assert box["x"] >= -1, f"tooltip clipped at left edge: {box}"
    assert box["y"] >= -1, f"tooltip clipped at top edge: {box}"
    assert box["x"] + box["width"] <= viewport["w"] + 1, (
        f"tooltip overflows right edge: {box} vw={viewport['w']}"
    )
    assert box["y"] + box["height"] <= viewport["h"] + 1, (
        f"tooltip overflows bottom edge: {box} vh={viewport['h']}"
    )


def test_click_pins_info_card(ready_page: Page) -> None:
    """Clicking a body cell pins its info card so it persists and stays interactive.

    A transient hover card vanishes on mouse-out; a pinned one must stay open (so
    its text can be moused over, selected and copied) until dismissed.
    """
    _goto_ionpi19(ready_page)
    ready_page.wait_for_function(
        "document.documentElement.classList.contains('mlpeg-tooltip-js')",
        timeout=TIMEOUT,
    )
    # Every model row carries a level-of-theory tooltip on its MLIP cell.
    cell = ready_page.locator('#IONPI19-table td[data-dash-column="MLIP"]').first
    expect(cell).to_be_visible(timeout=TIMEOUT)
    _settle(cell)  # wait out the post-load reflow so the cell stops moving
    cell.hover()  # let Dash render the cell's tooltip so there is content to pin
    ready_page.wait_for_selector(".mlpeg-tooltip-portal.is-visible", timeout=TIMEOUT)
    ready_page.wait_for_timeout(250)  # let the hovered tooltip settle before pinning
    cell.hover()  # re-assert hover in case the settle wait drifted the pointer
    cell.click()

    pinned = ready_page.locator(".mlpeg-tooltip-portal.is-pinned")
    expect(pinned).to_be_visible(timeout=TIMEOUT)

    # Move the pointer well away: a transient card would vanish; a pinned one stays.
    ready_page.mouse.move(20, 820)
    ready_page.wait_for_timeout(400)
    expect(pinned).to_be_visible()

    # The × dismisses it.
    ready_page.locator(".mlpeg-tooltip-close").click()
    expect(ready_page.locator(".mlpeg-tooltip-portal.is-visible")).to_have_count(
        0, timeout=TIMEOUT
    )
