"""Pin the session-state resync behaviour of dynamically injected tables.

Benchmark tables are baked at build time with the default view and their score
callbacks use ``prevent_initial_call=True``. That flag only suppresses the
app-load initial batch: a table *inserted* later (router navigation, lazy card
expand) fires its callbacks on insertion and re-reads the session stores. These
tests pin that load-bearing behaviour — a table injected after the user changed
the colour scheme, model filter, thresholds or normalise toggle must render the
session's state, not the baked defaults — and that a value flip does not revert
after the paint (the store write-back must not chain a stale re-render).
"""

from __future__ import annotations

from playwright.sync_api import Page, expect

TIMEOUT = 60_000

# Find a coloured (non-placeholder) cell: [row, col, bg] of the first cell whose
# background is neither the missing-data grey nor transparent/white, so colour
# assertions never land on a greyed placeholder-model row.
_FIND_COLOURED = """(sel) => {
    const rows = document.querySelectorAll(sel + ' tbody tr');
    for (let r = 0; r < rows.length; r++) {
        const cells = rows[r].querySelectorAll('td');
        for (let c = 0; c < cells.length; c++) {
            const bg = getComputedStyle(cells[c]).backgroundColor;
            if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'rgb(219, 219, 219)'
                && bg !== 'rgb(255, 255, 255)') {
                return [r, c, bg];
            }
        }
    }
    return null;
}"""

_CELL_BG = """([sel, r, c]) => {
    const row = document.querySelectorAll(sel + ' tbody tr')[r];
    if (!row) return null;
    const cell = row.querySelectorAll('td')[c];
    return cell ? getComputedStyle(cell).backgroundColor : null;
}"""


def _goto_ionpi19(page: Page) -> None:
    """Open the non-covalent-interactions category and wait for the table."""
    page.locator('#sidebar-nav a[href^="/category/"]').first.click()
    expect(page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)


def _expand_scoring(page: Page) -> None:
    """Expand the collapsed 'Adjust scoring' panel to reveal its controls."""
    page.get_by_text("Adjust scoring", exact=False).first.click()
    expect(page.locator("#IONPI19-table-normalized-toggle")).to_be_visible(
        timeout=TIMEOUT
    )


def test_normalised_values_stable_after_settle(ready_page: Page) -> None:
    """The normalise flip's values must survive the chained store round-trip.

    ``expect(...).not_to_have_text`` alone passes on the transient paint; this
    also asserts the value has not reverted to raw after callbacks settle.
    """
    _goto_ionpi19(ready_page)
    _expand_scoring(ready_page)
    mae = ready_page.locator('#IONPI19-table td[data-dash-column="MAE"]').first
    expect(mae).to_be_visible(timeout=TIMEOUT)
    raw_text = mae.inner_text()

    ready_page.locator("#IONPI19-table-normalized-toggle input").first.check()
    expect(mae).not_to_have_text(raw_text, timeout=TIMEOUT)

    ready_page.wait_for_timeout(2000)
    assert mae.inner_text() != raw_text, "normalised value reverted to raw"


def test_injected_table_uses_current_cmap(ready_page: Page) -> None:
    """A benchmark table injected after a colour-scheme change must use it."""
    _goto_ionpi19(ready_page)
    before = ready_page.evaluate(_FIND_COLOURED, "#IONPI19-table")
    assert before, "no coloured cell found in benchmark table"

    ready_page.locator('#sidebar-nav a[href="/"]').first.click()
    expect(ready_page.locator("#summary-table tbody tr").first).to_be_visible(
        timeout=TIMEOUT
    )
    ready_page.locator(".mlpeg-settings-summary").click()
    ready_page.locator("#cmap-dropdown").click()
    ready_page.locator('[role="option"]', has_text="Green-Red").first.click()
    ready_page.keyboard.press("Escape")
    ready_page.wait_for_timeout(1000)

    _goto_ionpi19(ready_page)
    ready_page.wait_for_timeout(1000)
    after = ready_page.evaluate(_CELL_BG, ["#IONPI19-table", before[0], before[1]])
    assert after != before[2], "injected benchmark table ignored the colour scheme"


def test_injected_table_uses_current_model_filter(ready_page: Page) -> None:
    """A benchmark table injected after filtering must drop the filtered model."""
    _goto_ionpi19(ready_page)
    initial_bench = ready_page.locator("#IONPI19-table tbody tr").count()
    assert initial_bench > 1

    ready_page.locator('#sidebar-nav a[href="/"]').first.click()
    rows = ready_page.locator("#summary-table tbody tr")
    expect(rows.first).to_be_visible(timeout=TIMEOUT)
    initial_home = rows.count()

    ready_page.locator("#model-filter-checklist").click()
    ready_page.locator('[role="option"][aria-selected="true"]:visible').first.click()
    expect(rows).to_have_count(initial_home - 1, timeout=30_000)
    ready_page.keyboard.press("Escape")

    _goto_ionpi19(ready_page)
    ready_page.wait_for_timeout(1000)
    final_bench = ready_page.locator("#IONPI19-table tbody tr").count()
    assert final_bench < initial_bench, "injected table ignored the model filter"


def test_reexpanded_card_keeps_session_edits(ready_page: Page) -> None:
    """A collapsed-then-expanded card must show the session's edited scores."""
    _goto_ionpi19(ready_page)
    _expand_scoring(ready_page)
    score = ready_page.locator('#IONPI19-table td[data-dash-column="Score"]').first
    expect(score).to_be_visible(timeout=TIMEOUT)
    default_score = score.inner_text()

    good = ready_page.locator("#IONPI19-table-MAE-good-threshold")
    good.fill("15")
    good.press("Enter")
    expect(score).not_to_have_text(default_score, timeout=TIMEOUT)
    edited_score = score.inner_text()

    header = ready_page.locator(".mlpeg-bench-header").first
    header.click()  # collapse -> body unmounts
    expect(ready_page.locator("#IONPI19-table")).to_have_count(0, timeout=TIMEOUT)
    header.click()  # expand -> lazy remount fires the sync on insertion
    expect(ready_page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)
    ready_page.wait_for_timeout(1000)

    final_score = ready_page.locator(
        '#IONPI19-table td[data-dash-column="Score"]'
    ).first.inner_text()
    assert final_score == edited_score, (
        f"re-injected table lost the session edit: default={default_score!r} "
        f"edited={edited_score!r} after-reexpand={final_score!r}"
    )
