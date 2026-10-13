"""Browser tests for the body-level DataTable tooltip portal.

The portal mirrors Dash's native tooltip into a fixed <div> at <body> level
(ml_peg/app/data/ui/tooltip_portal.js). These cover the parts that regress
silently: hovering a cell shows that cell's text, and moving between cells never
paints the previous cell's text against the new anchor.
"""

from __future__ import annotations

from playwright.sync_api import Page

TIMEOUT = 60_000

# Two summary-table headers with distinct tooltips. Both have to come from
# categories the fixture always supplies, or the test passes only on a machine
# that happens to hold the full dataset. The tooltip text is the category's
# description from <category>/<category>.yml.
COLUMN_A = "Bulk Crystals Score"
TOOLTIP_A = "Bulk crystal"
COLUMN_B = "Physicality Score"
TOOLTIP_B = "additivity"


def _hover(page: Page, column: str) -> None:
    """Point at a header cell (one mouse jump, so no cells in between)."""
    page.locator(f'#summary-table th[data-dash-column="{column}"]').first.hover(
        timeout=TIMEOUT
    )


def _wait_for_tooltip(page: Page, contains: str) -> None:
    """Block until the portal is showing a tooltip containing ``contains``."""
    page.wait_for_function(
        """(expected) => {
          const p = document.querySelector('.mlpeg-tooltip-portal');
          if (!p || !p.classList.contains('is-visible')) return false;
          // The whole point of the portal is to sit outside the table's
          // horizontal scroller, which would otherwise clip the card.
          if (p.parentElement !== document.body) return false;
          return p.querySelector('.dash-table-tooltip')
            .textContent.includes(expected);
        }""",
        arg=contains,
        timeout=TIMEOUT,
    )


def _portal_text(page: Page) -> str | None:
    """Text of the tooltip card, or None when it is not showing."""
    return page.evaluate(
        """() => {
          const p = document.querySelector('.mlpeg-tooltip-portal');
          if (!p || !p.classList.contains('is-visible')) return null;
          return p.querySelector('.dash-table-tooltip').textContent.trim();
        }"""
    )


def test_hover_shows_that_cells_tooltip(ready_page: Page) -> None:
    """Hovering a header shows its own tooltip in the portal."""
    _hover(ready_page, COLUMN_A)
    _wait_for_tooltip(ready_page, TOOLTIP_A)


def test_moving_between_cells_never_shows_the_previous_tooltip(
    ready_page: Page,
) -> None:
    """The card never paints the old cell's text against the new anchor.

    Dash updates its one tooltip node per table from a bubble-phase handler,
    i.e. after the portal's, so a naive read on the frame after the switch
    returns the *previous* cell's HTML.
    """
    _hover(ready_page, COLUMN_A)
    _wait_for_tooltip(ready_page, TOOLTIP_A)
    # Tied to column A, so a text that the bug could never reproduce cannot
    # make the comparison below pass without testing anything.
    text_a = _portal_text(ready_page)
    assert TOOLTIP_A in text_a

    # Sample every frame, so a single-frame flash is caught. Each sample records
    # the cell under the pointer, to tell pre-switch frames apart.
    ready_page.evaluate(
        """() => {
          window.__samples = [];
          const tick = () => {
            const p = document.querySelector('.mlpeg-tooltip-portal');
            const over = document.querySelector('#summary-table th:hover');
            if (p && p.classList.contains('is-visible')) {
              window.__samples.push([
                over ? over.getAttribute('data-dash-column') : null,
                p.querySelector('.dash-table-tooltip').textContent.trim(),
              ]);
            }
            window.__sampler = window.requestAnimationFrame(tick);
          };
          tick();
        }"""
    )
    _hover(ready_page, COLUMN_B)
    # Sampling continues across this wait, so a slow Dash update costs time
    # rather than coverage. Reaching the second tooltip is itself the assertion
    # that it appeared at all.
    _wait_for_tooltip(ready_page, TOOLTIP_B)
    samples = ready_page.evaluate(
        """() => {
          window.cancelAnimationFrame(window.__sampler);
          return window.__samples;
        }"""
    )

    over_b = [text for column, text in samples if column == COLUMN_B]
    assert over_b, "no frames sampled with the pointer over the second header"
    assert text_a not in over_b, (
        "the previous cell's tooltip was painted against the new anchor"
    )
