"""Browser tests for the header settings panel, dark mode and expand-all.

These pin the settings popover contents, the persisted dark-mode toggle (with
its no-flash reload path), the expand/collapse-all benchmark controls and the
summary-table accent card added in the frontend overhaul.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect

TIMEOUT = 60_000


def _open_settings(page: Page) -> None:
    """Open the header settings popover."""
    page.locator(".mlpeg-settings-summary").click()
    expect(page.locator(".mlpeg-settings-panel")).to_be_visible(timeout=TIMEOUT)


def _goto_category(page: Page) -> None:
    """Open the test category page and wait for the benchmark table."""
    # Target the benchmark's own category by href: several categories are
    # built now, so the first link in the nav is not this one.
    page.locator('#sidebar-nav a[href="/category/non-covalent-interactions"]').click()
    expect(page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)


def test_settings_panel_contents(ready_page: Page) -> None:
    """The settings popover holds theme, colour scheme, expand and cache controls."""
    _open_settings(ready_page)
    expect(ready_page.locator(".mlpeg-settings-panel #theme-toggle")).to_be_visible()
    expect(ready_page.locator("#cmap-dropdown")).to_be_visible()
    expect(ready_page.locator("#expand-pref-checklist")).to_be_visible()
    expect(ready_page.locator("#clear-storage-button")).to_be_visible()


def test_theme_toggle_lives_in_settings_panel(ready_page: Page) -> None:
    """The light/dark switch lives in the settings popover, not loose in the header."""
    # No longer a standalone header control (it used to sit beside the gear).
    expect(ready_page.locator(".mlpeg-header-actions > #theme-toggle")).to_have_count(0)
    _open_settings(ready_page)
    expect(ready_page.locator(".mlpeg-settings-panel #theme-toggle")).to_be_visible()


def test_dark_mode_toggle_and_persistence(ready_page: Page) -> None:
    """The settings toggle enables dark mode and it survives a reload (no flash)."""
    body_bg_before = ready_page.evaluate(
        "getComputedStyle(document.body).backgroundColor"
    )

    # The app starts light (headless default); open settings and one click of the
    # theme switch flips it to dark.
    _open_settings(ready_page)
    ready_page.locator("#theme-toggle").click()
    expect(ready_page.locator("html")).to_have_attribute(
        "data-theme", "dark", timeout=TIMEOUT
    )

    body_bg_after = ready_page.evaluate(
        "getComputedStyle(document.body).backgroundColor"
    )
    assert body_bg_after != body_bg_before, "dark mode did not restyle the body"

    # The preference is written to localStorage by the theme-store, and the
    # inline head script must re-apply it before first paint on reload.
    ready_page.wait_for_function(
        "window.localStorage.getItem('theme-store') !== null", timeout=TIMEOUT
    )
    ready_page.reload()
    assert ready_page.locator("html").get_attribute("data-theme") == "dark", (
        "dark theme not restored on reload"
    )


def test_settings_summary_has_aria_label(ready_page: Page) -> None:
    """The gear collapses to a glyph on narrow screens, so it needs an aria-label."""
    expect(ready_page.locator(".mlpeg-settings-summary")).to_have_attribute(
        "aria-label", "App settings"
    )


def test_settings_popover_closes_on_escape(ready_page: Page) -> None:
    """Escape closes the settings popover (a native <details> does not on its own)."""
    _open_settings(ready_page)
    ready_page.keyboard.press("Escape")
    expect(ready_page.locator(".mlpeg-settings-panel")).to_be_hidden(timeout=TIMEOUT)


def test_settings_popover_closes_on_outside_click(ready_page: Page) -> None:
    """Clicking outside the open popover closes it."""
    _open_settings(ready_page)
    # Click a neutral point in the content area, away from the top-right popover
    # and the left sidebar, so nothing else is activated.
    ready_page.mouse.click(640, 680)
    expect(ready_page.locator(".mlpeg-settings-panel")).to_be_hidden(timeout=TIMEOUT)


def test_expand_and_collapse_all(ready_page: Page) -> None:
    """Collapse all unmounts every card body; expand all remounts them."""
    _goto_category(ready_page)

    ready_page.locator("#collapse-all-benchmarks").click()
    expect(ready_page.locator("#IONPI19-table")).to_have_count(0, timeout=TIMEOUT)

    ready_page.locator("#expand-all-benchmarks").click()
    expect(ready_page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)


def test_expand_preference_applies_on_navigation(ready_page: Page) -> None:
    """The expand-all preference is persisted and applied when pages rebuild."""
    _open_settings(ready_page)
    ready_page.locator("#expand-pref-checklist label").click()
    ready_page.wait_for_function(
        "window.localStorage.getItem('bench-expand-store') !== null",
        timeout=TIMEOUT,
    )
    # Close the popover so it does not overlay the sidebar links.
    ready_page.locator(".mlpeg-settings-summary").click()

    _goto_category(ready_page)
    expect(ready_page.locator(".mlpeg-bench-header--open")).to_have_count(
        ready_page.locator(".mlpeg-bench-header").count(), timeout=TIMEOUT
    )


def test_summary_card_on_home_and_category(ready_page: Page) -> None:
    """The summary table sits in an accent card on both home and category pages."""
    # The home page has two (the Categories and Frameworks summaries, both now
    # wrapped in scroll cards); the category page has one. Assert the first is
    # visible on each so the locator isn't ambiguous under strict mode.
    expect(ready_page.locator(".mlpeg-summary-card").first).to_be_visible(
        timeout=TIMEOUT
    )
    _goto_category(ready_page)
    expect(ready_page.locator(".mlpeg-summary-card").first).to_be_visible(
        timeout=TIMEOUT
    )


def test_lazily_mounted_plot_follows_dark_theme(ready_page: Page) -> None:
    """A figure mounted by a callback is dark-themed without a theme toggle.

    Regression test: the plot re-theming observer used to look for the
    ``js-plotly-plot`` class on inserted nodes, but Dash inserts a ``dash-graph``
    wrapper and plotly adds that class only afterwards, so figures mounted after
    first paint kept their saved light template. The toggle tests above cannot
    catch this — flipping the theme re-themes everything and masks the miss.
    """
    # Persist dark mode and reload so the whole page loads dark from first
    # paint, without ever exercising the toggle (same idiom as the font test).
    ready_page.evaluate(
        "window.localStorage.setItem('theme-store', JSON.stringify('dark'))"
    )
    ready_page.reload()
    ready_page.wait_for_selector("#startup-mask", state="hidden", timeout=TIMEOUT)
    assert ready_page.locator("html").get_attribute("data-theme") == "dark"

    # Mount the IONPI19 parity plot: navigate to its category page and click an
    # MAE cell, which is what dispatches the figure into its placeholder. Scope
    # everything to IONPI19 — a developer checkout with real data renders other
    # benchmarks on this page too.
    ready_page.locator(
        '#sidebar-nav a[href="/category/non-covalent-interactions"]'
    ).click()
    expect(ready_page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)
    ready_page.locator('#IONPI19-table td[data-dash-column="MAE"]').first.click()

    ready_page.wait_for_function(
        """() => {
          const host = document.getElementById('IONPI19-figure');
          if (!host) return false;
          const gd = host.classList.contains('js-plotly-plot')
            ? host
            : host.querySelector('.js-plotly-plot');
          if (!gd || !gd.layout || !gd.layout.font) return false;
          const css = getComputedStyle(document.documentElement);
          return (
            gd.layout.font.color === css.getPropertyValue('--mlpeg-ink').trim() &&
            gd.layout.paper_bgcolor === css.getPropertyValue('--mlpeg-surface').trim()
          );
        }""",
        timeout=TIMEOUT,
    )


def test_clear_cache_preserves_theme(ready_page: Page) -> None:
    """Clearing cached storage keeps the dark-mode preference."""
    # Both the theme switch and the clear-cache button live in the popover now.
    _open_settings(ready_page)
    ready_page.locator("#theme-toggle").click()
    expect(ready_page.locator("html")).to_have_attribute(
        "data-theme", "dark", timeout=TIMEOUT
    )
    ready_page.wait_for_function(
        "window.localStorage.getItem('theme-store') !== null", timeout=TIMEOUT
    )

    ready_page.on("dialog", lambda dialog: dialog.accept())
    ready_page.locator("#clear-storage-button").click()
    ready_page.wait_for_selector("#startup-mask", state="hidden", timeout=TIMEOUT)
    assert ready_page.locator("html").get_attribute("data-theme") == "dark", (
        "theme preference lost after clearing the cache"
    )


def _mae_cell_colours(page: Page) -> list[str]:
    """Return the background colours of the benchmark table's MAE (heatmap) cells."""
    cells = page.locator('#IONPI19-table td[data-dash-column="MAE"]')
    expect(cells.first).to_be_visible(timeout=TIMEOUT)
    return cells.evaluate_all(
        "els => els.map((el) => getComputedStyle(el).backgroundColor)"
    )


def test_persisted_colour_scheme_applies_to_table_on_reload(ready_page: Page) -> None:
    """A colour scheme persisted in localStorage recolours tables on a hard reload.

    Guards the reload path for the benchmark/category tables: even though their
    styling callbacks use ``prevent_initial_call=True``, the tables are mounted by
    the router on load and so fire on mount and read the persisted ``cmap-store``.
    This pins that a returning user's saved colour scheme is honoured after a hard
    reload, not just after a live change.
    """
    _goto_category(ready_page)
    before = _mae_cell_colours(ready_page)
    assert before, "expected coloured MAE cells under the default colour scheme"

    # Persist a different scheme the way the cmap-store dcc.Store does (a JSON
    # string under the component id) and hard-reload so the table mounts at load.
    ready_page.evaluate(
        "window.localStorage.setItem('cmap-store', JSON.stringify('RdYlGn_r'))"
    )
    ready_page.reload()
    ready_page.wait_for_selector("#startup-mask", state="hidden", timeout=TIMEOUT)
    expect(ready_page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)

    after = _mae_cell_colours(ready_page)
    assert after != before, (
        "benchmark table kept the default colormap after reload; the persisted "
        "colour scheme was not applied to it on initial load"
    )


def test_settings_panel_has_font_option(ready_page: Page) -> None:
    """The Appearance section exposes a font choice (Inter / System)."""
    _open_settings(ready_page)
    expect(ready_page.locator("#font-dropdown")).to_be_visible()


def test_font_choice_persists_on_reload(ready_page: Page) -> None:
    """A persisted system-font choice is applied to data-font before first paint."""
    # Persist the choice the way the font-store dcc.Store does and reload; the
    # inline head script must apply it before first paint.
    ready_page.evaluate(
        "window.localStorage.setItem('font-store', JSON.stringify('system'))"
    )
    ready_page.reload()
    ready_page.wait_for_selector("#startup-mask", state="hidden", timeout=TIMEOUT)
    assert ready_page.locator("html").get_attribute("data-font") == "system", (
        "font choice not applied before first paint on reload"
    )
