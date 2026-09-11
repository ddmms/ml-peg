"""Browser tests for the header settings panel, dark mode, font and zoom.

These pin the settings popover contents, the persisted dark-mode toggle (with its
no-flash reload path), the font choice and the table-zoom preference added in the
theming/settings work. The expand-all and card tests live with the cards chunk.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect

TIMEOUT = 60_000


def _open_settings(page: Page) -> None:
    """Open the header settings popover."""
    page.locator(".mlpeg-settings-summary").click()
    expect(page.locator(".mlpeg-settings-panel")).to_be_visible(timeout=TIMEOUT)


def test_settings_panel_contents(ready_page: Page) -> None:
    """The popover holds theme, font, colour scheme, zoom and cache controls."""
    _open_settings(ready_page)
    expect(ready_page.locator(".mlpeg-settings-panel #theme-toggle")).to_be_visible()
    expect(ready_page.locator("#font-dropdown")).to_be_visible()
    expect(ready_page.locator("#cmap-dropdown")).to_be_visible()
    expect(ready_page.locator("#zoom-slider")).to_be_visible()
    expect(ready_page.locator("#expand-pref-checklist")).to_be_visible()
    expect(ready_page.locator("#clear-storage-button")).to_be_visible()


def test_theme_toggle_lives_in_settings_panel(ready_page: Page) -> None:
    """The light/dark switch lives in the settings popover, not loose in the header."""
    expect(ready_page.locator(".mlpeg-header-actions > #theme-toggle")).to_have_count(0)
    _open_settings(ready_page)
    expect(ready_page.locator(".mlpeg-settings-panel #theme-toggle")).to_be_visible()


def test_dark_mode_toggle_and_persistence(ready_page: Page) -> None:
    """The settings toggle enables dark mode and it survives a reload (no flash)."""
    body_bg_before = ready_page.evaluate(
        "getComputedStyle(document.body).backgroundColor"
    )

    _open_settings(ready_page)
    ready_page.locator("#theme-toggle").click()
    expect(ready_page.locator("html")).to_have_attribute(
        "data-theme", "dark", timeout=TIMEOUT
    )

    body_bg_after = ready_page.evaluate(
        "getComputedStyle(document.body).backgroundColor"
    )
    assert body_bg_after != body_bg_before, "dark mode did not restyle the body"

    # The preference is written to localStorage by the theme-store, and the inline
    # head script must re-apply it before first paint on reload.
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
    # A neutral point in the content area, away from the top-right popover and the
    # left sidebar, so nothing else is activated.
    ready_page.mouse.click(640, 680)
    expect(ready_page.locator(".mlpeg-settings-panel")).to_be_hidden(timeout=TIMEOUT)


def test_settings_panel_has_font_option(ready_page: Page) -> None:
    """The Appearance section exposes a font choice (Inter / System)."""
    _open_settings(ready_page)
    expect(ready_page.locator("#font-dropdown")).to_be_visible()


def test_font_choice_persists_on_reload(ready_page: Page) -> None:
    """A persisted system-font choice is applied to data-font before first paint."""
    # Persist the choice the way the font-store dcc.Store does (a JSON string under
    # the component id) and reload; the inline head script must apply it pre-paint.
    ready_page.evaluate(
        "window.localStorage.setItem('font-store', JSON.stringify('system'))"
    )
    ready_page.reload()
    ready_page.wait_for_selector("#startup-mask", state="hidden", timeout=TIMEOUT)
    assert ready_page.locator("html").get_attribute("data-font") == "system", (
        "font choice not applied before first paint on reload"
    )


def test_table_zoom_preference_persists(ready_page: Page) -> None:
    """A persisted table zoom is applied as the --mlpeg-table-zoom var pre-paint."""
    ready_page.evaluate(
        "window.localStorage.setItem('zoom-store', JSON.stringify(150))"
    )
    ready_page.reload()
    ready_page.wait_for_selector("#startup-mask", state="hidden", timeout=TIMEOUT)
    zoom = ready_page.evaluate(
        "getComputedStyle(document.documentElement)"
        ".getPropertyValue('--mlpeg-table-zoom').trim()"
    )
    assert zoom == "1.5", f"expected table zoom var 1.5, got {zoom!r}"


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
