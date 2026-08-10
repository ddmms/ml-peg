"""Responsive layout tests across phone / tablet / desktop / ultrawide sizes.

Each viewport asserts the "no bugs" invariants: the page itself never scrolls
horizontally (wide tables scroll *inside* their card, not the page), the nav
adapts (off-canvas hamburger at <=768px, inline sidebar above it, with the
hamburger vertically centred in the header band), and the model-name (MLIP)
column stays pinned to the left when a table is scrolled on desktop but is
deliberately un-pinned on phones (<=768px) so the whole table can scroll.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect
import pytest

TIMEOUT = 60_000

# name -> (width, height). Covers portrait/landscape phones, tablets, laptops,
# desktops, ultrawide/superwide monitors and an unusually tall-narrow window.
VIEWPORTS: dict[str, tuple[int, int]] = {
    "phone_portrait_sm": (360, 780),
    "phone_portrait_lg": (390, 844),
    "phone_landscape": (780, 360),
    "tablet_portrait": (768, 1024),
    "tablet_landscape": (1024, 768),
    "laptop": (1280, 800),
    "desktop": (1440, 900),
    "fhd": (1920, 1080),
    "ultrawide": (2560, 1080),
    "superwide": (3440, 1440),
    "tall_narrow": (500, 1200),
}

MOBILE_MAX = 768  # the app's single breakpoint (max-width: 768px)


def _load(page: Page, app_url: str, size: tuple[int, int]) -> None:
    """Size the viewport, dismiss onboarding, then load the app and hydrate."""
    page.set_viewport_size({"width": size[0], "height": size[1]})
    page.add_init_script(
        "try { window.localStorage.setItem('onboarding-state-store', "
        "JSON.stringify({completed: true})); } catch (e) {}"
    )
    page.goto(app_url)
    page.wait_for_selector("#startup-mask", state="hidden", timeout=TIMEOUT)


def _page_overflow(page: Page) -> float:
    """Horizontal overflow of the page in px (>0 means the page itself scrolls)."""
    return page.evaluate(
        "() => document.documentElement.scrollWidth - window.innerWidth"
    )


def _open_category(page: Page) -> None:
    """Navigate to the test category page (opening the hamburger first on mobile)."""
    if page.viewport_size["width"] <= MOBILE_MAX:
        page.locator("#mobile-nav-toggle").click()
        page.wait_for_function(
            "document.body.classList.contains('sidebar-open')", timeout=TIMEOUT
        )
    page.locator('#sidebar-nav a[href^="/category/"]').first.click()
    expect(page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)


@pytest.mark.parametrize("size_name", list(VIEWPORTS), ids=list(VIEWPORTS))
def test_responsive_layout(page: Page, app_url: str, size_name: str) -> None:
    """No page-level horizontal overflow, correct nav, and a working sticky column."""
    size = VIEWPORTS[size_name]
    _load(page, app_url, size)
    width = size[0]

    # 1) The home page (with its summary table) must not overflow the viewport.
    expect(page.locator("#summary-table")).to_be_visible(timeout=TIMEOUT)
    home_ovf = _page_overflow(page)
    assert home_ovf <= 2, f"{size_name}: home page overflows by {home_ovf}px"

    # 2) Navigation adapts to the breakpoint.
    if width <= MOBILE_MAX:
        expect(page.locator("#mobile-nav-toggle")).to_be_visible()
        # The fixed hamburger must sit on the same vertical line as the header
        # content (wordmark + top-right actions), i.e. centred in the topbar band.
        align = page.evaluate(
            """() => {
                const h = document.querySelector('#mobile-nav-toggle')
                    .getBoundingClientRect();
                const bar = document.querySelector('.mlpeg-topbar')
                    .getBoundingClientRect();
                return {
                    hCenter: h.top + h.height / 2,
                    barCenter: bar.top + bar.height / 2,
                };
            }"""
        )
        assert abs(align["hCenter"] - align["barCenter"]) <= 4, (
            f"{size_name}: hamburger not vertically centred in the header "
            f"(off by {align['hCenter'] - align['barCenter']:.1f}px)"
        )
        assert "sidebar-open" not in (
            page.locator("body").get_attribute("class") or ""
        ), f"{size_name}: sidebar should start closed"
        # Off-canvas: the sidebar sits to the left of the viewport until opened.
        assert page.locator("#sidebar-nav").bounding_box()["x"] < 0, (
            f"{size_name}: sidebar should be off-canvas when closed"
        )
        page.locator("#mobile-nav-toggle").click()
        # Wait for the drawer to finish sliding on-screen (0.25s transform).
        page.wait_for_function(
            "() => document.querySelector('#sidebar-nav')"
            ".getBoundingClientRect().left >= -1",
            timeout=TIMEOUT,
        )
        expect(page.locator("#sidebar-scrim")).to_be_visible()
        # Close via the scrim (click the right edge, clear of the 280px drawer)
        # and wait for it to slide back off-canvas.
        page.mouse.click(width - 8, size[1] // 2)
        page.wait_for_function(
            "() => !document.body.classList.contains('sidebar-open')"
            " && document.querySelector('#sidebar-nav')"
            ".getBoundingClientRect().left < 0",
            timeout=TIMEOUT,
        )
    else:
        expect(page.locator("#mobile-nav-toggle")).to_be_hidden()
        expect(page.locator("#sidebar-nav")).to_be_visible()
        assert page.locator("#sidebar-nav").bounding_box()["x"] >= -1, (
            f"{size_name}: sidebar should be on-screen on desktop widths"
        )

    # 3) A benchmark table page must also not overflow the page.
    _open_category(page)
    bench_ovf = _page_overflow(page)
    assert bench_ovf <= 2, f"{size_name}: benchmark page overflows by {bench_ovf}px"

    # 4) Model-name column: when the table is wider than its card (always true at
    #    phone widths for the IONPI19 table), scroll it right. On desktop the MLIP
    #    column must stay pinned to the container's left with an opaque background;
    #    on phones (<=768px) it is deliberately un-pinned, so it scrolls off left.
    res = page.evaluate(
        """() => {
            const table = document.querySelector('#IONPI19-table');
            const scroll = table && table.closest('.mlpeg-table-scroll');
            if (!scroll) return { err: 'no scroll container' };
            if (scroll.scrollWidth <= scroll.clientWidth + 4) {
                return { overflow: false };
            }
            scroll.scrollLeft = scroll.scrollWidth;
            const mlip = scroll.querySelector('td[data-dash-column="MLIP"]');
            if (!mlip) return { err: 'no MLIP cell' };
            const s = scroll.getBoundingClientRect();
            const m = mlip.getBoundingClientRect();
            return {
                overflow: true,
                scrollLeft: scroll.scrollLeft,
                delta: m.left - s.left,
                bg: getComputedStyle(mlip).backgroundColor,
            };
        }"""
    )
    assert "err" not in res, f"{size_name}: {res.get('err')}"
    if res.get("overflow"):
        assert res["scrollLeft"] > 0, f"{size_name}: table did not scroll"
        if width <= MOBILE_MAX:
            # Un-pinned on phones: the leftmost cell scrolls off to the left.
            assert res["delta"] < -2, (
                f"{size_name}: MLIP column should scroll (not pin) on mobile "
                f"(delta={res['delta']}px)"
            )
        else:
            assert abs(res["delta"]) <= 2, (
                f"{size_name}: MLIP column not pinned to left (delta={res['delta']}px)"
            )
            assert res["bg"] not in ("rgba(0, 0, 0, 0)", "transparent"), (
                f"{size_name}: sticky MLIP background is transparent ({res['bg']})"
            )


def test_mobile_controls_open(page: Page, app_url: str) -> None:
    """Settings popover and the Visible-models dropdown open on a 360px phone."""
    _load(page, app_url, (360, 780))
    page.locator(".mlpeg-settings-summary").click()
    expect(page.locator(".mlpeg-settings-panel")).to_be_visible(timeout=TIMEOUT)
    page.keyboard.press("Escape")
    page.locator("#model-filter-checklist").click()
    expect(page.locator(".dash-dropdown-content")).to_be_visible(timeout=TIMEOUT)


def test_table_zoom_persists(page: Page, app_url: str) -> None:
    """A persisted table-zoom % is applied as --mlpeg-table-zoom before paint."""
    page.add_init_script(
        "try{window.localStorage.setItem('zoom-store', JSON.stringify(70));}catch(e){}"
    )
    _load(page, app_url, (1440, 900))
    var = page.evaluate(
        "() => getComputedStyle(document.documentElement)"
        ".getPropertyValue('--mlpeg-table-zoom').trim()"
    )
    assert var, "no --mlpeg-table-zoom set on load"
    assert abs(float(var) - 0.7) < 0.01, f"persisted table zoom not applied: {var!r}"


def test_table_zoom_scales_tables_not_chrome(page: Page, app_url: str) -> None:
    """The table-zoom slider scales tables, not the sidebar/chrome."""
    _load(page, app_url, (1440, 900))
    expect(page.locator("#summary-table")).to_be_visible(timeout=TIMEOUT)

    def sizes() -> dict:
        return page.evaluate(
            """() => {
                const cell = document.querySelector(
                    '#summary-table td[data-dash-column="MLIP"]');
                const side = document.querySelector('#sidebar-nav');
                return { cell: cell.getBoundingClientRect().width,
                         side: side.getBoundingClientRect().width };
            }"""
        )

    before = sizes()
    page.locator(".mlpeg-settings-summary").click()
    expect(page.locator(".mlpeg-settings-panel")).to_be_visible(timeout=TIMEOUT)
    thumb = page.locator("#zoom-slider [role='slider']")  # Dash 4 slider thumb
    expect(thumb).to_be_visible(timeout=TIMEOUT)
    thumb.click()  # focus the thumb
    for _ in range(3):
        thumb.press("ArrowLeft")  # 100% -> 70% (step 10)
    page.wait_for_function(
        "() => { const z = getComputedStyle(document.documentElement)"
        ".getPropertyValue('--mlpeg-table-zoom').trim();"
        " return !!z && parseFloat(z) < 1; }",
        timeout=TIMEOUT,
    )
    page.wait_for_timeout(200)
    after = sizes()
    # The table cell shrank (~0.7x); the sidebar is unchanged (chrome not zoomed).
    assert after["cell"] < before["cell"] * 0.9, (
        f"table did not shrink: {before['cell']} -> {after['cell']}"
    )
    assert abs(after["side"] - before["side"]) <= 1, (
        f"sidebar should not scale: {before['side']} -> {after['side']}"
    )


def test_sidebar_pins_below_topbar(page: Page, app_url: str) -> None:
    """On a scrolled desktop page the sticky sidebar pins below the topbar.

    Regression guard: with ``top: 0`` the sidebar slid under the opaque sticky
    topbar (z-index 1600) once the page scrolled, permanently hiding the first
    ~--header-h of navigation. The sidebar must offset by the topbar height.
    """
    _load(page, app_url, VIEWPORTS["desktop"])
    _open_category(page)  # category pages are tall enough to scroll

    page.evaluate("window.scrollTo(0, 800)")
    page.wait_for_timeout(300)
    gap = page.evaluate(
        """() => {
            const topbar = document.querySelector('.mlpeg-topbar');
            const sidebar = document.querySelector('.mlpeg-sidebar');
            return sidebar.getBoundingClientRect().top
                - topbar.getBoundingClientRect().bottom;
        }"""
    )
    assert gap >= -1, f"sidebar tucked {-gap:.0f}px under the sticky topbar"
