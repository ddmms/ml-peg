"""Sidebar search: matches as you type and jumping straight to a benchmark card."""

from __future__ import annotations

import re
from urllib.parse import urlsplit

from playwright.sync_api import Page, expect

TIMEOUT = 60_000


def test_search_lists_matching_benchmark(ready_page: Page) -> None:
    """Typing part of a benchmark name lists it with its category."""
    ready_page.locator("#sidebar-search").fill("ionpi")

    result = ready_page.locator(
        '.mlpeg-search-result[href="/category/non-covalent-interactions#IONPI19"]'
    )
    expect(result).to_be_visible(timeout=TIMEOUT)
    expect(result).to_contain_text("IONPI19")
    expect(result).to_contain_text("Non-covalent interactions")


def test_search_shows_no_matches(ready_page: Page) -> None:
    """A query matching nothing says so instead of showing an empty list."""
    ready_page.locator("#sidebar-search").fill("zzzz-no-such-benchmark")

    expect(ready_page.locator(".mlpeg-search-empty")).to_have_text(
        "No matches", timeout=TIMEOUT
    )


def test_search_result_opens_collapsed_benchmark(ready_page: Page) -> None:
    """Picking a benchmark navigates to its category and expands its card."""
    # this benchmark is not the first card on its page, so it starts collapsed
    ready_page.locator("#sidebar-search").fill("oxid")
    ready_page.locator(
        '.mlpeg-search-result[href="/category/physicality#Iron%20Oxidation%20States"]'
    ).click()

    expect(ready_page).to_have_url(
        re.compile(r"/category/physicality#Iron%20Oxidation%20States$"),
        timeout=TIMEOUT,
    )
    expect(ready_page.locator("#Iron\\ Oxidation\\ States-table")).to_be_visible(
        timeout=TIMEOUT
    )
    expect(ready_page.locator("#sidebar-search")).to_have_value("")
    expect(ready_page.locator(".mlpeg-search-result")).to_have_count(0)


def test_search_reopens_current_target(ready_page: Page) -> None:
    """Selecting the current URL still reopens a manually collapsed card."""
    page = ready_page
    result = page.locator(
        '.mlpeg-search-result[href="/category/physicality#Iron%20Oxidation%20States"]'
    )
    toggle = page.get_by_role("button", name="Iron Oxidation States", exact=True)
    page.get_by_label("Search", exact=True).fill("oxid")
    result.click()
    expect(toggle).to_have_attribute("aria-expanded", "true")
    toggle.click()
    expect(toggle).to_have_attribute("aria-expanded", "false")
    page.get_by_label("Search", exact=True).fill("oxid")
    result.press("Enter")
    expect(toggle).to_have_attribute("aria-expanded", "true")
    expect(page.locator("#sidebar-search")).to_have_value("")


def test_search_closes_mobile_sidebar_on_same_category(ready_page: Page) -> None:
    """A hash-only result selection closes the mobile navigation overlay."""
    page = ready_page
    page.locator("#sidebar-search").fill("physicality")
    page.locator('.mlpeg-search-result[href="/category/physicality"]').click()
    expect(page).to_have_url(re.compile(r"/category/physicality$"))
    page.set_viewport_size({"width": 390, "height": 844})
    page.locator("#mobile-nav-toggle").click()
    expect(page.locator("body")).to_have_class(re.compile("sidebar-open"))
    page.locator("#sidebar-search").fill("oxid")
    page.locator(
        '.mlpeg-search-result[href="/category/physicality#Iron%20Oxidation%20States"]'
    ).click()
    expect(page.locator("body")).not_to_have_class(re.compile("sidebar-open"))
    expect(
        page.get_by_role("button", name="Iron Oxidation States", exact=True)
    ).to_have_attribute("aria-expanded", "true")


def test_search_matches_description(ready_page: Page) -> None:
    """A term absent from the benchmark name can match its description."""
    ready_page.locator("#sidebar-search").fill("aqueous")
    expect(
        ready_page.locator(
            '.mlpeg-search-result[href="/category/physicality#Iron%20Oxidation%20States"]'
        )
    ).to_be_visible()


def test_search_matches_benchmark_category(ready_page: Page) -> None:
    """Category searches include individual benchmarks, not only the category page."""
    ready_page.locator("#sidebar-search").fill("physicality")
    expect(
        ready_page.locator(
            '.mlpeg-search-result[href="/category/physicality#Iron%20Oxidation%20States"]'
        )
    ).to_be_visible()
    expect(
        ready_page.locator(
            '.mlpeg-search-result[href="/category/physicality#Extensivity"]'
        )
    ).to_be_visible()


def test_search_target_direct_link(ready_page: Page) -> None:
    """A bookmarked hash opens its card on initial page load too."""
    url = urlsplit(ready_page.url)
    ready_page.goto(
        f"{url.scheme}://{url.netloc}/category/physicality#Iron%20Oxidation%20States"
    )
    expect(
        ready_page.get_by_role("button", name="Iron Oxidation States", exact=True)
    ).to_have_attribute("aria-expanded", "true", timeout=TIMEOUT)
