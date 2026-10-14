"""Sidebar search: matches as you type and jumping straight to a benchmark card."""

from __future__ import annotations

import re

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
