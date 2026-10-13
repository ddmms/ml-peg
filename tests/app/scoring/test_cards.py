"""Card mounting and expansion preferences through production callbacks."""

from __future__ import annotations

from playwright.sync_api import expect
import pytest


def visit(page, path, summary_table):
    """Navigate and await the destination's summary table."""
    page.locator(f'#sidebar-nav a[href="{path}"]').first.click()
    expect(page.locator(f'[id="{summary_table}"]')).to_be_visible()


@pytest.mark.parametrize("expanded", [False, True])
def test_initial_expansion_preferences(ready_page, page_errors, expanded):
    """Default and expanded preferences apply on category and framework pages."""
    page = ready_page
    if expanded:
        page.locator(".mlpeg-settings-summary").click()
        page.locator("#expand-pref-checklist label").click()
        page.wait_for_function(
            "localStorage.getItem('bench-expand-store') === JSON.stringify('expanded')"
        )
        page.locator(".mlpeg-settings-summary").click()

    for path, table in [
        ("/category/alpha", "Alpha-summary-table"),
        ("/framework/mace-multihead", "mace-multihead-framework-summary-table"),
        ("/category/alpha", "Alpha-summary-table"),
    ]:
        visit(page, path, table)
        expect(page.locator(".mlpeg-bench-header")).to_have_count(2)
        expect(page.locator(".mlpeg-bench-header--open")).to_have_count(
            2 if expanded else 1
        )
        expect(page.locator("#B1-table")).to_be_visible()
        expect(page.locator("#B2-table")).to_have_count(1 if expanded else 0)


def test_collapse_preference_survives_navigation(ready_page, page_errors):
    """Collapse all survives navigation, cached-page revisits and reloads."""
    page = ready_page
    visit(page, "/category/alpha", "Alpha-summary-table")
    expect(page.locator("#B1-table")).to_be_visible()
    page.locator("#collapse-all-benchmarks").click()
    expect(page.locator(".mlpeg-bench-header--open")).to_have_count(0)
    page.wait_for_function(
        "localStorage.getItem('bench-expand-store') === JSON.stringify('collapsed')"
    )

    for path, table in [
        ("/category/beta", "Beta-summary-table"),
        ("/framework/mace-multihead", "mace-multihead-framework-summary-table"),
        ("/category/alpha", "Alpha-summary-table"),
    ]:
        visit(page, path, table)
        expect(page.locator(".mlpeg-bench-header--open")).to_have_count(0)
        expect(page.locator(".mlpeg-bench-body > *")).to_have_count(0)

    page.reload()
    expect(page.locator("#Alpha-summary-table")).to_be_visible()
    expect(page.locator(".mlpeg-bench-header--open")).to_have_count(0)
    page.get_by_role("button", name="B1", exact=True).click()
    expect(page.locator("#B1-table")).to_be_visible()


def test_individual_toggle_sends_only_expansion_state(ready_page, page_errors):
    """Closing and reopening a card never sends its component tree as State."""
    page = ready_page
    visit(page, "/category/alpha", "Alpha-summary-table")
    header = page.get_by_role("button", name="B1", exact=True)
    expect(page.locator("#B1-table")).to_be_visible()

    for before, after in [("true", "false"), ("false", "true")]:
        with page.expect_request(
            lambda request: (
                "_dash-update-component" in request.url
                and "bench-body" in request.post_data_json["output"]
            )
        ) as pending:
            header.click()
        state = pending.value.post_data_json["state"]
        assert len(state) == 1
        assert state[0]["property"] == "aria-expanded"
        assert state[0]["value"] == before
        expect(header).to_have_attribute("aria-expanded", after)
        expect(page.locator("#B1-table")).to_have_count(1 if after == "true" else 0)
