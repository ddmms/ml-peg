"""Numeric acceptance journeys through real controls, stores and rendered tables.

Model A starts with B1=.5, B2=.3, B3=.9, Alpha=.4, Beta=.9,
overall=.65. All expected values below are independent of production scoring.
"""

from __future__ import annotations

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import expect
import pytest

A = "mace-mp-0a"
B = "mace-mp-0b3"


def visit(page, path):
    """Use SPA navigation, retaining the same browser session."""
    page.locator(f'#sidebar-nav a[href="{path}"]').first.click()
    expect(page.locator("#page-content h1").first).to_be_visible()


def edit(page, field, value):
    """Commit a numeric edit through its real control."""
    control = page.locator(f'[id="{field}"]')
    control.fill(str(value))
    control.press("Enter")


def set_model_visible(page, model, visible):
    """Select by keyboard and await the exact backing-store selection."""
    before = page.evaluate(
        "JSON.parse(sessionStorage.getItem('selected-models-store'))"
    )
    assert (model in before) != visible
    expected = sorted(set(before) | {model} if visible else set(before) - {model})
    page.locator("#model-filter-checklist").click()
    search = page.get_by_role("searchbox")
    # Wait for the dropdown's opening autofocus before moving focus ourselves.
    expect(search).to_be_focused()
    search.fill(model)
    expect(page.get_by_role("option")).to_have_count(1)
    option = page.get_by_role("option", name=model, exact=True)
    expect(option).to_have_attribute("aria-selected", str(not visible).lower())
    # The dropdown can move while opening. Focus the named checkbox rather
    # than clicking coordinates that can land on a neighbouring option.
    option.get_by_role("checkbox").press("Space")
    expect(option).to_have_attribute("aria-selected", str(visible).lower())
    page.wait_for_function(
        """expected => JSON.stringify(JSON.parse(sessionStorage.getItem(
            'selected-models-store')).sort()) === JSON.stringify(expected)""",
        arg=expected,
    )
    page.keyboard.press("Escape")


def score(page, table, expected, model=A):
    """Wait for the named model's rendered score, allowing display rounding."""
    try:
        _wait_rendered_score(page, table, expected, model)
    except PlaywrightTimeoutError:
        state = page.evaluate(
            """table => ({
                selected: sessionStorage.getItem('selected-models-store'),
                computed: sessionStorage.getItem(table + '-computed-store'),
                rendered: document.getElementById(table)?.innerText
            })""",
            table,
        )
        pytest.fail(f"Expected {table}/{model}={expected}; state: {state}")


def _wait_rendered_score(page, table, expected, model):
    """Wait for a rendered score, separate from failure diagnostics."""
    page.wait_for_function(
        """([table, model, expected]) => {
        const rows = document.querySelectorAll(`[id="${table}"] tbody tr`);
        const row = [...rows].find(r =>
            r.querySelector('[data-dash-column="MLIP"]')?.textContent.trim() === model);
        const text = row?.querySelector('[data-dash-column="Score"]')
            ?.textContent.trim();
        return text && Math.abs(Number(text) - expected) < 0.00051;
    }""",
        arg=[table, model, expected],
        timeout=15000,
    )


def stored_score(page, table, expected, model=A):
    """Check full precision in the Store even when the table is unmounted."""
    try:
        _wait_stored_score(page, table, expected, model)
    except PlaywrightTimeoutError:
        pytest.fail(
            f"Expected {table}/{model}={expected}; stores: "
            f"{page.evaluate('() => ({...sessionStorage})')}"
        )


def _wait_stored_score(page, table, expected, model):
    """Wait for the full precision store value."""
    page.wait_for_function(
        """([table, model, expected]) => {
        const rows = JSON.parse(
            sessionStorage.getItem(table + '-computed-store') || 'null');
        const row = rows?.find(r => r.MLIP === model);
        return row && typeof row.Score === 'number'
            && Math.abs(row.Score - expected) < 1e-10;
    }""",
        arg=[table, model, expected],
        timeout=15000,
    )


def chain(page, benchmark, category, overall):
    """Verify both group branches before visiting their pages."""
    for table, value in [
        ("B1-table", benchmark),
        ("Alpha-summary-table", category),
        ("mace-polar-1-framework-summary-table", benchmark),
        ("mace-multihead-framework-summary-table", category),
    ]:
        stored_score(page, table, value)
    visit(page, "/")
    score(page, "summary-table", overall)
    score(page, "framework-summary-table", (benchmark + category + 0.9) / 3)
    visit(page, "/category/alpha")
    score(page, "B1-table", benchmark)
    score(page, "Alpha-summary-table", category)
    visit(page, "/framework/mace-polar-1")
    score(page, "B1-table", benchmark)
    score(page, "mace-polar-1-framework-summary-table", benchmark)


def test_weight_threshold_and_navigation(ready_page, page_errors):
    """Exact values propagate in both directions between category/framework views."""
    page = ready_page
    score(page, "summary-table", 0.65)
    visit(page, "/category/alpha")
    score(page, "B1-table", 0.5)
    edit(page, "B1-table-M1-input", 3)
    chain(page, 0.65, 0.475, 0.6875)
    edit(page, "B1-table-M1-bad-threshold", 20)
    chain(page, 0.725, 0.5125, 0.70625)
    expect(page.locator('[id="B1-table-M1-input"]')).to_have_value("3")
    expect(page.locator('[id="B1-table-M1-bad-threshold"]')).to_have_value("20")
    page.locator(".mlpeg-bench-header").first.click()
    expect(page.locator("#B1-table")).to_have_count(0)
    page.locator(".mlpeg-bench-header").first.click()
    score(page, "B1-table", 0.725)
    page.reload()
    score(page, "B1-table", 0.725)


def test_multiple_edits_and_resets(ready_page, page_errors):
    """Multiple settled edits survive and independent resets restore defaults."""
    page = ready_page
    visit(page, "/category/alpha")
    edit(page, "B1-table-M1-input", 3)
    score(page, "B1-table", 0.65)
    edit(page, "B1-table-M2-input", 2)
    score(page, "B1-table", 0.56)
    edit(page, "B1-table-M1-bad-threshold", 20)
    score(page, "B1-table", 0.62)
    edit(page, "B1-table-M2-bad-threshold", 20)
    chain(page, 0.78, 0.54, 0.72)
    page.locator("#B1-table-reset-button").click()
    chain(page, 0.75, 0.525, 0.7125)
    page.locator("#B1-table-reset-thresholds-button").click()
    chain(page, 0.5, 0.4, 0.65)


def test_hidden_model_receives_edits(ready_page, page_errors):
    """Visibility affects rendered rows, not the complete scoring state."""
    page = ready_page
    score(page, "summary-table", 0.65)
    set_model_visible(page, A, False)
    expect(
        page.locator('#summary-table td[data-dash-column="MLIP"]').get_by_text(
            A, exact=True
        )
    ).to_have_count(0)
    visit(page, "/category/alpha")
    edit(page, "B1-table-M1-input", 3)
    stored_score(page, "B1-table", 0.65)
    stored_score(page, "B1-table", 0.45, B)
    visit(page, "/")
    set_model_visible(page, A, True)
    score(page, "summary-table", 0.6875)
    chain(page, 0.65, 0.475, 0.6875)


def test_independent_browser_sessions(ready_page, browser, app_url, page_errors):
    """Edits in one context must not affect a second context sharing the server."""
    page = ready_page
    visit(page, "/category/alpha")
    edit(page, "B1-table-M1-input", 3)
    chain(page, 0.65, 0.475, 0.6875)
    with browser.new_context() as context:
        other = context.new_page()
        other.add_init_script(
            "localStorage.setItem('onboarding-state-store', "
            "JSON.stringify({completed:true}))"
        )
        other.goto(app_url)
        score(other, "summary-table", 0.65)
        visit(other, "/category/alpha")
        score(other, "B1-table", 0.5)
        edit(other, "B1-table-M1-input", 0)
        score(other, "B1-table", 0.2)
        score(page, "B1-table", 0.65)


@pytest.mark.parametrize("normalized", [False, True])
def test_display_options_preserve_scores(ready_page, page_errors, normalized):
    """Normalisation affects metric display, not any aggregate value."""
    page = ready_page
    visit(page, "/category/alpha")
    edit(page, "B1-table-M1-input", 3)
    score(page, "B1-table", 0.65)
    if normalized:
        page.locator("#B1-table-normalized-toggle input").check()
    chain(page, 0.65, 0.475, 0.6875)


def test_group_weights_are_independent(ready_page, page_errors):
    """Category, framework and overall weights have separate scopes."""
    page = ready_page
    visit(page, "/category/alpha")
    edit(page, "Alpha-summary-table-B1 Score-input", 3)
    score(page, "Alpha-summary-table", 0.45)
    visit(page, "/")
    score(page, "summary-table", 0.675)
    edit(page, "summary-table-Alpha Score-input", 3)
    score(page, "summary-table", 0.5625)
    visit(page, "/framework/mace-multihead")
    score(page, "mace-multihead-framework-summary-table", 0.4)
    edit(page, "mace-multihead-framework-summary-table-B1 Score-input", 3)
    score(page, "mace-multihead-framework-summary-table", 0.45)
    visit(page, "/")
    score(page, "summary-table", 0.5625)
    score(page, "framework-summary-table", (0.5 + 0.45 + 0.9) / 3)


def test_excluded_failure_can_reenter_summary(ready_page, page_errors):
    """A model with a failed benchmark becomes scoreable at zero group weight."""
    page = ready_page
    visit(page, "/category/alpha")
    # C has a valid B1 M1 but missing M2. Exclude that metric first.
    edit(page, "B1-table-M2-input", 0)
    score(page, "B1-table", 0.8, "mace-mpa-0")
    edit(page, "Alpha-summary-table-B2 Score-input", 0)
    score(page, "Alpha-summary-table", 0.8, "mace-mpa-0")
    visit(page, "/")
    score(page, "summary-table", 0.85, "mace-mpa-0")


def test_element_filter_round_trip(ready_page, page_errors):
    """Excluding B2's element changes scoring, then clearing restores edits."""
    page = ready_page
    visit(page, "/category/alpha")
    edit(page, "B1-table-M1-input", 3)
    score(page, "B1-table", 0.65)
    visit(page, "/")
    page.locator("#element-filter-details summary").click()
    page.get_by_role("button", name="O", exact=True).click()
    page.locator("#element-filter-apply").click()
    score(page, "summary-table", 0.775)
    visit(page, "/category/alpha")
    score(page, "B1-table", 0.65)
    score(page, "Alpha-summary-table", 0.65)
    visit(page, "/")
    page.locator("#element-filter-clear").click()
    page.locator("#element-filter-apply").click()
    chain(page, 0.65, 0.475, 0.6875)
