"""Characterization smoke tests for the ML-PEG Dash app.

They lock in current interactivity — boot, navigation, table rendering, model
filtering and in-browser plot rendering — so the frontend overhaul cannot
silently regress clicking behaviour or ship blank plots.
"""

from __future__ import annotations

from playwright.sync_api import Page, expect

READY_TIMEOUT = 60_000


def _open_ready(page: Page, app_url: str) -> None:
    """Load the app with the tutorial pre-dismissed and wait for hydration."""
    # Mark onboarding complete before load so its modal never overlays the page
    # (the modal is gated on the locally-persisted ``onboarding-state-store``).
    page.add_init_script(
        "try { window.localStorage.setItem('onboarding-state-store', "
        "JSON.stringify({completed: true})); } catch (e) {}"
    )
    page.goto(app_url)
    page.wait_for_selector("#startup-mask", state="hidden", timeout=READY_TIMEOUT)


def test_app_boots_with_title(page: Page, app_url: str) -> None:
    """The app serves and sets the ML-PEG document title."""
    page.goto(app_url)
    expect(page).to_have_title("ML-PEG")


def test_summary_table_renders_rows(page: Page, app_url: str) -> None:
    """The overall summary table renders with model rows."""
    _open_ready(page, app_url)
    rows = page.locator("#summary-table tbody tr")
    expect(rows.first).to_be_visible(timeout=READY_TIMEOUT)
    assert rows.count() > 0


def test_model_filter_reduces_rows(page: Page, app_url: str) -> None:
    """Deselecting a model in the multi-select filter drops its summary row."""
    _open_ready(page, app_url)
    rows = page.locator("#summary-table tbody tr")
    expect(rows.first).to_be_visible(timeout=READY_TIMEOUT)
    initial = rows.count()
    assert initial > 1

    # "Visible models" is a listbox dropdown; open it and deselect one model.
    # Only visible options: the settings-popover radio/checklist options also
    # carry role="option" but stay hidden inside the closed <details>.
    page.locator("#model-filter-checklist").click()
    page.locator('[role="option"][aria-selected="true"]:visible').first.click()
    expect(rows).to_have_count(initial - 1, timeout=30_000)


def test_navigate_to_category_renders_table(page: Page, app_url: str) -> None:
    """Clicking a category link renders its heading and a benchmark table."""
    _open_ready(page, app_url)
    page.locator('#sidebar-nav a[href^="/category/"]').first.click()
    expect(page.locator("#page-content h1").first).to_be_visible(timeout=READY_TIMEOUT)
    expect(page.locator("#page-content .dash-table-container").first).to_be_visible(
        timeout=READY_TIMEOUT
    )


def test_click_table_cell_shows_plot(page: Page, app_url: str) -> None:
    """Clicking a metric cell reveals its scatter plot (validates figure rendering)."""
    _open_ready(page, app_url)
    page.locator('#sidebar-nav a[href="/category/non-covalent-interactions"]').click()

    table = page.locator("#IONPI19-table")
    expect(table).to_be_visible(timeout=READY_TIMEOUT)
    page.locator('#IONPI19-table td[data-dash-column="MAE"]').first.click()

    expect(
        page.locator("#IONPI19-figure-placeholder .js-plotly-plot").first
    ).to_be_visible(timeout=READY_TIMEOUT)
