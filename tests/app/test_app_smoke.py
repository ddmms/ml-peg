"""Characterization smoke tests for the ML-PEG Dash app.

They lock in current interactivity — boot, navigation, table rendering, model
filtering and in-browser plot rendering — so the frontend overhaul cannot
silently regress clicking behaviour or ship blank plots.
"""

from __future__ import annotations

from conftest import wait_for_app_ready
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
    wait_for_app_ready(page)


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
    models = page.locator('#summary-table td[data-dash-column="MLIP"]')
    removed = models.get_by_text("mace-mp-0a", exact=True)
    retained = models.get_by_text("mace-mp-0b3", exact=True)
    expect(removed).to_be_visible(timeout=READY_TIMEOUT)
    expect(retained).to_be_visible(timeout=READY_TIMEOUT)

    # "Visible models" is a listbox dropdown; open it and deselect one model.
    # Only visible options: the settings-popover radio/checklist options also
    # carry role="option" but stay hidden inside the closed <details>.
    page.locator("#model-filter-checklist").click()
    page.get_by_role("option", name="mace-mp-0a", exact=True).click()
    expect(removed).to_have_count(0, timeout=30_000)
    expect(retained).to_be_visible(timeout=READY_TIMEOUT)


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
