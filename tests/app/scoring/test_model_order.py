"""Model order stays aligned with values through real browser interactions."""

from __future__ import annotations

import math

from conftest import set_model_visible as select_model
from playwright.sync_api import expect

from ml_peg.app.utils.model_order import order_model_rows

from .test_propagation import A, B, edit, score, visit


def set_model_visible(page, model, visible):
    """Toggle one of the three fixture models through the real dropdown."""
    if not page.locator("#model-filter-checklist").is_visible():
        page.get_by_text("Visible models", exact=True).click()
    select_model(page, model, visible)


def rows(page, table):
    """Read complete rendered rows keyed by their model name."""
    return page.locator(f'[id="{table}"] tbody tr').evaluate_all(
        """rows => rows.filter(r => r.querySelector('td[data-dash-column="MLIP"]'))
        .map(r => Object.fromEntries([...r.querySelectorAll('td[data-dash-column]')]
        .map(c => [c.dataset.dashColumn, c.textContent.trim()])))"""
    )


def names(page, table):
    """Return the visible row order."""
    return [row["MLIP"] for row in rows(page, table)]


def wait_order(page, table, expected):
    """Wait for server ordering and browser rendering to settle."""
    expect(page.locator(f'[id="{table}"] td[data-dash-column="MLIP"]')).to_have_text(
        expected, timeout=15000
    )


def random_names(page, baseline):
    """Calculate the expected order from the actual session seed."""
    state = page.evaluate("JSON.parse(sessionStorage.getItem('model-order-store'))")
    return [row["MLIP"] for row in order_model_rows(baseline, state)]


def test_order_filter_sort_and_navigation(ready_page, page_errors):
    """Keep every cell with its model across shuffle, filtering and native sorts."""
    page = ready_page
    table = "summary-table"
    baseline = rows(page, table)
    wait_order(page, table, sorted(row["MLIP"] for row in baseline))
    expected_cells = {row["MLIP"]: row for row in baseline}
    page.locator('#model-order-mode input[value="random"]').check()
    page.wait_for_function(
        "JSON.parse(sessionStorage.getItem('model-order-store'))?.mode === 'random'"
    )
    for _ in range(3):
        seed = page.evaluate("sessionStorage.getItem('model-order-store')")
        page.locator("#model-order-reshuffle").click()
        page.wait_for_function(
            "old => sessionStorage.getItem('model-order-store') !== old", arg=seed
        )
        wait_order(page, table, random_names(page, baseline))
        assert {row["MLIP"]: row for row in rows(page, table)} == expected_cells
    order = names(page, table)
    set_model_visible(page, A, False)
    wait_order(page, table, [name for name in order if name != A])
    set_model_visible(page, A, True)
    wait_order(page, table, order)
    # Column sorting overrides the random base order and survives model filtering.
    sorter = page.locator(f'#{table} th[data-dash-column="Score"] .column-header--sort')
    sorter.click()
    valid = [r for r in baseline if not math.isnan(float(r["Score"]))]
    missing = [r for r in baseline if math.isnan(float(r["Score"]))]
    ascending = sorted(valid, key=lambda row: float(row["Score"])) + missing
    wait_order(page, table, [row["MLIP"] for row in ascending])
    set_model_visible(page, B, False)
    wait_order(page, table, [row["MLIP"] for row in ascending if row["MLIP"] != B])
    set_model_visible(page, B, True)
    sorter.click()
    wait_order(
        page,
        table,
        [
            row["MLIP"]
            for row in sorted(valid, key=lambda r: -float(r["Score"])) + missing
        ],
    )
    page.locator('#model-order-mode input[value="alphabetical"]').check()
    wait_order(page, table, sorted(expected_cells))
    assert {row["MLIP"]: row for row in rows(page, table)} == expected_cells
    expect(page.locator("#model-order-reshuffle")).to_have_text("Reset A–Z")
    # Exercise metric and category tables, score recomputation, and page re-mount.
    page.locator('#model-order-mode input[value="random"]').check()
    page.wait_for_function(
        "JSON.parse(sessionStorage.getItem('model-order-store'))?.mode === 'random'"
    )
    order = random_names(page, baseline)
    visit(page, "/category/alpha")
    wait_order(page, "B1-table", order)
    wait_order(page, "Alpha-summary-table", [name for name in order if name in (A, B)])
    raw = {row["MLIP"]: row for row in rows(page, "B1-table")}
    edit(page, "B1-table-M1-input", 3)
    score(page, "B1-table", 0.65)
    wait_order(page, "B1-table", order)
    for row in rows(page, "B1-table"):
        assert row["M1"] == raw[row["MLIP"]]["M1"]
        assert row["M2"] == raw[row["MLIP"]]["M2"]
    visit(page, "/")
    wait_order(page, table, order)
    page.reload()
    wait_order(page, table, order)


def test_metric_sort_reset_empty_selection_and_tooltips(ready_page, page_errors):
    """Column arrows, missing values, tooltips and empty selections stay consistent."""
    page = ready_page
    visit(page, "/category/alpha")
    table = "B1-table"
    wait_order(page, table, [A, B, "mace-mpa-0"])
    baseline = rows(page, table)
    expected = {row["MLIP"]: row for row in baseline}
    metric_sort = page.locator(
        f'#{table} th[data-dash-column="M2"] .column-header--sort'
    )
    metric_sort.click()
    wait_order(page, table, [B, A, "mace-mpa-0"])
    assert {row["MLIP"]: row for row in rows(page, table)} == expected
    # Reset A–Z works even when the radio selection was already alphabetical.
    page.locator("#model-order-reshuffle").click()
    wait_order(page, table, [A, B, "mace-mpa-0"])
    page.locator('#model-order-mode input[value="random"]').check()
    page.wait_for_function(
        "JSON.parse(sessionStorage.getItem('model-order-store'))?.mode === 'random'"
    )
    wait_order(page, table, random_names(page, baseline))
    metric_sort.click()
    wait_order(page, table, [B, A, "mace-mpa-0"])
    metric_sort.click()
    wait_order(page, table, [A, B, "mace-mpa-0"])
    set_model_visible(page, A, False)
    wait_order(page, table, [B, "mace-mpa-0"])
    set_model_visible(page, A, True)
    wait_order(page, table, [A, B, "mace-mpa-0"])
    seed = page.evaluate("sessionStorage.getItem('model-order-store')")
    page.locator("#model-order-reshuffle").click()
    page.wait_for_function(
        "old => sessionStorage.getItem('model-order-store') !== old", arg=seed
    )
    order = random_names(page, baseline)
    wait_order(page, table, order)
    assert {row["MLIP"]: row for row in rows(page, table)} == expected
    # Check tooltip identity after both native and global sorting.
    for name in order:
        cell = page.locator(f'#{table} td[data-dash-column="MLIP"]').filter(
            has_text=name
        )
        cell.hover()
        expect(page.locator(".mlpeg-tooltip-portal.is-visible")).to_contain_text(name)
    page.mouse.move(0, 0)
    for name in order:
        set_model_visible(page, name, False)
    wait_order(page, table, [])
    for name in reversed(order):
        set_model_visible(page, name, True)
    wait_order(page, table, order)
    assert {row["MLIP"]: row for row in rows(page, table)} == expected
    # Changing display units must not swap raw/normalized values between models.
    page.locator("#B1-table-normalized-toggle input").check()
    page.wait_for_function(
        """() => [...document.querySelectorAll('#B1-table tbody tr')].some(r =>
        r.querySelector('td[data-dash-column="MLIP"]')?.textContent.trim()
            === 'mace-mp-0a'
        && Number(r.querySelector('td[data-dash-column="M1"]')?.textContent) === 0.8)"""
    )
    wait_order(page, table, order)
    normalized = {row["MLIP"]: row for row in rows(page, table)}
    assert float(normalized[B]["M1"]) == 0.4
    page.locator("#B1-table-normalized-toggle input").uncheck()
    expect(page.locator(f'#{table} td[data-dash-column="M1"]')).to_have_text(
        [expected[name]["M1"] for name in order]
    )


def test_order_controls_mobile_and_desktop(ready_page, page_errors, tmp_path):
    """Controls remain visible, usable and within the viewport at both sizes."""
    page = ready_page
    for width in [1280, 390]:
        page.set_viewport_size({"width": width, "height": 900})
        controls = page.locator(".mlpeg-model-order")
        controls.scroll_into_view_if_needed()
        expect(controls).to_be_visible()
        bounds = controls.bounding_box()
        assert bounds["x"] >= 0
        assert bounds["x"] + bounds["width"] <= width
        page.locator('#model-order-mode input[value="random"]').check()
        expect(page.locator("#model-order-reshuffle")).to_have_text("Reshuffle")
        page.screenshot(path=str(tmp_path / f"model-order-{width}.png"))
        page.locator('#model-order-mode input[value="alphabetical"]').check()
        expect(page.locator("#model-order-reshuffle")).to_have_text("Reset A–Z")
