"""Browser tests for GitHub bug-report drafts and optional annotated screenshots."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from playwright.sync_api import Page, expect
import pytest

TIMEOUT = 60_000


def _open_report(page: Page) -> None:
    """Open the reporter through its globally mounted header button."""
    page.locator("#bug-report-button").click()
    expect(page.locator("#bug-report-dialog")).to_be_visible(timeout=TIMEOUT)


def _fill_report(page: Page) -> None:
    """Enter a representative report, including characters needing URL encoding."""
    page.locator("#bug-report-title").fill("Plot & table disagree")
    page.locator("#bug-report-description").fill(
        "The plot is empty after selecting C & O."
    )
    page.locator("#bug-report-expected").fill("A curve should appear.")
    page.locator("#bug-report-steps").fill("1. Select a model\n2. Click its MAE cell")


def _intercept_github(page: Page) -> None:
    """Serve a stand-in draft page so no report data reaches GitHub during tests."""
    page.context.route(
        "https://github.com/ddmms/ml-peg/issues/new**",
        lambda route: route.fulfill(body="<html><title>GitHub draft</title></html>"),
    )


def _image(page: Page) -> bytes:
    """Create a small white PNG for annotation tests in the browser."""
    encoded = page.evaluate(
        """() => {
            const canvas = document.createElement('canvas');
            canvas.width = 240; canvas.height = 120;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'white'; ctx.fillRect(0, 0, 240, 120);
            return canvas.toDataURL('image/png').split(',')[1];
        }"""
    )
    return base64.b64decode(encoded)


def test_report_includes_current_filters_and_settings(ready_page: Page) -> None:
    """Snapshot live settings, including changed weights outside the current page."""
    ready_page.evaluate(
        """() => {
            window.dash_clientside.set_props(
                'selected-models-store', {data: ['mace-mp-0a']}
            );
            window.dash_clientside.set_props('element-filter', {data: ['O']});
            window.dash_clientside.set_props(
                'IONPI19-table-weight-store', {data: {MAE: 0.5}}
            );
        }"""
    )
    ready_page.wait_for_function(
        "JSON.parse(sessionStorage.getItem('selected-models-store')).length === 1"
    )
    _open_report(ready_page)
    context = json.loads(ready_page.locator("#bug-report-context").text_content())
    assert context["selected_models"] == ["mace-mp-0a"]
    assert context["excluded_elements"] == ["O"]
    assert context["changed_settings"]["IONPI19-table-weight-store"] == {"MAE": 0.5}
    assert context["url"] == ready_page.url
    assert context["version"]
    assert context["browser"]
    ready_page.keyboard.press("Escape")
    expect(ready_page.locator("#bug-report-dialog")).to_be_hidden()
    expect(ready_page.locator("#bug-report-button")).to_be_focused()


def test_report_dialog_preserves_scrolled_view(ready_page: Page) -> None:
    """Opening and closing the reporter keeps the area the user was viewing."""
    ready_page.locator(
        '#sidebar-nav a[href="/category/non-covalent-interactions"]'
    ).click()
    expect(ready_page.locator("#IONPI19-table")).to_be_visible(timeout=TIMEOUT)
    ready_page.evaluate("window.scrollTo(0, 600)")
    before = ready_page.evaluate("window.scrollY")
    button = ready_page.locator("#bug-report-button").bounding_box()
    ready_page.mouse.click(
        button["x"] + button["width"] / 2,
        button["y"] + button["height"] / 2,
    )
    expect(ready_page.locator("#bug-report-dialog")).to_be_visible()
    assert ready_page.evaluate("window.scrollY") == before
    context = json.loads(ready_page.locator("#bug-report-context").text_content())
    assert context["viewport"]["scroll_y"] == before
    ready_page.keyboard.press("Escape")
    expect(ready_page.locator("#bug-report-dialog")).to_be_hidden()
    assert ready_page.evaluate("window.scrollY") == before


def test_report_opens_prefilled_draft_without_submitting(ready_page: Page) -> None:
    """Open a URL-encoded issue draft with the user's report and reproduction data."""
    _intercept_github(ready_page)
    _open_report(ready_page)
    _fill_report(ready_page)
    with ready_page.expect_popup() as popup:
        ready_page.get_by_role("button", name="Open GitHub draft").click()
    draft = popup.value
    draft.wait_for_load_state()
    query = parse_qs(urlparse(draft.url).query)
    assert "[!IMPORTANT]" not in query["body"][0]
    assert query["title"] == ["Website bug: Plot & table disagree"]
    assert "selecting C & O" in query["body"][0]
    assert "A curve should appear." in query["body"][0]
    assert "2. Click its MAE cell" in query["body"][0]
    assert '"selected_models"' in query["body"][0]
    assert '"summary-table-weight-store"' in query["body"][0]


def test_report_includes_benchmark_curve_selection(ready_page: Page) -> None:
    """Record benchmark dropdown labels so a selected curve can be reproduced."""
    ready_page.locator('#sidebar-nav a[href="/category/bulk-crystals"]').click()
    curve = ready_page.locator('[id="Iron Properties-curve-dropdown"]')
    expect(curve).to_be_visible(timeout=TIMEOUT)
    curve.click()
    ready_page.locator(".dash-dropdown-content").get_by_text(
        "Bain Path", exact=True
    ).click()
    expect(curve.locator(".dash-dropdown-value")).to_have_text("Bain Path")
    _open_report(ready_page)
    context = json.loads(ready_page.locator("#bug-report-context").text_content())
    assert context["page_selections"]["Iron Properties-curve-dropdown"] == ["Bain Path"]
    assert "Iron Properties-table-weight-store" in context["page_settings"]


def test_screenshot_can_be_marked_and_downloaded(ready_page: Page) -> None:
    """Download the marked image and reference it in the GitHub draft."""
    _intercept_github(ready_page)
    _open_report(ready_page)
    _fill_report(ready_page)
    ready_page.locator("#bug-report-upload").set_input_files(
        {
            "name": "screenshot.png",
            "mimeType": "image/png",
            "buffer": _image(ready_page),
        }
    )
    canvas = ready_page.locator("#bug-report-canvas")
    expect(canvas).to_be_visible()
    canvas.scroll_into_view_if_needed()
    box = canvas.bounding_box()
    ready_page.mouse.move(box["x"] + 20, box["y"] + 20)
    ready_page.mouse.down()
    ready_page.mouse.move(box["x"] + 100, box["y"] + 60)
    ready_page.mouse.up()
    pixel = ready_page.evaluate(
        "Array.from(document.getElementById('bug-report-canvas')"
        ".getContext('2d').getImageData(40, 40, 1, 1).data)"
    )
    assert pixel[0] > pixel[1], "the annotation did not alter the screenshot"
    expect(ready_page.locator("#bug-report-attachment-note")).to_contain_text(
        "not uploaded automatically"
    )
    with ready_page.expect_download() as manual_download:
        ready_page.get_by_role("button", name="Download screenshot").click()
    assert manual_download.value.suggested_filename == "mlpeg-bug-screenshot.png"
    assert Path(manual_download.value.path()).read_bytes().startswith(b"\x89PNG")
    with ready_page.expect_download() as download, ready_page.expect_popup() as popup:
        ready_page.get_by_role("button", name="Open GitHub draft").click()
    assert download.value.suggested_filename == "mlpeg-bug-screenshot.png"
    assert Path(download.value.path()).read_bytes().startswith(b"\x89PNG")
    popup.value.wait_for_load_state()
    assert (
        "mlpeg-bug-screenshot.png"
        in parse_qs(urlparse(popup.value.url).query)["body"][0]
    )
    body = parse_qs(urlparse(popup.value.url).query)["body"][0]
    assert body.startswith("> [!IMPORTANT]")
    assert "- [ ] **Attach the screenshot:**" in body
    assert body.index("Attach the screenshot") < body.index("## What happened")
    expect(ready_page.locator("#bug-report-status")).to_contain_text(
        "into the issue description"
    )


@pytest.mark.parametrize("with_screenshot", [False, True])
def test_large_settings_use_compact_draft_without_report_file(
    ready_page: Page, with_screenshot: bool
) -> None:
    """Fit large settings into a draft while preserving text and changed values."""
    _intercept_github(ready_page)
    ready_page.evaluate(
        """weights => window.mlpegBugReport.open(
            {
                version: 'test',
                store_ids: ['summary-table-weight-store', 'IONPI19-table-weight-store'],
                defaults: {
                    'summary-table-weight-store': weights,
                    'IONPI19-table-weight-store': {MAE: 1},
                },
            },
            ['mace-mp-0a'], ['O'], [weights, {MAE: 0.5}]
        )""",
        {f"metric-{index}": 1 for index in range(600)},
    )
    expect(ready_page.locator("#bug-report-dialog")).to_be_visible()
    _fill_report(ready_page)
    downloads = []
    ready_page.on(
        "download", lambda download: downloads.append(download.suggested_filename)
    )
    if with_screenshot:
        ready_page.locator("#bug-report-upload").set_input_files(
            {
                "name": "screenshot.png",
                "mimeType": "image/png",
                "buffer": _image(ready_page),
            }
        )
        expect(ready_page.locator("#bug-report-canvas")).to_be_visible()
    with ready_page.expect_popup() as popup:
        ready_page.get_by_role("button", name="Open GitHub draft").click()
    popup.value.wait_for_load_state()
    assert len(popup.value.url) <= 7500
    body = parse_qs(urlparse(popup.value.url).query)["body"][0]
    assert "The plot is empty after selecting C & O." in body
    assert "A curve should appear." in body
    assert "2. Click its MAE cell" in body
    details = json.loads(body.split("```json\n")[1].split("\n```")[0])
    assert details["selected_models"] == ["mace-mp-0a"]
    assert details["excluded_elements"] == ["O"]
    assert details["changed_settings"]["IONPI19-table-weight-store"] == {"MAE": 0.5}
    assert "page_settings" not in details
    assert ("- [ ] **Attach the screenshot:**" in body) == with_screenshot
    assert "Attach the complete report" not in body
    assert downloads == (["mlpeg-bug-screenshot.png"] if with_screenshot else [])
    expect(
        ready_page.get_by_role("button", name="Download report", exact=True)
    ).to_have_count(0)


def test_overlong_text_is_preserved_for_editing(ready_page: Page) -> None:
    """Let the user shorten an oversized draft without losing text or opening it."""
    _intercept_github(ready_page)
    _open_report(ready_page)
    _fill_report(ready_page)
    description = "曲线错误 & 更多细节。\n" * 200
    ready_page.locator("#bug-report-description").fill(description)
    downloads = []
    ready_page.on(
        "download", lambda download: downloads.append(download.suggested_filename)
    )
    ready_page.get_by_role("button", name="Open GitHub draft").click()
    expect(ready_page.locator("#bug-report-status")).to_contain_text("too long")
    expect(ready_page.locator("#bug-report-description")).to_have_value(description)
    expect(ready_page.locator("#bug-report-expected")).to_have_value(
        "A curve should appear."
    )
    assert len(ready_page.context.pages) == 1
    assert downloads == []
    ready_page.locator("#bug-report-description").fill("The plot is empty.")
    with ready_page.expect_popup() as popup:
        ready_page.get_by_role("button", name="Open GitHub draft").click()
    popup.value.wait_for_load_state()
    assert "The plot is empty." in parse_qs(urlparse(popup.value.url).query)["body"][0]
    assert downloads == []


def test_capture_failure_keeps_text_reporting_available(ready_page: Page) -> None:
    """A screenshot failure still allows a report to be prepared."""
    _open_report(ready_page)
    ready_page.evaluate(
        "window.htmlToImage = {toPng: () => "
        "Promise.reject(new Error('capture failed'))}"
    )
    ready_page.get_by_role("button", name="Capture this page").click()
    expect(ready_page.locator("#bug-report-status")).to_contain_text(
        "Could not capture"
    )
    expect(ready_page.get_by_role("button", name="Open GitHub draft")).to_be_enabled()
    expect(ready_page.get_by_role("button", name="Capture this page")).to_be_enabled()


def test_capture_excludes_dialog_and_hidden_media(ready_page: Page) -> None:
    """Exclude the report itself and hidden media that break screenshot decoding."""
    _open_report(ready_page)
    encoded = base64.b64encode(_image(ready_page)).decode()
    ready_page.evaluate(
        """source => {
            const video = document.createElement('video');
            video.style.display = 'none';
            document.body.appendChild(video);
            const canvas = document.createElement('canvas');
            canvas.width = 0;
            document.body.appendChild(canvas);
            window.htmlToImage = {toPng: async (root, options) => {
                if (options.filter(document.getElementById('bug-report-dialog'))
                    || options.filter(video) || options.filter(canvas)) {
                    throw new Error('Hidden media or report dialog was included');
                }
                if (!options.filter(document.getElementById('summary-table'))) {
                    throw new Error('Page content was excluded');
                }
                return source;
            }};
        }""",
        f"data:image/png;base64,{encoded}",
    )
    ready_page.get_by_role("button", name="Capture this page").click()
    expect(ready_page.locator("#bug-report-canvas")).to_be_visible()
    expect(ready_page.locator("#bug-report-status")).to_contain_text("Drag on")
    ready_page.get_by_role("button", name="Remove screenshot").click()
    expect(ready_page.locator("#bug-report-canvas")).to_be_hidden()


@pytest.mark.parametrize("width", [360, 768])
def test_reporter_fits_mobile_screen(ready_page: Page, width: int) -> None:
    """Keep the global button and report dialog usable on narrow screens."""
    ready_page.set_viewport_size({"width": width, "height": 780})
    expect(ready_page.locator("#bug-report-button")).to_be_visible()
    assert ready_page.evaluate("document.documentElement.scrollWidth <= innerWidth + 2")
    _open_report(ready_page)
    box = ready_page.locator("#bug-report-dialog").bounding_box()
    assert box["x"] >= 0
    assert box["x"] + box["width"] <= width
