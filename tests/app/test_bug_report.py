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


def _draft_report(url: str) -> dict[str, str]:
    """Check the issue form URL and return its prefilled fields."""
    parsed = urlparse(url)
    assert parsed.scheme == "https"
    assert parsed.netloc == "github.com"
    assert parsed.path == "/ddmms/ml-peg/issues/new"
    query = parse_qs(parsed.query)
    assert query["template"] == ["6_website_bug.yml"]
    assert set(query) <= {
        "template",
        "title",
        "description",
        "expected",
        "steps",
        "screenshots",
        "details",
    }
    return {key: values[0] for key, values in query.items()}


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


def _capture_screenshot(page: Page) -> None:
    """Supply a small local image through the page-capture interface."""
    page.evaluate(
        "source => { window.htmlToImage = {toPng: async () => source}; }",
        "data:image/png;base64," + base64.b64encode(_image(page)).decode(),
    )
    page.get_by_role("button", name="Capture this page").click()
    expect(page.locator("#bug-report-canvas")).to_be_visible()


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
        ready_page.get_by_role("button", name="Open GitHub issue").click()
    draft = popup.value
    draft.wait_for_load_state()
    fields = _draft_report(draft.url)
    assert "screenshots" not in fields
    assert fields["title"] == "Plot & table disagree"
    assert fields["description"] == "The plot is empty after selecting C & O."
    assert fields["expected"] == "A curve should appear."
    assert fields["steps"] == "1. Select a model\n2. Click its MAE cell"
    details = json.loads(fields["details"])
    assert "selected_models" in details
    assert "summary-table-weight-store" in details["page_settings"]


def test_empty_report_can_open_github_with_page_details(ready_page: Page) -> None:
    """Allow all report fields to be blank and keep a usable title and context."""
    _intercept_github(ready_page)
    _open_report(ready_page)
    for field in ("title", "description", "expected", "steps"):
        ready_page.locator(f"#bug-report-{field}").fill("")
    expect(ready_page.locator("#bug-report-dialog [required]")).to_have_count(0)
    context = json.loads(ready_page.locator("#bug-report-context").text_content())
    with ready_page.expect_popup() as popup:
        ready_page.get_by_role("button", name="Open GitHub issue").click()
    popup.value.wait_for_load_state()
    fields = _draft_report(popup.value.url)
    assert fields["title"] == context["page"]
    assert not {"description", "expected", "steps"} & set(fields)
    details = json.loads(fields["details"])
    assert details["url"] == ready_page.url
    assert details["page"] == context["page"]


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
    _capture_screenshot(ready_page)
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
    assert manual_download.value.suggested_filename.startswith("mlpeg-bug-screenshot-")
    assert Path(manual_download.value.path()).read_bytes().startswith(b"\x89PNG")
    ready_page.get_by_role("button", name="Clear marks").click()
    with ready_page.expect_download() as download, ready_page.expect_popup() as popup:
        ready_page.get_by_role("button", name="Open GitHub issue").click()
    assert download.value.suggested_filename != manual_download.value.suggested_filename
    assert Path(download.value.path()).read_bytes().startswith(b"\x89PNG")
    assert (
        Path(download.value.path()).read_bytes()
        != Path(manual_download.value.path()).read_bytes()
    )
    popup.value.wait_for_load_state()
    screenshots = _draft_report(popup.value.url)["screenshots"]
    assert download.value.suggested_filename in screenshots
    assert manual_download.value.suggested_filename not in screenshots
    assert screenshots.startswith("> [!IMPORTANT]")
    assert "- [ ] **Attach the screenshot:**" in screenshots
    expect(ready_page.locator("#bug-report-status")).to_contain_text(
        "into the Screenshots field"
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
        _capture_screenshot(ready_page)
    with ready_page.expect_popup() as popup:
        ready_page.get_by_role("button", name="Open GitHub issue").click()
    popup.value.wait_for_load_state()
    assert len(popup.value.url) <= 7500
    fields = _draft_report(popup.value.url)
    assert fields["description"] == "The plot is empty after selecting C & O."
    assert fields["expected"] == "A curve should appear."
    assert fields["steps"] == "1. Select a model\n2. Click its MAE cell"
    details = json.loads(fields["details"])
    assert details["note"] == "Only changed weights and thresholds are included."
    assert details["selected_models"] == ["mace-mp-0a"]
    assert details["excluded_elements"] == ["O"]
    assert details["changed_settings"]["IONPI19-table-weight-store"] == {"MAE": 0.5}
    assert "page_settings" not in details
    assert ("screenshots" in fields) == with_screenshot
    assert len(downloads) == int(with_screenshot)
    if with_screenshot:
        assert downloads[0].startswith("mlpeg-bug-screenshot-")
        assert downloads[0] in fields["screenshots"]
    expect(
        ready_page.get_by_role("button", name="Download report", exact=True)
    ).to_have_count(0)


@pytest.mark.parametrize(
    "case", ["models", "changes", "both", "large_value", "unicode"]
)
@pytest.mark.parametrize("with_screenshot", [False, True])
def test_large_page_details_are_shortened_without_blocking_issue(
    ready_page: Page, case: str, with_screenshot: bool
) -> None:
    """Bound encoded links and account for omissions without losing report text."""
    _intercept_github(ready_page)
    ready_page.evaluate(
        """scenario => {
            const models = ['models', 'both', 'unicode'].includes(scenario)
                ? Array.from({length: 2000}, (_, i) =>
                    scenario === 'unicode' ? '模型🌍-' + i : 'test-model-' + i)
                : ['mace-mp-0a'];
            const changes = {};
            if (['changes', 'both', 'large_value'].includes(scenario)) {
                for (let i = 0; i < 1500; i++) {
                    changes['benchmark-' + i + '-table-weight-store'] = {MAE: 0.5};
                }
            }
            changes['summary-table-weight-store'] = scenario === 'large_value'
                ? Object.fromEntries(Array.from({length: 2000}, (_, i) =>
                    ['metric-' + i, 0.5]))
                : {MAE: 0.25};
            const ids = Object.keys(changes);
            window.mlpegBugReport.open({
                version: 'test',
                store_ids: ids,
                defaults: Object.fromEntries(ids.map(id => [id, {MAE: 1}])),
            }, models, ['O'], Object.values(changes));
        }""",
        case,
    )
    _fill_report(ready_page)
    full_details = ready_page.locator("#bug-report-context").text_content()
    original = json.loads(full_details)
    downloads = []
    ready_page.on(
        "download", lambda download: downloads.append(download.suggested_filename)
    )
    if with_screenshot:
        _capture_screenshot(ready_page)
    with ready_page.expect_popup() as popup:
        ready_page.get_by_role("button", name="Open GitHub issue").click()
    popup.value.wait_for_load_state()
    assert len(popup.value.url) <= 7500
    fields = _draft_report(popup.value.url)
    assert fields["description"] == "The plot is empty after selecting C & O."
    assert fields["expected"] == "A curve should appear."
    assert fields["steps"] == "1. Select a model\n2. Click its MAE cell"
    details = json.loads(fields["details"])
    assert details["note"].startswith("Page details shortened")
    assert details["url"] == ready_page.url
    assert details["version"] == "test"
    assert details["page"] == original["page"]
    for key in ("selected_models", "changed_settings"):
        assert len(details.get(key, [])) + details["omitted_details"].get(
            key, 0
        ) == len(original[key])
    if case == "large_value":
        assert "summary-table-weight-store" not in details["changed_settings"]
        assert details["changed_settings"], "smaller later changes should still fit"
    else:
        assert details["changed_settings"]["summary-table-weight-store"] == {
            "MAE": 0.25
        }
    assert ("screenshots" in fields) == with_screenshot
    assert len(downloads) == int(with_screenshot)
    if with_screenshot:
        assert downloads[0].startswith("mlpeg-bug-screenshot-")
        assert downloads[0] in fields["screenshots"]
    expect(ready_page.locator("#bug-report-context")).to_have_text(full_details)


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
    ready_page.get_by_role("button", name="Open GitHub issue").click()
    expect(ready_page.locator("#bug-report-status")).to_contain_text("too long")
    expect(ready_page.locator("#bug-report-description")).to_have_value(description)
    expect(ready_page.locator("#bug-report-expected")).to_have_value(
        "A curve should appear."
    )
    assert len(ready_page.context.pages) == 1
    assert downloads == []
    ready_page.locator("#bug-report-description").fill("The plot is empty.")
    with ready_page.expect_popup() as popup:
        ready_page.get_by_role("button", name="Open GitHub issue").click()
    popup.value.wait_for_load_state()
    assert _draft_report(popup.value.url)["description"] == "The plot is empty."
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
    expect(ready_page.get_by_role("button", name="Open GitHub issue")).to_be_enabled()
    expect(ready_page.get_by_role("button", name="Capture this page")).to_be_enabled()
    assert not ready_page.evaluate(
        "document.body.classList.contains('mlpeg-page-capturing')"
    )


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


@pytest.mark.parametrize("dismiss", ["close", "backdrop"])
def test_reporter_can_be_dismissed_after_scrolling(
    ready_page: Page, dismiss: str
) -> None:
    """Keep a close control visible and restore focus after either dismissal."""
    ready_page.set_viewport_size({"width": 360, "height": 500})
    _open_report(ready_page)
    close = ready_page.get_by_role("button", name="Close bug report", exact=True)
    before = close.bounding_box()
    body = ready_page.locator("#bug-report-body")
    body.evaluate("node => { node.scrollTop = node.scrollHeight; }")
    assert body.evaluate("node => node.scrollTop") > 0
    after = close.bounding_box()
    assert abs(after["y"] - before["y"]) < 1
    assert 0 <= after["y"] < after["y"] + after["height"] <= 500
    if dismiss == "close":
        close.click()
    else:
        ready_page.mouse.click(4, 4)
    expect(ready_page.locator("#bug-report-dialog")).to_be_hidden()
    expect(ready_page.locator("#bug-report-button")).to_be_focused()


def test_dragging_from_inside_reporter_does_not_dismiss_it(ready_page: Page) -> None:
    """Only a gesture that starts and ends on the backdrop dismisses the form."""
    _open_report(ready_page)
    title = ready_page.locator("#bug-report-title")
    title.fill("Keep this report")
    box = title.bounding_box()
    ready_page.mouse.move(box["x"] + 10, box["y"] + box["height"] / 2)
    ready_page.mouse.down()
    ready_page.mouse.move(4, 4)
    ready_page.mouse.up()
    expect(ready_page.locator("#bug-report-dialog")).to_be_visible()
    expect(title).to_have_value("Keep this report")


@pytest.mark.parametrize("phase", ["capture", "decode"])
@pytest.mark.parametrize("cancel", ["remove", "close"])
def test_cancelled_screenshot_cannot_restore_image(
    ready_page: Page, phase: str, cancel: str
) -> None:
    """Removing or closing invalidates pending work and allows another capture."""
    _open_report(ready_page)
    _capture_screenshot(ready_page)
    source = "data:image/png;base64," + base64.b64encode(_image(ready_page)).decode()
    ready_page.evaluate(
        """({source, phase}) => {
            if (phase === 'capture') {
                window.htmlToImage.toPng = () => new Promise(resolve => {
                    window.finishBugReportWork = () => resolve(source);
                });
            } else {
                window.bugReportDecode = Image.prototype.decode;
                Image.prototype.decode = function () {
                    return new Promise(resolve => {
                        window.finishBugReportWork = () =>
                            window.bugReportDecode.call(this).then(resolve);
                    });
                };
            }
        }""",
        {"source": source, "phase": phase},
    )
    capture = ready_page.get_by_role("button", name="Capture this page")
    capture.click()
    ready_page.wait_for_function("typeof window.finishBugReportWork === 'function'")
    expect(capture).to_be_disabled()
    if cancel == "remove":
        ready_page.get_by_role("button", name="Remove screenshot").click()
    else:
        ready_page.get_by_role("button", name="Close bug report").click()
        _open_report(ready_page)
    ready_page.evaluate("() => window.finishBugReportWork()")
    expect(ready_page.locator("#bug-report-canvas")).to_be_hidden()
    expect(capture).to_be_enabled()
    expect(ready_page.locator("#bug-report-status")).to_be_empty()
    assert not ready_page.evaluate(
        "document.body.classList.contains('mlpeg-page-capturing')"
    )
    if phase == "decode":
        ready_page.evaluate(
            "() => { Image.prototype.decode = window.bugReportDecode; }"
        )
    _capture_screenshot(ready_page)


def test_capture_hides_speed_panel_scrollbars_and_restores_styles(
    ready_page: Page,
) -> None:
    """Suppress capture scrollbars and retain the speed panel's scroll behaviour."""
    ready_page.set_viewport_size({"width": 360, "height": 780})
    panel = ready_page.locator('div[style="overflow-x: auto;"]').filter(
        has=ready_page.get_by_role("table").filter(
            has_text="Typical GPU runtime per test"
        )
    )
    expect(panel).to_be_visible()
    before = panel.evaluate("""node => ({
        scrollbar: getComputedStyle(node).scrollbarWidth,
        overflow: getComputedStyle(node).overflowX,
        left: node.scrollLeft,
    })""")
    _open_report(ready_page)
    ready_page.evaluate(
        """source => {
            window.htmlToImage = {toPng: async () => {
                const table = Array.from(document.querySelectorAll('table'))
                    .find(node => node.textContent.includes(
                        'Typical GPU runtime per test'));
                const panel = table.parentElement;
                window.bugReportPanelCapture = {
                    scrollbar: getComputedStyle(panel).scrollbarWidth,
                    overflow: getComputedStyle(panel).overflowX,
                    left: panel.scrollLeft,
                };
                return source;
            }};
        }""",
        "data:image/png;base64," + base64.b64encode(_image(ready_page)).decode(),
    )
    ready_page.get_by_role("button", name="Capture this page").click()
    expect(ready_page.locator("#bug-report-canvas")).to_be_visible()
    captured = ready_page.evaluate("window.bugReportPanelCapture")
    assert captured == {**before, "scrollbar": "none"}
    assert (
        panel.evaluate("""node => ({
        scrollbar: getComputedStyle(node).scrollbarWidth,
        overflow: getComputedStyle(node).overflowX,
        left: node.scrollLeft,
    })""")
        == before
    )
    expect(ready_page.locator('#bug-report-dialog input[type="file"]')).to_have_count(0)


@pytest.mark.parametrize("source", ["report", "table"])
def test_failed_library_load_can_retry_in_table_export(ready_page: Page, source: str):
    """A closed reporter or failed table export must not poison the shared loader."""
    _open_report(ready_page)
    ready_page.evaluate(
        """source => {
            delete window.htmlToImage;
            window._mlpegHtmlToImagePromise = null;
            const append = document.head.appendChild.bind(document.head);
            window.libraryAttempts = 0;
            document.head.appendChild = node => {
                if (!node.src?.includes('html-to-image')) return append(node);
                window.libraryAttempts++;
                if (window.libraryAttempts === 1) {
                    window.rejectLibrary = () => node.onerror();
                } else {
                    window.htmlToImage = {toPng: async () => source};
                    queueMicrotask(() => node.onload());
                }
                return node;
            };
        }""",
        "data:image/png;base64," + base64.b64encode(_image(ready_page)).decode(),
    )
    if source == "report":
        ready_page.get_by_role("button", name="Capture this page").click()
        ready_page.get_by_role("button", name="Close bug report").click()
        expect(ready_page.locator("#bug-report-dialog")).to_be_hidden()
    else:
        ready_page.evaluate("""() => {
            window.firstExport = window.dash_clientside.table_download.captureTable({
                element_id: 'summary-table', format: 'png'
            });
        }""")
    ready_page.wait_for_function("typeof window.rejectLibrary === 'function'")
    ready_page.evaluate("() => window.rejectLibrary()")
    ready_page.wait_for_function("window._mlpegHtmlToImagePromise === null")
    result = ready_page.evaluate("""() =>
        window.dash_clientside.table_download.captureTable({
            element_id: 'summary-table', format: 'png', filename: 'retry.png'
        })""")
    assert result["filename"] == "retry.png"
    assert ready_page.evaluate("window.libraryAttempts") == 2


def test_many_customised_tables_allow_empty_text_report(ready_page: Page):
    """Settings alone cannot exhaust the issue URL budget and block a draft."""
    _intercept_github(ready_page)
    ready_page.evaluate("""() => {
        const ids = Array.from({length: 30}, (_, i) =>
            'benchmark-' + i + '-table-thresholds-store');
        const values = ids.map(() => Object.fromEntries(
            Array.from({length: 10}, (_, i) => ['metric-' + i, {good: 0, bad: 10}])
        ));
        window.mlpegBugReport.open({version: 'test', store_ids: ids,
            defaults: Object.fromEntries(ids.map(id => [id, {}]))}, [], [], values);
    }""")
    for field in ("title", "description", "expected", "steps"):
        ready_page.locator(f"#bug-report-{field}").fill("")
    with ready_page.expect_popup() as popup:
        ready_page.get_by_role("button", name="Open GitHub issue").click()
    popup.value.wait_for_load_state()
    assert len(popup.value.url) <= 7500
    details = json.loads(_draft_report(popup.value.url)["details"])
    assert details["omitted_details"]["changed_settings"] > 0
    assert (
        len(details.get("changed_settings", {}))
        + details["omitted_details"]["changed_settings"]
        == 30
    )
