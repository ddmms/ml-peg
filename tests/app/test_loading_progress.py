"""Exercise loading completion, including paused browser animation frames."""

from __future__ import annotations

from pathlib import Path

from playwright.sync_api import Page
import pytest

SCRIPT = Path(__file__).parents[2] / "ml_peg/app/data/ui/loading_progress.js"


@pytest.fixture
def progress_page(page: Page) -> Page:
    """Load the real asset with manually delivered animation frames."""
    page.clock.install()
    page.set_content(
        '<div id="startup-mask"><div class="mlpeg-progress-ring"></div></div>'
    )
    page.evaluate(
        """() => {
            window.pendingFrames = new Map();
            let nextFrame = 0;
            window.requestAnimationFrame = fn => {
                pendingFrames.set(++nextFrame, fn);
                return nextFrame;
            };
            window.cancelAnimationFrame = id => pendingFrames.delete(id);
            window.tabHidden = false;
            Object.defineProperty(document, 'hidden', {get: () => tabHidden});
            window.stepCompletion = now => {
                const [id, fn] = Array.from(pendingFrames.entries()).at(-1);
                pendingFrames.delete(id);
                fn(now);
            };
            window.beginCompletion = () => {
                window.settled = false;
                window.completion = window.__mlpegLoadingProgress.finishStartup();
                completion.then(() => { window.settled = true; });
            };
        }"""
    )
    page.add_script_tag(path=str(SCRIPT))
    return page


def assert_complete(page: Page) -> None:
    """Check both promise settlement and the final displayed percentage."""
    assert page.evaluate("window.settled")
    assert (
        page.locator(".mlpeg-progress-ring").evaluate(
            "ring => ring.style.getPropertyValue('--mlpeg-pct')"
        )
        == "100"
    )


@pytest.mark.parametrize("advance_frame", [False, True])
def test_completion_survives_paused_frames(progress_page: Page, advance_frame: bool):
    """A fallback settles shared completion even if no more frames arrive."""
    page = progress_page
    page.evaluate("beginCompletion()")
    if advance_frame:
        page.evaluate("stepCompletion(0)")
    page.evaluate(
        """() => {
            window.lateFrame = Array.from(pendingFrames.values()).at(-1);
            window.tabHidden = true;
        }"""
    )
    assert page.evaluate("completion === window.__mlpegLoadingProgress.finishStartup()")
    page.clock.run_for(1000)
    assert_complete(page)
    # Even an already-queued callback must not repaint after fallback completion.
    page.evaluate("() => { window.tabHidden = false; lateFrame(0); }")
    assert_complete(page)
    assert page.evaluate("!Array.from(pendingFrames.values()).includes(lateFrame)")


@pytest.mark.parametrize("mode", ["hidden", "reduced-motion"])
def test_completion_skips_unnecessary_animation(progress_page: Page, mode: str):
    """Hidden tabs and reduced motion settle without frames or a hold timer."""
    page = progress_page
    if mode == "hidden":
        page.evaluate("window.tabHidden = true")
    else:
        page.emulate_media(reduced_motion="reduce")
    frames_before = page.evaluate("pendingFrames.size")
    page.evaluate("beginCompletion()")
    assert_complete(page)
    assert page.evaluate("pendingFrames.size") == frames_before


def test_visible_completion_holds_then_stops(progress_page: Page):
    """Visible completion retains its animation and hold, then stops painting."""
    page = progress_page
    page.evaluate("beginCompletion()")
    page.evaluate("stepCompletion(0)")
    page.evaluate("stepCompletion(180)")
    assert not page.evaluate("window.settled")
    page.clock.run_for(100)
    assert_complete(page)
    page.evaluate(
        """() => {
            window.paints = 0;
            const style = document.querySelector('.mlpeg-progress-ring').style;
            const original = style.setProperty.bind(style);
            style.setProperty = (...args) => { paints++; original(...args); };
        }"""
    )
    page.clock.run_for(1000)
    assert page.evaluate("paints") == 0


def test_completion_without_ring(progress_page: Page):
    """An absent ring must not prevent the shell from dismissing its mask."""
    page = progress_page
    page.locator("#startup-mask").evaluate("mask => mask.remove()")
    page.evaluate("beginCompletion()")
    assert page.evaluate("window.settled")


@pytest.mark.parametrize("mode", ["normal", "paused", "reduced-motion"])
def test_table_completion(progress_page: Page, mode: str):
    """Table removal completes visibly or dismisses without waiting for frames."""
    page = progress_page
    page.evaluate(
        """() => {
            const host = document.createElement('div');
            host.className = 'mlpeg-table-loading';
            host.innerHTML = '<div class="slot"><div class="mlpeg-table-spinner">'
                + '<div class="mlpeg-progress-ring"></div></div></div>';
            document.body.appendChild(host);
            window.stepProgress = now => {
                const [id, fn] = pendingFrames.entries().next().value;
                pendingFrames.delete(id);
                fn(now);
            };
            stepProgress(0);
            for (let now = 100; now <= 500; now += 100) stepProgress(now);
        }"""
    )

    def pct(selector):
        return page.locator(selector).evaluate(
            "ring => Number(ring.style.getPropertyValue('--mlpeg-pct'))"
        )

    assert pct(".mlpeg-table-spinner .mlpeg-progress-ring") > 60
    assert pct("#startup-mask .mlpeg-progress-ring") < 30
    if mode == "reduced-motion":
        page.emulate_media(reduced_motion="reduce")
    page.locator(".mlpeg-table-spinner").evaluate("node => node.remove()")
    if mode == "normal":
        page.evaluate("stepCompletion(500)")
        page.evaluate("stepCompletion(680)")
        assert pct(".mlpeg-table-completion .mlpeg-progress-ring") == 100
        page.clock.run_for(100)
    elif mode == "paused":
        page.clock.run_for(1250)
    assert page.locator(".mlpeg-table-completion").count() == 0


def test_real_dash_table_completion(page: Page):
    """Exercise real dcc.Loading unmounts, reloads, and final-frame cleanup."""
    import threading

    from dash import Dash, dash_table, html
    from playwright.sync_api import expect
    from werkzeug.serving import make_server

    from ml_peg.app.utils.build_components import build_filter_overlay

    app = Dash(__name__, include_assets_files=False)
    app.layout = html.Div(
        build_filter_overlay(
            "test-table",
            dash_table.DataTable(id="test-table", data=[{"value": 1}]),
            delay_hide=0,
        )
    )
    server = make_server("127.0.0.1", 0, app.server, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        page.goto(f"http://127.0.0.1:{server.server_port}")
        expect(page.locator(".dash-table-container")).to_be_visible()
        page.add_script_tag(path=str(SCRIPT))
        page.evaluate(
            """() => {
                window.displaySpinner = display => dash_clientside.set_props(
                    {type: 'filter-overlay', index: 'test-table'}, {display});
                window.sawComplete = false;
                new MutationObserver(() => {
                    const ring = document.querySelector(
                        '.mlpeg-table-completion .mlpeg-progress-ring');
                    if (ring && ring.style.getPropertyValue('--mlpeg-pct') === '100') {
                        window.sawComplete = true;
                    }
                }).observe(document.body, {
                    attributes: true, childList: true, subtree: true
                });
            }"""
        )
        page.evaluate("displaySpinner('show')")
        expect(page.locator(".mlpeg-table-spinner")).to_be_visible()
        page.evaluate("displaySpinner('auto')")
        page.wait_for_function("window.sawComplete")
        expect(page.locator(".mlpeg-table-completion")).to_have_count(0)
        # A new update supersedes a finishing overlay, with no stale copy left.
        page.evaluate("displaySpinner('show')")
        expect(page.locator(".mlpeg-table-spinner")).to_be_visible()
        page.evaluate("displaySpinner('auto')")
        expect(page.locator(".mlpeg-table-completion")).to_have_count(1)
        page.evaluate("displaySpinner('show')")
        expect(page.locator(".mlpeg-table-spinner")).to_be_visible()
        expect(page.locator(".mlpeg-table-completion")).to_have_count(0)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@pytest.mark.parametrize("resume_frames", [False, True])
def test_visible_fallback_paints_before_settling(
    progress_page: Page, resume_frames: bool
):
    """A visible fallback holds 100%, but cannot hang on paused animation frames."""
    page = progress_page
    page.evaluate("beginCompletion()")
    page.evaluate("window.lateFrame = Array.from(pendingFrames.values()).at(-1)")
    page.clock.run_for(1000)
    assert not page.evaluate("window.settled")
    assert (
        page.locator(".mlpeg-progress-ring").evaluate(
            "ring => ring.style.getPropertyValue('--mlpeg-pct')"
        )
        == "100"
    )
    # An animation callback queued before the timeout cannot lower the percentage.
    page.evaluate("lateFrame(0)")
    assert (
        page.locator(".mlpeg-progress-ring").evaluate(
            "ring => ring.style.getPropertyValue('--mlpeg-pct')"
        )
        == "100"
    )
    if resume_frames:
        page.evaluate("stepCompletion(1000)")
        page.clock.run_for(99)
        assert not page.evaluate("window.settled")
        page.clock.run_for(1)
    else:
        page.clock.run_for(249)
        assert not page.evaluate("window.settled")
        page.clock.run_for(1)
    assert_complete(page)
