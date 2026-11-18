"""Regression checks for browser timing readiness and readable reports."""

from __future__ import annotations

from ml_peg.app.utils import browser_profiling as profile_app


def test_measure_waits_for_delayed_clientside_update(page, tmp_path):
    """A quiet DOM must not hide an outstanding clientside callback."""
    page.set_content('<div id="result">Before</div>')
    page.evaluate(f"({profile_app.OBSERVE})()")
    page.evaluate("""() => {
        window.callbacks = {requested: [], executing: [], completed: 0, stored: []};
        window.store = {getState: () => ({callbacks: window.callbacks})};
    }""")
    journey = profile_app.Journey(page, tmp_path, timeout=3000, quiet_ms=50)

    def delayed_update():
        page.evaluate("""() => {
            window.callbacks.executing = ['rescore'];
            setTimeout(() => {
                document.getElementById('result').textContent = 'Updated';
                window.callbacks.executing = [];
            }, 600);
        }""")

    assert journey.measure("delayed update", delayed_update, repeat=1)
    assert page.locator("#result").inner_text() == "Updated"
    assert journey.results[0]["elapsed_ms"] >= 600


def test_table_waits_for_data_cells(page):
    """An already-visible table shell is insufficient for readiness."""
    page.set_content('<div id="table"><table><tbody></tbody></table></div>')
    page.evaluate("""() => setTimeout(() => {
        document.querySelector('tbody').innerHTML =
            '<tr><td data-dash-column="MAE">1.2</td></tr>';
    }, 300)""")
    profile_app.wait_for_table(page, "#table")
    assert page.locator("td").inner_text() == "1.2"


def test_summary_separates_failures_from_timings(tmp_path):
    """Failed actions remain visible but cannot inflate successful medians."""
    report = {
        "label": "Test data",
        "url": "http://localhost",
        "quiet_ms": 300,
        "trace": False,
        "results": [
            {"action": "weight", "status": "ok", "elapsed_ms": 1000, "repeat": 1},
            {
                "action": "weight",
                "status": "error",
                "elapsed_ms": 9000,
                "repeat": 2,
                "error": "Update timed out",
            },
        ],
    }
    profile_app.save_summary(report, tmp_path)
    text = (tmp_path / "summary.txt").read_text()
    assert "weight | 1.00 s | 1.00–1.00 s | 1/2" in text
    assert "Update timed out" in text
    assert (tmp_path / "summary.png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_empty_category_is_skipped_after_updates_finish(page, tmp_path):
    """An empty, fully loaded category must not wait for nonexistent data cells."""
    page.route(
        "http://profile.test/**",
        lambda route: route.fulfill(
            body="""
        <nav id="sidebar-nav"><a href="/category/empty">Empty</a></nav>
        <div id="page-content"><h1>Summary</h1>
        <div class="dash-table-container"><table></table></div></div>
        <script>
        window.callbacks = {requested: [], completed: 0, stored: []};
        window.store = {getState: () => ({callbacks: window.callbacks})};
        document.querySelector('a').onclick = event => {
            event.preventDefault();
            history.pushState({}, '', '/category/empty');
            document.querySelector('h1').textContent = 'Empty';
        };
        </script>
    """,
            content_type="text/html",
        ),
    )
    page.goto("http://profile.test/")
    page.evaluate(f"({profile_app.OBSERVE})()")
    journey = profile_app.Journey(page, tmp_path, timeout=3000, quiet_ms=50)
    assert not journey.measure(
        "empty category",
        lambda: profile_app.navigate(page, "/category/empty", journey.settle),
        repeat=1,
    )
    assert journey.results[0]["status"] == "skipped"
    assert "No table rows" in journey.results[0]["reason"]
    assert journey.results[0]["elapsed_ms"] < 3000


def test_visible_loading_ring_blocks_next_action(page, tmp_path):
    """Idle callback queues do not imply that a table overlay has disappeared."""
    page.set_content('<div class="mlpeg-progress-ring">60%</div>')
    page.evaluate(f"({profile_app.OBSERVE})()")
    page.evaluate("""() => {
        window.store = {getState: () => ({callbacks: {requested: []}})};
        setTimeout(() => {
            document.querySelector('.mlpeg-progress-ring').style.display = 'none';
        }, 600);
    }""")
    journey = profile_app.Journey(page, tmp_path, timeout=3000, quiet_ms=50)

    def action():
        assert not page.locator(".mlpeg-progress-ring").is_visible()

    assert journey.measure("wait for table overlay", action, repeat=1)
    assert not page.locator(".mlpeg-progress-ring").is_visible()
