"""Startup readiness waits for propagation rather than the first painted row."""

from __future__ import annotations

import pytest

from ml_peg.app.utils import shell


@pytest.fixture
def startup_page(page, monkeypatch):
    """Install the production callback with controllable renderer queues and time."""
    scripts = []
    monkeypatch.setattr(
        shell, "clientside_callback", lambda code, *a, **k: scripts.append(code)
    )
    shell.register_shell_callbacks()
    page.set_content(
        '<div id="page-content"><table><tbody><tr><td>1</td></tr></tbody></table></div>'
    )
    page.evaluate("""() => {
        window.now = 0;
        performance.now = () => now;
        window.dash_clientside = {no_update: 'NO_UPDATE'};
        window.queues = {requested: [], executing: [], stored: [{}], completed: 1};
        window.store = {getState: () => ({callbacks: queues})};
        window.finishes = 0;
        window.__mlpegLoadingProgress = {finishStartup: () => {
            finishes++;
            return Promise.resolve();
        }};
    }""")
    page.evaluate(f"window.pollStartup = {scripts[0]}")
    return page


def poll(page, now, tick=1):
    """
    Invoke the real readiness callback at a controlled time.

    Parameters
    ----------
    page
        Browser containing the callback.
    now
        Monotonic time in milliseconds.
    tick
        Startup polling interval count.

    Returns
    -------
    list
        Mask style and interval state returned by the callback.
    """
    return page.evaluate(
        "([time, tick]) => { now = time; return pollStartup(tick); }", [now, tick]
    )


@pytest.mark.parametrize(
    "queue", ["requested", "blocked", "executing", "executed", "prioritized", "watched"]
)
def test_first_row_does_not_mean_ready(startup_page, queue):
    """Each renderer work queue prevents dismissal despite a rendered table."""
    page = startup_page
    page.evaluate(
        "key => queues[key] = [{callback: {output: 'summary-table.data'}}]", queue
    )
    assert poll(page, 0) == ["NO_UPDATE"] * 2
    assert poll(page, 2000) == ["NO_UPDATE"] * 2
    page.evaluate("key => queues[key] = []", queue)
    assert poll(page, 2100) == ["NO_UPDATE"] * 2
    assert poll(page, 2599) == ["NO_UPDATE"] * 2
    assert poll(page, 2600) == [{"display": "none"}, True]
    assert page.evaluate("finishes") == 1


def test_own_poll_and_empty_page(startup_page):
    """The poll cannot deadlock itself and an empty results page can finish."""
    page = startup_page
    page.evaluate("""() => {
        document.getElementById('page-content').innerHTML = '<h1>No results</h1>';
        queues.executing = [{callback: {output:
            '..startup-mask.style...startup-mask-poll.disabled..'}}];
    }""")
    assert poll(page, 0) == ["NO_UPDATE"] * 2
    assert poll(page, 500) == [{"display": "none"}, True]


def test_loading_or_new_work_restarts_quiet_period(startup_page):
    """A second propagation wave must finish before the quiet window restarts."""
    page = startup_page
    poll(page, 0)
    page.locator("td").evaluate("el => el.setAttribute('data-dash-is-loading', 'true')")
    assert poll(page, 600) == ["NO_UPDATE"] * 2
    page.locator("td").evaluate("el => el.removeAttribute('data-dash-is-loading')")
    assert poll(page, 700) == ["NO_UPDATE"] * 2
    assert poll(page, 1199) == ["NO_UPDATE"] * 2
    assert poll(page, 1200) == [{"display": "none"}, True]


def test_page_content_own_loading_state_blocks(startup_page):
    """Dash marks the Loading target itself, and #page-content is one."""
    page = startup_page
    page.locator("#page-content").evaluate(
        "el => el.setAttribute('data-dash-is-loading', 'true')"
    )
    assert poll(page, 0) == ["NO_UPDATE"] * 2
    assert poll(page, 600) == ["NO_UPDATE"] * 2
    page.locator("#page-content").evaluate(
        "el => el.removeAttribute('data-dash-is-loading')"
    )
    assert poll(page, 700) == ["NO_UPDATE"] * 2
    assert poll(page, 1200) == [{"display": "none"}, True]


@pytest.mark.parametrize("failure", ["renderer", "page", "queue"])
def test_failed_startup_has_bounded_escape(startup_page, failure):
    """Unavailable renderer state or page content cannot trap the user forever."""
    page = startup_page
    page.evaluate(
        """failure => {
        if (failure === 'renderer') delete window.store;
        if (failure === 'page') {
            document.getElementById('page-content').replaceChildren();
        }
        if (failure === 'queue') queues.requested = [{}];
    }""",
        failure,
    )
    assert poll(page, 120000, shell.STARTUP_MASK_MAX_POLLS) == ["NO_UPDATE"] * 2
    assert poll(page, 120250, shell.STARTUP_MASK_MAX_POLLS + 1) == [
        {"display": "none"},
        True,
    ]
