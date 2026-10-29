"""Check profiler attribution and control flow without launching a browser."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from ml_peg.app.utils import browser_profiling as profiler


@pytest.mark.parametrize("duration,expected", [(-1, None), (0, 0), (12.5, 12.5)])
def test_request_duration_sentinel(tmp_path, duration, expected):
    """Unavailable timings become null while valid durations remain untouched."""
    journey = profiler.Journey(Mock(), tmp_path, 1000, 100)
    request = Mock()
    request.url = "http://localhost/_dash-layout"
    request.timing = {"responseEnd": duration}
    request.method = "GET"
    request.failure = "net::ERR_FAILED"
    request.response.return_value = None
    journey.finished(request)
    assert journey.requests[0]["duration_ms"] == expected


def test_measure_excludes_preparation_time_and_diagnostics(tmp_path, monkeypatch):
    """Previous work is neither charged to nor blamed on the next action."""
    clock = [0.0]
    monkeypatch.setattr(profiler, "perf_counter", lambda: clock[0])
    page = Mock()
    page.evaluate.side_effect = lambda script, *args: (
        clock[0] * 1000 if script == "performance.now()" else []
    )
    journey = profiler.Journey(page, tmp_path, 1000, 100)
    calls = []

    def settle():
        if not calls:
            clock[0] += 5
            journey.errors.append("old error")
            journey.requests.append({"status": 500, "failure": None})
        else:
            clock[0] += 0.1
        calls.append(True)

    journey.settle = settle
    assert journey.measure("next", lambda: None, 1)
    result = journey.results[0]
    assert result["elapsed_ms"] == 100
    assert result["browser_errors"] == []
    assert result["dash_requests"] == []


@pytest.mark.parametrize("skip", [False, True])
def test_diagnostics_only_promote_success_to_error(tmp_path, skip):
    """Retain skip reasons and diagnostics without counting skips as failures."""
    page = Mock()
    page.evaluate.side_effect = lambda script, *args: (
        0 if script == "performance.now()" else []
    )
    journey = profiler.Journey(page, tmp_path, 1000, 100)
    journey.settle = lambda: None

    def action():
        journey.errors.append("diagnostic")
        if skip:
            raise profiler.SkipActionError("no data")

    assert not journey.measure("action", action, 1)
    assert journey.results[0]["status"] == ("skipped" if skip else "error")
    assert journey.results[0]["browser_errors"] == ["diagnostic"]


def test_settle_does_not_add_a_second_quiet_window(tmp_path, monkeypatch):
    """Already quiet browser and network state can return immediately."""
    monkeypatch.setattr(profiler, "perf_counter", lambda: 1.0)
    page = Mock()
    page.evaluate.return_value = True
    page.wait_for_timeout.side_effect = AssertionError("extra quiet window")
    journey = profiler.Journey(page, tmp_path, 1000, 100)
    journey.last_network = 0
    journey.settle()
    page.wait_for_timeout.assert_not_called()


@pytest.mark.parametrize("value", ["bulk_crystal", "/category/bulk_crystal"])
def test_category_names_and_paths(value):
    """Both documented forms identify the same sidebar link."""
    assert profiler.category_path(value) == "/category/bulk_crystal"


def test_invalid_categories_are_rejected():
    """Malformed paths fail argument parsing rather than timing out in a browser."""
    with pytest.raises(argparse.ArgumentTypeError):
        profiler.category_path("/wrong/bulk_crystal")
    page = Mock()
    page.locator.return_value.first.count.return_value = 0
    with pytest.raises(ValueError, match="No sidebar link"):
        profiler.navigate(page, "/category/absent", lambda: None)
    page.locator.return_value.first.inner_text.assert_not_called()


def test_missing_benchmark_keeps_independent_steps(tmp_path, monkeypatch):
    """A dataset without the requested metric still exercises models and filters."""
    page = Mock()
    page.locator.return_value.evaluate_all.return_value = []
    page.locator.return_value.count.return_value = 2
    page.locator.return_value.first.inner_text.return_value = "model"
    context = Mock()
    context.new_page.return_value = page
    browser = Mock(version="test")
    browser.new_context.return_value = context
    playwright = Mock()
    playwright.chromium.launch.return_value = browser
    monkeypatch.setattr(profiler, "sync_playwright", lambda: nullcontext(playwright))
    mock_expect = Mock()
    monkeypatch.setattr(profiler, "expect", mock_expect)
    monkeypatch.setattr(profiler, "wait_for_table", lambda *args: None)
    monkeypatch.setattr(profiler, "edit", lambda *args: None)
    monkeypatch.setattr(profiler, "navigate", lambda *args: None)
    actions = []
    monkeypatch.setattr(profiler, "select_model", lambda *args: actions.append("model"))
    monkeypatch.setattr(
        profiler, "filter_elements", lambda *args, **kwargs: actions.append("filter")
    )
    reports = []
    monkeypatch.setattr(
        profiler, "save_summary", lambda report, output: reports.append(report)
    )

    class Journey:
        def __init__(self, *args):
            self.results = []

        def settle(self):
            pass

        def measure(self, name, action, repeat):
            status = "ok"
            try:
                action()
            except profiler.SkipActionError:
                status = "skipped"
            self.results.append({"action": name, "status": status, "elapsed_ms": 0})
            return status == "ok"

    monkeypatch.setattr(profiler, "Journey", Journey)
    args = SimpleNamespace(
        url="http://localhost",
        label="test",
        quiet_ms=100,
        timeout=1000,
        benchmark=None,
        benchmark_category=None,
        metric="MAE",
        trace=False,
        headed=False,
        repeats=1,
        category=None,
    )
    assert profiler.run_journey(args, tmp_path) == 0
    mock_expect.set_options.assert_called_once_with(timeout=args.timeout)
    assert actions == ["model", "model", "filter", "filter"]
    assert (
        next(r for r in reports[0]["results"] if r["action"] == "select benchmark")[
            "status"
        ]
        == "skipped"
    )
