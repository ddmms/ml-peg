"""Time common ML-PEG interactions using Playwright Chromium.

- Runs against an already-started app and its available data.
- Loads summary, category and framework pages, edits weights and a threshold,
  opens a plot, toggles a model, and applies and clears an element filter.
- Waits for populated tables and editable controls before interacting.
- Waits for each edit callback and the Dash queues to finish before moving on.
- Waits for visible loading rings to disappear and a 300 ms stable quiet period.
- Records browser long tasks, Dash requests and errors alongside timings.
- Starts each repetition with empty browser storage and cache, so no hard
  reset is needed. The server stays running and can retain server-side caches.
- Saves a PNG chart, text summary, terminal log and JSON under profiles/browser/.
- Skips empty categories after loading and reports errors separately.
- Prints each step as it starts, with no pass/fail speed limits.

Start the app, then run the browser tour in another terminal:

    ml_peg app --no-debug
    python dev/profile_app.py --repeats 3 --headed

Use --category to limit pages, --benchmark and --benchmark-category to select
an interaction target, and --metric to select its metric (default: MAE).
Add --trace for a replayable trace (adds overhead). See --help for all options.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
from functools import partial
import json
from pathlib import Path
import platform
import re
from statistics import median
import sys
from time import perf_counter
from urllib.parse import urlsplit

from playwright.sync_api import Error, expect, sync_playwright

# Observe rendering work without patching fetch, Dash or app callbacks.
OBSERVE = """() => {
    window.__mlpegPerf = {changed: performance.now(), longTasks: []};
    new MutationObserver(() => {
        window.__mlpegPerf.changed = performance.now();
    }).observe(document, {subtree: true, childList: true, characterData: true,
                          attributes: true, attributeFilter: ['data-dash-is-loading']});
    new PerformanceObserver(list => {
        for (const entry of list.getEntries()) {
            window.__mlpegPerf.longTasks.push({start: entry.startTime,
                                               duration: entry.duration});
        }
    }).observe({type: 'longtask', buffered: true});
}"""


class SkipActionError(Exception):
    """An action has no applicable data in the current app."""


class Journey:
    """Measure actions through a short period of network and DOM inactivity."""

    def __init__(self, page, output, timeout, quiet_ms):
        self.page = page
        self.output = output
        self.timeout = timeout
        self.quiet_ms = quiet_ms
        self.pending = set()
        self.last_network = perf_counter()
        self.requests = []
        self.errors = []
        self.results = []
        page.on("request", self.started)
        page.on("requestfinished", self.finished)
        page.on("requestfailed", self.finished)
        page.on("pageerror", lambda error: self.errors.append(str(error)))

    def started(self, request):
        """Track active page and callback requests."""
        if request.resource_type in {"fetch", "xhr", "document"}:
            self.pending.add(request)
            self.last_network = perf_counter()

    def finished(self, request):
        """Collect completed Dash requests without storing payloads."""
        if request in self.pending:
            self.pending.remove(request)
            self.last_network = perf_counter()
        if urlsplit(request.url).path.endswith(
            ("/_dash-layout", "/_dash-dependencies", "/_dash-update-component")
        ):
            response = request.response()
            timing = request.timing
            self.requests.append(
                {
                    "path": urlsplit(request.url).path,
                    "status": response.status if response else None,
                    "duration_ms": timing["responseEnd"]
                    if timing["responseEnd"] >= 0
                    else None,
                    "failure": request.failure,
                    "callback": (request.post_data_json or {}).get("output")
                    if request.method == "POST"
                    else None,
                    "response_bytes": response.headers.get("content-length")
                    if response
                    else None,
                }
            )

    def settle(self):
        """Wait for network, DOM and Dash loading indicators to settle."""
        deadline = perf_counter() + self.timeout / 1000
        while perf_counter() < deadline:
            ready = self.page.evaluate(
                """quiet => {
                const p = window.__mlpegPerf;
                // Dash 4's renderer queues include clientside callback work.
                const callbacks = window.store?.getState()?.callbacks;
                const idle = callbacks && Object.entries(callbacks).every(
                    ([key, value]) => ['stored', 'completed'].includes(key) ||
                        (Array.isArray(value) && value.length === 0));
                const visibleRing = [...document.querySelectorAll(
                    '.mlpeg-progress-ring')].some(ring =>
                        ring.getClientRects().length > 0 &&
                        getComputedStyle(ring).visibility !== 'hidden');
                const ready = !visibleRing && idle && p &&
                    !document.querySelector('[data-dash-is-loading="true"]');
                // Start one quiet window after both rendering and callbacks stop.
                if (p && !ready) p.changed = performance.now();
                return ready && performance.now() - p.changed >= quiet;
            }""",
                self.quiet_ms,
            )
            if (
                ready
                and not self.pending
                and (perf_counter() - self.last_network) * 1000 >= self.quiet_ms
            ):
                return
            self.page.wait_for_timeout(50)
        raise TimeoutError("Page did not settle within the timeout")

    def measure(self, name, action, repeat):
        """Record one action and its rendering and request diagnostics."""
        start = perf_counter()
        browser_start = self.page.evaluate("performance.now()")
        request_start, error_start = len(self.requests), len(self.errors)
        result = {"action": name, "repeat": repeat, "status": "ok"}
        print(f"Starting: {name}", flush=True)
        try:
            if name != "initial load":
                self.settle()
            # Preparation belongs to the previous action, not this measurement.
            start = perf_counter()
            browser_start = self.page.evaluate("performance.now()")
            request_start, error_start = len(self.requests), len(self.errors)
            action()
            self.settle()
        except SkipActionError as error:
            result.update(status="skipped", reason=str(error))
        except (Error, TimeoutError, ValueError, AssertionError) as error:
            result.update(status="error", error=str(error))
            try:
                self.page.screenshot(
                    path=str(self.output / f"failure-{repeat}-{len(self.results)}.png"),
                    timeout=5000,
                )
            except Error:
                pass  # A failed screenshot must not block the diagnostic report.
        result["elapsed_ms"] = round((perf_counter() - start) * 1000, 1)
        # Navigation starts a fresh performance clock.
        if name == "initial load":
            browser_start = 0
        tasks = self.page.evaluate(
            """start =>
            (window.__mlpegPerf?.longTasks || []).filter(t => t.start >= start)
        """,
            browser_start,
        )
        result["long_task_ms"] = round(sum(t["duration"] for t in tasks), 1)
        result["long_task_count"] = len(tasks)
        result["dash_requests"] = self.requests[request_start:]
        result["browser_errors"] = self.errors[error_start:]
        if result["status"] == "ok" and (
            result["browser_errors"]
            or any(
                (
                    r["failure"]
                    and not (r["status"] == 204 and r["failure"] == "net::ERR_ABORTED")
                )
                or (r["status"] is not None and r["status"] >= 400)
                for r in result["dash_requests"]
            )
        ):
            result["status"] = "error"
        self.results.append(result)
        print(
            f"{name:55} {result['elapsed_ms']:9.1f} ms  {result['status']}", flush=True
        )
        if result.get("reason") or result.get("error"):
            print("  " + (result.get("reason") or result["error"]).splitlines()[0])
        return result["status"] == "ok"


def by_id(identifier):
    """Quote component IDs safely for CSS selectors."""
    return f"[id={json.dumps(identifier)}]"


def edit(page, selector):
    """Commit a different value without assuming particular scoring defaults."""
    control = page.locator(selector).first
    expect(control).to_be_visible()
    expect(control).to_be_editable()
    identifier = control.get_attribute("id")
    table_id = identifier.split("-table-", 1)[0] + "-table"
    wait_for_table(page, by_id(table_id))
    page.wait_for_function(
        """id => {
            const input = document.getElementById(id);
            return input && input.value.trim() !== '' &&
                Number.isFinite(Number(input.value));
        }""",
        arg=identifier,
    )
    value = float(control.input_value())
    control.fill(str(value + 0.1 if value >= 0 else value - 0.1))
    # Wait for the callback caused by this edit, not an unrelated request.
    with page.expect_response(
        lambda response: (
            urlsplit(response.url).path.endswith("/_dash-update-component")
            and response.request.method == "POST"
            and identifier + ".value"
            in (response.request.post_data_json or {}).get("changedPropIds", [])
        )
    ) as update:
        control.press("Enter")
    if update.value.status != 200:
        raise ValueError(f"Edit was not applied: HTTP {update.value.status}")
    update.value.body()


def wait_for_table(page, selector):
    """Wait for actual rendered data cells, rather than an empty table shell."""
    table = page.locator(selector).first
    expect(table).to_be_visible()
    expect(table.locator("td[data-dash-column]").first).to_be_visible()


def navigate(page, path, settle):
    """Navigate through the sidebar so client state survives page changes."""
    link = page.locator(f"#sidebar-nav a[href={json.dumps(path)}]").first
    if not link.count():
        raise ValueError(f"No sidebar link for {path}")
    label = re.sub(r"^[^\w]+", "", link.inner_text().strip())
    link.click()
    page.wait_for_url(f"**{path}")
    if path != "/":
        heading = label + " Benchmarks" if path.startswith("/framework/") else label
        expect(page.locator("#page-content h1").first).to_have_text(heading)
    settle()
    cells = page.locator("#page-content .dash-table-container td[data-dash-column]")
    if not cells.count():
        raise SkipActionError(f"No table rows available on {path}")
    wait_for_table(page, "#page-content .dash-table-container")


def missing_benchmark():
    """Explain how to select a benchmark when automatic discovery finds none."""
    raise SkipActionError(
        "No compatible benchmark found. Set --benchmark, "
        "--benchmark-category and --metric for available plot data."
    )


def open_benchmark(page, category, benchmark, settle):
    """Open a benchmark card if lazy mounting has left it collapsed."""
    navigate(page, category, settle)
    header = page.get_by_role("button", name=benchmark, exact=True)
    header.wait_for(state="visible")
    if header.get_attribute("aria-expanded") == "false":
        header.click()
    wait_for_table(page, by_id(benchmark + "-table"))


def select_model(page, model, visible):
    """Toggle one model and wait for its summary row to appear or disappear."""
    page.locator("#model-filter-checklist").click()
    option = page.get_by_role("option", name=model, exact=True)
    expect(option).to_be_visible()
    option.get_by_role("checkbox").press("Space")
    expect(option).to_have_attribute("aria-selected", str(visible).lower())
    page.keyboard.press("Escape")
    row = page.locator('#summary-table td[data-dash-column="MLIP"]').get_by_text(
        model, exact=True
    )
    if visible:
        expect(row).to_be_visible()
    else:
        expect(row).to_have_count(0)


def filter_elements(page, clear=False):
    """Exclude hydrogen, or clear the filter, through the real Apply controls."""
    details = page.locator("#element-filter-details")
    if details.get_attribute("open") is None:
        details.locator("summary").click()
    if not clear:
        element = details.get_by_role("button", name="H", exact=True)
        element.click()
        expect(element).to_have_attribute("title", "H — excluded")
    button = "element-filter-clear" if clear else "element-filter-apply"
    with page.expect_response(
        lambda response: (
            urlsplit(response.url).path.endswith("/_dash-update-component")
            and response.request.method == "POST"
            and button + ".n_clicks"
            in (response.request.post_data_json or {}).get("changedPropIds", [])
        )
    ) as update:
        page.locator(by_id(button)).click()
    if update.value.status != 200:
        raise ValueError(f"Filter was not applied: HTTP {update.value.status}")
    update.value.body()
    expect(page.locator("#element-filter-apply")).to_be_enabled()


class Tee:
    """Keep console output visible while also saving it to a text file."""

    def __init__(self, console, logfile):
        self.console = console
        self.logfile = logfile

    def write(self, text):
        """Write each chunk to both destinations."""
        self.console.write(text)
        self.logfile.write(text)
        self.logfile.flush()
        return len(text)

    def flush(self):
        """Flush both streams."""
        self.console.flush()
        self.logfile.flush()


def run(args):
    """Save terminal output alongside the browser profiling artifacts."""
    output = args.output / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output.mkdir(parents=True)
    with (output / "terminal.txt").open("w") as logfile:
        with (
            redirect_stdout(Tee(sys.stdout, logfile)),
            redirect_stderr(Tee(sys.stderr, logfile)),
        ):
            return run_journey(args, output)


def run_journey(args, output):
    """Exercise the live app, saving results even if a journey fails."""
    results = []
    report = {
        "url": args.url,
        "label": args.label,
        "platform": platform.platform(),
        "quiet_ms": args.quiet_ms,
        "timeout_ms": args.timeout,
        "benchmark": args.benchmark,
        "metric": args.metric,
        "trace": args.trace,
        "results": results,
    }
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=not args.headed)
        report["browser"] = browser.version
        try:
            for repeat in range(1, args.repeats + 1):
                context = browser.new_context(viewport={"width": 1440, "height": 1000})
                context.add_init_script(f"({OBSERVE})()")
                context.add_init_script("""localStorage.setItem('onboarding-state-store',
                    JSON.stringify({completed: true}));""")
                if args.trace:
                    context.tracing.start(
                        screenshots=True, snapshots=True, sources=True
                    )
                page = context.new_page()
                page.set_default_timeout(args.timeout)
                expect.set_options(timeout=args.timeout)
                journey = Journey(page, output, args.timeout, args.quiet_ms)
                try:

                    def load(page=page):
                        page.goto(args.url)
                        page.wait_for_function("""() => {
                            const mask = document.getElementById('startup-mask');
                            return mask && mask.style.display === 'none';
                        }""")
                        wait_for_table(page, "#summary-table")

                    if not journey.measure("initial load", load, repeat):
                        continue
                    categories = args.category or page.locator(
                        '#sidebar-nav a[href^="/category/"]'
                    ).evaluate_all(
                        'els => [...new Set(els.map(e => e.getAttribute("href")))]'
                    )
                    report["categories"] = categories
                    report["summary_rows"] = page.locator(
                        "#summary-table tbody tr"
                    ).count()
                    journey.measure(
                        "summary weight",
                        lambda page=page: edit(
                            page, ".mlpeg-summary-card .mlpeg-scoring-grid input"
                        ),
                        repeat,
                    )
                    chosen = None
                    for path in categories:
                        navigated = journey.measure(
                            f"navigate {path}",
                            lambda p=path, page=page, journey=journey: navigate(
                                page, p, journey.settle
                            ),
                            repeat,
                        )
                        if not navigated:
                            continue
                        if not args.benchmark and chosen is None:
                            candidate = page.locator('[id$="-table"]').evaluate_all(
                                """(tables, metric) => tables.find(t =>
                                    [...t.querySelectorAll('td[data-dash-column]')]
                                    .some(c => c.dataset.dashColumn === metric) &&
                                    document.getElementById(t.id.slice(0, -6)
                                        + '-figure-placeholder'))?.id""",
                                args.metric,
                            )
                            if candidate:
                                chosen = (path, candidate[:-6])
                    frameworks = page.locator(
                        '#sidebar-nav a[href^="/framework/"]'
                    ).evaluate_all(
                        'els => [...new Set(els.map(e => e.getAttribute("href")))]'
                    )
                    report["frameworks"] = frameworks
                    for path in frameworks:
                        journey.measure(
                            f"navigate {path}",
                            partial(navigate, page, path, journey.settle),
                            repeat,
                        )
                    benchmark = args.benchmark or (chosen[1] if chosen else None)
                    category = args.benchmark_category or (
                        chosen[0] if chosen else None
                    )
                    if not benchmark or not category:
                        journey.measure(
                            "select benchmark",
                            lambda: missing_benchmark(),
                            repeat,
                        )
                    else:
                        report["benchmark"] = benchmark
                        report["benchmark_category"] = category
                        if journey.measure(
                            "open benchmark category",
                            partial(
                                open_benchmark,
                                page,
                                category,
                                benchmark,
                                journey.settle,
                            ),
                            repeat,
                        ):
                            table = f"[id={json.dumps(benchmark + '-table')}]"
                            journey.measure(
                                "benchmark weight",
                                lambda page=page, benchmark=benchmark: edit(
                                    page,
                                    by_id(
                                        benchmark + "-table-" + args.metric + "-input"
                                    ),
                                ),
                                repeat,
                            )
                            journey.measure(
                                "good threshold",
                                lambda page=page, benchmark=benchmark: edit(
                                    page,
                                    by_id(
                                        benchmark
                                        + "-table-"
                                        + args.metric
                                        + "-good-threshold"
                                    ),
                                ),
                                repeat,
                            )

                            def click_cell(page=page, table=table, benchmark=benchmark):
                                wait_for_table(page, table)
                                page.locator(
                                    f"{table} "
                                    f"td[data-dash-column={json.dumps(args.metric)}]"
                                ).first.click()
                                page.locator(
                                    by_id(benchmark + "-figure-placeholder")
                                    + " .js-plotly-plot"
                                ).first.wait_for(state="visible")

                            journey.measure("table cell to plot", click_cell, repeat)
                    journey.measure(
                        "return to summary",
                        lambda page=page, journey=journey: navigate(
                            page, "/", journey.settle
                        ),
                        repeat,
                    )
                    models = page.locator('#summary-table td[data-dash-column="MLIP"]')
                    if models.count() > 1:
                        model = models.first.inner_text().strip()
                        report["toggled_model"] = model
                        if journey.measure(
                            "deselect model",
                            partial(select_model, page, model, False),
                            repeat,
                        ):
                            journey.measure(
                                "restore model",
                                partial(select_model, page, model, True),
                                repeat,
                            )
                    journey.measure(
                        "apply element filter", partial(filter_elements, page), repeat
                    )
                    journey.measure(
                        "clear element filter",
                        partial(filter_elements, page, clear=True),
                        repeat,
                    )
                finally:
                    results.extend(journey.results)
                    if args.trace:
                        context.tracing.stop(path=str(output / f"trace-{repeat}.zip"))
                    context.close()
                    (output / "results.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
        finally:
            browser.close()
    save_summary(report, output)
    print("\nSuccessful action timings (including the quiet window):")
    for name in dict.fromkeys(r["action"] for r in results):
        samples = [
            r["elapsed_ms"]
            for r in results
            if r["action"] == name and r["status"] == "ok"
        ]
        if samples:
            print(
                f"{name:55} median {median(samples):9.1f} ms "
                f" max {max(samples):9.1f} ms"
            )
    print(f"\nReport: {output / 'results.json'}")
    return int(any(r["status"] == "error" for r in results))


def save_summary(report, output):
    """Save a readable chart and text report, keeping failed actions explicit."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for result in report["results"]:
        groups.setdefault(result["action"], []).append(result)
    if not groups:
        return
    names, medians, labels, colors = [], [], [], []
    lines = [
        "ML-PEG browser timings",
        report["label"] or report["url"],
        "",
        "Action | Median | Range | Successful / total",
    ]
    for name, results in groups.items():
        samples = [r["elapsed_ms"] / 1000 for r in results if r["status"] == "ok"]
        failed = sum(r["status"] == "error" for r in results)
        skipped = sum(r["status"] == "skipped" for r in results)
        names.append(
            name.replace("navigate /category/", "Page: ")
            .replace("navigate /framework/", "Framework: ")
            .replace("-", " ")
            .capitalize()
        )
        medians.append(median(samples) if samples else 0)
        colors.append("#cc7a29" if failed else "#4479b3")
        labels.append(
            (f"{median(samples):.2f} s" if samples else "No successful run")
            + (f"  ({failed} failed)" if failed else "")
            + (f"  ({skipped} skipped)" if skipped else "")
        )
        stats = (
            f"{median(samples):.2f} s | {min(samples):.2f}–{max(samples):.2f} s"
            if samples
            else "— | —"
        )
        lines.append(f"{name} | {stats} | {len(samples)}/{len(results)}")
        for result in results:
            if result["status"] != "ok":
                detail = (
                    result.get("error")
                    or result.get("reason")
                    or "Browser or request error (see JSON)"
                )
                lines.append(f"  Repeat {result['repeat']}: {detail}")
    note = (
        f"Includes {report['quiet_ms']} ms quiet window and automation overhead. "
        "Failed actions are excluded from timing statistics."
    )
    if report["trace"]:
        note += " Tracing enabled (adds overhead)."
    lines.extend(["", note])
    (output / "summary.txt").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(11, max(4, len(names) * 0.4 + 1.8)))
    positions = list(range(len(names)))
    ax.barh(positions, medians, color=colors, height=0.62)
    ax.set_yticks(positions, names)
    ax.invert_yaxis()
    maximum = max(max(medians), 0.1)
    for position, value, label in zip(positions, medians, labels, strict=True):
        ax.text(value + maximum * 0.02, position, label, va="center", fontsize=9)
    ax.set_xlim(0, maximum * 1.6)
    ax.set_xlabel("Median elapsed time (seconds)")
    ax.set_title("ML-PEG browser timings", loc="left", weight="bold", pad=20)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.grid(True, alpha=0.18)
    ax.set_axisbelow(True)
    fig.text(
        0.02,
        0.02,
        f"Quiet window: {report['quiet_ms']} ms · "
        f"Trace: {'on' if report['trace'] else 'off'} · "
        "Successful runs only, failures labelled",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fig.savefig(output / "summary.png", dpi=160)
    plt.close(fig)


def category_path(value):
    """Accept a category name or its sidebar path and reject malformed paths."""
    name = value.removeprefix("/category/")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", name):
        raise argparse.ArgumentTypeError("use a category name or /category/<name>")
    return f"/category/{name}"


def main():
    """Parse the diagnostic runner's command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8050")
    parser.add_argument("--output", type=Path, default=Path("profiles/browser"))
    parser.add_argument(
        "--label", default="", help="Commit and dataset description for comparison"
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="Fresh browser sessions, same server"
    )
    parser.add_argument(
        "--category",
        type=category_path,
        action="append",
        help=(
            "Category name or path to visit, repeatable. "
            "Default: all sidebar categories"
        ),
    )
    parser.add_argument(
        "--benchmark-category",
        type=category_path,
        help="Category name or path for an explicit benchmark",
    )
    parser.add_argument(
        "--benchmark", help="Benchmark component name. Default: auto-select"
    )
    parser.add_argument("--metric", default="MAE")
    parser.add_argument(
        "--timeout", type=int, default=60000, help="Action timeout in milliseconds"
    )
    parser.add_argument(
        "--quiet-ms",
        type=int,
        default=300,
        help="Network and DOM quiet window included in timings",
    )
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    parser.add_argument(
        "--trace", action="store_true", help="Save Playwright traces (adds overhead)"
    )
    args = parser.parse_args()
    if bool(args.benchmark) != bool(args.benchmark_category):
        parser.error("provide benchmark and benchmark-category together")
    if min(args.repeats, args.timeout, args.quiet_ms) <= 0:
        parser.error("repeats, timeout and quiet-ms must be positive")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
