"""Network-free scoring app, separate from the published real-data fixture."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys
import time

import pytest


@pytest.fixture(scope="session")
def app_url(tmp_path_factory):
    """Launch a fresh process with only the deterministic scoring dataset."""
    root = tmp_path_factory.mktemp("scoring-app")
    script = Path(__file__).with_name("synthetic_app.py")
    with (root / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                "import runpy,sys; runpy.run_path(sys.argv[1], run_name='__main__')",
                str(script),
                str(root),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            deadline = time.monotonic() + 60
            while not (root / "url").exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    pytest.fail((root / "server.log").read_text())
                time.sleep(0.1)
            yield (root / "url").read_text()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


@pytest.fixture
def page_errors(page):
    """Fail on browser exceptions or failed Dash callback responses."""
    errors = []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on(
        "response",
        lambda response: (
            errors.append(f"{response.status}: {response.url}")
            if response.status >= 400 and "_dash-update-component" in response.url
            else None
        ),
    )
    yield errors
    assert not errors, errors
