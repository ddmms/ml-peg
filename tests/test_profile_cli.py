"""Check optional profiler registration and shared command-line parsing."""

from __future__ import annotations

from unittest.mock import Mock

import pytest
from typer import Typer
from typer.testing import CliRunner

from ml_peg.app.utils import browser_profiling
from ml_peg.cli import profile as profile_cli
from ml_peg.cli.cli import app as main_app

runner = CliRunner()


def make_app(monkeypatch, missing=None):
    """Construct a CLI with a controlled optional-dependency inventory."""
    monkeypatch.setattr(
        profile_cli, "find_spec", lambda name: None if name == missing else object()
    )
    app = Typer()

    @app.callback()
    def root():
        pass

    profile_cli.register_profile_command(app)
    return app


@pytest.mark.parametrize("missing", ["playwright", "matplotlib"])
def test_profile_hidden_without_dependencies(monkeypatch, missing):
    """Ordinary installations neither advertise nor register developer profiling."""
    app = make_app(monkeypatch, missing)
    assert not app.registered_commands


def test_profile_help_is_available(monkeypatch):
    """Both command discovery and detailed help work without launching Chromium."""
    app = make_app(monkeypatch)
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "profile" in result.output
    result = runner.invoke(main_app, ["profile", "--help"])
    assert result.exit_code == 0
    assert "usage: ml_peg profile" in result.output
    assert "--benchmark-category" in result.output
    assert "--headed" in result.output


def test_profile_forwards_options_and_exit_status(monkeypatch, tmp_path):
    """The packaged command uses the existing parser and preserves failed-run status."""
    app = make_app(monkeypatch)
    run = Mock(return_value=1)
    monkeypatch.setattr(browser_profiling, "run", run)
    result = runner.invoke(
        app,
        [
            "profile",
            "--url",
            "http://localhost:9000",
            "--output",
            str(tmp_path),
            "--repeats",
            "2",
            "--category",
            "bulk_crystal",
            "--category",
            "/category/nebs",
            "--headed",
            "--trace",
            "--label",
            "test run",
        ],
    )
    assert result.exit_code == 1
    args = run.call_args.args[0]
    assert args.url == "http://localhost:9000"
    assert args.output == tmp_path
    assert args.repeats == 2
    assert args.category == ["/category/bulk_crystal", "/category/nebs"]
    assert args.headed and args.trace
    assert args.label == "test run"


@pytest.mark.parametrize(
    "options",
    [
        ["--repeats", "0"],
        ["--benchmark", "IONPI19"],
        ["--unknown"],
    ],
)
def test_profile_rejects_invalid_arguments(monkeypatch, options):
    """Invalid invocations fail before starting a profiling session."""
    app = make_app(monkeypatch)
    run = Mock()
    monkeypatch.setattr(browser_profiling, "run", run)
    result = runner.invoke(app, ["profile", *options])
    assert result.exit_code == 2
    run.assert_not_called()
