"""Test reporting the state of logged calculations."""

from __future__ import annotations

from collections import Counter
import json
from re import search, sub
import shutil

import pytest
from typer.testing import CliRunner

from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.run_log import (
    collect_runs,
    end_run,
    format_error,
    is_stale,
    latest_run,
    log_path,
    read_runs,
    start_run,
)
from ml_peg.cli.cli import app, describe_run, format_duration, format_timestamp
from ml_peg.models import get_models

#: Benchmark written into the calculations tree, as only those are reported.
TEMPORARY_BENCHMARK = CALCS_ROOT / "tmp_status_category" / "tmp_status_test"
BENCHMARK_ID = "tmp_status_category/tmp_status_test"

runner = CliRunner()


@pytest.fixture
def fake_model_names(monkeypatch):
    """
    Replace the defined models with three fake models, one of which is never run.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    """

    def get_model_names(models=None, filepath=None):
        """
        Get fake model names.

        Parameters
        ----------
        models
            Models to select, as a comma-separated list. Default is `None`,
            corresponding to all models.
        filepath
            Unused path to model definitions.

        Returns
        -------
        list[str]
            Selected model names, in the order they are defined.
        """
        all_models = ["model_1", "model_2", "model_3"]
        if models is None:
            return all_models
        selected = set(models.split(","))
        return [model for model in all_models if model in selected]

    monkeypatch.setattr(get_models, "get_model_names", get_model_names)


@pytest.fixture
def logged_runs(fake_model_names):
    """
    Log one run in each state for a temporary benchmark.

    Parameters
    ----------
    fake_model_names
        Fixture replacing the defined models with three fake models.

    Yields
    ------
    Path
        Output directory the temporary benchmark logged its runs to.
    """
    TEMPORARY_BENCHMARK.mkdir(parents=True)
    (TEMPORARY_BENCHMARK / "calc_tmp_status_test.py").write_text(
        '"""Temporary benchmark used to test status reporting."""\n', encoding="utf8"
    )
    out_path = TEMPORARY_BENCHMARK / "outputs"

    end_run(
        start_run(out_path, "model_1", BENCHMARK_ID),
        "completed",
        warning_counts=Counter({("UserWarning", "system 3 failed"): 7}),
    )
    end_run(
        start_run(out_path, "model_2", BENCHMARK_ID),
        "failed",
        format_error("RuntimeError", "CUDA out of memory"),
    )
    # Left as running, as a calculation killed by its scheduler would be
    start_run(out_path, "removed_model", BENCHMARK_ID)

    try:
        yield out_path
    finally:
        shutil.rmtree(TEMPORARY_BENCHMARK.parent)


def status(*options: str) -> str:
    """
    Run the status command.

    Parameters
    ----------
    *options
        Options passed to the command.

    Returns
    -------
    str
        Output of the command, with the table's borders and layout removed.
    """
    result = runner.invoke(app, ["status", *options], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    # Strip box drawing characters, so rows can be matched as plain text
    return " ".join(sub(r"[─-╿]", " ", result.output).split())


def test_read_runs(tmp_path):
    """
    Test reading back logged runs.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    assert read_runs(tmp_path, "model") == []
    assert latest_run(tmp_path, "model") is None

    end_run(start_run(tmp_path, "model"), "failed")
    end_run(start_run(tmp_path, "model"), "completed")

    assert len(read_runs(tmp_path, "model")) == 2
    assert latest_run(tmp_path, "model")["status"] == "completed"


def test_collect_runs(logged_runs):
    """
    Test collecting the most recent run of each model.

    Parameters
    ----------
    logged_runs
        Output directory of the temporary benchmark.
    """
    runs = collect_runs(test="tmp_status_test")
    assert list(runs) == [BENCHMARK_ID]

    models = runs[BENCHMARK_ID]
    # Defined models are listed first, then any that have been run since removed
    assert list(models) == ["model_1", "model_2", "model_3", "removed_model"]
    assert models["model_1"]["status"] == "completed"
    assert models["model_2"]["status"] == "failed"
    assert models["removed_model"]["status"] == "running"


def test_collect_runs_selects_models(logged_runs):
    """
    Test that collection is restricted to the requested models.

    Parameters
    ----------
    logged_runs
        Output directory of the temporary benchmark.
    """
    runs = collect_runs(test="tmp_status_test", models="model_2")
    assert list(runs[BENCHMARK_ID]) == ["model_2"]


def test_is_stale(logged_runs):
    """
    Test detecting results produced by a since-changed model definition.

    Parameters
    ----------
    logged_runs
        Output directory of the temporary benchmark.
    """
    run = latest_run(logged_runs, "model_1")
    # Fake models have no configuration to compare against
    assert not is_stale(run)

    run["model_config"]["config_sha256"] = "0" * 64
    assert is_stale(run)


def test_describe_run(logged_runs):
    """
    Test the state reported for each kind of run.

    Parameters
    ----------
    logged_runs
        Output directory of the temporary benchmark.
    """
    assert describe_run(None) == ("not run", "")
    assert describe_run(latest_run(logged_runs, "model_1")) == ("completed", "")

    state, note = describe_run(latest_run(logged_runs, "model_2"))
    assert state == "failed"
    assert note == "RuntimeError: CUDA out of memory"

    state, note = describe_run(latest_run(logged_runs, "removed_model"))
    assert state == "incomplete"
    assert note.startswith("started ")


def test_format_duration():
    """Test formatting run durations."""
    assert format_duration(None) == "-"
    assert format_duration(7.4) == "7s"
    assert format_duration(464.8) == "7m 44s"
    assert format_duration(7200) == "2h 00m"


def test_format_timestamp():
    """Test formatting logged timestamps."""
    assert format_timestamp(None) == "-"
    assert format_timestamp("2026-10-22T09:21:48+00:00") == "2026-10-22 09:21"
    assert format_timestamp("not a timestamp") == "not a timestamp"


def test_status_summary(logged_runs):
    """
    Test totalling the state of each model per benchmark.

    Parameters
    ----------
    logged_runs
        Output directory of the temporary benchmark.
    """
    output = status("--test", "tmp_status_test")
    assert "Category Test Completed Stale Failed Incomplete Skipped Not Run" in output
    # One completed, none stale, one failed, one incomplete, none skipped, one not run
    assert "tmp_status_category tmp_status_test 1 0 1 1 0 1" in output


def test_status_detail(logged_runs):
    """
    Test reporting each model separately.

    Parameters
    ----------
    logged_runs
        Output directory of the temporary benchmark.
    """
    output = status("--test", "tmp_status_test", "--detail")
    assert "model_2 failed" in output
    assert "RuntimeError: CUDA out of memory" in output
    # Warnings are reported after the time and duration, as they record failures
    assert search(r"model_1 completed \S+ \S+ \S+ 7\b", output)

    # Models that were never run are left out, as they would be most of the table
    assert "model_3" not in output
    assert "1 model(s) in other states" in output

    # The benchmark is named once, rather than repeated for each of its models
    assert output.count("tmp_status_category") == 1
    assert output.count("tmp_status_test") == 1


def test_status_reports_not_run(logged_runs):
    """
    Test reporting models that have never been run, when asked for.

    Parameters
    ----------
    logged_runs
        Output directory of the temporary benchmark.
    """
    output = status("--test", "tmp_status_test", "--state", "not-run")
    assert "model_3 not run" in output
    assert "model_1" not in output


def test_status_filters_state(logged_runs):
    """
    Test reporting only the states asked for.

    Parameters
    ----------
    logged_runs
        Output directory of the temporary benchmark.
    """
    output = status("--state", "failed")
    assert "model_2 failed" in output
    assert "model_1" not in output
    assert "removed_model" not in output


def test_status_stale(logged_runs):
    """
    Test reporting results produced by a since-changed model definition.

    Parameters
    ----------
    logged_runs
        Output directory of the temporary benchmark.
    """
    path = log_path(logged_runs, "model_1")
    log = json.loads(path.read_text(encoding="utf8"))
    log["runs"][-1]["model_config"]["config_sha256"] = "0" * 64
    path.write_text(json.dumps(log), encoding="utf8")

    output = status("--test", "tmp_status_test", "--detail")
    assert "model_1 stale" in output
    assert "model definition changed since run" in output


def test_status_no_tests():
    """Test that an unknown benchmark is reported as such."""
    result = runner.invoke(app, ["status", "--test", "not_a_test"])
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
