"""Test logging of when calculations ran and whether they completed."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
from warnings import catch_warnings, simplefilter, warn

import pytest

from ml_peg.calcs import CALCS_ROOT
from ml_peg.calcs.utils.run_log import (
    MAX_UNIQUE_WARNINGS,
    SCHEMA_VERSION,
    benchmark_id,
    end_run,
    log_path,
    log_run,
    start_run,
    summarise_warnings,
)

#: Benchmark written into the calculations tree to exercise the pytest plugin.
TEMPORARY_BENCHMARK = CALCS_ROOT / "tmp_run_log_category" / "tmp_run_log_test"
TEMPORARY_SCRIPT = TEMPORARY_BENCHMARK / "calc_tmp_run_log_test.py"

BENCHMARK_SCRIPT = '''
"""Temporary benchmark used to test run logging."""

from __future__ import annotations

from pathlib import Path
from warnings import warn

import pytest

OUT_PATH = Path(r"{out_path}")


@pytest.mark.parametrize("mlip", [("passes", None), ("warns", None), ("fails", None)])
def test_tmp_run_log(mlip):
    """
    Run a fake calculation.

    Parameters
    ----------
    mlip
        Name of model to use and model to get calculator.
    """
    model_name = mlip[0]
    if model_name == "warns":
        for index in range(5):
            warn(f"structure {{index % 2}} failed", stacklevel=2)
    if model_name == "fails":
        raise RuntimeError("CUDA out of memory")


@pytest.mark.slow
@pytest.mark.parametrize("mlip", [("not_run", None)])
def test_tmp_run_log_slow(mlip):
    """
    Run a fake calculation that is skipped.

    Parameters
    ----------
    mlip
        Name of model to use and model to get calculator.
    """
'''


def read_run(out_path: Path, model_name: str) -> dict:
    """
    Read the most recent logged run for a model.

    Parameters
    ----------
    out_path
        Output directory of the benchmark.
    model_name
        Name of the model.

    Returns
    -------
    dict
        Most recently logged run.
    """
    log = json.loads(log_path(out_path, model_name).read_text(encoding="utf8"))
    assert log["schema_version"] == SCHEMA_VERSION
    return log["runs"][-1]


@pytest.fixture
def benchmark(tmp_path):
    """
    Write a temporary benchmark into the calculations tree.

    The plugin only logs scripts within the calculations tree, so the benchmark
    cannot live in a temporary directory alone.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.

    Yields
    ------
    Path
        Output directory the temporary benchmark writes its logs to.
    """
    out_path = tmp_path / "outputs"
    TEMPORARY_BENCHMARK.mkdir(parents=True)
    TEMPORARY_SCRIPT.write_text(
        BENCHMARK_SCRIPT.format(out_path=out_path), encoding="utf8"
    )
    try:
        yield out_path
    finally:
        shutil.rmtree(TEMPORARY_BENCHMARK.parent)


def run_benchmark(script: Path, *options: str) -> subprocess.CompletedProcess:
    """
    Run the temporary benchmark through pytest.

    Calculations are named ``calc_*.py``, so are passed to pytest by path rather
    than collected from a directory.

    Parameters
    ----------
    script
        Path to the benchmark script.
    *options
        Additional options passed to pytest.

    Returns
    -------
    subprocess.CompletedProcess
        Completed pytest run.
    """
    return subprocess.run(
        [sys.executable, "-m", "pytest", str(script), "-q", *options],
        cwd=CALCS_ROOT.parents[1],
        capture_output=True,
        text=True,
        check=False,
    )


def test_benchmark_id():
    """Test identifying benchmarks from their script or directory."""
    script = CALCS_ROOT / "supramolecular" / "S30L" / "calc_S30L.py"
    assert benchmark_id(script) == "supramolecular/S30L"
    assert benchmark_id(script.parent) == "supramolecular/S30L"
    assert benchmark_id(CALCS_ROOT / "supramolecular") is None
    assert benchmark_id(Path("/elsewhere/calc_S30L.py")) is None


def test_start_and_end_run(tmp_path):
    """
    Test that a run is logged as running until it is finished.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    path = start_run(tmp_path, "model", "category/test", {"pressure_idx": 2})

    run = read_run(tmp_path, "model")
    assert run["status"] == "running"
    assert run["benchmark"] == "category/test"
    assert run["params"] == {"pressure_idx": 2}
    assert run["finished"] is None

    end_run(path, "completed")

    run = read_run(tmp_path, "model")
    assert run["status"] == "completed"
    assert run["finished"] is not None
    assert run["duration_s"] >= 0


def test_runs_appended(tmp_path):
    """
    Test that re-running a calculation keeps the previous run.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    for status in ("failed", "completed"):
        end_run(start_run(tmp_path, "model"), status)

    log = json.loads(log_path(tmp_path, "model").read_text(encoding="utf8"))
    assert [run["status"] for run in log["runs"]] == ["failed", "completed"]


def test_summarise_warnings():
    """Test that warning summaries are truncated but keep their total."""
    counts = Counter(
        {("UserWarning", f"message {index}"): index + 1 for index in range(30)}
    )
    summary = summarise_warnings(counts)

    assert summary["total"] == sum(counts.values())
    assert summary["n_unique"] == 30
    assert summary["truncated"]
    assert len(summary["messages"]) == MAX_UNIQUE_WARNINGS


def test_log_run_completed(tmp_path):
    """
    Test logging a calculation that pytest does not run directly.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    with catch_warnings(record=True) as recorded:
        simplefilter("always")
        with log_run(tmp_path, "model"):
            for index in range(4):
                warn(f"structure {index % 2} failed", stacklevel=2)

    # Warnings are counted without being swallowed
    assert len(recorded) == 4

    run = read_run(tmp_path, "model")
    assert run["status"] == "completed"
    assert run["warnings"]["total"] == 4
    assert run["warnings"]["n_unique"] == 2


def test_log_run_failed(tmp_path):
    """
    Test that a failing calculation is logged and still raises.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory fixture.
    """
    with pytest.raises(ValueError, match="bad structure"):
        with log_run(tmp_path, "model"):
            raise ValueError("bad structure")

    run = read_run(tmp_path, "model")
    assert run["status"] == "failed"
    assert run["error"]["type"] == "ValueError"
    assert run["error"]["message"] == "bad structure"
    assert "ValueError" in run["error"]["traceback"]


def test_plugin_logs_outcomes(benchmark):
    """
    Test that calculations run through pytest are logged without opting in.

    Parameters
    ----------
    benchmark
        Output directory of the temporary benchmark.
    """
    result = run_benchmark(TEMPORARY_SCRIPT)
    assert "1 failed, 2 passed" in result.stdout

    assert read_run(benchmark, "passes")["status"] == "completed"

    failed = read_run(benchmark, "fails")
    assert failed["status"] == "failed"
    assert failed["error"]["type"] == "RuntimeError"
    assert failed["error"]["message"] == "CUDA out of memory"

    # Repeated warnings are counted, as they record partial failures
    warns = read_run(benchmark, "warns")
    assert warns["status"] == "completed"
    assert warns["warnings"]["total"] == 5
    assert warns["warnings"]["n_unique"] == 2

    # A calculation that never ran has no log, rather than a logged outcome
    assert not log_path(benchmark, "not_run").exists()


def test_plugin_records_environment(benchmark):
    """
    Test that logs identify the code and model a calculation ran with.

    Parameters
    ----------
    benchmark
        Output directory of the temporary benchmark.
    """
    run_benchmark(TEMPORARY_SCRIPT)
    run = read_run(benchmark, "passes")

    assert run["benchmark"] == "tmp_run_log_category/tmp_run_log_test"
    assert run["model"] == "passes"
    assert run["environment"]["ml_peg_version"]
    assert run["environment"]["packages"]["ase"]
    # Unknown models are logged without a configuration digest
    assert run["model_config"] == {"name": "passes", "config_sha256": None}


def test_plugin_disabled(benchmark):
    """
    Test that logging can be turned off.

    Parameters
    ----------
    benchmark
        Output directory of the temporary benchmark.
    """
    run_benchmark(TEMPORARY_SCRIPT, "--no-run-log")
    assert not benchmark.exists()
