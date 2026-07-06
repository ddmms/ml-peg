"""Tests for skipping previously completed calculations."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pytest import Pytester

import ml_peg.analysis
import ml_peg.calcs
from ml_peg.calcs.utils import completion

CALCS_CONFTEST = Path(ml_peg.calcs.__file__).parent / "conftest.py"
ANALYSIS_CONFTEST = Path(ml_peg.analysis.__file__).parent / "conftest.py"

CALC_FILE = '''
"""Fake benchmark writing one line per model run."""

from pathlib import Path

OUT_PATH = Path(__file__).parent / "outputs"

MODELS = {"model-a": None, "model-b": None}


def test_fake():
    """Pretend to run calculations for each model."""
    for model_name, _model in MODELS.items():
        out_dir = OUT_PATH / model_name
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "runs.txt", "a") as file:
            file.write("run\\n")
'''

FAILING_CALC_FILE = (
    CALC_FILE
    + """
        if model_name == "model-b":
            raise ValueError("Calculation failed")
"""
)

PARAM_CALC_FILE = '''
"""Fake parametrized benchmark writing one line per model run."""

from pathlib import Path
from typing import Any

import pytest

OUT_PATH = Path(__file__).parent / "outputs"

MODELS = {"model-a": None, "model-b": None}


@pytest.mark.parametrize("mlip", MODELS.items())
def test_fake_param(mlip: tuple[str, Any]):
    """Pretend to run a calculation for one model."""
    model_name, _model = mlip
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs.txt", "a") as file:
        file.write("run\\n")
    if model_name == "model-b":
        raise ValueError("Calculation failed")
'''


STACKED_CALC_FILE = '''
"""Fake benchmark parametrized over models and cases, logging each model run."""

from pathlib import Path
from typing import Any

import pytest

OUT_PATH = Path(__file__).parent / "outputs"

MODELS = {"model-a": None, "model-b": None}


@pytest.mark.parametrize("mlip", MODELS.items())
@pytest.mark.parametrize("case_idx", range(2))
def test_fake_stacked(mlip: tuple[str, Any], case_idx: int):
    """Pretend to run a calculation for one model and case."""
    model_name, _model = mlip
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs.txt", "a") as file:
        file.write("run\\n")
'''


YAML_CALC_FILE = '''
"""Fake benchmark selecting models named in models_list.txt from models.yml."""

from pathlib import Path
from typing import Any

import pytest

from ml_peg import models

HERE = Path(__file__).parent
models.models_file = HERE / "models.yml"

OUT_PATH = HERE / "outputs"

MODELS = {name: None for name in (HERE / "models_list.txt").read_text().split()}


@pytest.mark.parametrize("mlip", MODELS.items())
def test_fake_yaml(mlip: tuple[str, Any]):
    """Pretend to run a calculation for one model."""
    model_name, _model = mlip
    out_dir = OUT_PATH / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "runs.txt", "a") as file:
        file.write("run\\n")
'''


ANALYSIS_FILE = '''
"""Fake analysis benchmark logging each run."""

from pathlib import Path

HERE = Path(__file__).parent
CALC_PATH = HERE / "calc_outputs"
OUT_PATH = HERE / "app_data"

MODELS = {"model-a": None, "model-b": None}


def test_analysis():
    """Pretend to analyse all models."""
    OUT_PATH.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH / "runs.txt", "a") as file:
        file.write("run\\n")
'''

FAILING_ANALYSIS_FILE = (
    ANALYSIS_FILE
    + """
    raise ValueError("Analysis failed")
"""
)

FLAKY_ANALYSIS_FILE = (
    ANALYSIS_FILE
    + """
    if (HERE / "fail.flag").exists():
        raise ValueError("Analysis failed")
"""
)

DATA_ANALYSIS_FILE = ANALYSIS_FILE.replace(
    "    OUT_PATH.mkdir(parents=True, exist_ok=True)",
    """    from ml_peg.calcs.utils import completion

    data_file = HERE / "cache.txt"
    data_file.write_text("data")
    completion.record_data_file(data_file)
    OUT_PATH.mkdir(parents=True, exist_ok=True)""",
)

NO_CALC_PATH_ANALYSIS_FILE = ANALYSIS_FILE.replace(
    'CALC_PATH = HERE / "calc_outputs"\n', ""
)


def _runs(pytester: Pytester, model_name: str) -> int:
    """
    Count runs recorded by the fake benchmark for a model.

    Parameters
    ----------
    pytester
        Pytester fixture the fake benchmark ran under.
    model_name
        Name of the model to count runs for.

    Returns
    -------
    int
        Number of recorded runs.
    """
    runs_file = pytester.path / "outputs" / model_name / "runs.txt"
    return runs_file.read_text().count("run") if runs_file.exists() else 0


def test_fingerprint_tracks_sources_and_config(tmp_path):
    """Test fingerprint changes with calc scripts and model configuration."""
    calc_file = tmp_path / "calc_fake.py"
    calc_file.write_text("A = 1\n")
    config = {"class_name": "mace"}

    fingerprint = completion.calc_fingerprint(tmp_path, "model", config=config)
    assert fingerprint == completion.calc_fingerprint(tmp_path, "model", config=config)

    calc_file.write_text("A = 2\n")
    new_fingerprint = completion.calc_fingerprint(tmp_path, "model", config=config)
    assert new_fingerprint != fingerprint

    assert (
        completion.calc_fingerprint(tmp_path, "model", config={"class_name": "orb"})
        != new_fingerprint
    )


def test_fingerprint_tracks_local_config_files(tmp_path):
    """Test fingerprint changes when local files in the configuration change."""
    (tmp_path / "calc_fake.py").write_text("A = 1\n")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("weights")
    config = {"kwargs": {"model_path": str(checkpoint)}}

    fingerprint = completion.calc_fingerprint(tmp_path, "model", config=config)
    checkpoint.write_text("retrained weights")
    assert completion.calc_fingerprint(tmp_path, "model", config=config) != fingerprint


def test_fingerprint_ignores_checkpoint_mtime(tmp_path):
    """Test fingerprint is unchanged when a checkpoint is re-copied unmodified."""
    (tmp_path / "calc_fake.py").write_text("A = 1\n")
    checkpoint = tmp_path / "model.ckpt"
    checkpoint.write_text("weights")
    config = {"kwargs": {"model_path": str(checkpoint)}}

    fingerprint = completion.calc_fingerprint(tmp_path, "model", config=config)

    # Bump mtime without touching the content, as re-staging with cp does
    os.utime(checkpoint, ns=(1, 1))
    assert completion.calc_fingerprint(tmp_path, "model", config=config) == fingerprint

    # Same size, new content: the file is re-hashed and the fingerprint changes
    checkpoint.write_text("weightz")
    assert completion.calc_fingerprint(tmp_path, "model", config=config) != fingerprint


def test_marker_roundtrip(tmp_path):
    """Test completion markers record fingerprints and data files."""
    out_path = tmp_path / "outputs"
    data_file = tmp_path / "data.zip"
    data_file.write_text("data")

    assert not completion.is_complete(out_path, "model", "test_x", "abc")

    completion.mark_complete(out_path, "model", "test_x", "abc", [str(data_file)])
    assert completion.is_complete(out_path, "model", "test_x", "abc")
    assert not completion.is_complete(out_path, "model", "test_x", "other")
    assert not completion.is_complete(out_path, "model", "test_y", "abc")

    # Data is treated as changed once the cached file is gone
    data_file.unlink()
    assert not completion.is_complete(out_path, "model", "test_x", "abc")

    # Removing an entry invalidates it; absent entries are a no-op
    completion.mark_complete(out_path, "model", "test_y", "abc", [])
    completion.unmark_complete(out_path, "model", "test_y")
    assert not completion.is_complete(out_path, "model", "test_y", "abc")
    completion.unmark_complete(out_path, "model", "missing")


def test_completed_calcs_skipped(pytester: Pytester):
    """Test completed calculations are skipped until their inputs change."""
    pytester.makeconftest(CALCS_CONFTEST.read_text())
    pytester.makepyfile(calc_fake=CALC_FILE)

    result = pytester.runpytest_subprocess("calc_fake.py")
    result.assert_outcomes(passed=1)
    assert _runs(pytester, "model-a") == 1
    assert _runs(pytester, "model-b") == 1

    marker_file = pytester.path / "outputs" / "model-a" / completion.MARKER_FILENAME
    assert "test_fake" in json.loads(marker_file.read_text())

    # Unchanged inputs: test is skipped and calculations do not re-run
    result = pytester.runpytest_subprocess("calc_fake.py")
    result.assert_outcomes(skipped=1)
    assert _runs(pytester, "model-a") == 1

    # Missing outputs for one model: only that model re-runs
    marker_file_b = pytester.path / "outputs" / "model-b" / completion.MARKER_FILENAME
    marker_file_b.unlink()
    result = pytester.runpytest_subprocess("calc_fake.py")
    result.assert_outcomes(passed=1)
    assert _runs(pytester, "model-a") == 1
    assert _runs(pytester, "model-b") == 2

    # --force-calcs: everything re-runs
    result = pytester.runpytest_subprocess("calc_fake.py", "--force-calcs")
    result.assert_outcomes(passed=1)
    assert _runs(pytester, "model-a") == 2

    # Changed benchmark code: everything re-runs
    (pytester.path / "calc_fake.py").write_text(CALC_FILE + "\n# changed\n")
    result = pytester.runpytest_subprocess("calc_fake.py")
    result.assert_outcomes(passed=1)
    assert _runs(pytester, "model-a") == 3


def test_parametrized_calcs_skipped_per_model(pytester: Pytester):
    """Test model-parametrized calculations are skipped and marked per model."""
    pytester.makeconftest(CALCS_CONFTEST.read_text())
    pytester.makepyfile(calc_param=PARAM_CALC_FILE)

    # model-a passes, model-b fails
    result = pytester.runpytest_subprocess("calc_param.py")
    result.assert_outcomes(passed=1, failed=1)
    assert _runs(pytester, "model-a") == 1
    assert _runs(pytester, "model-b") == 1

    # Only the passed model is skipped; the failed model re-runs
    result = pytester.runpytest_subprocess("calc_param.py")
    result.assert_outcomes(skipped=1, failed=1)
    assert _runs(pytester, "model-a") == 1
    assert _runs(pytester, "model-b") == 2

    # Changed benchmark code: everything re-runs
    fixed_calc = PARAM_CALC_FILE.replace(
        '    if model_name == "model-b":\n        raise ValueError'
        '("Calculation failed")\n',
        "",
    )
    (pytester.path / "calc_param.py").write_text(fixed_calc)
    result = pytester.runpytest_subprocess("calc_param.py")
    result.assert_outcomes(passed=2)
    assert _runs(pytester, "model-a") == 2
    assert _runs(pytester, "model-b") == 3

    # --force-calcs: everything re-runs
    result = pytester.runpytest_subprocess("calc_param.py", "--force-calcs")
    result.assert_outcomes(passed=2)
    assert _runs(pytester, "model-a") == 3


def test_stacked_parametrized_calcs_skipped_per_case(pytester: Pytester):
    """Test calculations stacking extra parameters on the model track each case."""
    pytester.makeconftest(CALCS_CONFTEST.read_text())
    pytester.makepyfile(calc_stacked=STACKED_CALC_FILE)

    # Every model and case combination runs
    result = pytester.runpytest_subprocess("calc_stacked.py")
    result.assert_outcomes(passed=4)
    assert _runs(pytester, "model-a") == 2
    assert _runs(pytester, "model-b") == 2

    # Completion is marked per case, not once per model
    marker_file = pytester.path / "outputs" / "model-a" / completion.MARKER_FILENAME
    marker = json.loads(marker_file.read_text())
    assert set(marker) == {
        "test_fake_stacked[case_idx=0]",
        "test_fake_stacked[case_idx=1]",
    }

    # Unchanged inputs: nothing re-runs
    result = pytester.runpytest_subprocess("calc_stacked.py")
    result.assert_outcomes(skipped=4)
    assert _runs(pytester, "model-a") == 2

    # One case's marker missing for one model: only that case re-runs
    del marker["test_fake_stacked[case_idx=1]"]
    marker_file.write_text(json.dumps(marker))
    result = pytester.runpytest_subprocess("calc_stacked.py")
    result.assert_outcomes(passed=1, skipped=3)
    assert _runs(pytester, "model-a") == 3
    assert _runs(pytester, "model-b") == 2


def test_yaml_changes_invalidate_only_changed_models(pytester: Pytester):
    """Test only models with unchanged configuration are skipped across runs."""
    pytester.makeconftest(CALCS_CONFTEST.read_text())
    pytester.makepyfile(calc_yaml=YAML_CALC_FILE)
    models_yml = pytester.path / "models.yml"
    models_list = pytester.path / "models_list.txt"

    models_yml.write_text(
        "model-a:\n  version: 1\nmodel-b:\n  version: 1\nmodel-c:\n  version: 1\n"
    )
    models_list.write_text("model-a model-b")

    result = pytester.runpytest_subprocess("calc_yaml.py")
    result.assert_outcomes(passed=2)

    # Change model-b's configuration only, and select model-c as well
    models_yml.write_text(
        "model-a:\n  version: 1\nmodel-b:\n  version: 2\nmodel-c:\n  version: 1\n"
    )
    models_list.write_text("model-a model-b model-c")

    result = pytester.runpytest_subprocess("calc_yaml.py")
    result.assert_outcomes(skipped=1, passed=2)
    assert _runs(pytester, "model-a") == 1  # overlapping and unchanged: skipped
    assert _runs(pytester, "model-b") == 2  # overlapping but changed: re-ran
    assert _runs(pytester, "model-c") == 1  # newly selected: ran

    # Nothing changed: all models are skipped
    result = pytester.runpytest_subprocess("calc_yaml.py")
    result.assert_outcomes(skipped=3)


def test_dry_run_reports_pending_calcs(pytester: Pytester):
    """Test --dry-run reports pending calculations without running any."""
    pytester.makeconftest(CALCS_CONFTEST.read_text())
    pytester.makepyfile(calc_param=PARAM_CALC_FILE)

    # Nothing has run yet: all models pending, nothing executes
    result = pytester.runpytest_subprocess("calc_param.py", "--dry-run")
    result.assert_outcomes(skipped=2)
    assert _runs(pytester, "model-a") == 0
    result.stdout.fnmatch_lines(
        [
            "*would run:*model-a*",
            "*would run:*model-b*",
            "*2 calculation(s) to run, 0 up to date*",
        ]
    )

    # model-a passes, model-b fails
    pytester.runpytest_subprocess("calc_param.py")

    # Only the failed model is still pending; dry run executes nothing
    result = pytester.runpytest_subprocess("calc_param.py", "--dry-run")
    result.assert_outcomes(skipped=2)
    assert _runs(pytester, "model-a") == 1
    assert _runs(pytester, "model-b") == 1
    result.stdout.fnmatch_lines(
        [
            "*up to date:*model-a*",
            "*would run:*model-b*",
            "*1 calculation(s) to run, 1 up to date*",
        ]
    )

    # --collect-only also reports calculation statuses
    result = pytester.runpytest_subprocess("calc_param.py", "--collect-only")
    assert _runs(pytester, "model-a") == 1
    result.stdout.fnmatch_lines(
        [
            "*up to date:*model-a*",
            "*would run:*model-b*",
            "*1 calculation(s) to run, 1 up to date*",
        ]
    )


def test_failed_calcs_not_marked(pytester: Pytester):
    """Test failed calculations are not marked as completed."""
    pytester.makeconftest(CALCS_CONFTEST.read_text())
    pytester.makepyfile(calc_fake=FAILING_CALC_FILE)

    result = pytester.runpytest_subprocess("calc_fake.py")
    result.assert_outcomes(failed=1)
    marker_file = pytester.path / "outputs" / "model-a" / completion.MARKER_FILENAME
    assert not marker_file.exists()

    # Both models re-run after a failure
    result = pytester.runpytest_subprocess("calc_fake.py")
    result.assert_outcomes(failed=1)
    assert _runs(pytester, "model-a") == 2
    assert _runs(pytester, "model-b") == 2


def test_analysis_fingerprint_tracks_inputs(tmp_path):
    """Test analysis fingerprint tracks sources, configs and calc markers."""
    analysis_dir = tmp_path / "analysis"
    analysis_dir.mkdir()
    calc_path = tmp_path / "calc_outputs"
    (analysis_dir / "analyse_fake.py").write_text("A = 1\n")
    (analysis_dir / "metrics.yml").write_text("metric: 1\n")
    configs = {"model-a": {"class_name": "mace"}, "model-b": {}}

    def fingerprint(models=configs, config_map=None):
        names = list(models)
        return completion.analysis_fingerprint(
            analysis_dir, names, calc_path, configs=config_map or models
        )

    reference = fingerprint()
    assert fingerprint() == reference

    # Analysis source changed
    (analysis_dir / "analyse_fake.py").write_text("A = 2\n")
    changed_source = fingerprint()
    assert changed_source != reference

    # metrics.yml changed
    (analysis_dir / "metrics.yml").write_text("metric: 2\n")
    changed_yaml = fingerprint()
    assert changed_yaml != changed_source

    # A model's configuration changed
    changed_config = fingerprint(
        config_map={"model-a": {"class_name": "orb"}, "model-b": {}}
    )
    assert changed_config != changed_yaml

    # A model added
    assert fingerprint({**configs, "model-c": {}}) != changed_yaml

    # A calc marker appearing, then changing, each change the fingerprint
    marker = calc_path / "model-a" / completion.MARKER_FILENAME
    marker.parent.mkdir(parents=True)
    marker.write_text("{}")
    marker_written = fingerprint()
    assert marker_written != changed_yaml
    marker.write_text('{"test": {}}')
    assert fingerprint() != marker_written

    # MODELS as a dict and as a list of names fingerprint identically
    assert fingerprint(list(configs), config_map=configs) == fingerprint()


def test_analysis_fingerprint_absent_markers_distinct(tmp_path):
    """Test absent calc markers for different models do not collide."""
    calc_path = tmp_path / "calc_outputs"

    fingerprint_a = completion.analysis_fingerprint(
        tmp_path, ["model-a"], calc_path, configs={"model-a": {}}
    )
    fingerprint_b = completion.analysis_fingerprint(
        tmp_path, ["model-b"], calc_path, configs={"model-b": {}}
    )
    assert fingerprint_a != fingerprint_b


def _analysis_runs(pytester: Pytester) -> int:
    """
    Count runs recorded by the fake analysis benchmark.

    Parameters
    ----------
    pytester
        Pytester fixture the fake benchmark ran under.

    Returns
    -------
    int
        Number of recorded runs.
    """
    runs_file = pytester.path / "app_data" / "runs.txt"
    return runs_file.read_text().count("run") if runs_file.exists() else 0


def test_completed_analysis_skipped(pytester: Pytester):
    """Test completed analysis is skipped until its inputs change."""
    pytester.makeconftest(ANALYSIS_CONFTEST.read_text())
    pytester.makepyfile(analyse_fake=ANALYSIS_FILE)
    calc_marker = (
        pytester.path / "calc_outputs" / "model-a" / completion.MARKER_FILENAME
    )
    calc_marker.parent.mkdir(parents=True)
    calc_marker.write_text("{}")

    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(passed=1)
    assert _analysis_runs(pytester) == 1

    marker_file = pytester.path / "app_data" / completion.MARKER_FILENAME
    assert "test_analysis" in json.loads(marker_file.read_text())

    # Unchanged inputs: analysis is skipped
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(skipped=1)
    assert _analysis_runs(pytester) == 1

    # Skipping preserves the marker: a still-unchanged third run skips again
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(skipped=1)
    assert _analysis_runs(pytester) == 1

    # Changed analysis code: re-runs
    (pytester.path / "analyse_fake.py").write_text(ANALYSIS_FILE + "\n# changed\n")
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(passed=1)
    assert _analysis_runs(pytester) == 2

    # Changed calc marker (calculation re-ran): analysis re-runs
    calc_marker.write_text('{"test_calc": {}}')
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(passed=1)
    assert _analysis_runs(pytester) == 3

    # New metrics.yml next to the analysis script: re-runs, then skips again
    metrics_file = pytester.path / "metrics.yml"
    metrics_file.write_text("metric: 1\n")
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(passed=1)
    assert _analysis_runs(pytester) == 4
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(skipped=1)

    # Changed metrics.yml: re-runs
    metrics_file.write_text("metric: 2\n")
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(passed=1)
    assert _analysis_runs(pytester) == 5

    # --force-analysis: re-runs despite a valid marker, and rewrites it; the
    # stale fingerprint proves the skip below relies on the rewritten marker
    marker = json.loads(marker_file.read_text())
    marker["test_analysis"]["fingerprint"] = "stale"
    marker_file.write_text(json.dumps(marker))
    result = pytester.runpytest_subprocess("analyse_fake.py", "--force-analysis")
    result.assert_outcomes(passed=1)
    assert _analysis_runs(pytester) == 6
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(skipped=1)


def test_failed_analysis_not_marked(pytester: Pytester):
    """Test failed analysis is not marked as completed."""
    pytester.makeconftest(ANALYSIS_CONFTEST.read_text())
    pytester.makepyfile(analyse_fake=FAILING_ANALYSIS_FILE)

    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(failed=1)
    marker_file = pytester.path / "app_data" / completion.MARKER_FILENAME
    assert not marker_file.exists()

    # A later passing run writes the marker
    (pytester.path / "analyse_fake.py").write_text(ANALYSIS_FILE)
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(passed=1)
    assert "test_analysis" in json.loads(marker_file.read_text())


def test_failed_rerun_invalidates_marker(pytester: Pytester):
    """Test a failed re-run with unchanged inputs does not skip afterwards."""
    pytester.makeconftest(ANALYSIS_CONFTEST.read_text())
    pytester.makepyfile(analyse_fake=FLAKY_ANALYSIS_FILE)

    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(passed=1)

    # Force a re-run, with fingerprinted inputs unchanged, that fails partway
    flag = pytester.path / "fail.flag"
    flag.write_text("")
    result = pytester.runpytest_subprocess("analyse_fake.py", "--force-analysis")
    result.assert_outcomes(failed=1)

    # The earlier success must not mask the failure: the analysis re-runs
    flag.unlink()
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(passed=1)
    assert _analysis_runs(pytester) == 3


def test_analysis_data_files_tracked(pytester: Pytester):
    """Test analysis re-runs once a recorded data file is gone from the cache."""
    pytester.makeconftest(ANALYSIS_CONFTEST.read_text())
    pytester.makepyfile(analyse_fake=DATA_ANALYSIS_FILE)

    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(passed=1)
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(skipped=1)

    (pytester.path / "cache.txt").unlink()
    result = pytester.runpytest_subprocess("analyse_fake.py")
    result.assert_outcomes(passed=1)
    assert _analysis_runs(pytester) == 2


def test_analysis_without_convention_always_runs(pytester: Pytester):
    """Test analysis modules missing CALC_PATH always run and keep no marker."""
    pytester.makeconftest(ANALYSIS_CONFTEST.read_text())
    pytester.makepyfile(analyse_fake=NO_CALC_PATH_ANALYSIS_FILE)

    for _ in range(2):
        result = pytester.runpytest_subprocess("analyse_fake.py")
        result.assert_outcomes(passed=1)
    assert _analysis_runs(pytester) == 2
    assert not (pytester.path / "app_data" / completion.MARKER_FILENAME).exists()
