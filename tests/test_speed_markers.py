"""Tests for benchmark speed markers."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_peg.app.utils.speed import (
    SPEED_LEVELS,
    SPEED_ORDER,
    get_benchmark_speed,
)


def write_calc(tmp_path: Path, body: str) -> Path:
    """
    Write a fake calc file into a temporary benchmark directory.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    body
        Python source to write into the calc file.

    Returns
    -------
    Path
        Directory containing the written calc file.
    """
    calc_dir = tmp_path / "mycat" / "mybench"
    calc_dir.mkdir(parents=True)
    (calc_dir / "calc_mybench.py").write_text(body)
    return calc_dir


def test_levels_are_ordered_fast_to_multi_day():
    """Levels are ordered cheapest to most expensive."""
    assert SPEED_ORDER == ("fast", "medium", "slow", "very_slow", "multi_day")
    assert tuple(SPEED_LEVELS) == SPEED_ORDER


def test_every_level_has_display_metadata():
    """Each level carries the metadata the badge needs."""
    for level, config in SPEED_LEVELS.items():
        assert set(config) == {
            "label",
            "runtime",
            "tooltip",
            "color",
            "text_color",
        }, level


def test_single_marker_is_returned(tmp_path):
    """A benchmark with one speed marker reports that marker."""
    calc_dir = write_calc(
        tmp_path,
        "import pytest\n\n@pytest.mark.slow\ndef test_a():\n    pass\n",
    )
    assert get_benchmark_speed(calc_dir) == "slow"


def test_multiple_markers_return_the_slowest(tmp_path):
    """A benchmark with several speed markers reports the slowest."""
    calc_dir = write_calc(
        tmp_path,
        "import pytest\n\n"
        "@pytest.mark.slow\ndef test_a():\n    pass\n\n"
        "@pytest.mark.very_slow\ndef test_b():\n    pass\n\n"
        "@pytest.mark.multi_day\ndef test_c():\n    pass\n",
    )
    assert get_benchmark_speed(calc_dir) == "multi_day"


def test_very_slow_is_not_matched_as_slow(tmp_path):
    """very_slow is matched exactly, not as a substring of slow."""
    calc_dir = write_calc(
        tmp_path,
        "import pytest\n\n@pytest.mark.very_slow\ndef test_a():\n    pass\n",
    )
    assert get_benchmark_speed(calc_dir) == "very_slow"


def test_unmarked_benchmark_returns_none(tmp_path):
    """A benchmark with no speed marker reports None."""
    calc_dir = write_calc(
        tmp_path,
        "import pytest\n\n@pytest.mark.parametrize('x', [1])\n"
        "def test_a(x):\n    pass\n",
    )
    assert get_benchmark_speed(calc_dir) is None


def test_missing_directory_returns_none(tmp_path):
    """A directory with no calc file reports None rather than raising."""
    assert get_benchmark_speed(tmp_path / "does_not_exist") is None


def test_speed_badge_renders_label_and_tooltip():
    """The badge names what it measures, shows the level, and has a tooltip."""
    from ml_peg.app.utils.build_components import build_speed_badge

    badge = build_speed_badge("very_slow")
    # The tooltip is drawn by speed_badge.css from data-tooltip, not by the
    # native title attribute, whose show delay is not configurable.
    assert getattr(badge, "title", None) is None
    assert badge.className == "speed-badge"
    props = badge.to_plotly_json()["props"]
    assert props["data-tooltip"] == "Tests run in 10 hours to a day on GPU"
    assert [segment.children for segment in badge.children] == [
        "Test speed",
        "Very slow",
    ]


def test_multi_day_speed_badge_renders():
    """The multi-day badge describes tests requiring multiple GPU days."""
    from ml_peg.app.utils.build_components import build_speed_badge

    badge = build_speed_badge("multi_day")
    props = badge.to_plotly_json()["props"]
    assert props["data-tooltip"] == "Tests require multiple GPU days"
    assert [segment.children for segment in badge.children] == [
        "Test speed",
        "Multi-day",
    ]


def test_speed_badge_is_none_when_unmarked():
    """No badge is built for an unmarked benchmark."""
    from ml_peg.app.utils.build_components import build_speed_badge

    assert build_speed_badge(None) is None


def test_speed_badge_is_none_for_unknown_level():
    """No badge is built for an unrecognised level."""
    from ml_peg.app.utils.build_components import build_speed_badge

    assert build_speed_badge("blazing") is None


def test_speed_for_table_path_maps_to_calc_dir():
    """A table output path resolves to the matching benchmark's speed."""
    from ml_peg.app import APP_ROOT
    from ml_peg.app.utils.speed import speed_for_table_path

    table_path = APP_ROOT / "data" / "bulk_crystal" / "phonons" / "x.json"
    assert speed_for_table_path(table_path) == "slow"


def test_speed_for_unknown_table_path_is_none():
    """A table path with no matching calc directory reports None."""
    from ml_peg.app.utils.speed import speed_for_table_path

    assert speed_for_table_path(Path("/tmp/nope/also_nope/x.json")) is None


def test_other_markers_are_ignored(tmp_path):
    """Non-speed markers do not interfere with detection."""
    calc_dir = write_calc(
        tmp_path,
        "import pytest\n\n@pytest.mark.framework('mace-multihead')\n"
        "@pytest.mark.medium\ndef test_a():\n    pass\n",
    )
    assert get_benchmark_speed(calc_dir) == "medium"


@pytest.mark.fast
def test_fast_marker_does_not_skip():
    """A fast-marked test must run without any extra CLI flag."""
    assert True


@pytest.mark.medium
def test_medium_marker_does_not_skip():
    """A medium-marked test must run without any extra CLI flag."""
    assert True


@pytest.mark.multi_day
def test_multi_day_marker_runs_when_enabled():
    """A multi-day test runs only when explicitly enabled."""
    assert True


def test_summarise_speeds_counts_each_level():
    """Each level is counted, with unmarked benchmarks kept separate."""
    from ml_peg.app.utils.speed import summarise_speeds

    counts = summarise_speeds(["fast", "fast", "slow", None])
    assert counts == {
        "fast": 2,
        "medium": 0,
        "slow": 1,
        "very_slow": 0,
        "multi_day": 0,
        "unclassified": 1,
    }


def test_summarise_speeds_handles_empty_input():
    """An empty input still reports every level."""
    from ml_peg.app.utils.speed import summarise_speeds

    counts = summarise_speeds([])
    assert set(counts) == {
        "fast",
        "medium",
        "slow",
        "very_slow",
        "multi_day",
        "unclassified",
    }
    assert sum(counts.values()) == 0


def test_speed_panel_reports_counts():
    """The panel reports per-level counts and categories, but not unclassified."""
    from ml_peg.app.utils.build_components import build_speed_panel

    speeds = {f"cat/fast{i}": "fast" for i in range(42)}
    speeds.update({f"cat/slow{i}": "slow" for i in range(7)})
    speeds["cat/unmarked"] = None

    text = str(build_speed_panel(speeds))
    assert "42" in text
    assert "Test speeds" in text
    assert text.count("Typical GPU runtime per test") == 1
    assert "Under 10 min" in text
    assert "Multiple days" in text
    assert "unclassified" not in text.lower()
    assert "Everything" not in text
    assert "Total runtime" not in text
    assert "Timings are for" not in text


def test_runtimes_skip_blank_entries(tmp_path):
    """Blank benchmarks and a blank device are treated as not recorded."""
    from ml_peg.analysis.utils import runtimes

    path = tmp_path / "runtimes.yml"
    path.write_text(
        "measured_with:\n"
        "  model: mace-mp-0\n"
        "  device:\n"
        "benchmarks:\n"
        "  molecular:\n"
        "    GMTKN55: 2.5\n"
        "    Wiggle150:\n"
    )
    original = runtimes.RUNTIMES_FILE
    try:
        runtimes.RUNTIMES_FILE = path
        provenance, measured = runtimes.load_runtimes()
    finally:
        runtimes.RUNTIMES_FILE = original

    assert provenance == {"model": "mace-mp-0"}
    assert measured == {"molecular/GMTKN55": 2.5}


def test_app_speed_keys_use_directory_identifiers():
    """Speed keys do not depend on human-facing benchmark names."""
    from types import SimpleNamespace

    from ml_peg.app.build_app import _collect_benchmark_speeds

    tables = {
        "bulk_crystal": {
            "Iron Properties": SimpleNamespace(
                benchmark_key="bulk_crystal/iron_properties",
                speed="medium",
            )
        }
    }
    assert _collect_benchmark_speeds(tables) == {
        "bulk_crystal/iron_properties": "medium"
    }


def test_runtimes_scaffold_keys_are_real_benchmarks():
    """Every key in the shipped scaffold names an existing benchmark."""
    import yaml

    from ml_peg.analysis.utils.runtimes import RUNTIMES_FILE
    from ml_peg.calcs import CALCS_ROOT

    data = yaml.safe_load(RUNTIMES_FILE.read_text(encoding="utf8"))
    for category, benchmarks in data["benchmarks"].items():
        for benchmark in benchmarks or {}:
            assert (CALCS_ROOT / category / benchmark).is_dir(), (
                f"{category}/{benchmark}"
            )


def test_sub_minute_runtime_uses_one_minute_floor():
    """Positive sub-minute timings include a one-minute execution overhead."""
    from conftest import _round_runtime_minutes

    assert _round_runtime_minutes(0.01) == 1
    assert _round_runtime_minutes(0.99) == 1
    assert _round_runtime_minutes(1.24) == 1.2


def test_speed_badge_is_below_the_framework_row(tmp_path):
    """The app derives speed and puts its badge below the framework row."""
    import json

    from ml_peg.app.utils.build_components import build_test_layout
    from ml_peg.app.utils.load import rebuild_table

    source = {
        "data": [],
        "columns": [
            {"id": "MLIP", "name": "MLIP"},
            {"id": "Metric", "name": "Metric"},
            {"id": "Score", "name": "Score"},
        ],
        "thresholds": {"Metric": {"good": 0.0, "bad": 1.0, "unit": "eV"}},
        "weights": {"Metric": 1.0},
        "tooltip_header": {"Metric": "Test metric"},
    }
    path = tmp_path / "supramolecular" / "S30L" / "table.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(source))
    table = rebuild_table(path, id="t", description="d")
    assert table.speed == "fast"

    layout = build_test_layout(
        name="S30L",
        description="d",
        framework_ids=[],
        table=table,
        thresholds=table.thresholds,
        speed=table.speed,
    )
    title_row = layout.children[0].children
    assert all("Test speed" not in str(child.children) for child in title_row[1:])
    assert "Test speed" in str(layout.children[1].children.children)


def test_timings_out_ignores_tests_that_did_not_run(tmp_path):
    """Skipped tests must not be recorded as zero-minute measurements."""
    import subprocess
    import sys

    import yaml

    out = tmp_path / "timings.yml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "ml_peg/calcs/physicality/oxidation_states/calc_oxidation_states.py",
            "--models",
            "mace-mp-0a",
            "--mock-only",
            "--run-mock",
            "--timings-out",
            str(out),
            "-q",
        ],
        capture_output=True,
        check=False,
    )
    recorded = yaml.safe_load(out.read_text())["benchmarks"] or {}
    # oxidation_states is very_slow, so it is skipped without --run-very-slow.
    assert "physicality" not in recorded


def test_timings_out_ignores_non_calc_tests_and_writes_reference_schema(tmp_path):
    """Unit tests do not become benchmarks or leave the reference schema."""
    import subprocess
    import sys

    import yaml

    from ml_peg.analysis.utils import runtimes

    out = tmp_path / "timings.yml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_speed_markers.py::test_levels_are_ordered_fast_to_multi_day",
            "--models",
            "mace-mp-0a",
            "--timings-out",
            str(out),
            "-q",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert yaml.safe_load(out.read_text()) == {
        "measured_with": {"model": "mace-mp-0a", "device": None},
        "benchmarks": {},
    }

    original = runtimes.RUNTIMES_FILE
    try:
        runtimes.RUNTIMES_FILE = out
        provenance, measured = runtimes.load_runtimes()
    finally:
        runtimes.RUNTIMES_FILE = original
    assert provenance == {"model": "mace-mp-0a"}
    assert measured == {}


@pytest.mark.parametrize("models", [None, "model-a,model-b"])
def test_timings_out_requires_one_model(tmp_path, models):
    """Timing collection requires exactly one attributable model."""
    import subprocess
    import sys

    model_args = ["--models", models] if models else []
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_speed_markers.py::test_levels_are_ordered_fast_to_multi_day",
            *model_args,
            "--timings-out",
            str(tmp_path / "timings.yml"),
            "-q",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "--timings-out requires exactly one model via --models" in output


def test_timings_out_refuses_mixed_model_file(tmp_path):
    """Existing measurements from another model are never relabelled."""
    import subprocess
    import sys

    out = tmp_path / "timings.yml"
    original = (
        "measured_with:\n"
        "  model: another-model\n"
        "  device: GPU\n"
        "benchmarks:\n"
        "  molecular:\n"
        "    GMTKN55: 2.5\n"
    )
    out.write_text(original)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_speed_markers.py::test_levels_are_ordered_fast_to_multi_day",
            "--models",
            "mace-mp-0a",
            "--timings-out",
            str(out),
            "-q",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "contains another-model measurements, expected mace-mp-0a" in output
    assert out.read_text() == original


@pytest.mark.parametrize("model", ["mace-mp-0a", "mace-mp-0"])
def test_calc_cli_configures_selected_timing_run(monkeypatch, tmp_path, model):
    """The calc command forwards one selected model and disables the mock model."""
    import pytest as pytest_module
    from typer.testing import CliRunner

    from ml_peg.cli.cli import app

    calls = []
    monkeypatch.setattr(pytest_module, "main", lambda options: calls.append(options))
    out = tmp_path / "runtimes.yml"
    result = CliRunner().invoke(
        app,
        [
            "calc",
            "--category",
            "surfaces",
            "--test",
            "S24",
            "--models",
            model,
            "--timings-out",
            str(out),
            "--no-verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    options = calls[0]
    model_index = options.index("--models")
    timing_index = options.index("--timings-out")
    assert options[model_index + 1] == model
    assert options[timing_index + 1] == out
    assert "--run-mock" not in options


def test_calc_cli_requires_timing_model(tmp_path):
    """The calc timing option requires an explicit model selection."""
    from typer.testing import CliRunner

    from ml_peg.cli.cli import app

    result = CliRunner().invoke(
        app,
        [
            "calc",
            "--category",
            "surfaces",
            "--test",
            "S24",
            "--timings-out",
            str(tmp_path / "runtimes.yml"),
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "Timing mode requires exactly one model via --models" in str(
        result.exception
    )


def test_calc_cli_forwards_run_multi_day(monkeypatch):
    """The calc command forwards the multi-day opt-in to pytest."""
    import pytest as pytest_module
    from typer.testing import CliRunner

    from ml_peg.cli.cli import app

    calls = []
    monkeypatch.setattr(pytest_module, "main", lambda options: calls.append(options))
    result = CliRunner().invoke(
        app,
        [
            "calc",
            "--category",
            "surfaces",
            "--test",
            "S24",
            "--run-multi-day",
            "--no-verbose",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "--run-multi-day" in calls[0]


def test_multi_day_marker_requires_explicit_flag():
    """Multi-day tests are skipped unless the dedicated flag is supplied."""
    import subprocess
    import sys

    nodeid = f"{__file__}::test_multi_day_marker_runs_when_enabled"
    skipped = subprocess.run(
        [sys.executable, "-m", "pytest", nodeid, "-q"],
        capture_output=True,
        text=True,
        check=False,
    )
    enabled = subprocess.run(
        [sys.executable, "-m", "pytest", nodeid, "--run-multi-day", "-q"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert "1 skipped" in skipped.stdout, skipped.stdout
    assert "1 passed" in enabled.stdout, enabled.stdout


@pytest.mark.parametrize(
    ("flag", "expected"),
    [("--fast-only", "fast"), ("--medium-only", "medium")],
)
def test_only_flags_restrict_to_one_tier(flag, expected):
    """--fast-only and --medium-only run just that tier."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            __file__,
            "-k",
            "marker_does_not_skip",
            flag,
            "-q",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # This file has one fast-marked and one medium-marked test, so exactly one
    # of them survives each flag.
    assert "1 passed" in result.stdout, result.stdout
    assert "1 skipped" in result.stdout, result.stdout
    assert expected in flag
