"""Tests for benchmark speed markers."""

from __future__ import annotations

from collections.abc import Sequence
from itertools import combinations
from pathlib import Path

import pytest

from ml_peg.app.utils.speed import (
    SPEED_LEVELS,
    SPEED_ORDER,
    get_benchmark_speed,
)


def write_calc(tmp_path: Path, marker_groups: Sequence[Sequence[str]]) -> Path:
    """
    Write a fake calc file into a temporary benchmark directory.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    marker_groups
        Marker expressions for each generated test function.

    Returns
    -------
    Path
        Directory containing the written calc file.
    """
    calc_dir = tmp_path / "mycat" / "mybench"
    calc_dir.mkdir(parents=True)
    tests = []
    for index, markers in enumerate(marker_groups):
        decorators = "\n".join(f"@pytest.mark.{marker}" for marker in markers)
        tests.append(
            f"""{decorators}
def test_{index}():
    pass
"""
        )
    (calc_dir / "calc_mybench.py").write_text("import pytest\n\n" + "\n".join(tests))
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
        [["slow"]],
    )
    assert get_benchmark_speed(calc_dir) == "slow"


def test_different_test_markers_return_the_slowest(tmp_path):
    """A benchmark with tests in several tiers reports the slowest."""
    calc_dir = write_calc(
        tmp_path,
        [["slow"], ["very_slow"], ["multi_day"]],
    )
    assert get_benchmark_speed(calc_dir) == "multi_day"


@pytest.mark.parametrize("markers", tuple(combinations(SPEED_ORDER, 2)))
def test_conflicting_markers_on_one_test_raise(tmp_path, markers):
    """Any pair of speed markers on one test is rejected."""
    calc_dir = write_calc(tmp_path, [markers])
    with pytest.raises(ValueError, match="conflicting speed markers"):
        get_benchmark_speed(calc_dir)


def test_very_slow_is_not_matched_as_slow(tmp_path):
    """very_slow is matched exactly, not as a substring of slow."""
    calc_dir = write_calc(
        tmp_path,
        [["very_slow"]],
    )
    assert get_benchmark_speed(calc_dir) == "very_slow"


def test_unmarked_benchmark_returns_none(tmp_path):
    """A benchmark with no speed marker reports None."""
    calc_dir = write_calc(
        tmp_path,
        [["parametrize('x', [1])"]],
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
        [["framework('mace-multihead')", "medium"]],
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


def test_app_speed_keys_use_unique_test_names():
    """Speed summaries use the existing unique benchmark names."""
    from types import SimpleNamespace

    from ml_peg.app.build_app import _collect_benchmark_speeds

    tables = {"bulk_crystal": {"Iron Properties": SimpleNamespace(speed="medium")}}
    assert _collect_benchmark_speeds(tables) == {"Iron Properties": "medium"}


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


def test_calc_cli_forwards_builtin_pytest_options(monkeypatch):
    """The calc command forwards marker and duration options to pytest."""
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
            "--no-verbose",
            "-m",
            "fast",
            "--durations=0",
        ],
    )

    assert result.exit_code == 0, result.output
    assert calls[0][-3:] == ["-m", "fast", "--durations=0"]


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


@pytest.mark.parametrize("marker", ["fast", "medium"])
def test_pytest_marker_expression_restricts_to_one_tier(marker):
    """Pytest's built-in marker expression runs just the selected tier."""
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            f"{__file__}::test_fast_marker_does_not_skip",
            f"{__file__}::test_medium_marker_does_not_skip",
            "-m",
            marker,
            "-q",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    # This file has one fast-marked and one medium-marked test, so only the
    # requested tier survives the marker expression.
    assert "1 passed" in result.stdout, result.stdout
    assert "1 deselected" in result.stdout, result.stdout
