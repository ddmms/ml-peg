"""Speed levels for benchmarks, and reading them from calc files."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

import yaml

# Ordered cheapest to most expensive. The ordering is what resolves benchmarks
# carrying more than one marker, since the badge shows the slowest one present.
SPEED_LEVELS: dict[str, dict[str, str]] = {
    "fast": {
        "label": "Fast",
        "runtime": "Under 10 min",
        "tooltip": "Tests run in seconds to minutes on GPU",
        "color": "#dcfce7",
        "text_color": "#166534",
    },
    "medium": {
        "label": "Medium",
        "runtime": "10 min to 1 hour",
        "tooltip": "Tests run in tens of minutes on GPU",
        "color": "#fef9c3",
        "text_color": "#854d0e",
    },
    "slow": {
        "label": "Slow",
        "runtime": "1 to 10 hours",
        "tooltip": "Tests run in hours on GPU",
        "color": "#ffedd5",
        "text_color": "#9a3412",
    },
    "very_slow": {
        "label": "Very slow",
        "runtime": "10 hours to 1 day",
        "tooltip": "Tests run in 10 hours to a day on GPU",
        "color": "#fee2e2",
        "text_color": "#991b1b",
    },
    "multi_day": {
        "label": "Multi-day",
        "runtime": "Multiple days",
        "tooltip": "Tests require multiple GPU days",
        "color": "#ede9fe",
        "text_color": "#5b21b6",
    },
}

SPEED_ORDER: tuple[str, ...] = tuple(SPEED_LEVELS)


def _marker_names(tree: ast.Module) -> set[str]:
    """
    Collect ``pytest.mark.<name>`` decorator names used in a parsed module.

    Parameters
    ----------
    tree
        Parsed module to inspect.

    Returns
    -------
    set[str]
        Marker names applied to any function definition in the module.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            # Strip the call form, so @pytest.mark.framework("x") is handled
            # alongside the bare @pytest.mark.slow form.
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Attribute):
                names.add(target.attr)
    return names


def get_benchmark_speed(calc_dir: Path) -> str | None:
    """
    Return the slowest pytest speed marker found in a benchmark's calc file.

    The calc file is parsed rather than imported, because importing pulls in
    torch and the model registry, which is far too heavy for a metadata lookup.

    Parameters
    ----------
    calc_dir
        Directory holding the benchmark's ``calc_*.py`` file.

    Returns
    -------
    str | None
        Slowest speed level present, or None when the benchmark carries no
        speed marker or has no readable calc file.
    """
    found: set[str] = set()
    for calc_file in sorted(Path(calc_dir).glob("calc_*.py")):
        try:
            tree = ast.parse(calc_file.read_text(encoding="utf8"))
        except (OSError, SyntaxError):
            continue
        found |= _marker_names(tree) & set(SPEED_ORDER)

    for level in reversed(SPEED_ORDER):
        if level in found:
            return level
    return None


def summarise_speeds(speeds: Iterable[str | None]) -> dict[str, int]:
    """
    Count benchmarks per speed level.

    Parameters
    ----------
    speeds
        Speed level of each benchmark, with None for unmarked benchmarks.

    Returns
    -------
    dict[str, int]
        Count for each level, plus an ``unclassified`` count for benchmarks
        with no recognised speed marker.
    """
    counts = dict.fromkeys(SPEED_ORDER, 0)
    counts["unclassified"] = 0
    for speed in speeds:
        counts[speed if speed in SPEED_ORDER else "unclassified"] += 1
    return counts


def speed_for_table_path(table_path: Path | str) -> str | None:
    """
    Resolve a benchmark's speed level from the path its table is written to.

    Parameters
    ----------
    table_path
        Full path of the table JSON, of the form
        ``<APP_ROOT>/data/<category>/<benchmark>/<name>.json``.

    Returns
    -------
    str | None
        Slowest speed marker for the matching benchmark, or None when it has no
        marker or no matching calc directory.
    """
    from ml_peg.calcs import CALCS_ROOT

    table_dir = Path(table_path).parent
    return get_benchmark_speed(CALCS_ROOT / table_dir.parent.name / table_dir.name)


RUNTIMES_FILE = Path(__file__).with_name("runtimes.yml")


def load_runtimes() -> tuple[dict[str, str], dict[str, float]]:
    """
    Load measured benchmark runtimes from ``runtimes.yml``.

    Benchmarks left blank in the file are ignored.

    Returns
    -------
    tuple[dict[str, str], dict[str, float]]
        Provenance of the measurements (the model and device they were taken
        on), and a mapping of ``<category>/<benchmark>`` to minutes per model.
    """
    data = yaml.safe_load(RUNTIMES_FILE.read_text(encoding="utf8")) or {}

    measured_with = data.get("measured_with") or {}
    if not isinstance(measured_with, dict):
        raise ValueError(
            f"measured_with in {RUNTIMES_FILE} must contain model/device metadata"
        )
    provenance = {key: value for key, value in measured_with.items() if value}
    measured = {
        f"{category}/{benchmark}": float(minutes)
        for category, benchmarks in (data.get("benchmarks") or {}).items()
        for benchmark, minutes in (benchmarks or {}).items()
        if minutes is not None
    }
    return provenance, measured
