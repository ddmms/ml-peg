"""Benchmark speed classification and display metadata."""

from __future__ import annotations

import ast
from collections.abc import Iterable
from pathlib import Path

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
_SPEED_MARKERS = frozenset(SPEED_ORDER)


def _marker_names(tree: ast.Module) -> set[str]:
    """
    Collect ``pytest.mark.<name>`` decorator names from a parsed module.

    Parameters
    ----------
    tree
        Parsed module to inspect.

    Returns
    -------
    set[str]
        Marker names applied to functions in the module.

    Raises
    ------
    ValueError
        If one function has more than one speed marker.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        function_names: set[str] = set()
        for decorator in node.decorator_list:
            # Strip the call form, so @pytest.mark.framework("x") is handled
            # alongside the bare @pytest.mark.slow form.
            target = decorator.func if isinstance(decorator, ast.Call) else decorator
            if isinstance(target, ast.Attribute):
                function_names.add(target.attr)
        speed_markers = function_names & _SPEED_MARKERS
        if len(speed_markers) > 1:
            ordered_markers = [
                marker for marker in SPEED_ORDER if marker in speed_markers
            ]
            raise ValueError(
                f"{node.name} has conflicting speed markers: "
                f"{', '.join(ordered_markers)}"
            )
        names.update(function_names)
    return names


def get_benchmark_speed(calc_dir: Path) -> str | None:
    """
    Return the slowest pytest speed marker in a benchmark's calc file.

    Parameters
    ----------
    calc_dir
        Directory holding the benchmark's ``calc_*.py`` file.

    Returns
    -------
    str | None
        Slowest speed level present, or None when no speed marker is found.
    """
    found: set[str] = set()
    for calc_file in sorted(Path(calc_dir).glob("calc_*.py")):
        try:
            tree = ast.parse(calc_file.read_text(encoding="utf8"))
        except (OSError, SyntaxError):
            continue
        found |= _marker_names(tree) & _SPEED_MARKERS

    for level in reversed(SPEED_ORDER):
        if level in found:
            return level
    return None


def summarise_speeds(speeds: Iterable[str | None]) -> dict[str, int]:
    """
    Count benchmarks per speed level, including unclassified benchmarks.

    Parameters
    ----------
    speeds
        Speed level of each benchmark, with None for unmarked benchmarks.

    Returns
    -------
    dict[str, int]
        Count for each speed level and for unclassified benchmarks.
    """
    counts = dict.fromkeys(SPEED_ORDER, 0)
    counts["unclassified"] = 0
    for speed in speeds:
        counts[speed if speed in SPEED_ORDER else "unclassified"] += 1
    return counts


def speed_for_table_path(table_path: Path | str) -> str | None:
    """
    Resolve a benchmark's speed from the path of its app table JSON.

    Parameters
    ----------
    table_path
        Full path to the benchmark's table JSON.

    Returns
    -------
    str | None
        Speed marker for the matching benchmark, or None when none is found.
    """
    from ml_peg.calcs import CALCS_ROOT

    table_dir = Path(table_path).parent
    return get_benchmark_speed(CALCS_ROOT / table_dir.parent.name / table_dir.name)
