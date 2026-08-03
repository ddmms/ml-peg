"""
Configure pytest.

Based on https://docs.pytest.org/en/latest/example/simple.html.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from ml_peg import models


def _round_runtime_minutes(minutes: float) -> int | float:
    """Round a measured runtime while retaining a one-minute overhead floor."""
    if minutes < 1:
        return 1
    if minutes < 10:
        return round(minutes, 1)
    return round(minutes)


def pytest_addoption(parser):
    """Add flag to run tests for extra MLIPs."""
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow benchmarks",
    )
    parser.addoption(
        "--run-very-slow",
        action="store_true",
        default=False,
        help="Run very slow benchmarks",
    )
    parser.addoption(
        "--models",
        action="store",
        default=None,
        help="MLIPs, in comma-separated list. Default is all models",
    )
    parser.addoption(
        "--models-file",
        action="store",
        default=None,
        help="Filepath to model definitions. Default models.yml in models directory.",
    )
    parser.addoption(
        "--framework",
        action="store",
        default=None,
        help=(
            "Run only tests belonging to these MLIP framework(s), as a "
            "comma-separated list of framework ids. Default is all tests."
        ),
    )
    for level in ("fast", "medium"):
        parser.addoption(
            f"--{level}-only",
            action="store_true",
            help=f"Run only benchmarks marked {level}",
        )
    parser.addoption(
        "--timings-out",
        action="store",
        default=None,
        help=("Write measured runtimes for one model using the runtimes.yml schema."),
    )


def pytest_configure(config):
    """Configure pytest to custom markers and CLI inputs."""
    # Create custom marker for slow tests
    config.addinivalue_line("markers", "slow: mark test as slow calculations")
    config.addinivalue_line("markers", "very_slow: mark test as very slow calculations")
    config.addinivalue_line(
        "markers",
        "framework(*ids): mark test as belonging to MLIP framework(s)",
    )

    for marker in ("fast: seconds to minutes on GPU", "medium: tens of minutes on GPU"):
        config.addinivalue_line("markers", marker)

    # Set current models from CLI input
    models.current_models = config.getoption("--models")
    if config.getoption("--timings-out") and (
        not models.current_models or "," in models.current_models
    ):
        raise pytest.UsageError("--timings-out requires exactly one model via --models")
    model_file = config.getoption("--models-file")
    if model_file:
        models.models_file = model_file


def pytest_collection_modifyitems(config, items):
    """Skip tests outside the requested speed tier and framework(s)."""
    skip_slow = pytest.mark.skip(reason="need --run-slow option to run")
    skip_very_slow = pytest.mark.skip(reason="need --run-very-slow option to run")

    only = next(
        (level for level in ("fast", "medium") if config.getoption(f"--{level}-only")),
        None,
    )

    for item in items:
        if only:
            if only not in item.keywords:
                item.add_marker(
                    pytest.mark.skip(reason=f"only running {only} benchmarks")
                )
            continue
        if "very_slow" in item.keywords and not config.getoption("--run-very-slow"):
            item.add_marker(skip_very_slow)
        elif "slow" in item.keywords and not config.getoption("--run-slow"):
            item.add_marker(skip_slow)

    # Keep only tests tagged with one of the requested frameworks
    framework = config.getoption("--framework")
    if not framework:
        return
    requested = {name.strip() for name in framework.split(",") if name.strip()}
    selected = []
    deselected = []
    for item in items:
        item_frameworks = {
            fw for marker in item.iter_markers(name="framework") for fw in marker.args
        }
        (selected if item_frameworks & requested else deselected).append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
    items[:] = selected


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Write measured benchmark runtimes when requested."""
    out = config.getoption("--timings-out")
    if not out:
        return

    model = config.getoption("--models").strip()
    out_path = Path(out)
    output = (
        yaml.safe_load(out_path.read_text(encoding="utf8")) if out_path.exists() else {}
    ) or {}
    provenance = output.setdefault("measured_with", {"model": model, "device": None})
    if provenance.get("model") not in (None, model):
        raise pytest.UsageError(
            f"Timing file {out_path} contains {provenance['model']} measurements, "
            f"expected {model}"
        )
    provenance["model"] = model

    durations: dict[tuple[str, str], float] = {}
    for report in terminalreporter.stats.get("passed", []):
        if getattr(report, "when", None) != "call":
            continue
        calc_file = Path(str(report.nodeid).split("::")[0])
        try:
            relative_calc = calc_file.relative_to("ml_peg/calcs")
        except ValueError:
            continue
        if len(relative_calc.parts) != 3 or not relative_calc.name.startswith("calc_"):
            continue
        key = relative_calc.parts[0], relative_calc.parts[1]
        durations[key] = durations.get(key, 0.0) + report.duration / 60

    benchmarks = output.setdefault("benchmarks", {})
    for (category, benchmark), minutes in durations.items():
        benchmarks.setdefault(category, {})[benchmark] = _round_runtime_minutes(minutes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(yaml.safe_dump(output, sort_keys=False), encoding="utf8")
    terminalreporter.write_line(f"Wrote benchmark runtimes to {out_path}")
