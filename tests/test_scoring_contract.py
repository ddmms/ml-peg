"""Independent numerical expectations for scoring and group propagation."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from dash.dash_table import DataTable
import numpy as np
import pytest

from ml_peg.analysis.utils.utils import calc_table_scores, normalize_metric
from ml_peg.app.utils import register_callbacks


@pytest.mark.xfail(
    strict=True,
    raises=AssertionError,
    reason="Concurrent threshold edits replace other metrics' values",
)
def test_threshold_callbacks_with_stale_state(monkeypatch):
    """Two in-flight requests must merge edits, not replace the full dictionary."""
    callbacks = []

    def capture(*args, **kwargs):
        def decorate(function):
            callbacks.append(function)
            return function

        return decorate

    monkeypatch.setattr(register_callbacks, "callback", capture)
    defaults = {key: {"good": 0, "bad": 10, "unit": "eV"} for key in ("M1", "M2")}
    register_callbacks.register_normalization_callbacks(
        "B1", ["M1", "M2"], defaults, register_toggle=False
    )
    edits = [f for f in callbacks if f.__name__ == "store_threshold_values"]
    result = deepcopy(defaults)
    for metric, callback in zip(defaults, edits, strict=True):
        monkeypatch.setattr(
            register_callbacks,
            "ctx",
            SimpleNamespace(triggered_id=f"B1-{metric}-bad-threshold"),
        )
        update = callback(0, 20, 0, deepcopy(defaults))
        if isinstance(update, dict):
            result = update
        else:
            # Support an eventual partial-update fix without changing this test.
            for operation in update.to_plotly_json()["operations"]:
                target = result
                for key in operation["location"][:-1]:
                    target = target[key]
                target[operation["location"][-1]] = operation["params"]["value"]
    assert [result[key]["bad"] for key in defaults] == [20, 20]


@pytest.mark.parametrize(
    ("value", "good", "bad", "expected"),
    [
        (2, 0, 10, 0.8),
        (8, 0, 10, 0.2),
        (2, 0, 20, 0.9),
        (2, 10, 0, 0.2),
        (-1, 0, 10, 1),
        (11, 0, 10, 0),
        (5, 5, 5, 1),
        (4, 5, 5, 0),
        (None, 0, 10, None),
    ],
)
def test_normalization_contract(value, good, bad, expected):
    """Threshold direction, boundaries and equal thresholds have fixed answers."""
    result = normalize_metric(value, good, bad)
    assert result == pytest.approx(expected) if expected is not None else result is None


@pytest.mark.parametrize(
    ("values", "weights", "expected"),
    [
        ((0.8, 0.2), (1, 1), 0.5),
        ((0.8, 0.2), (3, 1), 0.65),
        ((0.8, 0.2), (30, 10), 0.65),
        ((0.8, 0.2), (0, 1), 0.2),
        ((0.8, 0.2), (0, 0), None),
        ((0.8, None), (1, 1), 0.8),
        ((0.8, "NaN"), (1, 1), "NaN"),
        ((0.8, np.nan), (1, 1), "NaN"),
        ((0.8, "NaN"), (1, 0), 0.8),
        ((None, None), (1, 1), None),
    ],
)
def test_weighted_mean_contract(values, weights, expected):
    """Missing, failed and excluded values have distinct semantics."""
    rows = [{"MLIP": "A", "x": values[0], "y": values[1]}]
    actual = calc_table_scores(
        rows, weights=dict(zip(("x", "y"), weights, strict=False))
    )[0]["Score"]
    if isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


def test_registered_group_callback_preserves_all_models(monkeypatch):
    """Shared benchmarks propagate to both groups without filtering hidden rows."""
    callbacks = []

    def capture(*args, **kwargs):
        def decorate(function):
            callbacks.append(function)
            return function

        return decorate

    monkeypatch.setattr(register_callbacks, "callback", capture)
    table = DataTable(id="B1-table")
    register_callbacks.register_benchmark_to_group_callback(
        {"category": {"B1": table}, "framework": {"B1": table}},
        {"category": "category", "framework": "framework"},
    )
    rows = [{"MLIP": name, "B1 Score": 0.5, "Score": 0.5} for name in ("A", "B")]
    before = deepcopy(rows)
    patches = callbacks[0](
        {"B1 Score": 1},
        rows,
        {"B1 Score": 1},
        deepcopy(rows),
        [{"MLIP": "A", "Score": 0.65}, {"MLIP": "B", "Score": 0.45}],
    )
    for patch in patches:
        updates = {
            tuple(op["location"]): op["params"]["value"]
            for op in patch.to_plotly_json()["operations"]
        }
        assert updates == {
            (0, "B1 Score"): 0.65,
            (0, "Score"): 0.65,
            (1, "B1 Score"): 0.45,
            (1, "Score"): 0.45,
        }
    assert rows == before
