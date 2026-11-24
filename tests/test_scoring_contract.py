"""Independent numerical expectations for scoring and group propagation."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from dash import no_update
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
        # A single model is the only model, so a column missing for it is
        # missing for everyone, and is discarded however it is missing.
        ((0.8, None), (1, 1), 0.8),
        ((0.8, "NaN"), (1, 1), 0.8),
        ((0.8, np.nan), (1, 1), 0.8),
        ((0.8, "NaN"), (1, 0), 0.8),
        ((None, None), (1, 1), None),
        (("NaN", "NaN"), (1, 1), None),
    ],
)
def test_weighted_mean_contract(values, weights, expected):
    """Excluded and wholly missing columns drop out of the weighted mean."""
    rows = [{"MLIP": "A", "x": values[0], "y": values[1]}]
    actual = calc_table_scores(
        rows, weights=dict(zip(("x", "y"), weights, strict=False))
    )[0]["Score"]
    if isinstance(expected, float):
        assert actual == pytest.approx(expected)
    else:
        assert actual == expected


@pytest.mark.parametrize("absent", [None, "NaN", np.nan])
def test_column_with_data_for_one_model_nans_the_others(absent):
    """A column any model scores is scored for all of them, however it is missing.

    Which of None/NaN a missing cell holds depends on where it came from, not on
    the model, so the two must not give different scores.
    """
    rows = [
        {"MLIP": "A", "x": 0.8, "y": 0.4},
        {"MLIP": "B", "x": 0.6, "y": absent},
    ]
    scored = calc_table_scores(rows, weights={"x": 1, "y": 1})
    assert scored[0]["Score"] == pytest.approx(0.6)
    assert scored[1]["Score"] == "NaN"


@pytest.mark.parametrize("absent", [None, "NaN", np.nan])
def test_column_with_data_for_no_model_is_discarded(absent):
    """A column scoring nothing for anyone drops out, as a zero weight would.

    This covers a benchmark that has not been run, and one whose systems are all
    removed by an element filter: neither may block every model's score.
    """
    rows = [
        {"MLIP": "A", "x": 0.8, "y": absent},
        {"MLIP": "B", "x": 0.6, "y": None},
    ]
    scored = calc_table_scores(rows, weights={"x": 1, "y": 1})
    assert scored[0]["Score"] == pytest.approx(0.8)
    assert scored[1]["Score"] == pytest.approx(0.6)


def test_every_model_scores_the_same_columns():
    """Each model's score must average exactly the same combination of columns."""
    rows = [
        # "unrun" is missing for everyone, so no model is scored on it. "patchy"
        # is missing only for C, so C alone is NaN rather than scoring on a
        # smaller set of columns than A and B.
        {"MLIP": "A", "weighted": 0.8, "off": 0.1, "unrun": None, "patchy": 0.4},
        {"MLIP": "B", "weighted": 0.6, "off": 0.9, "unrun": "NaN", "patchy": 0.2},
        {"MLIP": "C", "weighted": 0.5, "off": 0.5, "unrun": None, "patchy": None},
    ]
    weights = {"weighted": 3, "off": 0, "unrun": 1, "patchy": 1}
    scored = calc_table_scores(rows, weights=weights)
    assert scored[0]["Score"] == pytest.approx((3 * 0.8 + 0.4) / 4)
    assert scored[1]["Score"] == pytest.approx((3 * 0.6 + 0.2) / 4)
    assert scored[2]["Score"] == "NaN"


def test_registered_group_callback_preserves_all_models(monkeypatch):
    """Shared benchmarks propagate to both groups without filtering hidden rows."""
    callbacks = []

    def capture(*args, **kwargs):
        def decorate(function):
            callbacks.append(function)
            return function

        return decorate

    monkeypatch.setattr(register_callbacks, "callback", capture)
    monkeypatch.setattr(
        register_callbacks,
        "ctx",
        SimpleNamespace(
            triggered_prop_ids={
                "B1-table-computed-store.data": "B1-table-computed-store"
            }
        ),
    )
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


@pytest.mark.parametrize("trigger_group", ["Category title", "framework"])
def test_group_weight_only_trigger_rescores_without_benchmark_changes(
    monkeypatch, trigger_group
):
    """Weight-only edits rescore all cached models in just the triggered group."""
    callbacks = []

    def capture(*args, **kwargs):
        def decorate(function):
            callbacks.append(function)
            return function

        return decorate

    group_ids = {"Category title": "category-slug", "framework": "framework-slug"}
    weight_id = f"{group_ids[trigger_group]}-summary-table-weight-store"
    monkeypatch.setattr(register_callbacks, "callback", capture)
    monkeypatch.setattr(
        register_callbacks,
        "ctx",
        SimpleNamespace(triggered_prop_ids={f"{weight_id}.data": weight_id}),
    )
    tables = {name: DataTable(id=f"{name}-table") for name in ("B1", "B2")}
    register_callbacks.register_benchmark_to_group_callback(
        dict.fromkeys(group_ids, tables), group_ids
    )
    rows = [
        {"MLIP": "A", "B1 Score": 0.2, "B2 Score": 0.8, "Score": 0.5},
        {"MLIP": "B", "B1 Score": 0.4, "B2 Score": 0.8, "Score": 0.6},
    ]
    before = deepcopy(rows)
    args = []
    for group in sorted(group_ids):
        args.extend(
            [{"B1 Score": 3 if group == trigger_group else 1, "B2 Score": 1}, rows]
        )
    for benchmark in ("B1", "B2"):
        args.append(
            [{"MLIP": row["MLIP"], "Score": row[f"{benchmark} Score"]} for row in rows]
        )
    outputs = callbacks[0](*args)
    for group, output in zip(sorted(group_ids), outputs, strict=True):
        if group != trigger_group:
            assert output is no_update
            continue
        updates = {
            tuple(op["location"]): op["params"]["value"]
            for op in output.to_plotly_json()["operations"]
        }
        assert updates == {
            (0, "Score"): pytest.approx(0.35),
            (1, "Score"): pytest.approx(0.5),
        }
    assert rows == before
