"""Settings callbacks suppress echoes while preserving edits and resets."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from dash.exceptions import PreventUpdate
import pytest

from ml_peg.app.utils import register_callbacks
from ml_peg.app.utils.utils import default_weight_store_data


@pytest.fixture
def setting_callbacks(monkeypatch):
    """Capture the registered handlers without creating a Dash application."""
    handlers = {}

    def capture(*args, **kwargs):
        def decorate(function):
            handlers[function.__name__] = function
            return function

        return decorate

    monkeypatch.setattr(register_callbacks, "callback", capture)
    return handlers


def test_weight_edit_echo_and_reset(setting_callbacks, monkeypatch):
    """A changed weight is stored once, and reset still restores all defaults."""
    register_callbacks.register_weight_callbacks("B-MAE", "B", "MAE", {"MAE": 2})
    update = setting_callbacks["store_input_value"]
    monkeypatch.setattr(
        register_callbacks, "ctx", SimpleNamespace(triggered_id="B-MAE-input")
    )
    weights = update(3, 0, {"MAE": 2, "RMSE": 1})
    assert weights == {"MAE": 3, "RMSE": 1}
    with pytest.raises(PreventUpdate):
        update(3, 0, deepcopy(weights))

    monkeypatch.setattr(
        register_callbacks, "ctx", SimpleNamespace(triggered_id="B-reset-button")
    )
    reset = update(3, 1, weights)
    assert reset == {"MAE": 2, "RMSE": 1}
    assert update(3, 2, deepcopy(reset)) == reset


def test_weight_sync_tolerates_incomplete_store(setting_callbacks):
    """A store missing the column, or absent entirely, falls back to the default."""
    # A store persisted before this column existed must not 500 the callback
    register_callbacks.register_weight_callbacks("B-MAE", "B", "MAE", {"MAE": 2})
    sync = setting_callbacks["sync_inputs"]

    assert sync({"MAE": 3}) == 3
    assert sync({"RMSE": 1}) == 2
    assert sync({}) == 2
    assert sync(None) == 2


def test_weight_edit_tolerates_incomplete_store(setting_callbacks, monkeypatch):
    """Storing a weight works when the store is stale or absent."""
    register_callbacks.register_weight_callbacks("B-MAE", "B", "MAE", {"MAE": 2})
    update = setting_callbacks["store_input_value"]
    monkeypatch.setattr(
        register_callbacks, "ctx", SimpleNamespace(triggered_id="B-MAE-input")
    )

    assert update(3, 0, None) == {"MAE": 3}
    assert update(3, 0, {}) == {"MAE": 3}


def test_weight_reset_repairs_incomplete_store(setting_callbacks, monkeypatch):
    """Reset restores defaults for columns a stale store is missing."""
    register_callbacks.register_weight_callbacks(
        "B-MAE", "B", "MAE", {"MAE": 2, "RMSE": 3}
    )
    update = setting_callbacks["store_input_value"]
    monkeypatch.setattr(
        register_callbacks, "ctx", SimpleNamespace(triggered_id="B-reset-button")
    )

    assert update(5, 1, {"MAE": 5}) == {"MAE": 2, "RMSE": 3}
    assert update(5, 1, {}) == {"MAE": 2, "RMSE": 3}


def test_default_weight_store_data_fills_gaps():
    """Configurable columns absent from the table's weights default to 1.0."""
    table = SimpleNamespace(
        weights={"MAE": 2},
        columns=[{"id": "MLIP"}, {"id": "Score"}, {"id": "MAE"}, {"id": "RMSE"}],
    )

    assert default_weight_store_data(table) == {"MAE": 2, "RMSE": 1.0}
    # Must not alias table.weights, which is filled in place elsewhere
    assert table.weights == {"MAE": 2}


def test_default_weight_store_data_without_weights():
    """A table with no weights gets a full set of defaults."""
    table = SimpleNamespace(weights=None, columns=[{"id": "MLIP"}, {"id": "MAE"}])

    assert default_weight_store_data(table) == {"MAE": 1.0}


def test_weight_edit_does_not_mutate_store(setting_callbacks, monkeypatch):
    """The stored dict passed in by Dash is not modified in place."""
    register_callbacks.register_weight_callbacks("B-MAE", "B", "MAE", {"MAE": 2})
    update = setting_callbacks["store_input_value"]
    monkeypatch.setattr(
        register_callbacks, "ctx", SimpleNamespace(triggered_id="B-MAE-input")
    )

    stored = {"MAE": 2, "RMSE": 1}
    assert update(3, 0, stored) == {"MAE": 3, "RMSE": 1}
    assert stored == {"MAE": 2, "RMSE": 1}


@pytest.mark.parametrize("field,value", [("good", 2), ("bad", 20)])
def test_threshold_edit_echo_and_reset(setting_callbacks, monkeypatch, field, value):
    """Threshold edits survive, their echoes stop, and reset retains metadata."""
    defaults = {"MAE": {"good": 0, "bad": 10, "unit": "eV"}}
    register_callbacks.register_normalization_callbacks(
        "B", ["MAE"], defaults, register_toggle=False
    )
    update = setting_callbacks["store_threshold_values"]
    monkeypatch.setattr(
        register_callbacks,
        "ctx",
        SimpleNamespace(triggered_id=f"B-MAE-{field}-threshold"),
    )
    good, bad = (value, 10) if field == "good" else (0, value)
    result = update(good, bad, 0, deepcopy(defaults))
    assert result["MAE"][field] == value
    assert result["MAE"]["unit"] == "eV"
    assert defaults["MAE"] == {"good": 0, "bad": 10, "unit": "eV"}
    with pytest.raises(PreventUpdate):
        update(good, bad, 0, deepcopy(result))

    monkeypatch.setattr(
        register_callbacks,
        "ctx",
        SimpleNamespace(triggered_id="B-reset-thresholds-button"),
    )
    reset = update(good, bad, 1, result)
    assert reset == defaults
    assert update(good, bad, 2, deepcopy(reset)) == defaults


def test_threshold_rounding_still_syncs_inputs(setting_callbacks, monkeypatch):
    """An input rounded back to its stored value must still be corrected onscreen."""
    defaults = {"MAE": {"good": 0, "bad": 10, "unit": "eV"}}
    register_callbacks.register_normalization_callbacks(
        "B", ["MAE"], defaults, register_toggle=False
    )
    monkeypatch.setattr(
        register_callbacks,
        "ctx",
        SimpleNamespace(triggered_id="B-MAE-good-threshold"),
    )
    update = setting_callbacks["store_threshold_values"]
    assert update(1e-12, 10, 0, deepcopy(defaults)) == defaults
