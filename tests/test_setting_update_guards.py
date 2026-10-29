"""Settings callbacks suppress echoes while preserving edits and resets."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from dash.exceptions import PreventUpdate
import pytest

from ml_peg.app.utils import register_callbacks


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
