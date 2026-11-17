"""Browser editors preserve defaults when saved session stores are incomplete."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def editor_page(page):
    """Load the production editors with a controllable Dash callback context."""
    page.evaluate(
        "window.dash_clientside = {no_update: 'NO_UPDATE', "
        "callback_context: {triggered: []}}"
    )
    page.add_script_tag(
        path=str(Path(__file__).parents[2] / "ml_peg/app/data/ui/settings_editors.js")
    )
    return page


@pytest.mark.parametrize("stored", [None, {}, {"other": 0}])
def test_weight_defaults_and_reset(editor_page, stored):
    """Missing weights restore configured defaults and reset repairs missing keys."""
    result = editor_page.evaluate(
        """stored => {
        const config = {columns: ['M1', 'M2'], inputs: ['one', 'two'],
            defaults: {M1: 2, M2: 3}, reset: 'reset'};
        const sync = dash_clientside.settingsEditors.weights(
            config, stored, null, null, 0);
        dash_clientside.callback_context.triggered = [{prop_id: 'reset.n_clicks'}];
        const reset = dash_clientside.settingsEditors.weights(
            config, stored, null, null, 1);
        return {sync, reset};
    }""",
        stored,
    )
    assert result["sync"] == ["NO_UPDATE", 2, 3]
    assigned = {
        op["location"][0]: op["params"]["value"]
        for op in result["reset"][0]["operations"]
    }
    assert assigned == {**({"other": 1} if stored else {}), "M1": 2, "M2": 3}


def test_missing_threshold_can_be_edited(editor_page):
    """A new metric accepts an edit without requiring a preliminary reset."""
    result = editor_page.evaluate("""() => {
        const config = {metrics: ['M1'], inputs: ['good', 'bad'],
            defaults: {M1: {good: 0, bad: 10, unit: 'eV'}},
            reset: 'reset', step: 0.1, digits: 2};
        dash_clientside.callback_context.triggered = [{prop_id: 'bad.value'}];
        return dash_clientside.settingsEditors.thresholds(
            config, {}, 0, 20, 0, 0, 0, 1, 0);
    }""")
    assert result[0]["operations"][0]["params"]["value"] == {
        "good": 0,
        "bad": 20,
        "unit": "eV",
    }
