"""Browser restoration preserves stored values and suppresses unchanged outputs."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml_peg.app.utils import register_callbacks


@pytest.fixture
def restorers(page, monkeypatch):
    """Capture production clientside callbacks without constructing a full app."""
    scripts = []
    monkeypatch.setattr(
        register_callbacks,
        "clientside_callback",
        lambda code, *a, **k: scripts.append(code),
    )
    monkeypatch.setattr(register_callbacks, "callback", lambda *a, **k: lambda fn: fn)
    page.evaluate(
        """window.dash_clientside = {no_update: 'NO_UPDATE',
            callback_context: {triggered: []}}"""
    )
    page.add_script_tag(
        path=str(Path(__file__).parents[2] / "ml_peg/app/data/ui/settings_editors.js")
    )
    return scripts


@pytest.mark.parametrize("value", [0, 1, 2.5, 1e-12, 1000000])
def test_weight_echo_is_suppressed(page, restorers, value):
    """Equal values produce no output, while a stale input receives its exact weight."""
    columns = ['Energy "MAE"', "Δ force / atom"]
    register_callbacks.register_weight_callbacks("table", columns, ["a", "b"])
    page.evaluate(f"window.restore = {restorers[0]}")
    actual = page.evaluate(
        """([columns, value]) => restore(
            {[columns[0]]: value, [columns[1]]: value}, value, null)""",
        [columns, value],
    )
    assert actual == ["NO_UPDATE", "NO_UPDATE", value]


@pytest.mark.parametrize("bounds", [(0, 10), (10, 0), (-2.5, 0), (1, 1)])
def test_threshold_restoration_preserves_direction(page, restorers, bounds):
    """Restoration converts stored numeric bounds without applying new corrections."""
    good, bad = bounds
    register_callbacks.register_normalization_callbacks(
        "table",
        ["M1", "M2"],
        {key: {"good": good, "bad": bad, "unit": "eV"} for key in ("M1", "M2")},
        register_toggle=False,
    )
    page.evaluate(f"window.restore = {restorers[-1]}")
    actual = page.evaluate(
        """([good, bad]) => restore({
        M1: {good: String(good), bad: String(bad), unit: 'eV'},
        M2: {good, bad, unit: 'eV'}
    }, good, bad, null, null)""",
        [good, bad],
    )
    assert actual == ["NO_UPDATE", "NO_UPDATE", "NO_UPDATE", good, bad]


def test_rounding_matches_python_binary_floats(page, restorers):
    """Decimal ties and random finite binary floats round exactly as in Python."""
    import math
    import random
    import struct

    rng = random.Random(501)
    values = [0.0, -0.0, 0.00048828125, -0.00048828125, 2.675, 1e-12]
    values.extend(i / 2048 for i in range(-101, 102, 2))
    for _ in range(10000):
        value = struct.unpack("d", rng.getrandbits(64).to_bytes(8, "little"))[0]
        if math.isfinite(value):
            values.append(value)
    actual = page.evaluate(
        """values => values.map(value => {
            const result = dash_clientside.settingsEditors.roundEven(value, 10);
            const view = new DataView(new ArrayBuffer(8));
            view.setFloat64(0, result);
            return view.getBigUint64(0).toString(16).padStart(16, '0');
        })""",
        values,
    )
    assert actual == [struct.pack(">d", round(value, 10)).hex() for value in values]
    assert page.evaluate(
        "Object.is(dash_clientside.settingsEditors.roundEven(-1e-12, 10), -0)"
    )


def test_threshold_correction_matches_python(page, restorers):
    """Browser corrections preserve the existing direction and rounding contract."""
    import random

    from ml_peg.app.utils.register_callbacks import enforce_threshold_direction

    rng = random.Random(502)
    cases = []
    expected = []
    for default_good, default_bad in [(0, 10), (10, 0), (1, 1)]:
        for field in ("good", "bad"):
            for _ in range(500):
                good, bad = rng.uniform(-100, 100), rng.uniform(-100, 100)
                cases.append(
                    [field, good, bad, {"good": default_good, "bad": default_bad}]
                )
                expected.append(
                    list(
                        enforce_threshold_direction(
                            edited_field=field,
                            good=good,
                            bad=bad,
                            default_good=default_good,
                            default_bad=default_bad,
                        )
                    )
                )
    actual = page.evaluate(
        """cases => cases.map(args =>
            dash_clientside.settingsEditors.correctBounds(...args, 0.0001, 10))""",
        cases,
    )
    assert actual == expected


@pytest.mark.parametrize("field,value", [("good", 2), ("bad", 20), ("good", 1e-12)])
def test_threshold_edit_echo_and_reset(page, restorers, field, value):
    """A local edit preserves metadata, suppresses echoes and supports reset."""
    defaults = {"MAE": {"good": 0, "bad": 10, "unit": "eV", "level_of_theory": "PBE"}}
    register_callbacks.register_normalization_callbacks(
        "B", ["MAE"], defaults, register_toggle=False
    )
    page.evaluate(f"window.editThreshold = {restorers[0]}")
    good, bad = (value, 10) if field == "good" else (0, value)
    actual = page.evaluate(
        """([stored, field, good, bad]) => {
        dash_clientside.callback_context.triggered = [{
            prop_id: 'B-MAE-' + field + '-threshold.value'}];
        return editThreshold(stored, good, bad, 0, 1, 0, 1, 0);
    }""",
        [defaults, field, good, bad],
    )
    corrected_good = round(good, 10)
    expected = {**defaults["MAE"], "good": corrected_good, "bad": bad}
    if value == 1e-12:
        assert actual == ["NO_UPDATE", 0, "NO_UPDATE"]
    else:
        assert actual[0]["operations"] == [
            {"operation": "Assign", "location": ["MAE"], "params": {"value": expected}}
        ]
    echoed = page.evaluate(
        """([entry, good, bad]) =>
        editThreshold({MAE: entry}, good, bad, 0, 1, 0, 1, 0)""",
        [expected, corrected_good, bad],
    )
    assert echoed == ["NO_UPDATE"] * 3
    reset = page.evaluate(
        """entry => {
        dash_clientside.callback_context.triggered = [{
            prop_id: 'B-reset-thresholds-button.n_clicks'}];
        return editThreshold({MAE: entry}, entry.good, entry.bad, 1);
    }""",
        expected,
    )
    assert reset[0]["operations"][0]["params"]["value"] == defaults["MAE"]


def test_weight_edit_echo_and_reset(page, restorers):
    """Weight edits patch one key, retain other weights, and reset to metadata."""
    register_callbacks.register_weight_callbacks("B", ["MAE"], ["B-MAE"], {"MAE": 2})
    page.evaluate(f"window.editWeight = {restorers[0]}")
    actual = page.evaluate("""() => {
        dash_clientside.callback_context.triggered = [{
            prop_id: 'B-MAE-input.value'}];
        return editWeight({MAE: 2, RMSE: 1}, 3, 0, 1, 0);
    }""")
    assert actual[0]["operations"] == [
        {"operation": "Assign", "location": ["MAE"], "params": {"value": 3}}
    ]
    assert (
        page.evaluate("editWeight({MAE: 3, RMSE: 1}, 3, 0, 1, 0)") == ["NO_UPDATE"] * 2
    )
    assert page.evaluate("editWeight({MAE: 3, RMSE: 1}, null, 0, 1, 0)") == [
        "NO_UPDATE",
        3,
    ]
    reset = page.evaluate("""() => {
        dash_clientside.callback_context.triggered = [{
            prop_id: 'B-reset-button.n_clicks'}];
        return editWeight({MAE: 3, RMSE: 4}, 3, 1, 1, 0);
    }""")
    assert reset[0]["operations"] == [
        {"operation": "Assign", "location": ["MAE"], "params": {"value": 2}},
        {"operation": "Assign", "location": ["RMSE"], "params": {"value": 1}},
    ]
    assert reset[1] == 2


def test_threshold_patches_merge_stale_snapshots(page, restorers):
    """Independent threshold patches cannot replace another metric's newer bounds."""
    defaults = {key: {"good": 0, "bad": 10, "unit": "eV"} for key in ("M1", "M2")}
    register_callbacks.register_normalization_callbacks(
        "B", ["M1", "M2"], defaults, register_toggle=False
    )
    page.evaluate(f"window.editThreshold = {restorers[0]}")
    patches = page.evaluate(
        """defaults => ['M1', 'M2'].map(metric => {
        dash_clientside.callback_context.triggered = [{
            prop_id: 'B-' + metric + '-bad-threshold.value'}];
        return editThreshold(defaults, 0, metric === 'M1' ? 20 : 10,
            0, metric === 'M2' ? 20 : 10, 0, 1, 0, 1, 0, 1, 0, 1, 0)[0];
    })""",
        defaults,
    )
    merged = dict(defaults)
    for patch in patches:
        assert len(patch["operations"]) == 1
        operation = patch["operations"][0]
        assert operation["operation"] == "Assign"
        merged[operation["location"][0]] = operation["params"]["value"]
    assert [merged[key]["bad"] for key in defaults] == [20, 20]
