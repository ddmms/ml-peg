"""Tests for the data included by the browser bug reporter."""

from __future__ import annotations

from dash.dcc import Store

from ml_peg.app.utils import bug_report


def test_report_callback_uses_current_settings_without_results(monkeypatch):
    """Read settings on demand without copying computed results or unrelated data."""
    calls = []
    monkeypatch.setattr(
        bug_report,
        "clientside_callback",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    bug_report.register_bug_report_callbacks(
        [
            Store(id="test-table-weight-store", data={"MAE": 1}),
            Store(id="test-table-thresholds-store", data={"MAE": {"good": 0}}),
            Store(id="theme-store", data="dark"),
            Store(id="test-table-computed-store", data={"private_result": 123}),
            Store(id="unrelated", data={"not_reported": "value"}),
        ]
    )

    args, kwargs = calls[0]
    assert kwargs["prevent_initial_call"] is True
    assert args[2].component_id == "bug-report-button"
    assert [state.component_id for state in args[3:]] == [
        "selected-models-store",
        "element-filter",
        "test-table-weight-store",
        "test-table-thresholds-store",
        "theme-store",
    ]
    assert "private_result" not in args[0]
    assert "not_reported" not in args[0]
