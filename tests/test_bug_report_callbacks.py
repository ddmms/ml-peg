"""Tests for the data included by the browser bug-report callback."""

from __future__ import annotations

from dash import ClientsideFunction
from dash.dcc import Store

from ml_peg.app.utils import bug_report

STORES = [
    Store(id="test-table-weight-store", data={"MAE": 1}),
    Store(id="test-table-thresholds-store", data={"MAE": {"good": 0}}),
    Store(id="theme-store", data="dark"),
    Store(id="test-table-computed-store", data={"private_result": 123}),
    Store(id="unrelated", data={"not_reported": "value"}),
]


def _find(component: Store, component_id: str):
    """Return the first child of ``component`` with the given id."""
    return next(child for child in component.children if child.id == component_id)


def test_report_callback_reads_current_settings_without_results(monkeypatch):
    """Read settings on demand without copying computed results or unrelated data."""
    calls = []
    monkeypatch.setattr(
        bug_report,
        "clientside_callback",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    bug_report.register_bug_report_callbacks(STORES)

    args, kwargs = calls[0]
    assert kwargs["prevent_initial_call"] is True
    assert args[2].component_id == "bug-report-button"
    assert [state.component_id for state in args[3:]] == [
        "bug-report-config",
        "selected-models-store",
        "element-filter",
        "test-table-weight-store",
        "test-table-thresholds-store",
        "theme-store",
    ]


def test_report_callback_is_a_named_asset_function(monkeypatch):
    """
    Keep the callback's Dash identity independent of the benchmark data.

    An inline clientside callback is named by hashing its source, so embedding
    the defaults there renamed the function whenever benchmark data changed. A
    page still holding the previous name then failed to find it on click.
    """
    calls = []
    monkeypatch.setattr(
        bug_report,
        "clientside_callback",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    bug_report.register_bug_report_callbacks(STORES)

    function = calls[0][0][0]
    assert isinstance(function, ClientsideFunction)
    assert function.namespace == "bug_report"
    assert function.function_name == "open"


def test_config_store_records_defaults_without_results():
    """Compare against defaults for settings stores only, omitting computed data."""
    config = _find(bug_report.build_bug_report_button(STORES), "bug-report-config").data

    assert config["store_ids"] == [
        "test-table-weight-store",
        "test-table-thresholds-store",
        "theme-store",
    ]
    assert config["defaults"]["test-table-weight-store"] == {"MAE": 1}
    assert config["defaults"]["test-table-thresholds-store"] == {"MAE": {"good": 0}}
    assert config["version"]
    assert "test-table-computed-store" not in config["defaults"]
    assert "unrelated" not in config["defaults"]
