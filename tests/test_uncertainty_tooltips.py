"""Test tooltips showing metric uncertainties."""

from __future__ import annotations

from ml_peg.app.utils.utils import add_uncertainty_tooltips, filter_rows_by_models

UNCERTAINTIES = {
    "model_1": {
        "MAE": {
            "ci_low": 0.012,
            "ci_high": 0.018,
            "ci_low_ref": 0.01,
            "ci_high_ref": 0.02,
            "n_eff_min": 83.4,
            "method": "autocorrelation (Chodera)",
        }
    },
    "model_2": {"MAE": {"note": "no time series"}},
}
THRESHOLDS = {"MAE": {"good": 0.0, "bad": 0.1, "unit": "g/cm^3"}}


def rows():
    """
    Get table rows.

    Returns
    -------
    list[dict]
        Rows for three models.
    """
    return [
        {"MLIP": "model_1", "id": "model_1", "MAE": 0.015},
        {"MLIP": "model_2", "id": "model_2", "MAE": 0.025},
        {"MLIP": "model_3", "id": "model_3", "MAE": 0.035},
    ]


def test_uncertainty_tooltips():
    """Test confidence intervals are added to the tooltips of metric cells."""
    table_rows = rows()
    tooltips = add_uncertainty_tooltips(
        table_rows,
        [{"MLIP": "config"} for _ in table_rows],
        UNCERTAINTIES,
        THRESHOLDS,
    )

    assert tooltips[0]["MLIP"] == "config"
    assert tooltips[0]["MAE"]["value"].split("  \n") == [
        "95 % CI (sampling): [0.0120, 0.0180] g/cm^3",
        "95 % CI (incl. reference): [0.0100, 0.0200] g/cm^3",
        "Min. effective samples: 83",
        "Method: autocorrelation (Chodera)",
    ]
    assert tooltips[1]["MAE"]["value"] == "Note: no time series"
    assert "MAE" not in tooltips[2]


def test_uncertainty_tooltips_filtered_rows():
    """Test tooltips follow rows by model ID after filtering."""
    table_rows = filter_rows_by_models(rows(), ["model_2", "model_1"])
    table_rows = [row for row in table_rows if row["id"] != "model_1"]
    tooltips = add_uncertainty_tooltips(
        table_rows, [{} for _ in table_rows], UNCERTAINTIES, THRESHOLDS
    )

    assert [row["id"] for row in table_rows] == ["model_2"]
    assert tooltips[0]["MAE"]["value"] == "Note: no time series"


def test_uncertainty_tooltips_hidden():
    """Test tooltips are omitted for normalised views and filtered cells."""
    table_rows = rows()
    tooltips = add_uncertainty_tooltips(
        table_rows,
        [{} for _ in table_rows],
        UNCERTAINTIES,
        THRESHOLDS,
        show_normalized=True,
    )
    assert tooltips == [{}, {}, {}]

    # Cleared by the element filter, so the saved CI no longer applies
    table_rows[0]["MAE"] = None
    # Notes explain why a metric is missing, so are still shown
    table_rows[1]["MAE"] = "NaN"
    tooltips = add_uncertainty_tooltips(
        table_rows, [{} for _ in table_rows], UNCERTAINTIES, THRESHOLDS
    )
    assert tooltips == [
        {},
        {"MAE": {"type": "markdown", "value": "Note: no time series"}},
        {},
    ]
