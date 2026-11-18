"""Unit tests for per-column decimal formatting of table numbers."""

from __future__ import annotations

import pytest

from ml_peg.app.utils.utils import apply_column_decimals, column_decimals


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # Smallest value 0.355 needs 3 dp for 3 s.f., so 11 shows as 11.000.
        ([11.0, 0.355, 2.004], 3),
        ([0.0272, 0.5], 4),
        ([11.0, 15.2], 1),
        ([1.5, 9.9], 2),
        ([2760.0, 135.0], 0),
        # Capped so a near-zero value can't demand a dozen decimals.
        ([1e-6, 3.0], 4),
        # Zero, NaN strings and None carry no scale information.
        ([0.0, "NaN", None, 0.42], 3),
    ],
)
def test_column_decimals(values: list, expected: int) -> None:
    """Decimals keep the column's smallest non-zero value at 3 s.f."""
    assert column_decimals(values) == expected


def test_column_decimals_without_numbers_falls_back() -> None:
    """A column with no usable numbers still gets a sensible default."""
    assert column_decimals(["NaN", None, 0.0]) == 2


def test_apply_column_decimals_sets_fixed_format_on_numeric_columns() -> None:
    """Numeric columns get one fixed-point format each, others are untouched."""
    columns = [
        {"id": "MLIP", "name": "MLIP"},
        {"id": "MAE", "name": "MAE", "type": "numeric"},
        {"id": "Score", "name": "Score", "type": "numeric"},
    ]
    rows = [
        {"MLIP": "a", "MAE": 11.0, "Score": 0.5},
        {"MLIP": "b", "MAE": 0.355, "Score": 0.0272},
    ]

    result = apply_column_decimals(columns, rows)

    assert "format" not in result[0]
    specifiers = {
        col["id"]: col["format"].to_plotly_json()["specifier"] for col in result[1:]
    }
    assert specifiers == {"MAE": ".3f", "Score": ".4f"}
    assert "format" not in columns[1], "input columns were mutated"
