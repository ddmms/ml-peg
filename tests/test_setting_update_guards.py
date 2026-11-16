"""Weight defaults remain complete for stale session stores."""

from types import SimpleNamespace

from ml_peg.app.utils.utils import default_weight_store_data


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
