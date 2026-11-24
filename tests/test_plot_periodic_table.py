"""Tests for the periodic-table heatmap decorator."""

from __future__ import annotations

import numpy as np
from plotly.io import read_json

from ml_peg.analysis.utils.decorators import plot_periodic_table


def element_labels(filename) -> dict[str, str]:
    """
    Get the label of each element in a saved periodic table.

    Parameters
    ----------
    filename
        Path to the saved figure.

    Returns
    -------
    dict[str, str]
        Label of each element, by symbol.
    """
    labels = read_json(filename).data[1].text
    return {label.split("<br>")[0]: label for label in labels}


def test_plot_periodic_table_labels(tmp_path):
    """Test that values are printed in the cells only when a format is given."""
    values = {"Fe": 12.345, "O": np.nan}

    plain = tmp_path / "plain.json"
    assert plot_periodic_table(filename=str(plain))(lambda: values)() == values
    labels = element_labels(plain)
    assert (labels["Fe"], labels["O"], labels["Cu"]) == ("Fe", "O", "Cu")

    printed = tmp_path / "printed.json"
    plot_periodic_table(filename=str(printed), value_format=".1f")(lambda: values)()
    labels = element_labels(printed)
    assert labels["Fe"] == "Fe<br><span style='font-size:9px'>12.3</span>"
    # Missing values and elements without data are not printed
    assert (labels["O"], labels["Cu"]) == ("O", "Cu")
