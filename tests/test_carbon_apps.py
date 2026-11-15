"""Test carbon app helpers against saved figures."""

from __future__ import annotations

import json

import pytest

from ml_peg.app.carbon.lattice_parameters import app_lattice_parameters
from ml_peg.app.carbon.surface_energies import app_surface_energies
from ml_peg.app.utils.load import read_plot


@pytest.mark.parametrize(
    ("app", "benchmark"),
    [
        (app_lattice_parameters, "lattice_parameters"),
        (app_surface_energies, "surface_energies"),
    ],
)
def test_mock_structs_from_saved_figure(tmp_path, app, benchmark):
    """Test structure paths are read from a saved figure's first trace."""
    figure = tmp_path / "figure.json"
    figure.write_text(json.dumps({"data": [{"customdata": [["Diamond"]]}]}))
    assert app.get_mock_structs(read_plot(figure)) == [
        f"/assets/carbon/{benchmark}/mock/Diamond.extxyz"
    ]
