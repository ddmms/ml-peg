"""Test Al3Ti calculation, analysis, and app behavior."""

from __future__ import annotations

from importlib import import_module
from inspect import unwrap
import json
from pathlib import Path
from unittest.mock import patch

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import write
import numpy as np
from plotly.io import read_json
import pytest

from ml_peg.analysis.alloy_metallurgy.al3ti_polymorphs import (
    analyse_al3ti_polymorphs as al3ti_analysis,
)
from ml_peg.app.alloy_metallurgy.al3ti_polymorphs import (
    app_al3ti_polymorphs as al3ti_app,
)


@pytest.fixture
def benchmark():
    """Import the calculation module without loading configured models."""
    with patch("ml_peg.models.get_models.load_models", return_value={}):
        return import_module(
            "ml_peg.calcs.alloy_metallurgy.al3ti_polymorphs.calc_al3ti_polymorphs"
        )


def test_old_outputs_removed_before_calculator_setup(benchmark, monkeypatch, tmp_path):
    """A calculator setup failure must not leave previous results visible."""

    class FailingModel:
        def get_calculator(self, *, precision):
            raise RuntimeError("calculator setup failed")

    monkeypatch.setattr(benchmark, "OUT_PATH", tmp_path)
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    for name in benchmark.STRUCTURE_NAMES:
        (output_dir / f"{name}.xyz").write_text("stale")

    with pytest.raises(RuntimeError, match="calculator setup failed"):
        benchmark.test_al3ti_polymorphs(("model", FailingModel()), tmp_path)

    assert not list(output_dir.glob("*.xyz"))


def test_failed_write_does_not_publish_partial_output(benchmark, monkeypatch, tmp_path):
    """A write error leaves neither a final nor temporary XYZ file."""

    def fail_after_partial_write(path: Path, *_args, **_kwargs):
        path.write_text("partial")
        raise OSError("write failed")

    monkeypatch.setattr(benchmark, "write", fail_after_partial_write)

    with pytest.warns(UserWarning, match="Error writing structure D022"):
        assert not benchmark.write_structure("D022", object(), tmp_path)

    assert not (tmp_path / "D022.xyz").exists()
    assert not (tmp_path / "D022.xyz.tmp").exists()


def test_structure_grid_uses_selected_model_and_handles_missing(monkeypatch, tmp_path):
    """Each viewer uses the selected model and failed structures stay separate."""
    monkeypatch.setattr(al3ti_app, "DATA_PATH", tmp_path)
    for model, name in (("model-a", "D022"), ("model-b", "D023")):
        model_dir = tmp_path / model
        model_dir.mkdir()
        write(model_dir / f"{name}.xyz", Atoms("Al3Ti", cell=(2, 2, 4)))
    (tmp_path / "model-b" / "L12.xyz").write_text("invalid structure")

    cards = al3ti_app.structure_grid("model-b").children

    assert cards[0].children[1].children == "Structure unavailable."
    assert "/model-b/D023.xyz" in cards[1].children[1].srcDoc
    assert cards[2].children[1].children == "Structure unavailable."


def test_parity_highlight_keeps_guide_and_selected_trace_last(monkeypatch):
    """The selected model stands out without hiding the parity guide."""
    monkeypatch.setattr(al3ti_app, "MODELS", ["model-a", "model-b"])
    figure = {
        "data": [
            {"name": "model-b", "marker": {"color": "blue"}},
            {"name": "model-a"},
            {"name": "__parity_line__"},
        ]
    }

    highlighted = al3ti_app.highlight_model(figure, "model-b")

    assert [trace["name"] for trace in highlighted["data"]] == [
        "model-a",
        "__parity_line__",
        "model-b",
    ]
    assert highlighted["data"][0]["opacity"] == 0.25
    assert "opacity" not in highlighted["data"][1]
    assert highlighted["data"][2]["marker"] == {
        "color": "blue",
        "size": 13,
        "line": {"color": "#222222", "width": 2},
    }
    assert "opacity" not in figure["data"][0]
    assert al3ti_app.highlight_model(figure, "missing") == figure


@pytest.mark.parametrize("failed_structure", ["Al_fcc", "Ti_hcp", "D023"])
def test_energy_failure_keeps_other_models_scoreable(
    monkeypatch, tmp_path, failed_structure
):
    """An unreadable energy leaves NaN and does not stop later models."""

    class EnergyAtoms:
        def __init__(self, energy: float | None = -4.0):
            self.energy = energy

        def __len__(self):
            return 4

        def get_potential_energy(self):
            if self.energy is None:
                raise RuntimeError("energy unavailable")
            return self.energy

    monkeypatch.setattr(al3ti_analysis, "MODELS", ["bad", "good"])
    monkeypatch.setattr(al3ti_analysis, "CALC_PATH", tmp_path)
    for model in ("bad", "good"):
        (tmp_path / model).mkdir()

    def load_reference(structure, model_dir):
        if model_dir.name == "bad" and structure == failed_structure:
            return EnergyAtoms(None)
        return EnergyAtoms()

    monkeypatch.setattr(al3ti_analysis, "load_structure", load_reference)
    relaxed = {
        model: [EnergyAtoms(-5.0) for _ in al3ti_analysis.POLYMORPHS]
        for model in ("bad", "good")
    }
    if failed_structure == "D023":
        relaxed["bad"][1] = EnergyAtoms(None)
    references = {
        name: {"formation_energy_0k_ev_atom": {"value": -0.25}}
        for name in al3ti_analysis.POLYMORPHS
    }

    with pytest.warns(UserWarning, match=f"model bad, structure {failed_structure}"):
        results = unwrap(al3ti_analysis.formation_energies)(references, relaxed)

    assert np.isnan(results["bad"][1])
    assert results["good"] == pytest.approx([-0.25] * len(al3ti_analysis.POLYMORPHS))
    if failed_structure in ("Al_fcc", "Ti_hcp"):
        assert np.isnan(results["bad"]).all()
    else:
        assert np.isfinite([results["bad"][0], results["bad"][2]]).all()


def test_formation_energies_use_al3ti_stoichiometry_and_polymorph_order(
    monkeypatch, tmp_path
):
    """Formation energies use three Al atoms per Ti and retain input order."""

    def atoms_with_energy(symbols, energy):
        atoms = Atoms(symbols)
        atoms.calc = SinglePointCalculator(atoms, energy=energy)
        return atoms

    monkeypatch.setattr(al3ti_analysis, "MODELS", ["model"])
    monkeypatch.setattr(al3ti_analysis, "CALC_PATH", tmp_path)
    (tmp_path / "model").mkdir()
    elemental = {
        "Al_fcc": atoms_with_energy("Al4", -8.0),
        "Ti_hcp": atoms_with_energy("Ti4", -16.0),
    }
    monkeypatch.setattr(
        al3ti_analysis, "load_structure", lambda name, _path: elemental[name]
    )
    relaxed = {
        "model": [
            atoms_with_energy("Al3Ti", energy) for energy in (-12.0, -11.2, -10.4)
        ]
    }
    references = {
        name: {"formation_energy_0k_ev_atom": {"value": value}}
        for name, value in zip(
            al3ti_analysis.POLYMORPHS, (-0.51, -0.31, -0.11), strict=True
        )
    }

    results = unwrap(al3ti_analysis.formation_energies)(references, relaxed)

    assert results["model"] == pytest.approx([-0.5, -0.3, -0.1])
    assert results["ref"] == pytest.approx([-0.51, -0.31, -0.11])


@pytest.mark.parametrize(
    ("prediction", "expected"),
    [
        ([1.0, 2.0, 3.0], 100.0),
        ([2.0, 1.0, 3.0], 200.0 / 3.0),
        ([1.0, 1.0, 3.0], 200.0 / 3.0),
        ([3.0, 2.0, 1.0], 0.0),
        ([1.0, np.nan, 3.0], np.nan),
        ([1.0, 2.0], np.nan),
    ],
)
def test_pairwise_order_accuracy(prediction, expected):
    """Ordering counts matching pairs and rejects incomplete predictions."""
    accuracy = al3ti_analysis.pairwise_order_accuracy([1.0, 2.0, 3.0], prediction)

    if np.isnan(expected):
        assert np.isnan(accuracy)
    else:
        assert accuracy == pytest.approx(expected)


def test_ordering_plots_show_agreements_and_unavailable_data(monkeypatch, tmp_path):
    """Ordering plots use DFT order and show missing energies without a score."""
    monkeypatch.setattr(al3ti_analysis, "MODELS", ["inverted", "missing"])
    monkeypatch.setattr(al3ti_analysis, "OUT_PATH", tmp_path)
    energies = {
        "ref": [2.0, 1.0, 3.0],
        "inverted": [1.0, 2.0, 3.0],
        "missing": [2.0, np.nan, 3.0],
    }

    al3ti_analysis.write_formation_energy_ordering_plots(energies)

    inverted = json.loads(
        (tmp_path / "inverted" / "figure_formation_energy_ordering.json").read_text()
    )
    missing = json.loads(
        (tmp_path / "missing" / "figure_formation_energy_ordering.json").read_text()
    )
    assert inverted["data"][0]["x"] == ["D023", "D022", "L12"]
    assert inverted["data"][1]["marker"]["symbol"] == [
        "diamond",
        "diamond",
        "circle",
    ]
    assert "66.7% (2/3 pairs)" in inverted["layout"]["title"]["text"]
    assert missing["data"][1]["marker"]["color"] == "#888888"
    assert "unavailable" in missing["layout"]["title"]["text"]


def test_structure_copy_clears_stale_and_unreadable_assets(monkeypatch, tmp_path):
    """Only current, readable calculation results reach the app."""
    calc_path = tmp_path / "calcs"
    app_path = tmp_path / "app"
    source_dir = calc_path / "model"
    target_dir = app_path / "model"
    source_dir.mkdir(parents=True)
    target_dir.mkdir(parents=True)
    write(source_dir / "D022.xyz", Atoms("Al3Ti", cell=(2, 2, 4)))
    (source_dir / "D023.xyz").write_text("invalid structure")
    for name in al3ti_analysis.POLYMORPHS:
        (target_dir / f"{name}.xyz").write_text("stale")
    monkeypatch.setattr(al3ti_analysis, "MODELS", ["model"])
    monkeypatch.setattr(al3ti_analysis, "CALC_PATH", calc_path)
    monkeypatch.setattr(al3ti_analysis, "OUT_PATH", app_path)

    with pytest.warns(UserWarning, match="Unable to copy structure .*D023.xyz"):
        al3ti_analysis.copy_structures_to_app_data()

    assert (target_dir / "D022.xyz").is_file()
    assert not (target_dir / "D023.xyz").exists()
    assert not (target_dir / "L12.xyz").exists()
    assert not list(target_dir.glob("*.tmp"))


def test_geometry_metrics_weight_polymorphs_and_parameters():
    """Length errors stay visible when opposing changes nearly cancel in volume."""
    references = {
        "D022": {
            "lattice_a_angstrom": {"value": 2.0},
            "lattice_c_angstrom": {"value": 4.0},
            "volume_per_atom_angstrom3": {"value": 4.0},
        },
        "D023": {
            "lattice_a_angstrom": {"value": 3.0},
            "lattice_c_angstrom": {"value": 6.0},
            "volume_per_atom_angstrom3": {"value": 13.5},
        },
        "L12": {
            "lattice_a_angstrom": {"value": 4.0},
            "volume_per_atom_angstrom3": {"value": 16.0},
        },
    }
    relaxed = {
        "model": [
            Atoms("Al3Ti", cell=(2.2, 1.8, 4.0)),
            Atoms("Al3Ti", cell=(3.0, 3.0, 6.6)),
            Atoms("Al3Ti", cell=(4.4, 4.0, 4.0)),
        ]
    }

    volumes = unwrap(al3ti_analysis.volumes_per_atom)(references, relaxed)
    volume_errors = unwrap(al3ti_analysis.volume_errors)(volumes)
    lattice_errors = unwrap(al3ti_analysis.lattice_parameter_errors)(
        references, relaxed
    )
    lengths = unwrap(al3ti_analysis.lattice_parameters)(references, relaxed)

    assert volumes["ref"] == pytest.approx([4.0, 13.5, 16.0])
    assert volumes["model"] == pytest.approx([3.96, 14.85, 17.6])
    assert volume_errors["model"] == pytest.approx((0.04 + 1.35 + 1.6) / 3)
    assert lattice_errors["model"] == pytest.approx((5.0 + 5.0 + 10.0 / 3) / 3)
    assert lengths["ref"] == pytest.approx([2, 2, 4, 3, 3, 6, 4, 4, 4])
    assert lengths["model"] == pytest.approx([2.2, 1.8, 4, 3, 3, 6.6, 4.4, 4, 4])


def test_lattice_plot_separates_signed_errors(monkeypatch, tmp_path):
    """Each polymorph displays positive and negative length deviations."""
    monkeypatch.setattr(al3ti_analysis, "OUT_PATH", tmp_path)
    al3ti_analysis.write_lattice_parameter_plot(
        {
            "ref": [2, 2, 4, 3, 3, 6, 4, 4, 4],
            "model": [2.2, 1.8, 4, 3, 3, 6.6, 4.4, 4, 4],
            "symmetric": [2, 2, 4, 3, 3, 6, 4, 4, 4],
            "partial": [np.nan] * 3 + [3, 3, 6] + [np.nan] * 3,
            "missing": [np.nan] * 9,
        }
    )

    figure = read_json(tmp_path / "figure_lattice_parameters.json")
    assert [trace.name for trace in figure.data] == (
        ["model"] * 3 + ["symmetric"] * 3 + ["partial"]
    )
    assert [trace.xaxis for trace in figure.data[:3]] == ["x", "x2", "x3"]
    assert list(figure.data[0].y) == pytest.approx([10, -10, 0])
    assert list(figure.data[1].x) == ["a", "c"]
    assert list(figure.data[1].y) == pytest.approx([0, 10])
    assert list(figure.data[2].x) == ["a", "b", "c"]
    assert [list(trace.x) for trace in figure.data[3:6]] == [
        ["a", "c"],
        ["a", "c"],
        ["a"],
    ]
    assert list(figure.layout.xaxis3.categoryarray) == ["a", "b", "c"]
    assert figure.data[-1].xaxis == "x2"
    assert figure.data[-1].showlegend
    assert figure.data[0].customdata[0][2] == pytest.approx(0.2)


def test_missing_polymorph_makes_geometry_scores_nan():
    """A missing relaxed structure cannot yield a complete geometry score."""
    references = {
        name: {
            "lattice_a_angstrom": {"value": 2.0},
            "lattice_c_angstrom": {"value": 4.0},
            "volume_per_atom_angstrom3": {"value": 4.0},
        }
        for name in al3ti_analysis.POLYMORPHS
    }
    relaxed = {"model": [Atoms("Al3Ti", cell=(2.0, 2.0, 4.0)), None, None]}

    volumes = unwrap(al3ti_analysis.volumes_per_atom)(references, relaxed)
    volume_errors = unwrap(al3ti_analysis.volume_errors)(volumes)
    lattice_errors = unwrap(al3ti_analysis.lattice_parameter_errors)(
        references, relaxed
    )

    assert np.isnan(volumes["model"][1:]).all()
    assert np.isnan(volume_errors["model"])
    assert np.isnan(lattice_errors["model"])
