"""Tests for the parity symmetry benchmark calculation and analysis."""

from __future__ import annotations

from collections.abc import Iterator
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write
import numpy as np
import pytest
from test_permutation_symmetry import PairSpringCalculator

from ml_peg.calcs.physicality.parity_symmetry import (
    calc_parity_symmetry as calc_module,
)
from ml_peg.calcs.physicality.parity_symmetry.calc_parity_symmetry import (
    RATTLE_SEED,
    RATTLE_STDEV,
    STRUCTURES,
    evaluate,
    rattled,
)
from ml_peg.models.mock import MockErrorCalculator

ANALYSIS_MODULE = "ml_peg.analysis.physicality.parity_symmetry.analyse_parity_symmetry"


class CubicFieldCalculator(Calculator):
    """
    Toy calculator that breaks parity by construction.

    The energy is odd in the positions, E = sum_i x_i^3, so inversion gives
    E -> -E, and the forces, F_i = (-3 * x_i^2, 0, 0), are even and do not
    change sign. The expected changes are therefore |ΔE| = 2|E| / N and
    max |F_inv + F_orig| = max |2 F|. The energy is a sum over atoms, so it
    remains permutation invariant and only parity is broken. Periodic images
    are ignored.
    """

    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: list[str] | None = None,
        system_changes: list[str] = all_changes,
        **kwargs,
    ) -> None:
        """
        Calculate cubic-field energy and analytic forces.

        Parameters
        ----------
        atoms
            Atoms object to calculate properties for.
        properties
            List of properties to calculate.
        system_changes
            List of system changes to consider for calculation.
        **kwargs
            Any additional keyword arguments.
        """
        super().calculate(atoms, properties, system_changes, **kwargs)
        positions = self.atoms.positions
        energy = np.sum(positions[:, 0] ** 3)
        forces = np.zeros((len(positions), 3))
        forces[:, 0] = -3.0 * positions[:, 0] ** 2
        self.results["energy"] = energy
        self.results["forces"] = forces


def calc_outputs_path(tmp_path: Path) -> Path:
    """
    Get the calculation outputs directory used by the analysis module.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.

    Returns
    -------
    Path
        Outputs directory, matching the analysis module's `CALC_PATH` once
        `ml_peg.calcs.CALCS_ROOT` is patched to `tmp_path / "calcs"`.
    """
    return tmp_path / "calcs" / "physicality" / "parity_symmetry" / "outputs"


def run_calc(
    monkeypatch: pytest.MonkeyPatch, out_path: Path, model_name: str, calc: Calculator
) -> None:
    """
    Run the benchmark calculation for one toy model, writing to `out_path`.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    out_path
        Directory to use as the calculation module's `OUT_PATH`.
    model_name
        Name to write outputs under.
    calc
        Toy calculator to evaluate structures with.
    """
    monkeypatch.setattr(calc_module, "OUT_PATH", out_path)
    # The real calculation loop, called with a fake model whose calculator is
    # the toy one
    calc_module.test_parity_symmetry(
        (model_name, SimpleNamespace(get_calculator=lambda precision: calc))
    )

@pytest.fixture
def analysis_module(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[ModuleType]:
    """
    Import the analysis module with its input and output paths in `tmp_path`.

    The analysis module calls `get_struct_info` at import, which raises if
    `outputs/mock` does not exist, so the real outputs must not be relied on.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    monkeypatch
        Pytest monkeypatch fixture.

    Returns
    -------
    ModuleType
        Freshly imported analysis module, reading from and writing to
        `tmp_path`.
    """
    monkeypatch.setattr("ml_peg.calcs.CALCS_ROOT", tmp_path / "calcs")
    monkeypatch.setattr("ml_peg.app.APP_ROOT", tmp_path / "app")

    # get_struct_info reads outputs/mock at import, so the equivariant toy model
    # must be written under the name "mock"
    run_calc(monkeypatch, calc_outputs_path(tmp_path), "mock", PairSpringCalculator())
    run_calc(monkeypatch, calc_outputs_path(tmp_path), "cubic_field", CubicFieldCalculator())
    monkeypatch.delitem(sys.modules, ANALYSIS_MODULE, raising=False)
    module = importlib.import_module(ANALYSIS_MODULE)
    monkeypatch.setattr(module, "MODELS", ["mock", "cubic_field"])
    yield module
    # Clean up after the test
    sys.modules.pop(ANALYSIS_MODULE, None)

def test_rattled_does_not_modify_input():
    """Test rattled returns a displaced copy and leaves the input unchanged."""
    water = molecule("H2O")
    water_copy = molecule("H2O")
    rattled_water = rattled(water)
    assert not np.allclose(water.positions, rattled_water.positions)
    assert np.allclose(water.positions, water_copy.positions)


def test_rattled_is_deterministic():
    """Test rattling the same structure twice gives identical positions."""
    water = molecule("H2O")
    rattled_water_1 = rattled(water)
    rattled_water_2 = rattled(water)
    assert np.allclose(rattled_water_1.positions, rattled_water_2.positions)


def test_rattled_structures_are_chiral():
    """Test rattled structures are not mapped onto themselves by inversion."""
    methane = molecule("CH4")
    rattled_methane = rattled(methane)
    inverted_rattled_methane = rattled_methane.copy()
    inverted_rattled_methane.positions *= -1.0
    results = evaluate(CubicFieldCalculator(), rattled_methane, inverted=False)
    results_inverted = evaluate(CubicFieldCalculator(), inverted_rattled_methane, inverted=True)
    assert np.isclose(results.info["energy"], -results_inverted.info["energy"])

def test_evaluate_stores_results():
    """Test evaluate stores energy, forces and the inverted flag, and detaches calc."""
    # Use PairSpringCalculator on a small molecule
    water = molecule("H2O")
    results = evaluate(PairSpringCalculator(), water, inverted=False)
    assert results is not None
    assert np.isfinite(results.info["energy"])
    assert results.arrays["forces"].shape == (3, 3)
    assert not results.info["inverted"]
    assert results.calc is None


def test_evaluate_failure_stores_nan():
    """Test a failed evaluation warns and stores NaN energy and forces."""
    water = molecule("H2O")
    with pytest.warns(RuntimeWarning):
        results = evaluate(MockErrorCalculator(), water, inverted=False)
    assert np.isnan(results.info["energy"])
    assert np.all(np.isnan(results.arrays["forces"]))


def test_equivariant_calculator_gives_zero_deltas(analysis_module):
    """Test a parity-equivariant calculator gives zero energy and force changes."""
    # Run PairSpringCalculator; every get_deltas entry should be ~0. This also
    # checks the F_inv + F_orig sign convention
    for struct_name in STRUCTURES:
        deltas = analysis_module.get_deltas("mock", struct_name)
        assert deltas is not None
        delta_energy, delta_forces = deltas
        assert delta_energy == pytest.approx(0.0, abs=1e-6)
        assert delta_forces == pytest.approx(0.0, abs=1e-6)


def test_non_equivariant_calculator_gives_expected_deltas(analysis_module):
    """Test a parity-breaking calculator gives the analytically expected changes."""
    # Run CubicFieldCalculator on a small molecule and compare get_deltas to
    # 2|E| / N (meV/atom) and max |2F| (meV/Å)
    for struct_name, struct in STRUCTURES.items():
        deltas = analysis_module.get_deltas("cubic_field", struct_name)
        assert deltas is not None
        delta_energy, delta_forces = deltas
        calc = CubicFieldCalculator()
        expected_energy = 2.0 * abs(calc.get_potential_energy(struct)) / len(struct)
        expected_forces = np.max(np.abs(2.0 * calc.get_forces(struct)))
        assert delta_energy == pytest.approx(expected_energy * 1e3, rel=1e-6)
        assert delta_forces == pytest.approx(expected_forces * 1e3, rel=1e-6)


def test_get_deltas_missing_file_returns_none(analysis_module):
    """Test a structure without an output file returns None."""
    deltas = analysis_module.get_deltas("mock", "kürtőskalács")
    assert deltas is None


def test_get_deltas_unreadable_file_returns_nan(analysis_module, tmp_path):
    """Test an unreadable output file gives a failed (NaN) datum, not an error."""
    bad_path = calc_outputs_path(tmp_path) / "mock" / "truncated.xyz"
    bad_path.write_text("3\nnot a valid extxyz comment line\nH 0.0\n")
    with pytest.warns(RuntimeWarning):
        deltas = analysis_module.get_deltas("mock", "truncated")
    assert deltas is not None
    assert all(np.isnan(d) for d in deltas)

def test_missing_model_gives_no_scores(analysis_module, monkeypatch):
    """Test a model with no outputs is given None for every metric."""
    monkeypatch.setattr(analysis_module, "MODELS", ["mock", "cubic_field", "missing_model"])
    deltas = analysis_module.deltas_by_structure.__wrapped__()
    scores = analysis_module.parity_metrics.__wrapped__(deltas)
    for metric in scores.values():
        assert metric["missing_model"] is None
        assert np.isclose(metric["mock"], 0.0)
        assert not np.isclose(metric["cubic_field"], 0.0)
