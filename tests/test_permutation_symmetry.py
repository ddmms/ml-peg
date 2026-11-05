"""Tests for the permutation symmetry benchmark calculation and analysis."""

from __future__ import annotations

from collections.abc import Iterator
import importlib
from itertools import combinations
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from ase import Atoms
from ase.build import molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.io import write
import numpy as np
import pytest

from ml_peg.calcs.physicality.permutation_symmetry import (
    calc_permutation_symmetry as calc_module,
)
from ml_peg.calcs.physicality.permutation_symmetry.calc_permutation_symmetry import (  # noqa: E501
    MAX_PERMUTATIONS,
    STRUCTURES,
    evaluate,
    generate_permutations,
)
from ml_peg.models.mock import MockErrorCalculator

ANALYSIS_MODULE = (
    "ml_peg.analysis.physicality.permutation_symmetry.analyse_permutation_symmetry"
)


class PairSpringCalculator(Calculator):
    """
    Toy calculator that is permutation invariant by construction.

    The energy is a sum of harmonic springs over all atom pairs,
    E = sum_{i<j} 0.5 * (r_ij - 1)^2, so it depends only on the set of
    interatomic distances. Relabelling identical atoms therefore leaves the
    energy unchanged and permutes the forces with the atoms. Periodic images
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
        Calculate pair-spring energy and analytic forces.

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
        energy = 0.0
        forces = np.zeros((len(positions), 3))
        for i, j in combinations(range(len(positions)), 2):
            r_ij = positions[j] - positions[i]
            r = np.linalg.norm(r_ij)
            force_vector = (r - 1.0) * r_ij / r
            energy += 0.5 * (r - 1.0) ** 2
            forces[i] += force_vector
            forces[j] -= force_vector
        self.results["energy"] = energy
        self.results["forces"] = forces


class IndexWeightedCalculator(Calculator):
    """
    Toy calculator that breaks permutation invariance by construction.

    Each atom is tethered to the origin with a spring whose stiffness depends
    on its index in the input, E = sum_i 0.5 * (i + 1) * |r_i|^2, so
    F_i = -(i + 1) * r_i. Swapping two identical atoms changes the energy
    unless they are equidistant from the origin. Periodic images are ignored.
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
        Calculate index-weighted energy and analytic forces.

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
        super().calculate(atoms, properties, system_changes)
        positions = self.atoms.positions
        weights = np.arange(1, len(positions) + 1)[:, None]
        self.results["energy"] = float(0.5 * np.sum(weights * positions**2))
        self.results["forces"] = -weights * positions


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
    return tmp_path / "calcs" / "physicality" / "permutation_symmetry" / "outputs"


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
    calc_module.test_permutation_symmetry(
        (model_name, SimpleNamespace(get_calculator=lambda precision: calc))
    )


@pytest.fixture
def analysis_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[ModuleType]:
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

    Yields
    ------
    ModuleType
        Freshly imported analysis module, reading from and writing to
        `tmp_path`.
    """
    # The analysis module builds its paths from these at import time
    monkeypatch.setattr("ml_peg.calcs.CALCS_ROOT", tmp_path / "calcs")
    monkeypatch.setattr("ml_peg.app.APP_ROOT", tmp_path / "app")

    # Outputs for the "mock" model are required by get_struct_info at import
    run_calc(monkeypatch, calc_outputs_path(tmp_path), "mock", PairSpringCalculator())

    # Force a fresh import so the patched paths are picked up
    monkeypatch.delitem(sys.modules, ANALYSIS_MODULE, raising=False)
    module = importlib.import_module(ANALYSIS_MODULE)
    monkeypatch.setattr(module, "MODELS", ["mock"])
    yield module
    # Do not leave the module (with tmp_path paths) cached for other tests
    sys.modules.pop(ANALYSIS_MODULE, None)


def test_generate_permutations():
    """Test permutations are distinct, non-identity, and only swap like atoms."""
    for struct in STRUCTURES.values():
        numbers = struct.get_atomic_numbers()
        perms = generate_permutations(struct)

        assert 0 < len(perms) <= MAX_PERMUTATIONS
        assert len({tuple(p) for p in perms}) == len(perms)
        for perm in perms:
            assert sorted(perm) == list(range(len(struct)))
            assert not np.array_equal(perm, np.arange(len(struct)))
            np.testing.assert_array_equal(numbers[perm], numbers)


def test_generate_permutations_enumerates_small_cases():
    """Test all permutations are returned when there are few enough."""
    # H2O: only the two hydrogens can be swapped
    water = molecule("H2O")
    perms = generate_permutations(water)
    assert len(perms) == 1
    np.testing.assert_array_equal(perms[0], [0, 2, 1])


def test_generate_permutations_is_deterministic():
    """Test random sampling with the fixed seed gives the same permutations."""
    # C6H6 has 6! * 6! - 1 permutations, so they are sampled at random
    benzene = STRUCTURES["C6H6"]
    first = generate_permutations(benzene)
    second = generate_permutations(benzene)
    assert len(first) == MAX_PERMUTATIONS
    for p1, p2 in zip(first, second, strict=True):
        np.testing.assert_array_equal(p1, p2)


def test_evaluate_stores_results():
    """Test evaluate stores energy, forces and permutation, and detaches calc."""
    water = molecule("H2O")
    perm = np.array([0, 2, 1])
    result = evaluate(
        PairSpringCalculator(), water, permutation_index=3, permutation=perm
    )

    assert np.isfinite(result.info["energy"])
    assert result.info["permutation_index"] == 3
    assert result.arrays["forces"].shape == (3, 3)
    np.testing.assert_array_equal(result.arrays["permutation"], perm)
    assert result.calc is None


def test_evaluate_failure_stores_nan():
    """Test a failed evaluation warns and stores NaN energy and forces."""
    water = molecule("H2O")
    with pytest.warns(RuntimeWarning):
        result = evaluate(
            MockErrorCalculator(), water, permutation_index=0, permutation=np.arange(3)
        )
    assert np.isnan(result.info["energy"])
    assert np.isnan(result.arrays["forces"]).all()


def test_invariant_calculator_gives_zero_deltas(analysis_module):
    """Test a permutation-invariant calculator gives zero energy and force changes."""
    # "mock" was run with PairSpringCalculator by the fixture. This also checks
    # forces are mapped back to the original atom order correctly
    for struct_name in STRUCTURES:
        deltas = analysis_module.list_deltas("mock", struct_name)
        assert deltas is not None
        np.testing.assert_allclose(deltas, 0, atol=1e-6)


def test_non_invariant_calculator_gives_nonzero_deltas(
    analysis_module, tmp_path, monkeypatch
):
    """Test a calculator that depends on atom order gives non-zero changes."""
    run_calc(
        monkeypatch,
        calc_outputs_path(tmp_path),
        "index_weighted",
        IndexWeightedCalculator(),
    )
    mean_e, max_e, mean_f, max_f = analysis_module.list_deltas(
        "index_weighted", "CH3OH"
    )
    assert mean_e > 0
    assert max_e >= mean_e
    assert mean_f > 0
    assert max_f >= mean_f


def test_list_deltas_missing_file_returns_none(analysis_module):
    """Test a structure without an output file returns None."""
    assert analysis_module.list_deltas("mock", "not_a_structure") is None
    assert analysis_module.list_deltas("not_a_model", "H2O") is None


def test_list_deltas_missing_original_returns_nan(analysis_module):
    """Test an output file without the unpermuted frame is a failed (NaN) datum."""
    water = molecule("H2O")
    perm = np.array([0, 2, 1])
    permuted = evaluate(
        PairSpringCalculator(), water[perm], permutation_index=1, permutation=perm
    )
    out_dir = analysis_module.CALC_PATH / "partial"
    out_dir.mkdir(parents=True)
    write(out_dir / "H2O.xyz", [permuted])

    with pytest.warns(RuntimeWarning):
        deltas = analysis_module.list_deltas("partial", "H2O")
    assert all(np.isnan(d) for d in deltas)


def test_list_deltas_missing_permutation_returns_nan(analysis_module):
    """Test an output file missing a permuted frame is a failed (NaN) datum."""
    methane = molecule("CH4")
    original = evaluate(
        PairSpringCalculator(),
        methane.copy(),
        permutation_index=0,
        permutation=np.arange(len(methane)),
    )
    # Only the first of the expected permutations is written
    perm = generate_permutations(STRUCTURES["CH4"])[0]
    permuted = evaluate(
        PairSpringCalculator(), methane[perm], permutation_index=1, permutation=perm
    )
    out_dir = analysis_module.CALC_PATH / "partial"
    out_dir.mkdir(parents=True)
    write(out_dir / "CH4.xyz", [original, permuted])

    with pytest.warns(RuntimeWarning):
        deltas = analysis_module.list_deltas("partial", "CH4")
    assert all(np.isnan(d) for d in deltas)


def test_list_deltas_unreadable_file_returns_nan(analysis_module):
    """Test an unreadable output file is a failed (NaN) datum, not an error."""
    out_dir = analysis_module.CALC_PATH / "partial"
    out_dir.mkdir(parents=True)
    (out_dir / "H2O.xyz").write_text("3\nnot a valid extxyz comment line\nH 0.0\n")

    with pytest.warns(RuntimeWarning):
        deltas = analysis_module.list_deltas("partial", "H2O")
    assert all(np.isnan(d) for d in deltas)


def test_mean_max_requires_complete_set(analysis_module):
    """Test missing structures or NaN values give (None, None)."""
    n_structs = analysis_module.N_STRUCTURES
    complete = [float(i) for i in range(n_structs)]

    assert analysis_module._mean_max(complete) == (np.mean(complete), max(complete))
    assert analysis_module._mean_max(complete[:-1]) == (None, None)
    assert analysis_module._mean_max([*complete[:-1], np.nan]) == (None, None)


def test_missing_model_gives_no_scores(analysis_module, monkeypatch):
    """Test a model with no outputs is given None for every metric."""
    monkeypatch.setattr(analysis_module, "MODELS", ["mock", "missing_model"])

    # These are pytest fixtures, which cannot be called directly, so call the
    # functions they wrap
    deltas = analysis_module.deltas_by_structure.__wrapped__()
    scores = analysis_module.permutation_metrics.__wrapped__(deltas)

    for metric in scores.values():
        assert metric["missing_model"] is None
        assert metric["mock"] == pytest.approx(0, abs=1e-6)
