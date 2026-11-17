"""Tests for the NABP30 base pair benchmark calculation and analysis."""

from __future__ import annotations

from collections.abc import Iterator
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

from ase import Atoms, units
from ase.calculators.calculator import Calculator, all_changes
from ase.collections import s22
from ase.data.s22 import get_number_of_dimer_atoms
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.calcs.non_covalent_interactions.NABP30 import calc_NABP30 as calc_module
from ml_peg.calcs.non_covalent_interactions.NABP30.calc_NABP30 import (
    BASE_RATTLE_STDEV,
    BASE_STARTS,
    FRAMES,
    PLANE_TOL,
    RATTLE_SEED,
    constrain_to_plane,
    get_monomers,
    relax_base,
)
from ml_peg.models.mock import MockCalculator, MockErrorCalculator

ANALYSIS_MODULE = "ml_peg.analysis.non_covalent_interactions.NABP30.analyse_NABP30"

EV_TO_KCAL = units.mol / units.kcal
KCAL_TO_EV = 1 / EV_TO_KCAL

# Hydrogen-bonded pairs from S22 (ASE), as stand-ins for NABP30 base pairs
AT = "Adenine-thymine_Watson-Crick_complex"  # adenine (15 atoms), thymine (15)
PA = "2-pyridoxine_2-aminopyridine_complex"  # 2-pyridoxine (12), 2-aminopyridine (13)
N_A, N_B = get_number_of_dimer_atoms(AT)

# Toy base pairs: label -> (S22 structure, names of the two bases, planar,
# reference binding energy, reference deformation energy), energies in kcal/mol
TOY_PAIRS = {
    "AT_planar": (AT, ("A", "T"), True, -15.0, 1.5),
    "PA_nonplanar": (PA, ("P", "AP"), False, -16.0, 1.0),
}
FORMULAS = {"A": "C5H5N5", "T": "C5H6N2O2", "P": "C5H5NO", "AP": "C5H6N2"}

# Energy coefficient (eV) of SizeCalculator
SIZE_COEFF = 0.001


class SizeCalculator(Calculator):
    """
    Toy calculator whose energy depends only on the number of atoms.

    E = -SIZE_COEFF * N**2 and the forces are zero. A relaxation therefore stops
    where it starts (after any rattling or flattening), and the energy of a base pair
    relative to its bases is known exactly: E(AB) - E(A) - E(B) =
    -2 * SIZE_COEFF * N_A * N_B, at the input geometry and after relaxation.
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
        Calculate the energy and (zero) forces.

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
        self.results["energy"] = -SIZE_COEFF * len(self.atoms) ** 2
        self.results["forces"] = np.zeros((len(self.atoms), 3))


def make_pair(label: str, frag_charges: tuple[int, int] = (0, 0)) -> Atoms:
    """
    Build a toy NABP30 input structure from an S22 pair.

    Parameters
    ----------
    label
        Label of the base pair in `TOY_PAIRS`.
    frag_charges
        Charges of the two bases. Default is (0, 0).

    Returns
    -------
    Atoms
        Base pair with the arrays and info of the NABP30 input data.
    """
    s22_name, names, planar, ref_binding, ref_e_def = TOY_PAIRS[label]
    n_a, n_b = get_number_of_dimer_atoms(s22_name)
    pair = s22[s22_name]
    pair.arrays["frag"] = np.array([0] * n_a + [1] * n_b)
    pair.info = {
        "label": label,
        "charge": sum(frag_charges),
        "spin": 1,
        "frag_charges": list(frag_charges),
        "frag_names": list(names),
        "planar": planar,
        "ref_binding_energy_kcal": ref_binding,
        "ref_E_def_kcal": ref_e_def,
    }
    return pair


def size_energy(label: str) -> float:
    """
    Get the energy of a toy base pair relative to its bases with `SizeCalculator`.

    Parameters
    ----------
    label
        Label of the base pair in `TOY_PAIRS`.

    Returns
    -------
    float
        E(AB) - E(A) - E(B) in kcal/mol.
    """
    n_a, n_b = get_number_of_dimer_atoms(TOY_PAIRS[label][0])
    return -2 * SIZE_COEFF * n_a * n_b * EV_TO_KCAL


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
    return tmp_path / "calcs" / "non_covalent_interactions" / "NABP30" / "outputs"


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
        Toy calculator to calculate structures with.
    """
    monkeypatch.setattr(calc_module, "OUT_PATH", out_path)
    # The real calculation loop, called with a fake model whose calculator is the toy
    # one, without dispersion correction
    model = SimpleNamespace(
        get_calculator=lambda precision: calc, add_d3_calculator=lambda calc: calc
    )
    calc_module.test_nabp30((model_name, model))


@pytest.fixture
def calc_outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """
    Run the calculation for three toy models on the toy base pairs.

    The calculation reads the toy base pairs instead of downloading the data. The
    models are "mock" (zero energy and forces), "size" (`SizeCalculator`) and
    "error" (every calculation raises an error).

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    monkeypatch
        Pytest monkeypatch fixture.

    Returns
    -------
    Path
        Directory with one subdirectory of output files per model.
    """
    data_dir = tmp_path / "data"
    (data_dir / "NABP30").mkdir(parents=True)
    write(
        data_dir / "NABP30" / "dimers.extxyz", [make_pair(label) for label in TOY_PAIRS]
    )
    monkeypatch.setattr(calc_module, "download_s3_data", lambda **kwargs: data_dir)

    out_path = calc_outputs_path(tmp_path)
    run_calc(monkeypatch, out_path, "mock", MockCalculator())
    run_calc(monkeypatch, out_path, "size", SizeCalculator())
    with pytest.warns(UserWarning, match="failed"):
        run_calc(monkeypatch, out_path, "error", MockErrorCalculator())
    return out_path


@pytest.fixture
def analysis_module(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, calc_outputs: Path
) -> Iterator[ModuleType]:
    """
    Import the analysis module with its input and output paths in `tmp_path`.

    The analysis module calls `get_struct_info` at import, which reads
    `outputs/mock`, so the real outputs must not be relied on.

    Parameters
    ----------
    tmp_path
        Temporary directory provided by pytest.
    monkeypatch
        Pytest monkeypatch fixture.
    calc_outputs
        Outputs of the toy models.

    Yields
    ------
    ModuleType
        Freshly imported analysis module, reading from and writing to `tmp_path`.
    """
    monkeypatch.setattr("ml_peg.calcs.CALCS_ROOT", tmp_path / "calcs")
    monkeypatch.setattr("ml_peg.app.APP_ROOT", tmp_path / "app")
    monkeypatch.delitem(sys.modules, ANALYSIS_MODULE, raising=False)
    module = importlib.import_module(ANALYSIS_MODULE)
    monkeypatch.setattr(module, "MODELS", ["mock", "size", "error"])
    yield module
    # Clean up after the test
    sys.modules.pop(ANALYSIS_MODULE, None)


# Calculation


def test_get_monomers_splits_by_fragment():
    """Test a base pair is split into its bases, with their charges, unmoved."""
    pair = make_pair("AT_planar", frag_charges=(0, 1))
    base_a, base_b = get_monomers(pair)

    assert base_a.get_chemical_formula() == FORMULAS["A"]
    assert base_b.get_chemical_formula() == FORMULAS["T"]
    assert base_a.info == {"charge": 0, "spin": 1}
    assert base_b.info == {"charge": 1, "spin": 1}
    assert np.array_equal(base_a.positions, pair.positions[:N_A])
    assert np.array_equal(base_b.positions, pair.positions[N_A:])


def test_constrain_to_plane_keeps_methyl_hydrogens_free():
    """Test planar pairs are flattened, except atoms well out of the plane."""
    pair = make_pair("AT_planar")
    distances = pair.get_all_distances()
    constrain_to_plane(pair)

    fixed = pair.constraints[0].index
    free = np.setdiff1d(np.arange(len(pair)), fixed)
    # Only the two out-of-plane hydrogens of the thymine methyl group are free
    assert len(free) == 2
    assert {pair.get_chemical_symbols()[i] for i in free} == {"H"}
    assert np.all(pair.positions[fixed, 2] == 0.0)
    assert np.all(np.abs(pair.positions[free, 2]) > PLANE_TOL)
    # Only rotated and flattened: distances change by less than the flattening
    assert np.allclose(pair.get_all_distances(), distances, atol=0.05)


@pytest.mark.parametrize(
    "ok, expected",
    [
        ([True, False, True], -2.0),  # lowest start failed: next lowest
        ([True, True, True], -3.0),
        ([False, False, False], -3.0),  # all failed: lowest of all
    ],
)
def test_relax_base_keeps_lowest_good_start(
    monkeypatch: pytest.MonkeyPatch, ok: list[bool], expected: float
):
    """Test a base is relaxed from several starts and the lowest good one is kept."""
    energies = [-1.0, -3.0, -2.0]
    calls = []

    def fake_relax(atoms, calc, planar=False, rattle=0.0, seed=RATTLE_SEED):
        """Return the base with the energy and flags of the start `seed`."""
        calls.append((rattle, seed))
        start = seed - RATTLE_SEED
        relaxed = atoms.copy()
        relaxed.info.update(
            model_energy=energies[start],
            converged=ok[start],
            bonds_unchanged=True,
            n_steps=1,
        )
        return relaxed

    monkeypatch.setattr(calc_module, "relax", fake_relax)
    base = get_monomers(make_pair("AT_planar"))[0]
    best = relax_base(base, calc=None)

    # Each start is displaced, with its own random seed
    assert BASE_STARTS == 3
    assert calls == [(BASE_RATTLE_STDEV, RATTLE_SEED + start) for start in range(3)]
    assert best.info["model_energy"] == expected
    assert best.info["start_energies"] == energies


def test_calc_writes_six_tagged_frames(calc_outputs: Path):
    """Test each output file has the pair and its bases, unrelaxed and relaxed."""
    for label, (_, names, _, ref_binding, ref_e_def) in TOY_PAIRS.items():
        frames = read(calc_outputs / "size" / f"{label}.xyz", ":")

        assert [(f.info["geometry"], f.info["fragment"]) for f in frames] == FRAMES
        pair_formula = make_pair(label).get_chemical_formula()
        base_formulas = [FORMULAS[name] for name in names]
        assert [f.get_chemical_formula() for f in frames] == 2 * [
            pair_formula,
            *base_formulas,
        ]
        # References (eV) are stored on the first frame
        assert frames[0].info["ref_binding_energy"] == pytest.approx(
            ref_binding * KCAL_TO_EV
        )
        assert frames[0].info["ref_interaction_energy"] == pytest.approx(
            (ref_binding - ref_e_def) * KCAL_TO_EV
        )
        for frame in frames[3:]:
            assert frame.info["converged"]
            assert frame.info["bonds_unchanged"]
        assert len(frames[4].info["start_energies"]) == BASE_STARTS
        assert len(frames[5].info["start_energies"]) == BASE_STARTS


# Analysis


def test_energies_of_toy_model(analysis_module: ModuleType):
    """Test interaction and binding energies equal the known toy model value."""
    for label in TOY_PAIRS:
        frames = analysis_module.read_frames("size", label)
        expected = size_energy(label)
        assert analysis_module.get_interaction_energy(frames) == pytest.approx(expected)
        assert analysis_module.get_binding_energy(frames) == pytest.approx(expected)


def test_failed_relaxation_only_removes_binding_energy(
    analysis_module: ModuleType, tmp_path: Path
):
    """Test a failed relaxation gives NaN binding energy, but keeps the interaction."""
    path = calc_outputs_path(tmp_path) / "size" / "AT_planar.xyz"
    frames = read(path, ":")
    # The relaxation of the pair (fourth frame) did not converge
    frames[3].info["converged"] = False
    write(path, frames)

    frames = analysis_module.read_frames("size", "AT_planar")
    expected = size_energy("AT_planar")
    assert analysis_module.get_interaction_energy(frames) == pytest.approx(expected)
    assert np.isnan(analysis_module.get_binding_energy(frames))


def test_failed_model_gives_nan(analysis_module: ModuleType):
    """Test a model that cannot calculate anything gives NaN for both energies."""
    for label in TOY_PAIRS:
        frames = analysis_module.read_frames("error", label)
        assert frames is not None
        assert np.isnan(analysis_module.get_interaction_energy(frames))
        assert np.isnan(analysis_module.get_binding_energy(frames))


def test_metrics(
    analysis_module: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    """Test both MAEs of each model, including a failed and a missing model."""
    # Make the binding energy of one pair differ from its interaction energy:
    # lower the energy of its relaxed pair (fourth frame) by 0.1 eV
    path = calc_outputs_path(tmp_path) / "size" / "AT_planar.xyz"
    frames = read(path, ":")
    frames[3].info["model_energy"] -= 0.1
    write(path, frames)
    monkeypatch.setattr(analysis_module, "MODELS", ["mock", "size", "error", "none"])

    interaction = analysis_module.interaction_energies.__wrapped__()
    binding = analysis_module.binding_energies.__wrapped__()
    interaction_mae = analysis_module.interaction_mae.__wrapped__(interaction)
    binding_mae = analysis_module.binding_mae.__wrapped__(binding)
    metrics = analysis_module.metrics.__wrapped__(interaction_mae, binding_mae)

    assert metrics == {
        "Interaction energy MAE": interaction_mae,
        "Binding energy MAE": binding_mae,
    }
    labels = analysis_module.LABELS
    size_interaction = np.array([size_energy(label) for label in labels])
    size_binding = size_interaction - 0.1 * EV_TO_KCAL * (
        np.array(labels) == "AT_planar"
    )
    for maes, refs, size_energies in [
        (interaction_mae, analysis_module.REF_INTERACTION_ENERGIES, size_interaction),
        (binding_mae, analysis_module.REF_BINDING_ENERGIES, size_binding),
    ]:
        refs = np.array(refs)
        # The mock model gives zero energies, so zero interaction and binding energies
        assert maes["mock"] == pytest.approx(np.mean(np.abs(refs)))
        assert maes["size"] == pytest.approx(np.mean(np.abs(size_energies - refs)))
        assert np.isnan(maes["error"])
        assert np.isnan(maes["none"])
