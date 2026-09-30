"""Unit tests for the MOF phonon benchmark helpers."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose
import pytest

from ml_peg.calcs.porous_materials.mof_phonons.mof_phonon_utils import (
    CM1_PER_MEV,
    INS_BACKENDS,
    diagonal_supercell_matrix,
    heat_capacity_per_gram,
    imaginary_mode_percentage,
    load_ins_reference,
    resolve_ins_backend,
    wasserstein_spectrum_distance,
)

GAS_CONSTANT = 8.31446261815324


@pytest.fixture(scope="module")
def argon_phonons():
    """
    Build a converged phonon calculation for fcc argon.

    A Lennard-Jones solid is used so the test needs no MLIP: it is dynamically
    stable, its classical heat capacity limit is known analytically, and it
    runs in a couple of seconds.

    Returns
    -------
    Phonopy
        Phonopy object with force constants, mesh and thermal properties.
    """
    from ase import Atoms
    from ase.build import bulk
    from ase.calculators.lj import LennardJones
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    atoms = bulk("Ar", "fcc", a=5.31, cubic=True)
    calc = LennardJones(sigma=3.405, epsilon=0.0104, rc=10.0)

    phonons = Phonopy(
        PhonopyAtoms(
            symbols=list(atoms.symbols),
            cell=atoms.cell.array,
            positions=atoms.positions,
        ),
        supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        primitive_matrix="auto",
    )
    phonons.generate_displacements(distance=0.01, is_plusminus=True)

    forces = []
    for supercell in phonons.supercells_with_displacements:
        image = Atoms(
            supercell.symbols,
            cell=supercell.cell,
            positions=supercell.positions,
        )
        image.calc = calc
        forces.append(image.get_forces())

    phonons.forces = np.array(forces)
    phonons.produce_force_constants(show_drift=False)
    phonons.run_mesh([8, 8, 8])
    phonons.run_thermal_properties(t_min=0, t_max=900, t_step=10, cutoff_frequency=0.0)
    return phonons


def test_diagonal_supercell_matrix_reaches_min_length() -> None:
    """Supercell repeats should be the fewest that reach the minimum length."""
    from ase import Atoms

    atoms = Atoms("H", cell=[5.0, 10.0, 21.0], pbc=True)
    matrix = diagonal_supercell_matrix(atoms, min_length=20.0)

    assert matrix == [[4, 0, 0], [0, 2, 0], [0, 0, 1]]
    lengths = np.array([5.0, 10.0, 21.0]) * np.array([4, 2, 1])
    assert (lengths >= 20.0).all()


def test_diagonal_supercell_matrix_never_shrinks() -> None:
    """A cell already larger than the minimum length is not expanded."""
    from ase import Atoms

    atoms = Atoms("H", cell=[37.1, 37.1, 30.0], pbc=True)
    assert diagonal_supercell_matrix(atoms, min_length=20.0) == [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
    ]


def test_heat_capacity_reaches_classical_limit(argon_phonons) -> None:
    """
    Heat capacity per gram should recover the Dulong-Petit limit.

    At high temperature Cv tends to 3*N*R per cell, so multiplying the
    per-gram value by the primitive cell mass must recover 3*N_prim*R. This
    pins down phonopy's per-primitive-cell convention, which the conversion
    to J/g/K depends on.
    """
    cv_per_gram = heat_capacity_per_gram(argon_phonons, 900.0)
    molar_mass = float(np.sum(argon_phonons.primitive.masses))
    expected = 3 * len(argon_phonons.primitive) * GAS_CONSTANT

    assert_allclose(cv_per_gram * molar_mass, expected, rtol=2e-3)


def test_heat_capacity_independent_of_cell_choice(argon_phonons) -> None:
    """
    The per-gram heat capacity must not depend on the cell phonopy was run on.

    Dividing by the mass of the same cell phonopy reports for cancels the
    choice of primitive versus conventional cell.
    """
    from ase import Atoms
    from ase.build import bulk
    from ase.calculators.lj import LennardJones
    from phonopy import Phonopy
    from phonopy.structure.atoms import PhonopyAtoms

    atoms = bulk("Ar", "fcc", a=5.31, cubic=True)
    calc = LennardJones(sigma=3.405, epsilon=0.0104, rc=10.0)
    phonons = Phonopy(
        PhonopyAtoms(
            symbols=list(atoms.symbols),
            cell=atoms.cell.array,
            positions=atoms.positions,
        ),
        supercell_matrix=[[2, 0, 0], [0, 2, 0], [0, 0, 2]],
        primitive_matrix=None,
    )
    phonons.generate_displacements(distance=0.01, is_plusminus=True)
    forces = []
    for supercell in phonons.supercells_with_displacements:
        image = Atoms(
            supercell.symbols,
            cell=supercell.cell,
            positions=supercell.positions,
        )
        image.calc = calc
        forces.append(image.get_forces())
    phonons.forces = np.array(forces)
    phonons.produce_force_constants(show_drift=False)
    phonons.run_mesh([8, 8, 8])
    phonons.run_thermal_properties(t_min=0, t_max=900, t_step=10, cutoff_frequency=0.0)

    assert len(phonons.primitive) == 4 * len(argon_phonons.primitive)
    assert_allclose(
        heat_capacity_per_gram(phonons, 900.0),
        heat_capacity_per_gram(argon_phonons, 900.0),
        rtol=1e-6,
    )


def test_imaginary_percentage_zero_for_stable_crystal(argon_phonons) -> None:
    """A dynamically stable crystal should report no imaginary modes."""
    assert imaginary_mode_percentage(argon_phonons) == pytest.approx(0.0)


def test_imaginary_percentage_matches_weighted_mode_count(argon_phonons) -> None:
    """
    The metric must equal a direct q-point-weighted count of non-positive modes.

    phonopy integrates modes strictly above its cutoff, so at zero cutoff the
    excluded modes are exactly those with a frequency of zero or below.
    """
    mesh = argon_phonons.get_mesh_dict()
    frequencies = np.asarray(mesh["frequencies"], dtype=float)
    weights = np.asarray(mesh["weights"], dtype=float)
    weights_per_mode = np.broadcast_to(weights[:, None], frequencies.shape)

    expected = (
        100.0
        * float(weights_per_mode[frequencies <= 0.0].sum())
        / float(weights_per_mode.sum())
    )
    assert imaginary_mode_percentage(argon_phonons) == pytest.approx(expected)


def test_wasserstein_zero_for_identical_spectra() -> None:
    """Comparing a spectrum against itself must give exactly zero."""
    energy = np.linspace(0.0, 100.0, 201)
    intensity = np.exp(-((energy - 40.0) ** 2) / 50.0)

    assert wasserstein_spectrum_distance(
        energy, intensity, energy, intensity
    ) == pytest.approx(0.0, abs=1e-12)


def test_wasserstein_invariant_to_intensity_scale() -> None:
    """Arbitrary intensity units must not change the distance."""
    energy = np.linspace(0.0, 100.0, 201)
    reference = np.exp(-((energy - 40.0) ** 2) / 50.0)
    predicted = np.exp(-((energy - 45.0) ** 2) / 50.0)

    unscaled = wasserstein_spectrum_distance(energy, reference, energy, predicted)
    scaled = wasserstein_spectrum_distance(
        energy, 1000.0 * reference, energy, 0.001 * predicted
    )
    assert scaled == pytest.approx(unscaled, rel=1e-9)


def test_wasserstein_recovers_known_shift() -> None:
    """A rigid shift within the shared window should return that shift."""
    energy = np.linspace(0.0, 200.0, 2001)
    reference = np.exp(-((energy - 80.0) ** 2) / 20.0)
    predicted = np.exp(-((energy - 95.0) ** 2) / 20.0)

    distance = wasserstein_spectrum_distance(energy, reference, energy, predicted)
    assert distance == pytest.approx(15.0, abs=0.1)


def test_wasserstein_clips_negative_intensities() -> None:
    """Digitisation noise below zero must not produce negative weights."""
    energy = np.linspace(0.0, 100.0, 201)
    intensity = np.exp(-((energy - 40.0) ** 2) / 50.0)
    noisy = intensity - 0.05

    distance = wasserstein_spectrum_distance(energy, noisy, energy, intensity)
    assert np.isfinite(distance)
    assert distance >= 0.0


def test_wasserstein_insensitive_to_reference_sampling_density() -> None:
    """
    Resampling must remove any dependence on the native grid density.

    The same underlying spectrum, sampled densely at low energy and sparsely
    at high energy, must give the same distance as a uniformly sampled copy.
    Weighting raw intensities on the native grid would not.
    """

    def shape(e):
        """
        Evaluate a fixed two-peak test spectrum.

        Parameters
        ----------
        e
            Energies in meV.

        Returns
        -------
        np.ndarray
            Intensities at ``e``.
        """
        return np.exp(-((e - 40.0) ** 2) / 50.0) + np.exp(-((e - 160.0) ** 2) / 50.0)

    uniform = np.linspace(0.0, 200.0, 801)
    # Dense below 100 meV, sparse above: a 16x density contrast.
    skewed = np.concatenate(
        [np.linspace(0.0, 100.0, 700), np.linspace(100.2, 200.0, 45)]
    )
    pred_e = np.linspace(0.0, 200.0, 801)
    pred_i = shape(pred_e - 6.0)

    from_uniform = wasserstein_spectrum_distance(
        uniform, shape(uniform), pred_e, pred_i
    )
    from_skewed = wasserstein_spectrum_distance(skewed, shape(skewed), pred_e, pred_i)

    assert from_uniform == pytest.approx(from_skewed, rel=0.02)


def test_wasserstein_handles_unsorted_and_duplicate_energies() -> None:
    """Digitised traces with shuffled or repeated abscissae must still work."""
    energy = np.linspace(0.0, 100.0, 201)
    intensity = np.exp(-((energy - 40.0) ** 2) / 50.0)

    messy_e = np.concatenate([energy, energy[50:60]])
    messy_i = np.concatenate([intensity, intensity[50:60]])
    order = np.argsort(np.sin(np.arange(messy_e.size)))  # deterministic shuffle
    messy_e, messy_i = messy_e[order], messy_i[order]

    clean = wasserstein_spectrum_distance(energy, intensity, energy, intensity + 0.0)
    messy = wasserstein_spectrum_distance(messy_e, messy_i, energy, intensity)
    assert np.isfinite(messy)
    assert messy == pytest.approx(clean, abs=1e-9)


def test_wasserstein_respects_explicit_window() -> None:
    """An explicit window must restrict the comparison to that range."""
    energy = np.linspace(0.0, 200.0, 801)
    reference = np.exp(-((energy - 40.0) ** 2) / 20.0)
    # Identical below 100 meV, divergent above.
    predicted = reference + np.exp(-((energy - 160.0) ** 2) / 20.0)

    windowed = wasserstein_spectrum_distance(
        energy, reference, energy, predicted, window=(0.0, 100.0)
    )
    full = wasserstein_spectrum_distance(energy, reference, energy, predicted)

    assert windowed == pytest.approx(0.0, abs=1e-6)
    assert full > 10.0


@pytest.mark.parametrize("scale", [1 / 3000, 0.1, 10.0, 30.0, 36.0])
def test_wasserstein_invariant_to_reference_scaling(scale: float) -> None:
    """
    Pre-scaling the reference intensity cannot change the distance.

    Experimental INS intensities are in arbitrary units, so a per-spectrum
    factor is often applied by hand to bring a measured trace onto the same
    scale as a calculation. The metric normalises both spectra by area over
    the common grid, so any such factor is absorbed and the score is
    unaffected. The scales here span the range of hand-applied factors used
    on this benchmark's frameworks.
    """
    energy = np.linspace(0.0, 200.0, 801)
    reference = np.exp(-((energy - 60.0) ** 2) / 40.0)
    predicted = np.exp(-((energy - 75.0) ** 2) / 40.0)

    unscaled = wasserstein_spectrum_distance(energy, reference, energy, predicted)
    scaled = wasserstein_spectrum_distance(energy, reference * scale, energy, predicted)
    assert scaled == pytest.approx(unscaled, rel=1e-12)


def test_wasserstein_reference_area_matches_prediction() -> None:
    """
    Explicitly area-matching the reference beforehand changes nothing.

    Scaling the reference so its area equals the prediction's is the same
    operation the metric already performs internally, so it is a no-op.
    """
    energy = np.linspace(0.0, 200.0, 801)
    reference = 1e-4 * np.exp(-((energy - 60.0) ** 2) / 40.0)
    predicted = 7.5e3 * np.exp(-((energy - 75.0) ** 2) / 40.0)

    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    factor = trapezoid(predicted, energy) / trapezoid(reference, energy)
    matched = reference * factor
    assert trapezoid(matched, energy) == pytest.approx(
        trapezoid(predicted, energy), rel=1e-12
    )

    assert wasserstein_spectrum_distance(
        energy, matched, energy, predicted
    ) == pytest.approx(
        wasserstein_spectrum_distance(energy, reference, energy, predicted),
        rel=1e-12,
    )


def test_wasserstein_nan_without_overlap() -> None:
    """Spectra covering disjoint energy ranges cannot be compared."""
    low = np.linspace(0.0, 10.0, 51)
    high = np.linspace(50.0, 60.0, 51)
    ones = np.ones_like(low)

    assert np.isnan(wasserstein_spectrum_distance(low, ones, high, ones))


def test_wavenumber_conversion_factor() -> None:
    """The packaged conversion must match the CODATA value."""
    assert CM1_PER_MEV == pytest.approx(8.0655439, rel=1e-7)


def test_resolve_ins_backend_rejects_unknown() -> None:
    """An unrecognised backend name is an error, not a silent fallback."""
    with pytest.raises(ValueError, match="Unknown INS backend"):
        resolve_ins_backend("not-a-backend")


def test_resolve_ins_backend_auto_selects_installed() -> None:
    """Automatic selection must return a backend that is actually available."""
    from importlib.util import find_spec

    if find_spec("euphonic") is None and find_spec("mantid") is None:
        pytest.skip("No INS backend installed")

    backend = resolve_ins_backend("auto")
    assert backend in INS_BACKENDS


def test_euphonic_backend_produces_finite_spectrum(argon_phonons, tmp_path) -> None:
    """The Euphonic backend should return one finite intensity per energy bin."""
    pytest.importorskip("euphonic")
    from ml_peg.calcs.porous_materials.mof_phonons.mof_phonon_utils import (
        ins_spectrum_euphonic,
    )

    bins = np.arange(0.0, 20.0 + 0.05, 0.05)
    intensity = ins_spectrum_euphonic(
        argon_phonons, tmp_path / "phonopy", bins, 10.0, (4, 4, 4)
    )

    assert intensity.shape == (len(bins) - 1,)
    assert np.isfinite(intensity).all()
    assert intensity.sum() > 0.0


def test_every_shipped_ins_reference_declares_a_window() -> None:
    """Each INS reference must name the range it is meaningful over."""
    reference = load_ins_reference()
    if not reference:
        pytest.skip("INS reference data not staged")

    missing = [mof for mof, entry in reference.items() if not entry.get("window_meV")]

    assert not missing, f"no window_meV declared for {missing}"


def test_declared_ins_windows_are_ordered_and_positive() -> None:
    """A declared window must be a positive, increasing pair of energies."""
    reference = load_ins_reference()
    if not reference:
        pytest.skip("INS reference data not staged")

    for mof, entry in reference.items():
        low, high = entry["window_meV"]
        assert 0.0 <= low < high, f"{mof} has a malformed window {(low, high)}"


def test_declared_window_overlaps_the_reference_range() -> None:
    """A window that misses its own reference data would score nothing."""
    reference = load_ins_reference()
    if not reference:
        pytest.skip("INS reference data not staged")

    for mof, entry in reference.items():
        low, high = entry["window_meV"]
        for kind, spectrum in entry.items():
            if not isinstance(spectrum, dict) or "energy_meV" not in spectrum:
                continue
            energy = spectrum["energy_meV"]
            assert max(energy) > low and min(energy) < high, (
                f"{mof} {kind} spans {min(energy)}-{max(energy)} meV, "
                f"outside its window {(low, high)}"
            )
