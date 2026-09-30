"""Unit tests for the MOF bulk modulus benchmark helpers."""

from __future__ import annotations

from ase.eos import birchmurnaghan
from ase.units import kJ
import numpy as np
import pytest

from ml_peg.analysis.porous_materials.mof_bulk_modulus.analyse_mof_bulk_modulus import (
    fit_bulk_modulus,
)
from ml_peg.calcs.porous_materials.mof_phonons.mof_phonon_utils import (
    N_VOLUMES,
    VOLUME_RANGE,
    unconverged_eos_points,
    volume_scales,
)


def test_volume_scales_span_requested_range() -> None:
    """The scan must be centred on the relaxed volume and span the range."""
    scales = volume_scales()

    assert len(scales) == N_VOLUMES
    assert scales[0] == pytest.approx(1.0 - VOLUME_RANGE)
    assert scales[-1] == pytest.approx(1.0 + VOLUME_RANGE)
    assert scales[len(scales) // 2] == pytest.approx(1.0)
    assert np.all(np.diff(scales) > 0)


@pytest.mark.parametrize("bulk_gpa", [2.0, 10.0, 37.9, 120.0])
def test_fit_recovers_known_bulk_modulus(bulk_gpa: float) -> None:
    """
    Fitting a synthetic Birch-Murnaghan curve must return its bulk modulus.

    The curve is generated from ASE's own Birch-Murnaghan form with a known
    modulus, so this checks both the fit and the eV/Angstrom^3 to GPa
    conversion, which is the step most easily got wrong.
    """
    v0, e0, bulk_prime = 1000.0, -500.0, 4.0
    bulk_ev_per_ang3 = bulk_gpa * kJ / 1.0e24

    volumes = v0 * volume_scales()
    energies = birchmurnaghan(volumes, e0, bulk_ev_per_ang3, bulk_prime, v0)

    assert fit_bulk_modulus(volumes, energies) == pytest.approx(bulk_gpa, rel=1e-3)


def test_fit_returns_nan_on_degenerate_input() -> None:
    """
    A curve with no curvature must be rejected rather than scored.

    ASE does not raise on a flat energy-volume curve; it returns a bulk
    modulus of essentially zero. That is not a prediction, so the fit is
    required to report it as missing.
    """
    volumes = list(1000.0 * volume_scales())
    energies = [-500.0] * len(volumes)

    assert np.isnan(fit_bulk_modulus(volumes, energies))


def test_fit_accepts_softest_experimental_value() -> None:
    """The degenerate-fit floor must not reject a genuinely soft framework."""
    v0, e0, bulk_prime = 1000.0, -500.0, 4.0
    volumes = v0 * volume_scales()
    energies = birchmurnaghan(volumes, e0, 0.35 * kJ / 1.0e24, bulk_prime, v0)

    assert fit_bulk_modulus(volumes, energies) == pytest.approx(0.35, rel=1e-3)


def test_excluded_references_are_not_scored() -> None:
    """
    Records flagged as excluded must never enter the scored framework list.

    MIL-53 carries an experimental value that cannot be compared with a
    single-structure equation-of-state fit, so it is retained in the
    reference file with a reason but kept out of every statistic.
    """
    from ml_peg.analysis.porous_materials.mof_bulk_modulus import (
        analyse_mof_bulk_modulus as module,
    )

    assert not set(module.FRAMEWORKS) & set(module.EXCLUDED)
    for name in module.EXCLUDED:
        assert module.BULK_MODULUS_REFERENCE[name].get("exclusion_reason")


def test_unconverged_eos_points_flags_capped_relaxations() -> None:
    """Volume scales whose fixed-cell relaxation hit the step limit are named."""
    flags = [True] * N_VOLUMES
    flags[0] = False
    flags[-1] = False
    record = {"inner_converged": flags}

    flagged = unconverged_eos_points(record)

    assert flagged == pytest.approx([1.0 - VOLUME_RANGE, 1.0 + VOLUME_RANGE])


def test_unconverged_eos_points_empty_when_all_converged() -> None:
    """A fully converged scan reports nothing, as does a record predating the field."""
    assert unconverged_eos_points({"inner_converged": [True] * N_VOLUMES}) == []
    assert unconverged_eos_points({"volumes_ang3": [1.0]}) == []


def test_unconverged_eos_points_uses_the_grid_of_the_record() -> None:
    """A record written with a different scan grid is read on its own terms."""
    record = {
        "n_volumes": 3,
        "volume_range": 0.1,
        "inner_converged": [False, True, True],
    }

    assert unconverged_eos_points(record) == pytest.approx([0.9])


def test_unconverged_eos_points_ignores_a_mismatched_flag_list() -> None:
    """Flags that do not line up with the recorded grid are not guessed at."""
    assert unconverged_eos_points({"n_volumes": 7, "inner_converged": [False]}) == []
