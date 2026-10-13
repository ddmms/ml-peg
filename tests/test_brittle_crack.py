"""Tests for the brittle_crack benchmark metric and tools."""

from __future__ import annotations

import numpy as np
import pytest

from ml_peg.analysis.fracture.brittle_crack import s_curves as sc

# crackcell (ml-peg-hackathon a2e0f46): errors(curve(1.03, 1.1), curve(), "A'")
EXPECTED_EK = 0.044970452637459225
EXPECTED_EE = 0.3508842049166176


def _curve(scale_k: float = 1.0, scale_e: float = 1.0, n: int = 120) -> sc.SCurve:
    """
    Cubic S-curve with folds, as in crackcell's metric tests.

    Parameters
    ----------
    scale_k
        Scale of k.
    scale_e
        Scale of the energy.
    n
        Number of points.

    Returns
    -------
    SCurve
        The curve.
    """
    b = np.linspace(2.6, 4.8, n)
    u = b - 2.8
    k = (1 + 0.4 * u - 1.2 * u**2 + 0.8 * u**3) * scale_k
    e = scale_e * (2.0 * k**2 - 0.3 * np.tanh(3 * (u - 0.5)))
    return sc.SCurve(k, 0.4 * k, b, e, 0.4, 5.4686, 1.511, 3.866)


def test_identical_curves_have_zero_errors_and_full_coverage():
    """Identical model and reference give zero K and E errors."""
    e = sc.errors(_curve(), _curve())
    assert e["E_K"] == pytest.approx(0, abs=1e-12)
    assert e["E_E"] == pytest.approx(0, abs=1e-12)
    assert e["coverage"] == 1.0


def test_constant_k_shift_is_the_k_error():
    """A constant shift in k is the K error."""
    m = _curve()
    m.k = m.k + 0.05
    assert sc.errors(m, _curve())["E_K"] == pytest.approx(0.05, rel=1e-6)


def test_elastic_term_uses_the_reference_c_tilde():
    """An extra elastic energy in the model counts as energy error."""
    m = _curve()
    m.E = m.E + 0.7 * m.k**2
    assert sc.errors(m, _curve())["E_E"] > 0.05


def test_a_model_outside_the_window_has_zero_coverage():
    """A model curve entirely outside the reference window is uncovered."""
    m = _curve()
    m.b = m.b + 10.0
    assert sc.errors(m, _curve())["coverage"] == 0.0


def test_padding_scales_the_score_by_coverage():
    """Padding the uncovered fraction at b gives c times the score."""
    g, b = 0.1, 0.3
    for x in (0.0, 0.2, 0.5):
        for c in (0.0, 0.4, 1.0):
            assert sc.normalised(sc.padded(x, c, g, b), g, b) == pytest.approx(
                c * sc.normalised(x, g, b)
            )


def test_the_port_reproduces_crackcell():
    """The ported errors equal crackcell's on the same curves."""
    e = sc.errors(_curve(scale_k=1.03, scale_e=1.1), _curve())
    assert e["E_K"] == pytest.approx(EXPECTED_EK, rel=1e-9)
    assert e["E_E"] == pytest.approx(EXPECTED_EE, rel=1e-9)


from ase.calculators.calculator import Calculator, all_changes  # noqa: E402

from ml_peg.calcs.fracture.brittle_crack import crack_tools as ct  # noqa: E402


def _sw():
    """
    Build the Stillinger-Weber Si calculator.

    Returns
    -------
    Calculator
        A matscipy Manybody SW calculator.
    """
    from matscipy.calculators.manybody import Manybody
    from matscipy.calculators.manybody.explicit_forms.stillinger_weber import (
        Stillinger_Weber_PRB_31_5262_Si,
        StillingerWeber,
    )

    return Manybody(**StillingerWeber(Stillinger_Weber_PRB_31_5262_Si))


class _SiOnly(Calculator):
    """SW on the Si atoms only; H caps get zero force (SW has no H parameters)."""

    implemented_properties = ["energy", "free_energy", "forces"]

    def calculate(
        self, atoms=None, properties=("energy",), system_changes=all_changes
    ) -> None:
        """
        Evaluate SW on the Si subsystem.

        Parameters
        ----------
        atoms
            Structure.
        properties
            Requested properties.
        system_changes
            Changes since the last call.
        """
        super().calculate(atoms, properties, system_changes)
        si = atoms.numbers == 14
        sub = atoms[si]
        sub.calc = _sw()
        forces = np.zeros((len(atoms), 3))
        forces[si] = sub.get_forces()
        e = sub.get_potential_energy()
        self.results = {"energy": e, "free_energy": e, "forces": forces}


def test_reference_clusters_have_the_dft_atom_counts():
    """The DFT topology clusters have the DFT atom counts."""
    cases = ct.load_cases()
    counts = {"Si_110_110": 135, "Si_110_001": 100, "Si_111_110": 102}
    for name, n in counts.items():
        assert len(ct.reference_cluster(cases[name])) == n


def test_sw_constants_match_the_known_values():
    """SW a0, C11 and gamma(110) match the values used in the crackcell runs."""
    from ase.units import GPa, J, m

    c = ct.material_constants(_sw())
    assert c["a0"] == pytest.approx(5.43095, abs=1e-4)
    assert c["C11"] / GPa == pytest.approx(151.42, abs=0.1)
    g = ct.cleavage_energy(_sw(), c["a0"], [1, -1, 0], [1, 1, 0])
    assert g / (J / m**2) == pytest.approx(1.6657, abs=1e-3)


def test_trace_case_runs_on_the_reference_topology():
    """A short trace on (111) runs and keeps the reference atom order."""
    case = ct.load_cases()["Si_111_110"]
    consts = ct.material_constants(_sw())
    consts["gamma"] = ct.cleavage_energy(
        _sw(), consts["a0"], case["crack_surface"], case["crack_front"]
    )
    frames, status = ct.trace_case(case, _SiOnly(), consts, max_points=4)
    assert status == "complete" and len(frames) == 4
    ref = ct.reference_cluster(case)
    assert frames[0].get_chemical_symbols() == ref.get_chemical_symbols()
    assert frames[0].info["K_over_KG"] == pytest.approx(case["start_k"])


def test_trace_case_reports_failure_instead_of_raising(monkeypatch):
    """A tracer RuntimeError becomes a failed status with no frames."""
    import matscipy.fracture_mechanics.continuation as cont

    def boom(*args, **kwargs):
        """
        Raise like a start relaxation that misses ftol.

        Parameters
        ----------
        *args
            Ignored.
        **kwargs
            Ignored.
        """
        raise RuntimeError("start relaxation did not reach ftol at fixed k")

    monkeypatch.setattr(cont, "trace_s_curve", boom)
    case = ct.load_cases()["Si_111_110"]
    consts = {"a0": 5.43095, "C11": 0.945, "C12": 0.477, "C44": 0.352, "gamma": 0.104}
    frames, status = ct.trace_case(case, _SiOnly(), consts)
    assert frames == [] and status.startswith("failed:")
