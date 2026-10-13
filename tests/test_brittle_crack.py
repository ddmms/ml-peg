"""Tests for the brittle_crack benchmark metric and tools."""

from __future__ import annotations

import os
from pathlib import Path

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


HAVE_DATA = bool(os.environ.get("BRITTLE_CRACK_DATA"))


def test_raw_metrics_pad_by_coverage_with_the_loaded_thresholds():
    """Raw values are padded with the metrics.yml thresholds; zero coverage gives b."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    th = an.DEFAULT_THRESHOLDS
    errs = {
        "Si_110_110": {
            "E_K": 0.0,
            "E_E": 0.0,
            "E_D": 0.0,
            "coverage": 0.5,
            "coverage_D": 1.0,
        },
        "Si_110_001": {
            "E_K": np.nan,
            "E_E": np.nan,
            "E_D": None,
            "coverage": 0.0,
            "coverage_D": 0.0,
        },
    }
    raw = an.raw_metrics(errs)
    k = "Si (110)[1-10] K error"
    g, b = th[k]["good"], th[k]["bad"]
    assert raw[k] == pytest.approx(0.5 * g + 0.5 * b)
    assert raw["Si (110)[1-10] D error"] == pytest.approx(
        th["Si (110)[1-10] D error"]["good"]
    )
    assert raw["Si (110)[001] D error"] == pytest.approx(
        th["Si (110)[001] D error"]["bad"]
    )


def test_empty_model_trace_has_zero_coverage():
    """A model with no frames gets NaN errors and zero coverage, not an exception."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    case = ct.load_cases()["Si_111_110"]
    t = ct.topology_constants(case)
    e = an.case_errors([], [object()], case, t, t)
    assert e["coverage"] == 0.0 and e["coverage_D"] == 0.0


@pytest.mark.skipif(not HAVE_DATA, reason="set BRITTLE_CRACK_DATA to the packaged data")
def test_reference_against_itself_has_zero_errors():
    """The DFT reference path scored against itself has zero errors."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = an.reference_frames("Si_111_110")
    case = ct.load_cases()["Si_111_110"]
    t = ct.topology_constants(case)
    e = an.case_errors(ref, ref, case, t, t)
    assert e["E_K"] == pytest.approx(0, abs=1e-12) and e["coverage"] == 1.0
    assert e["E_E"] == pytest.approx(0, abs=1e-12)
    assert e["E_D"] == pytest.approx(0, abs=1e-12)


def test_missing_reference_names_the_s3_key(monkeypatch):
    """An offline download failure names the S3 key."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    def fail(**kwargs):
        """
        Fail like an offline download.

        Parameters
        ----------
        **kwargs
            Ignored.
        """
        raise OSError("offline")

    monkeypatch.setattr(an, "download_s3_data", fail)
    monkeypatch.setattr(an, "LOCAL_DATA", Path("/nonexistent"))
    with pytest.raises(FileNotFoundError, match="inputs/fracture/brittle_crack"):
        an.reference_frames("Si_111_110")


def test_structure_info_comes_from_the_reference_clusters(tmp_path, monkeypatch):
    """Element info uses the reference clusters: model traces may be empty files."""
    from ase.io import read

    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    monkeypatch.setattr(an, "OUT_PATH", tmp_path)
    paths = an.reference_structures()
    assert len(paths) == 3
    assert {a.get_chemical_formula() for a in (read(p) for p in paths)} >= {"H30Si72"}


def test_model_errors_score_a_missing_gamma_or_constants_as_uncovered():
    """Failed constants or a failed cleavage energy give zero coverage, not KeyError."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    refs = {name: [object()] for name in an.CASES}
    frames = {name: [] for name in an.CASES}
    no_consts = {"status": dict.fromkeys(an.CASES, "failed: x")}
    for consts in (
        no_consts,
        {"a0": 5.4, "C11": 0.9, "C12": 0.4, "C44": 0.4, "gamma": {}},
    ):
        errs = an.model_errors(consts, frames, refs)
        assert set(errs) == set(an.CASES)
        assert all(e["coverage"] == 0.0 for e in errs.values())


def test_frames_on_another_topology_are_rejected_clearly():
    """Frames whose atoms differ from the case topology raise a clear ValueError."""
    from ase import Atoms

    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    case = ct.load_cases()["Si_111_110"]
    t = ct.topology_constants(case)
    wrong = Atoms("Si4", positions=np.zeros((4, 3)), cell=[10, 10, 3.8])
    wrong.info.update(K_over_KG=1.0, REF_energy=0.0)
    with pytest.raises(ValueError, match="topology"):
        an.case_errors([wrong], [wrong], case, t, t)


def test_s_curve_figure_lets_only_model_points_open_structures():
    """The DFT curve has no customdata; model points carry the model label."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    fig = an.s_curve_figure(
        "m1", "Si_111_110", [2.5, 3.0], [0.6, 1.0], [2.6, 3.1], [0.6, 1.1], (2.5, 3.0)
    )
    dft, model = fig.data
    assert dft.customdata is None
    assert [c[-1] for c in model.customdata] == ["m1", "m1"]


def test_app_registers_traces_only_for_models_with_a_trace_file(tmp_path, monkeypatch):
    """Structure lookup maps a model label to its trace file only when it exists."""
    from ml_peg.app.fracture.brittle_crack import app_brittle_crack as app

    monkeypatch.setattr(app, "DATA_PATH", tmp_path)
    assert app.trace_structs("m1", "Si_111_110") == {}
    (tmp_path / "Si_111_110" / "m1").mkdir(parents=True)
    (tmp_path / "Si_111_110" / "m1" / "m1-trace.extxyz").write_text("x")
    assert app.trace_structs("m1", "Si_111_110") == {
        "m1": "/assets/fracture/brittle_crack/Si_111_110/m1/m1-trace.extxyz"
    }


def test_calc_is_marked_fast():
    """About 7 minutes per model on a GPU is ML-PEG's fast tier."""
    from ml_peg.calcs.fracture.brittle_crack import calc_brittle_crack as calc

    marks = {m.name for m in calc.test_brittle_crack.pytestmark}
    assert "fast" in marks and "slow" not in marks
