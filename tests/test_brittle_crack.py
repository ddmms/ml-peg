"""Tests for the brittle_crack benchmark metric and tools."""

from __future__ import annotations

import importlib

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
from ml_peg.calcs.utils.utils import BENCHMARK_DATA_DIR  # noqa: E402


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


class _InfoRecorder(_SiOnly):
    """``_SiOnly`` that records ``charge`` and ``spin`` of every evaluated structure."""

    def __init__(self) -> None:
        """Start with no recorded structures."""
        super().__init__()
        self.seen = set()

    def calculate(
        self, atoms=None, properties=("energy",), system_changes=all_changes
    ) -> None:
        """
        Record the structure's charge and spin, then evaluate.

        Parameters
        ----------
        atoms
            Structure.
        properties
            Requested properties.
        system_changes
            Changes since the last call.
        """
        self.seen.add((atoms.info.get("charge"), atoms.info.get("spin")))
        super().calculate(atoms, properties, system_changes)


def test_evaluated_structures_are_neutral_singlets():
    """Slab and crack structures carry charge 0 and spin 1 for the model."""
    case = ct.load_cases()["Si_111_110"]
    rec = _InfoRecorder()
    consts = {"a0": 5.43095, "C11": 0.945, "C12": 0.477, "C44": 0.352}
    consts["gamma"] = ct.cleavage_energy(
        rec, consts["a0"], case["crack_surface"], case["crack_front"]
    )
    ct.trace_case(case, rec, consts, max_points=2)
    assert rec.seen == {(0, 1)}


HAVE_DATA = (BENCHMARK_DATA_DIR / "brittle_crack.zip").exists()
HAVE_ORDERING_DATA = (
    BENCHMARK_DATA_DIR / "brittle_crack" / "Si_111_110_431_ordering.extxyz"
).exists()


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


@pytest.mark.skipif(not HAVE_DATA, reason="reference data not in the ml_peg cache")
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
    with pytest.raises(FileNotFoundError, match="inputs/fracture/brittle_crack"):
        an.reference_frames("Si_111_110")


def test_reference_loads_only_through_the_s3_helper(tmp_path, monkeypatch):
    """Reference data comes from download_s3_data (cache or S3); no env override."""
    from ase import Atoms
    from ase.io import write

    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    for d, n in (("s3", 1), ("env", 2)):
        (tmp_path / d / "brittle_crack").mkdir(parents=True)
        write(tmp_path / d / "brittle_crack" / "Si_111_110.extxyz", [Atoms("H")] * n)
    monkeypatch.setenv("BRITTLE_CRACK_DATA", str(tmp_path / "env" / "brittle_crack"))
    an = importlib.reload(an)
    monkeypatch.setattr(an, "download_s3_data", lambda **kwargs: tmp_path / "s3")
    assert len(an.reference_frames("Si_111_110")) == 1


def test_structure_info_comes_from_the_reference_clusters(tmp_path, monkeypatch):
    """Element info uses the reference clusters: model traces may be empty files."""
    from ase.io import read

    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    monkeypatch.setattr(an, "OUT_PATH", tmp_path)
    paths = an.reference_structures()
    assert len(paths) == 3
    assert {a.get_chemical_formula() for a in (read(p) for p in paths)} >= {"H30Si72"}


def test_failed_calculations_give_nan_metrics():
    """Failed constants, gamma or trace give NaN metrics (a NaN benchmark score)."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    refs = {name: [object()] for name in an.CASES}
    frames = {name: [] for name in an.CASES}
    ok = {"a0": 5.4, "C11": 0.9, "C12": 0.4, "C44": 0.4}
    for consts in (
        {"status": dict.fromkeys(an.CASES, "failed: constants: x")},
        {**ok, "gamma": {}},
        {
            **ok,
            "gamma": dict.fromkeys(an.CASES, 0.1),
            "status": dict.fromkeys(an.CASES, "failed: x"),
        },
    ):
        errs = an.model_errors(consts, frames, refs)
        assert set(errs) == set(an.CASES)
        assert all(np.isnan(v) for v in an.raw_metrics(errs).values())


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


def test_an_error_in_one_case_gives_nan_for_that_case_only(monkeypatch):
    """An unexpected per-case error warns and gives NaN; other cases are scored."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    names = list(an.CASES)

    def fake_case_errors(model_frames, ref_frames, case, consts, ref_consts):
        """
        Fail on the first case, return zero errors on the others.

        Parameters
        ----------
        model_frames, ref_frames, case, consts, ref_consts
            As ``case_errors``.

        Returns
        -------
        dict
            Zero errors with full coverage.
        """
        if case is an.CASES[names[0]]:
            raise RuntimeError("non-finite positions")
        return dict.fromkeys(an.FAILED_KEYS, 0.0) | {"coverage": 1.0}

    monkeypatch.setattr(an, "case_errors", fake_case_errors)
    consts = {"a0": 5.4, "C11": 0.9, "C12": 0.4, "C44": 0.4}
    consts["gamma"] = dict.fromkeys(names, 0.1)
    with pytest.warns(UserWarning, match=f"mymodel {names[0]}: non-finite"):
        errs = an.model_errors(consts, {}, dict.fromkeys(names), model="mymodel")
    assert np.isnan(errs[names[0]]["coverage"])
    assert all(errs[n]["coverage"] == 1.0 for n in names[1:])


def test_a_topology_mismatch_still_stops_the_analysis():
    """Frames on another topology (stale outputs) raise through model_errors."""
    from ase import Atoms

    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    wrong = Atoms("Si4", positions=np.zeros((4, 3)), cell=[10, 10, 3.8])
    wrong.info.update(K_over_KG=1.0, REF_energy=0.0)
    consts = {"a0": 5.4, "C11": 0.9, "C12": 0.4, "C44": 0.4}
    consts["gamma"] = dict.fromkeys(an.CASES, 0.1)
    frames = dict.fromkeys(an.CASES, [wrong])
    with pytest.raises(ValueError, match="topology"):
        an.model_errors(consts, frames, frames, model="mymodel")


def test_an_unconverged_cleavage_slab_raises(monkeypatch):
    """A slab relaxation that hits the step limit raises instead of giving gamma."""
    from ase.optimize import LBFGS

    monkeypatch.setattr(LBFGS, "run", lambda self, **kwargs: False)
    with pytest.raises(RuntimeError, match="did not converge"):
        ct.cleavage_energy(_sw(), 5.43095, [1, -1, 0], [1, 1, 0])


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


ORDER = {
    "label": "X 5-7",
    "states": ["5-7", "sharp"],
    "q_triple": [0, 1, 2],
    "q_tol": 0.5,
    "n_grid": 7,
}


def _ramp_frames(de57, key: str, ks=(1.5, 1.6, 1.7), q57=1.5, qsharp=-1.3) -> list:
    """
    Ordering-over-load frames: the 5-7 and sharp states at each K, front length 4 A.

    The q coordinate d(1, 2) - d(0, 2) of the three Si atoms is q57 (5-7) or qsharp
    (sharp); atoms 0 and 2 are clamped (``fixed``).

    Parameters
    ----------
    de57
        E(5-7) - E(sharp) (eV) as a function of K; E(sharp) = 0.
    key
        Info key of the energy (``REF_energy`` or ``model_energy``).
    ks
        Loads (K/K_G) of the frames, the same for both states.
    q57
        q of the 5-7 frames (A).
    qsharp
        q of the sharp frames (A).

    Returns
    -------
    list
        The 5-7 frames by K, then the sharp ones.
    """
    from ase import Atoms

    out = []
    for state, q in (("5-7", q57), ("sharp", qsharp)):
        for k in ks:
            d02 = 2.5 if q > 0 else 3.7
            a = Atoms(
                "Si3",
                positions=[[d02, 0, 0], [0, d02 + q, 0], [0, 0, 0]],
                cell=[10, 10, 4],
                pbc=[False, False, True],
            )
            a.set_array("fixed", np.array([1, 0, 1], dtype=int))
            e = de57(k) if state == "5-7" else 0.0
            a.info.update(state=state, K_over_KG=k, **{key: e})
            if key == "model_energy":
                a.info.update(model_q=q, model_converged=True)
            out.append(a)
    return out


def _linear(k):
    """
    DFT-like E(5-7) - E(sharp): -0.2 eV at K 1.5, crossing zero at K 1.6.

    Parameters
    ----------
    k
        Load (K/K_G).

    Returns
    -------
    float
        Energy difference (eV).
    """
    return 2.0 * (k - 1.6)


def test_ordering_curve_of_the_reference_itself_has_no_error():
    """A model equal to DFT: zero ordering error, full coverage, the same crossover."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ramp_frames(_linear, "REF_energy")
    c = an.ordering_curve(_ramp_frames(_linear, "model_energy"), ref, ORDER)
    assert c["error"] == pytest.approx(0.0, abs=1e-9) and c["coverage"] == 1.0
    assert c["k_cross"] == pytest.approx(c["k_cross_ref"]) == pytest.approx(1.6)


def test_an_energy_offset_is_the_ordering_error_and_shifts_the_crossover():
    """DE off by 0.04 eV: 10 meV/A on a 4 A front; the crossover moves by 0.04 / 2.0."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ramp_frames(_linear, "REF_energy")
    model = _ramp_frames(lambda k: _linear(k) + 0.04, "model_energy")
    c = an.ordering_curve(model, ref, ORDER)
    assert c["error"] == pytest.approx(10.0)
    assert c["k_cross"] - c["k_cross_ref"] == pytest.approx(-0.02)


def test_a_state_that_leaves_its_basin_is_not_scored_at_that_load():
    """
    Leave a load unscored where a state left its basin or did not converge.

    A 5-7 frame that relaxes to the sharp side (q off by more than q_tol) drops out: the
    loads beyond the last 5-7 frame still in its basin are not covered.
    """
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ramp_frames(_linear, "REF_energy")
    model = _ramp_frames(_linear, "model_energy")
    model[2].info["model_q"] = -1.2  # the 5-7 frame at K 1.7
    c = an.ordering_curve(model, ref, ORDER)
    assert c["coverage"] == pytest.approx(4 / 7)  # grid 1.5..1.7 in 7: 1.5..1.6 covered
    model[2].info.update(
        model_q=1.5, model_converged=False
    )  # an unconverged relaxation
    assert an.ordering_curve(model, ref, ORDER)["coverage"] == pytest.approx(4 / 7)


def test_loads_above_k_max_are_not_scored():
    """
    Score only the loads up to ``k_max``.

    A 5-7 frame above k_max that leaves its basin (a state close to its own fold) costs
    no coverage, and the window, grid and crossover fit stop at the last load kept.
    """
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ks = (1.5, 1.6, 1.7, 1.8)
    ref = _ramp_frames(_linear, "REF_energy", ks=ks)
    model = _ramp_frames(lambda k: _linear(k) + 0.04, "model_energy", ks=ks)
    model[3].info["model_q"] = -1.2  # the 5-7 frame at K 1.8 collapses
    assert an.ordering_curve(model, ref, ORDER)["coverage"] < 1.0
    c = an.ordering_curve(model, ref, {**ORDER, "k_max": 1.75})
    assert c["coverage"] == 1.0 and c["error"] == pytest.approx(10.0)
    assert c["k_cross_ref"] == pytest.approx(1.6)
    ks3 = (1.5, 1.6, 1.7)
    capped = _ramp_frames(_linear, "REF_energy", ks=ks3)
    capped_model = _ramp_frames(lambda k: _linear(k) + 0.04, "model_energy", ks=ks3)
    # as if the frames ended at K 1.7
    assert c == an.ordering_curve(capped_model, capped, ORDER)


def test_ordering_frames_that_differ_from_the_reference_raise():
    """Stale outputs (other states or loads) raise, as a topology mismatch does."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ramp_frames(_linear, "REF_energy")
    with pytest.raises(ValueError, match="do not match"):
        an.ordering_curve(_ramp_frames(_linear, "model_energy")[::-1], ref, ORDER)
    with pytest.raises(ValueError, match="do not match"):
        an.ordering_curve(
            _ramp_frames(_linear, "model_energy", ks=(1.5, 1.6, 1.8)), ref, ORDER
        )


def test_ordering_metrics_pad_by_coverage_and_give_nan_for_failed_runs(monkeypatch):
    """
    Pad both metrics by coverage; failed or incomplete outputs are NaN.

    Failed, absent, truncated or energy-less outputs give NaN (the last two warn).
    """
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    monkeypatch.setattr(an, "ORDERINGS", {"o": ORDER})
    th = {
        "X 5-7 ordering error": {"good": 12.5, "bad": 37.6},
        "X 5-7 crossover error": {"good": 0.020, "bad": 0.060},
    }
    monkeypatch.setattr(an, "DEFAULT_THRESHOLDS", {**an.DEFAULT_THRESHOLDS, **th})
    ref = {"o": _ramp_frames(_linear, "REF_energy")}
    model = _ramp_frames(lambda k: _linear(k) + 0.2, "model_energy")
    out = an.ordering_metrics({"o": model}, {}, ref)
    assert out["X 5-7 ordering error"] == pytest.approx(50.0)
    assert out["X 5-7 crossover error"] == pytest.approx(0.1)
    for frames, status in (({"o": []}, {}), ({"o": model}, {"o": "failed: x"})):
        assert all(
            np.isnan(v) for v in an.ordering_metrics(frames, status, ref).values()
        )
    with pytest.warns(UserWarning, match="mymodel o: 5 of 6 frames"):
        out = an.ordering_metrics({"o": model[:5]}, {}, ref, model="mymodel")
    assert all(np.isnan(v) for v in out.values())
    del model[0].info["model_energy"]
    with pytest.warns(UserWarning, match="mymodel o: no model energy"):
        out = an.ordering_metrics({"o": model}, {}, ref, model="mymodel")
    assert all(np.isnan(v) for v in out.values())


def test_ordering_metric_thresholds():
    """
    Check the thresholds: ordering bad = the DFT gap at K/K_G 1.50, good = bad / 3.

    Crossover: the K shift that bad gap makes at the DFT slope, and a third of it.
    """
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    th = an.DEFAULT_THRESHOLDS["Si (111)[1-10] 5-7 ordering error"]
    assert th["good"] == pytest.approx(12.5) and th["bad"] == pytest.approx(37.6)
    th = an.DEFAULT_THRESHOLDS["Si (111)[1-10] 5-7 crossover error"]
    assert th["good"] == pytest.approx(0.020) and th["bad"] == pytest.approx(0.060)


def test_ordering_relaxations_keep_the_clamped_atoms_and_record_the_basin():
    """
    Relax each state at fixed K as a neutral lowest-spin structure.

    The clamped atoms stay put; the output carries the relaxed and DFT-geometry model
    energies, q and whether the relaxation converged.
    """
    rec = _InfoRecorder()
    ref = _ramp_frames(_linear, "REF_energy", ks=(1.5,))
    out = ct.ordering_relaxations(ref, rec, ORDER["q_triple"], fmax=0.05)
    assert rec.seen == {(0, 1)}
    assert [f.info["state"] for f in out] == ["5-7", "sharp"]
    for f, r in zip(out, ref, strict=True):
        clamped = r.arrays["fixed"] == 1
        assert np.allclose(f.positions[clamped], r.positions[clamped])
        assert f.info["model_energy"] <= f.info["model_energy_dft_geom"] + 1e-9
        assert isinstance(f.info["model_converged"], bool | np.bool_)
        assert np.isfinite(f.info["model_q"])
        assert f.info["REF_energy"] == r.info["REF_energy"]


def test_an_ordering_relaxation_has_a_step_limit(monkeypatch):
    """The fixed-K relaxations run with a finite step limit."""
    from ase.optimize import LBFGS

    calls = []
    run = LBFGS.run

    def spy(self, **kwargs):
        """
        Record the run's keyword arguments, then run.

        Parameters
        ----------
        **kwargs
            Optimiser run arguments.

        Returns
        -------
        bool
            Whether the run converged.
        """
        calls.append(kwargs)
        return run(self, **kwargs)

    monkeypatch.setattr(LBFGS, "run", spy)
    ct.ordering_relaxations(
        _ramp_frames(_linear, "REF_energy", ks=(1.5,)), _SiOnly(), [0, 1, 2]
    )
    assert calls and all(0 < k.get("steps", np.inf) < np.inf for k in calls)


def test_ordering_relaxations_survive_the_output_file(tmp_path):
    """Model ordering frames written by the calc are read back and scored."""
    from ase.io import read, write

    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ramp_frames(_linear, "REF_energy")
    out = ct.ordering_relaxations(ref, _SiOnly(), ORDER["q_triple"], fmax=0.05)
    write(tmp_path / "o.extxyz", out, format="extxyz")
    back = read(tmp_path / "o.extxyz", ":")
    assert all(isinstance(f.info["model_converged"], bool | np.bool_) for f in back)
    assert np.isfinite(an.ordering_curve(back, ref, ORDER)["coverage"])


@pytest.mark.skipif(
    not HAVE_ORDERING_DATA, reason="ordering reference not in the ml_peg cache"
)
def test_packaged_dft_ordering_crosses_over_near_k_1_56():
    """DFT: 5-7 below sharp at low load, above it past K* ~ 1.56, on 11 loads each."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    name = "Si_111_110_431_ordering"
    ref = an.reference_frames(name)
    assert len(ref) == 22 and len(ref[0]) == 216 and "fixed" in ref[0].arrays
    c = an.ordering_curve(ref_as_model(ref), ref, an.ORDERINGS[name])
    assert c["k_cross_ref"] == pytest.approx(1.564, abs=0.001)
    assert c["error"] == pytest.approx(0.0, abs=1e-9) and c["coverage"] == 1.0


def ref_as_model(ref: list) -> list:
    """
    Copy reference frames as a perfect model output.

    Parameters
    ----------
    ref
        Reference frames.

    Returns
    -------
    list
        Frames with ``model_energy`` = ``REF_energy``, converged, in their basin.
    """
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    q_triple = an.ORDERINGS["Si_111_110_431_ordering"]["q_triple"]
    out = []
    for f in ref:
        g = f.copy()
        g.info.update(
            model_energy=f.info["REF_energy"],
            model_converged=True,
            model_q=an.q_coordinate(f, q_triple),
        )
        out.append(g)
    return out


def test_an_unreadable_output_file_gives_no_frames_with_a_warning(tmp_path):
    """A damaged output file warns and reads as no frames (NaN metrics)."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    bad = tmp_path / "Si_111_110.extxyz"
    bad.write_text("3\nnot an extxyz header\nSi 0 0\n")
    with pytest.warns(UserWarning, match="mymodel Si_111_110.extxyz: unreadable"):
        assert an.read_frames(bad, "mymodel") == []
    assert an.read_frames(tmp_path / "missing.extxyz", "mymodel") == []


class _SWRecorder(Calculator):
    """Stillinger-Weber (with stress) recording ``charge`` and ``spin`` of each call."""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(self) -> None:
        """Start with no recorded structures."""
        super().__init__()
        self.seen = set()
        self.sw = _sw()

    def calculate(
        self, atoms=None, properties=("energy",), system_changes=all_changes
    ) -> None:
        """
        Record the structure's charge and spin, then evaluate SW.

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
        self.seen.add((atoms.info.get("charge"), atoms.info.get("spin")))
        a = atoms.copy()
        a.calc = self.sw
        self.results = {
            "energy": a.get_potential_energy(),
            "free_energy": a.get_potential_energy(),
            "forces": a.get_forces(),
            "stress": a.get_stress(),
        }


def test_material_constants_give_every_structure_charge_and_spin():
    """The lattice and elastic-constant cells carry charge and spin from the start."""
    rec = _SWRecorder()
    c = ct.material_constants(rec)
    assert rec.seen == {(0, 1)}
    assert c["a0"] == pytest.approx(5.43095, abs=1e-4)


def test_an_unconverged_lattice_relaxation_raises(monkeypatch):
    """A lattice relaxation that hits its step limit raises (failed constants)."""
    from ase.optimize import LBFGS

    monkeypatch.setattr(LBFGS, "run", lambda self, **kwargs: False)
    with pytest.raises(RuntimeError, match="did not converge"):
        ct.material_constants(_sw())


def test_the_elastic_constant_relaxations_have_a_step_limit(monkeypatch):
    """Every optimiser run in material_constants is given a finite step limit."""
    from ase.optimize import LBFGS

    calls = []
    run = LBFGS.run

    def spy(self, **kwargs):
        """
        Record the run's keyword arguments, then run.

        Parameters
        ----------
        **kwargs
            Optimiser run arguments.

        Returns
        -------
        bool
            Whether the run converged.
        """
        calls.append(kwargs)
        return run(self, **kwargs)

    monkeypatch.setattr(LBFGS, "run", spy)
    ct.material_constants(_sw())
    assert calls and all(0 < k.get("steps", np.inf) < np.inf for k in calls)


def test_spin_multiplicity_follows_the_electron_count():
    """Even electron count: singlet; odd ((110)[1-10], 96 Si + 39 H): doublet."""
    from ase import Atoms

    assert ct.neutral_lowest_spin(Atoms("Si2")).info == {"charge": 0, "spin": 1}
    assert ct.neutral_lowest_spin(Atoms("SiH")).info == {"charge": 0, "spin": 2}
    cl = ct.reference_cluster(ct.load_cases()["Si_110_110"])
    assert ct.neutral_lowest_spin(cl).info["spin"] == 2


def test_the_110_crack_is_traced_as_a_doublet():
    """The odd-electron (110)[1-10] cluster is evaluated with spin 2."""
    case = ct.load_cases()["Si_110_110"]
    rec = _InfoRecorder()
    consts = {"a0": 5.43095, "C11": 0.945, "C12": 0.477, "C44": 0.352, "gamma": 0.104}
    ct.trace_case(case, rec, consts, max_points=2)
    assert rec.seen == {(0, 2)}


def test_case_outputs_are_skipped_for_unscored_cases_and_plot_errors(
    tmp_path, monkeypatch
):
    """No figure or trace for a case that failed scoring; a plot error only warns."""
    from ase import Atoms

    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    monkeypatch.setattr(an, "OUT_PATH", tmp_path)
    bad = [Atoms("Si2", positions=[[0, 0, 0], [2.3, 0, 0]])]  # no K_over_KG
    nan_errs = dict.fromkeys(an.FAILED_KEYS, np.nan)
    assert not an.write_case_outputs("m", "Si_111_110", bad, bad, nan_errs)
    assert not list(tmp_path.rglob("*"))
    ok_errs = {**nan_errs, "coverage": 0.5}
    with pytest.warns(UserWarning, match="m Si_111_110: no plot"):
        assert not an.write_case_outputs("m", "Si_111_110", bad, bad, ok_errs)
