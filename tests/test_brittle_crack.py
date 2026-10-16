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


def _ordering_frames(e57: float, esharp: float, key: str) -> list:
    """
    Two-state ordering frames with energies under ``key``, front length 4 A.

    Parameters
    ----------
    e57
        Energy of the 5-7 state (eV).
    esharp
        Energy of the sharp state (eV).
    key
        Info key of the energy (``REF_energy`` or ``model_energy``).

    Returns
    -------
    list
        The two frames.
    """
    from ase import Atoms

    out = []
    for state, e in (("5-7", e57), ("sharp", esharp)):
        a = Atoms("Si2", positions=[[0, 0, 0], [1.2, 1.2, 1.2]], cell=[10, 10, 4])
        a.info.update(state=state, K_over_KG=1.5, **{key: e})
        out.append(a)
    return out


def test_ordering_error_is_the_per_length_energy_difference_error():
    """|dE_model - dE_DFT| per unit crack-front length, in meV/A."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ordering_frames(-1.0, 0.0, "REF_energy")
    model = _ordering_frames(0.5, 0.0, "model_energy")
    assert an.ordering_error(model, ref) == pytest.approx(375.0)


def test_failed_or_missing_ordering_gives_nan():
    """A failed or absent ordering calculation is not scored."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ordering_frames(-1.0, 0.0, "REF_energy")
    model = _ordering_frames(0.5, 0.0, "model_energy")
    name = next(iter(an.ORDERINGS))
    refs = {name: ref}
    ok = an.ordering_metrics({name: model}, {}, refs)
    assert list(ok.values()) == [pytest.approx(375.0)]
    for frames, status in (({name: []}, {}), ({name: model}, {name: "failed: x"})):
        assert all(
            np.isnan(v) for v in an.ordering_metrics(frames, status, refs).values()
        )


def test_ordering_metric_thresholds():
    """Thresholds: bad = |dE_DFT| per front length (37.6 meV/A), good = bad / 3."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    th = an.DEFAULT_THRESHOLDS["Si (111)[1-10] 5-7 ordering error"]
    assert th["good"] == pytest.approx(12.5) and th["bad"] == pytest.approx(37.6)


def test_ordering_energies_are_neutral_singlet_model_energies():
    """The calc helper returns copies with the model energy; DFT info is kept."""
    rec = _InfoRecorder()
    frames = ct.ordering_energies(_ordering_frames(-1.0, 0.0, "REF_energy"), rec)
    assert rec.seen == {(0, 1)}
    assert [f.info["state"] for f in frames] == ["5-7", "sharp"]
    assert [f.info["REF_energy"] for f in frames] == [-1.0, 0.0]
    assert all(np.isfinite(f.info["model_energy"]) for f in frames)


@pytest.mark.skipif(not HAVE_DATA, reason="reference data not in the ml_peg cache")
def test_packaged_dft_ordering_is_5_7_below_sharp():
    """The DFT 5-7 state is 145.4 meV (37.6 meV/A of front) below the sharp state."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = an.reference_frames("Si_111_110_431_order")
    e = {f.info["state"]: f.info["REF_energy"] for f in ref}
    assert 1000 * (e["5-7"] - e["sharp"]) == pytest.approx(-145.36, abs=0.01)
    assert len(ref[0]) == 216


def test_ordering_energies_survive_the_output_file(tmp_path):
    """Model ordering energies written by the calc are read back by the analysis."""
    from ase.io import read, write

    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ordering_frames(-1.0, 0.0, "REF_energy")
    write(tmp_path / "o.extxyz", ct.ordering_energies(ref, _SiOnly()), format="extxyz")
    assert np.isfinite(an.ordering_error(read(tmp_path / "o.extxyz", ":"), ref))


def test_a_truncated_ordering_file_gives_nan_with_a_warning():
    """An ordering output with fewer states than the reference is not scored."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ordering_frames(-1.0, 0.0, "REF_energy")
    model = _ordering_frames(0.5, 0.0, "model_energy")[:1]
    name = next(iter(an.ORDERINGS))
    with pytest.warns(UserWarning, match="mymodel .*1 of 2 states"):
        out = an.ordering_metrics({name: model}, {}, {name: ref}, model="mymodel")
    assert all(np.isnan(v) for v in out.values())


def test_ordering_states_that_differ_from_the_reference_still_raise():
    """Stale outputs on other states raise, as a topology mismatch does."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ordering_frames(-1.0, 0.0, "REF_energy")
    model = _ordering_frames(0.5, 0.0, "model_energy")[::-1]
    name = next(iter(an.ORDERINGS))
    with pytest.raises(ValueError, match="do not match"):
        an.ordering_metrics({name: model}, {}, {name: ref})


def test_an_unreadable_output_file_gives_no_frames_with_a_warning(tmp_path):
    """A damaged output file warns and reads as no frames (NaN metrics)."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    bad = tmp_path / "Si_111_110.extxyz"
    bad.write_text("3\nnot an extxyz header\nSi 0 0\n")
    with pytest.warns(UserWarning, match="mymodel Si_111_110.extxyz: unreadable"):
        assert an.read_frames(bad, "mymodel") == []
    assert an.read_frames(tmp_path / "missing.extxyz", "mymodel") == []


def test_an_ordering_output_without_model_energies_gives_nan_with_a_warning():
    """Frames lacking ``model_energy`` (an incomplete calculation) are not scored."""
    from ml_peg.analysis.fracture.brittle_crack import analyse_brittle_crack as an

    ref = _ordering_frames(-1.0, 0.0, "REF_energy")
    model = _ordering_frames(0.5, 0.0, "model_energy")
    del model[1].info["model_energy"]
    name = next(iter(an.ORDERINGS))
    with pytest.warns(UserWarning, match="mymodel .*no model energy"):
        out = an.ordering_metrics({name: model}, {}, {name: ref}, model="mymodel")
    assert all(np.isnan(v) for v in out.values())
