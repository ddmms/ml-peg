"""Test MAD-BENCH subset preparation, prediction checks and diagnostics."""

from __future__ import annotations

from ase import Atoms, units
from ase.calculators.singlepoint import SinglePointCalculator
import numpy as np
import pytest

from ml_peg.analysis.bulk_crystal.mad_bench.analyse_mad_bench import (
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    PRED_FORCES_KEY,
    PRED_STRESS_KEY,
    REFERENCE_PROTOCOLS,
    SCORED_REFERENCE,
    align_predictions,
    available_references,
    collect_property,
    density_point_sources,
    error_statistics,
    p8_error,
    property_coverage,
    reference_property_data,
    sup_error,
)
from ml_peg.analysis.utils.utils import calc_table_scores
from ml_peg.app.bulk_crystal.mad_bench.app_mad_bench import (
    MODELS as APP_MODELS,
)
from ml_peg.app.bulk_crystal.mad_bench.app_mad_bench import (
    diagnostics_table_content,
)
from ml_peg.calcs.bulk_crystal.mad_bench.calc_mad_bench import predict_property
from ml_peg.calcs.bulk_crystal.mad_bench.prepare_mad_bench import (
    find_invalid_labels,
    select_indices,
)

FORCES_KEY = REFERENCE_PROTOCOLS[SCORED_REFERENCE]["forces_key"]
STRESS_KEY = REFERENCE_PROTOCOLS[SCORED_REFERENCE]["stress_key"]


def make_structure(subset: str, frame_id: int, symbols: str = "HO") -> Atoms:
    """Build a small periodic structure with a MAD-1.6 identity."""
    atoms = Atoms(
        symbols,
        positions=np.arange(3 * len(symbols), dtype=float).reshape(-1, 3),
        cell=np.eye(3) * 10,
        pbc=True,
    )
    atoms.info["subset"] = subset
    atoms.info["frame_id"] = frame_id
    atoms.info["source_index"] = frame_id

    return atoms


def make_source(forces: np.ndarray, stress: np.ndarray | None = None) -> Atoms:
    """Build a source structure with reference labels."""
    atoms = make_structure("mc3d", 0)
    atoms.info["atomization_energy"] = -1.0

    labels = {"energy": -10.0, "forces": forces}

    if stress is not None:
        labels["stress"] = stress

    atoms.calc = SinglePointCalculator(atoms, **labels)

    return atoms


def make_prediction(subset: str, frame_id: int, **statuses: str) -> Atoms:
    """Build a prediction with per-property statuses and zero values."""
    atoms = make_structure(subset, frame_id)
    atoms.arrays[PRED_FORCES_KEY] = np.zeros((2, 3))
    atoms.info[PRED_STRESS_KEY] = np.zeros(6)

    for name in ("forces", "energy", "stress"):
        atoms.info[f"{name}_status"] = statuses.get(name, "ok")

    return atoms


def test_select_indices_is_exact_and_reproducible():
    """Test selection gives unique sorted indices that depend only on the seed."""
    selected = select_indices(n_source=500, n_select=200, seed=42)

    assert len(selected) == len(set(selected.tolist())) == 200
    assert (np.diff(selected) > 0).all()
    assert selected.min() >= 0
    assert selected.max() < 500

    assert (selected == select_indices(n_source=500, n_select=200, seed=42)).all()
    assert (selected != select_indices(n_source=500, n_select=200, seed=0)).any()


def test_find_invalid_labels_reports_each_problem():
    """Test invalid forces and stresses are reported and valid labels pass."""
    valid = make_source(np.zeros((2, 3)), stress=np.zeros(6))
    misshapen = make_source(np.zeros((3, 3)))
    non_finite = make_source(np.full((2, 3), np.nan))

    molecular = make_source(np.zeros((2, 3)), stress=np.zeros(6))
    molecular.pbc = False

    invalid = find_invalid_labels([valid, misshapen, non_finite, molecular])

    assert [record["source_index"] for record in invalid] == [1, 2, 3]
    assert [record["problems"] for record in invalid] == [
        ["forces shape (3, 3)"],
        ["non-finite forces"],
        ["stress on a structure that is not fully periodic"],
    ]


def test_predict_property_records_the_actual_failure():
    """Test failures are flagged and unsupported elements need an explicit error."""

    def raise_error(message):
        raise ValueError(message)

    value, status, message = predict_property(lambda: np.ones((2, 3)), (2, 3), "forces")

    assert (status, message) == ("ok", "")
    assert (value == 1).all()

    cases = [
        (lambda: raise_error("neighbour list overflow"), (2, 3), "error"),
        (lambda: raise_error("unknown element"), (2, 3), "error"),
        (
            lambda: raise_error("this model does not support the atomic type '97'"),
            (2, 3),
            "unsupported_elements",
        ),
        (lambda: np.ones(4), (2, 3), "bad_shape"),
        (lambda: np.nan, (), "non_finite"),
        (lambda: np.inf, (), "non_finite"),
    ]

    for get_property, shape, expected in cases:
        with pytest.warns(UserWarning):
            value, status, message = predict_property(get_property, shape, "forces")

        assert status == expected
        assert np.isnan(value).all()
        assert bool(message) == (expected in ("error", "unsupported_elements"))


def test_align_predictions_matches_ids_and_marks_missing():
    """Test predictions are matched by identity and missing ones are None."""
    reference = [make_structure("mc3d", 0), make_structure("madcat", 0)]
    prediction = make_prediction("madcat", 0)

    aligned = align_predictions(reference, [prediction])

    assert aligned[0] is None
    assert aligned[1] is prediction


def test_align_predictions_rejects_stale_outputs():
    """Test duplicate or unknown identities and changed geometries are rejected."""
    reference = make_structure("mc3d", 0)

    # far from the origin a relative tolerance would accept the shifted structure
    reference.positions += 1e4

    nudged = reference.copy()
    nudged.positions[0, 0] += 1e-7

    shifted = reference.copy()
    shifted.positions[0, 0] += 1e-4

    reordered = make_structure("mc3d", 0, symbols="OH")
    reordered.positions += 1e4

    assert align_predictions([reference], [nudged])[0] is nudged

    with pytest.raises(ValueError, match="Duplicate"):
        align_predictions([reference], [nudged, nudged])

    with pytest.raises(ValueError, match="unknown"):
        align_predictions([reference], [make_structure("mc3d", 1)])

    for changed in (shifted, reordered):
        with pytest.raises(ValueError, match="Geometry"):
            align_predictions([reference], [changed])


def test_density_point_sources_skip_failed_components():
    """Test each plotted point maps to its structure after failed points are dropped."""
    values = {
        "ref": np.array([0.0, 1.0, 2.0, 3.0]),
        "pred": np.array([np.nan, 1.0, 2.0, 3.0]),
        "source_index": np.array([10, 11, 11, 12]),
    }

    assert density_point_sources(values) == [11, 11, 12]


def test_p8_and_sup_match_their_definitions():
    """Test p8 and sup values, their order with MAE and RMSE, and p8 overflow."""
    ref = np.zeros(4)

    assert p8_error(ref, np.array([0.0, 0.0, 0.0, 2.0])) == pytest.approx(
        2 * 0.25**0.125
    )
    assert sup_error(ref, np.array([0.5, -3.0, 1.0, 0.0])) == 3.0

    # the naive eighth power of 1e40 would overflow
    assert p8_error(ref, np.array([1e40, 1e40, 0.0, 0.0])) == pytest.approx(
        1e40 * 0.5**0.125
    )

    rng = np.random.default_rng(0)
    statistics = error_statistics(np.zeros(1000), rng.normal(size=1000))

    assert statistics["MAE"] <= statistics["RMSE"] <= statistics["p8"]
    assert statistics["p8"] <= statistics["sup"]


def test_stress_failure_keeps_valid_forces():
    """Test one failed stress makes every stress statistic NaN but keeps forces."""
    population = []

    for frame_id, stress_status in ((0, "ok"), (1, "error")):
        reference = make_structure("mc3d_rattled", frame_id)
        reference.arrays[FORCES_KEY] = np.ones((2, 3))
        reference.info[STRESS_KEY] = np.ones(6)

        prediction = make_prediction("mc3d_rattled", frame_id, stress=stress_status)
        population.append((reference, prediction))

    forces = collect_property(
        population,
        "forces",
        lambda atoms: atoms.arrays[FORCES_KEY],
        lambda atoms: atoms.arrays[PRED_FORCES_KEY],
    )
    stress = collect_property(
        population,
        "stress",
        lambda atoms: atoms.info[STRESS_KEY],
        lambda atoms: atoms.info[PRED_STRESS_KEY],
    )
    stress_statistics = error_statistics(stress["ref"], stress["pred"])

    assert error_statistics(forces["ref"], forces["pred"])["MAE"] == 1.0
    assert all(np.isnan(value) for value in stress_statistics.values())
    assert property_coverage(population, "stress", None)["outcomes"] == {
        "ok": 1,
        "error": 1,
    }


def test_coverage_keeps_statuses_and_counts_outside_declared_separately():
    """Test declared training elements never change the recorded status."""
    inside = make_structure("mc3d", 0, symbols="HH")
    outside = make_structure("mc3d", 1, symbols="HO")

    population = [
        (inside, make_prediction("mc3d", 0, forces="non_finite")),
        (outside, make_prediction("mc3d", 1, forces="error")),
        (outside, None),
    ]

    coverage = property_coverage(population, "forces", frozenset({"H"}))

    assert coverage["outcomes"] == {"non_finite": 1, "error": 1, "missing": 1}
    assert coverage["outside_declared_elements"] == {"error": 1, "missing": 1}

    undeclared = property_coverage(population, "forces", None)

    assert undeclared["outside_declared_elements"] is None


def test_available_references_require_force_labels_on_every_structure():
    """Test only reference protocols present in the input data are offered."""
    labelled = make_structure("mc3d", 0)
    labelled.arrays[FORCES_KEY] = np.zeros((2, 3))

    unlabelled = make_structure("mc3d", 1)

    assert available_references([labelled]) == [SCORED_REFERENCE]
    assert available_references([labelled, unlabelled]) == []


def test_reference_property_data_uses_protocol_keys_and_gpa():
    """Test any protocol's label keys are used and stress is converted to GPa."""
    reference = make_structure("mc3d_rattled", 0)
    reference.arrays["other_forces"] = np.full((2, 3), 2.0)
    reference.info["other_energy"] = -1.0
    reference.info["other_stress"] = np.full(6, 0.01)

    protocol = {
        "forces_key": "other_forces",
        "energy_key": "other_energy",
        "stress_key": "other_stress",
    }

    prediction = make_prediction("mc3d_rattled", 0)
    prediction.info[PRED_STRESS_KEY] = np.full(6, 0.005)

    data = reference_property_data([(reference, prediction)], protocol, None)

    assert error_statistics(data["forces"]["ref"], data["forces"]["pred"])["MAE"] == 2.0
    assert data["energy"]["outcomes"] == {"ok": 1}
    assert data["stress"]["ref"] == pytest.approx(np.full(6, 0.01 / units.GPa))
    assert data["stress"]["pred"] == pytest.approx(np.full(6, 0.005 / units.GPa))

    without_stress = reference_property_data(
        [(reference, prediction)], protocol | {"stress_key": "absent"}, None
    )

    assert without_stress["stress"]["ref"].size == 0


def make_reference_diagnostics(n_stress: int) -> dict:
    """Build the diagnostics of one reference protocol for one model."""
    unavailable = dict.fromkeys(("MAE", "RMSE", "p8", "sup"))

    return {
        "units": {"forces": "eV/Å", "energy": "eV", "stress": "GPa"},
        "n_structures": {"forces": 20, "energy": 20, "stress": n_stress},
        "models": {
            APP_MODELS[0]: {
                "forces": {
                    "outcomes": {"ok": 17, "unsupported_elements": 3},
                    "outside_declared_elements": {"ok": 1, "unsupported_elements": 3},
                    **unavailable,
                },
                "energy": {
                    "outcomes": {"ok": 17, "error": 3},
                    "outside_declared_elements": {"ok": 1, "error": 3},
                },
                "stress": {
                    "outcomes": {"ok": n_stress},
                    "outside_declared_elements": {},
                    "MAE": 0.001,
                    "RMSE": 0.002,
                    "p8": 0.003,
                    "sup": 0.004,
                },
            }
        },
    }


def test_diagnostics_table_shows_counts_and_hides_unlabelled_properties():
    """Test counts and N/A are shown, and stress columns only when labelled."""
    columns, rows = diagnostics_table_content(make_reference_diagnostics(n_stress=5))
    row = rows[0]

    assert row["Force evaluated / expected"] == "17 / 20"
    assert row["Energy evaluated / expected"] == "17 / 20"
    assert row["Stress evaluated / expected"] == "5 / 5"
    assert row["Force MAE / eV/Å"] == "N/A"
    assert row["Force RMSE / eV/Å"] == "N/A"
    assert row["Stress MAE / GPa"] == "0.001"
    assert row["Failures"] == "forces unsupported_elements 3, energy error 3"
    assert row["Outside declared training elements"] == "4 (3 failed)"
    assert not any("Score" in column["name"] for column in columns)

    columns, _ = diagnostics_table_content(make_reference_diagnostics(n_stress=0))
    names = [column["name"] for column in columns]

    assert "Force RMSE / eV/Å" in names
    assert not any(name.startswith("Stress") for name in names)


def test_unavailable_stress_mae_does_not_change_the_score():
    """Test Stress MAE has weight 0, so N/A stress keeps the force-based score."""
    rows = [
        {"MLIP": "with stress", "Force MAE": 0.2, "Stress MAE": 0.5},
        {"MLIP": "without stress", "Force MAE": 0.2, "Stress MAE": np.nan},
    ]

    scored = calc_table_scores(
        rows, weights=DEFAULT_WEIGHTS, thresholds=DEFAULT_THRESHOLDS
    )

    assert DEFAULT_WEIGHTS["Stress MAE"] == 0
    assert np.isfinite(scored[1]["Score"])
    assert scored[0]["Score"] == scored[1]["Score"]
