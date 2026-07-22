"""Tests for diatomic metric orchestration and aggregation."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ml_peg.analysis.physicality.diatomics import metrics
from ml_peg.analysis.physicality.diatomics.metrics import (
    DEFAULT_DFT_REFERENCE_PATH,
    DIATOMIC_METRIC_KEYS,
    ENERGY_JUMP,
    PBE_ENERGY_MAE,
    PBE_FORCE_MAE,
    TORTUOSITY,
    DiatomicCurve,
    DiatomicCurves,
    aggregate_finite_means,
    calc_diatomic_metrics,
    eval_window,
    find_low_quality_dft_refs,
    load_dft_reference_curves,
)


def _forces_from_energy(
    distances: np.ndarray,
    energies: np.ndarray,
) -> np.ndarray:
    """Construct equal-and-opposite radial forces."""
    forces = np.zeros((len(distances), 2, 3))
    forces[:, 0, 0] = -np.gradient(energies, distances)
    forces[:, 1, 0] = -forces[:, 0, 0]
    return forces


def _make_curves(
    curves_by_element: dict[str, np.ndarray],
    distances: np.ndarray,
) -> DiatomicCurves:
    """Wrap element energy arrays as homonuclear curves."""
    return DiatomicCurves(
        distances=distances,
        homo_nuclear={
            element_symbol: DiatomicCurve(
                distances, energies, _forces_from_energy(distances, energies)
            )
            for element_symbol, energies in curves_by_element.items()
        },
    )


@pytest.mark.parametrize("prediction_key", ["H", "H-H"])
def test_all_metrics_and_reference_key_normalization(prediction_key: str) -> None:
    """Matching element and pair keys both produce all 12 finite metrics."""
    distances = np.linspace(0.2, 3.0, 50)
    energies = 100 * (distances - 1.0) ** 2 - 2
    references = _make_curves({"H": energies}, distances)
    predictions = _make_curves({"H": energies}, distances)
    if prediction_key == "H-H":
        predictions.homo_nuclear[prediction_key] = predictions.homo_nuclear.pop("H")
    result = calc_diatomic_metrics(references, predictions, interpolate=200)
    assert set(result["H"]) == DIATOMIC_METRIC_KEYS
    assert result["H"][PBE_ENERGY_MAE] == pytest.approx(0)
    for metric_value in result["H"].values():
        assert np.isfinite(metric_value)


def test_duplicate_normalized_element_keys_are_rejected() -> None:
    """Reject separate element and pair keys for the same homonuclear curve."""
    distances = np.linspace(0.2, 3.0, 50)
    energies = (distances - 1.0) ** 2
    curves = _make_curves({"H": energies}, distances)
    curves.homo_nuclear["H-H"] = curves.homo_nuclear["H"]

    with pytest.raises(ValueError, match="Duplicate homonuclear curve"):
        calc_diatomic_metrics(None, curves)


def test_homonuclear_only_and_non_mp_filtering() -> None:
    """Imported metrics ignore heteronuclear curves and unsupported MP elements."""
    distances = np.linspace(0.3, 3.0, 20)
    energies = (distances - 1.0) ** 2
    homo_curves = _make_curves(
        {"H": energies, "Po": energies, "Og": energies}, distances
    )
    homo_curves.hetero_nuclear["H-He"] = DiatomicCurve(
        distances, energies, _forces_from_energy(distances, energies)
    )

    result = calc_diatomic_metrics(None, homo_curves)

    assert set(result) == {"H"}
    assert TORTUOSITY in result["H"]


def test_nonfinite_repulsive_wall_sample_skips_element() -> None:
    """Non-finite values in the wider wall window exclude the whole curve."""
    distances = np.linspace(0.5, 3.0, 101)
    energies = (distances - 1.5) ** 2
    curves = _make_curves({"C": energies}, distances)
    wall_only_index = int(np.flatnonzero((distances >= 0.608) & (distances < 0.684))[0])
    curves.homo_nuclear["C"].energies[wall_only_index] = np.nan

    assert calc_diatomic_metrics(None, curves) == {}


def test_low_quality_reference_gate_and_nonfinite_predictions() -> None:
    """Jumpy refs lose PBE metrics while non-finite predictions are skipped."""
    distances = np.linspace(0.3, 6.0, 40)
    smooth = (distances - 1.5) ** 2
    jumpy = smooth + 5 * (-1) ** np.arange(len(distances))
    nonfinite = smooth.copy()
    nonfinite[10] = np.nan
    reference_curves = _make_curves(
        {"H": smooth, "Ho": jumpy, "Er": np.full(len(distances), np.nan)},
        distances,
    )
    predicted_curves = _make_curves(
        {"H": smooth, "Ho": smooth, "Er": smooth, "He": nonfinite},
        distances,
    )

    assert find_low_quality_dft_refs(reference_curves) == {"Ho", "Er"}
    result = calc_diatomic_metrics(reference_curves, predicted_curves)
    assert set(result) == {"H", "Ho", "Er"}
    assert PBE_ENERGY_MAE in result["H"]
    for gated_element in ("Ho", "Er"):
        assert PBE_ENERGY_MAE not in result[gated_element]
        assert TORTUOSITY in result[gated_element]


def test_missing_forces_raise_clear_error() -> None:
    """A scored prediction missing force samples is rejected."""
    distances = np.linspace(0.3, 3.0, 10)
    energies = (distances - 1.0) ** 2
    predicted_curves = _make_curves({"H": energies}, distances)
    predicted_curves.homo_nuclear["H"].forces = np.array([])

    with pytest.raises(ValueError, match="H diatomic curve is missing forces"):
        calc_diatomic_metrics(None, predicted_curves)


def test_full_pipeline_interpolation() -> None:
    """Full relative metrics require matching grids unless interpolation is enabled."""
    reference_distances = np.linspace(0.3, 3.0, 20)
    predicted_distances = reference_distances * 1.001
    reference_energies = (reference_distances - 1.0) ** 2
    predicted_energies = (predicted_distances - 1.0) ** 2
    reference_curves = _make_curves({"H": reference_energies}, reference_distances)
    predicted_curves = _make_curves({"H": predicted_energies}, predicted_distances)
    curve_pair = (reference_curves, predicted_curves)

    with pytest.raises(ValueError, match="distances must be same"):
        calc_diatomic_metrics(*curve_pair, interpolate=False)
    result = calc_diatomic_metrics(*curve_pair, interpolate=200)
    assert set(result["H"]) == DIATOMIC_METRIC_KEYS


def test_force_interpolation_omits_point_only_overlap() -> None:
    """Omit force MAE when reference and prediction ranges only touch."""
    reference_distances = np.linspace(0.4, 1.0, 5)
    predicted_distances = np.linspace(1.0, 3.0, 5)
    reference_curves = _make_curves({"H": reference_distances**2}, reference_distances)
    predicted_curves = _make_curves({"H": predicted_distances**2}, predicted_distances)
    result = calc_diatomic_metrics(
        reference_curves,
        predicted_curves,
        metrics={PBE_FORCE_MAE: {"interpolate": True}},
    )
    assert result["H"] == {}


def test_eval_window_and_repulsive_exclusion() -> None:
    """Element windows use physical radii and exclude deep-overlap spikes."""
    radius_min, radius_max = eval_window("H-H", 2.5)
    atomic_number = metrics.atomic_numbers["H"]
    assert radius_min == pytest.approx(0.9 * metrics.covalent_radii[atomic_number])
    assert radius_max == pytest.approx(2.5)

    distances = np.linspace(0.1, 6.0, 60)
    energies = np.exp(-distances)
    energies[distances < 0.2] = 1e6
    result = calc_diatomic_metrics(
        None,
        _make_curves({"H": energies}, distances),
    )
    assert result["H"][ENERGY_JUMP] == pytest.approx(0)

    source_keyword_window = eval_window(
        elem_symbol="H-H", seps_max=2.5, r_min_factor=0.8
    )
    assert source_keyword_window[0] == pytest.approx(
        0.8 * metrics.covalent_radii[atomic_number]
    )


def test_bundled_pbe_quality_gate_regression() -> None:
    """Bundled PBE data retains the eight known jumpy lanthanide references."""
    reference_curves = load_dft_reference_curves()
    assert find_low_quality_dft_refs(reference_curves) == {
        "Pr",
        "Pm",
        "Sm",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
    }
    self_metrics = calc_diatomic_metrics(
        reference_curves, reference_curves, interpolate=200
    )
    assert len(self_metrics) == 87
    mean_metrics = aggregate_finite_means(self_metrics)
    assert mean_metrics == {
        "tortuosity": 1.043,
        "force_flips": 1.632,
        "energy_jump": 6.28,
        "energy_diff_flips": 2.575,
        "force_total_variation": 193.5,
        "force_jump": 7.164,
        "pbe_wall_dist_mae": 0.0,
        "pbe_energy_mae": 0.0,
        "pbe_bond_length_error": 0.0,
        "pbe_well_depth_error": 0.0,
        "pbe_force_mae": 0.0,
        "pbe_vib_freq_error": 0.0,
    }


def test_bundled_pbe_reference_hash() -> None:
    """Pin the bundled DFT reference file."""
    with open(DEFAULT_DFT_REFERENCE_PATH, "rb") as file:
        digest = hashlib.sha256(file.read()).hexdigest()

    assert digest == "1fe6334a82e98208ea74169a3beaf98cd5188bdc7ac40e518697fd36c7196e3d"


def test_finite_mean_aggregation() -> None:
    """Aggregation unions metric keys and ignores missing or non-finite values."""
    metrics_by_element = {
        "H": {TORTUOSITY: 1.0, ENERGY_JUMP: 2.0},
        "He": {TORTUOSITY: np.nan, ENERGY_JUMP: 4.0},
        "Li": {TORTUOSITY: np.inf},
    }
    assert aggregate_finite_means(metrics_by_element) == {
        TORTUOSITY: 1.0,
        ENERGY_JUMP: 3.0,
    }
    assert aggregate_finite_means(
        {"H": {TORTUOSITY: 1e308}, "He": {TORTUOSITY: 1e308}}
    ) == {TORTUOSITY: 1e308}


def test_unknown_metric_rejected() -> None:
    """The orchestrator rejects names outside the stable 12-key set."""
    distances = np.linspace(0.3, 3.0, 10)
    curves = _make_curves({"H": distances**2}, distances)
    with pytest.raises(ValueError, match="unknown_metrics"):
        calc_diatomic_metrics(None, curves, metrics={"not_a_metric": {}})
