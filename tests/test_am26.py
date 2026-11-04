"""Small synthetic tests of the fixed AM26 metric definitions."""

from __future__ import annotations

import numpy as np
import pytest

from ml_peg.analysis.amorphous_materials.AM26.metrics import (
    SYSTEM_LABELS,
    SYSTEM_ORDER,
    calculate_am26_metrics,
    calculate_system_metrics,
    mean_system_metrics,
    reference_normalisers,
    relrmse_from_rmse,
)


@pytest.fixture
def unequal_structures():
    """Provide two frames with one and three atoms and nonconstant references."""
    return {
        "reference_energies": [0.0, 12.0],  # Per atom: 0, 4.
        "predicted_energies": [2.0, 12.0],  # Per-atom residuals: 2, 0.
        "atom_counts": [1, 3],
        "reference_forces": [
            np.array([[1.0, -1.0, 0.0]]),
            np.array([[1.0, -1.0, 0.0]] * 3),
        ],
        "predicted_forces": [
            np.array([[-1.0, 1.0, 0.0]]),
            np.array([[1.0, 0.0, -1.0]] * 3),
        ],
    }


def test_per_atom_energy_conversion_and_structure_weighting(unequal_structures):
    """A large frame does not receive extra weight in energy RMSE."""
    metrics = calculate_system_metrics(**unequal_structures)
    # sqrt((2**2 + 0**2) / 2), rather than atom-weighted RMSE (= 1).
    assert metrics["energy_rmse"] == pytest.approx(np.sqrt(2.0))
    assert metrics["Energy RelRMSE"] == pytest.approx(50 * np.sqrt(2.0))
    assert metrics["energy_rmse"] != pytest.approx(1.0)


def test_population_reference_standard_deviations(unequal_structures):
    """Use ddof=0 for both the per-atom energies and force components."""
    normalisers = reference_normalisers(
        unequal_structures["reference_energies"],
        unequal_structures["atom_counts"],
        unequal_structures["reference_forces"],
    )
    assert normalisers["energy"] == pytest.approx(2.0)
    assert normalisers["energy"] != pytest.approx(np.sqrt(8.0))
    assert normalisers["force"] == pytest.approx(np.sqrt(2.0 / 3.0))
    assert normalisers["force"] != pytest.approx(np.sqrt(8.0 / 11.0))


def test_force_components_are_flattened_and_weighted_equally(unequal_structures):
    """All 12 components count; do not average the two frame-level RMSEs."""
    metrics = calculate_system_metrics(**unequal_structures)
    # Squared errors: 8 in the small frame and 6 in the large frame.
    assert metrics["force_rmse"] == pytest.approx(np.sqrt(14.0 / 12.0))
    assert metrics["Force RelRMSE"] == pytest.approx(100 * np.sqrt(14.0 / 8.0))
    frame_rmse_average = (np.sqrt(8.0 / 3.0) + np.sqrt(6.0 / 9.0)) / 2
    assert metrics["force_rmse"] != pytest.approx(frame_rmse_average)


def test_force_vector_magnitudes_are_not_used(unequal_structures):
    """Reversing forces preserves vector norms but must incur component error."""
    for ref, pred in zip(
        unequal_structures["reference_forces"],
        unequal_structures["predicted_forces"],
        strict=True,
    ):
        np.testing.assert_array_equal(
            np.linalg.norm(ref, axis=1), np.linalg.norm(pred, axis=1)
        )
    assert calculate_system_metrics(**unequal_structures)["force_rmse"] > 0


def test_raw_constant_energy_offset_is_retained(unequal_structures):
    """A constant +3 eV/atom bias has RMSE 3, without residual centering."""
    unequal_structures["predicted_energies"] = [3.0, 21.0]
    metrics = calculate_system_metrics(**unequal_structures)
    assert metrics["energy_rmse"] == pytest.approx(3.0)
    assert metrics["Energy RelRMSE"] == pytest.approx(150.0)


@pytest.mark.parametrize("property_name", ["energy", "force"])
def test_required_nan_propagates_independently(unequal_structures, property_name):
    """One failed value invalidates only its property's system result."""
    if property_name == "energy":
        unequal_structures["predicted_energies"][1] = np.nan
    else:
        unequal_structures["predicted_forces"][1][2, 1] = np.nan
    metrics = calculate_system_metrics(**unequal_structures)
    affected = "Energy" if property_name == "energy" else "Force"
    unaffected = "Force" if property_name == "energy" else "Energy"
    assert np.isnan(metrics[f"{property_name}_rmse"])
    assert np.isnan(metrics[f"{affected} RelRMSE"])
    assert np.isfinite(metrics[f"{unaffected} RelRMSE"])


def test_fixed_five_system_mapping_and_order():
    """Stored names and presentation labels have the authoritative order."""
    assert SYSTEM_ORDER == ("carbon", "silicon", "silica", "gst", "lips")
    assert tuple(SYSTEM_LABELS.values()) == (
        "a-C",
        "a-Si",
        "a-SiO2",
        "a-GST",
        "a-LiPS",
    )


def test_grouping_and_equal_system_weighting():
    """Different frame counts and input order do not change system weights."""
    systems, reference, predicted, counts, forces, predicted_forces = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for index, system in reversed(list(enumerate(SYSTEM_ORDER))):
        # Each system has a different number of repeated reference pairs.
        for _ in range(index + 1):
            systems.extend([system, system])
            reference.extend([0.0, 2.0])
            predicted.extend([float(index), float(index + 2)])
            counts.extend([1, 1])
            forces.extend([np.array([[1.0, -1.0, 0.0]])] * 2)
            predicted_forces.extend([np.array([[1.0, -1.0, 0.0]]) + index] * 2)
    metrics, means = calculate_am26_metrics(
        systems, reference, predicted, counts, forces, predicted_forces
    )
    assert tuple(metrics) == SYSTEM_ORDER
    for index, system in enumerate(SYSTEM_ORDER):
        assert metrics[system]["Energy RelRMSE"] == pytest.approx(index * 100)
        assert metrics[system]["Force RelRMSE"] == pytest.approx(
            index * 100 / np.sqrt(2.0 / 3.0)
        )
    assert means["Mean Energy RelRMSE"] == pytest.approx(200.0)
    assert means["Mean Force RelRMSE"] == pytest.approx(200 / np.sqrt(2.0 / 3.0))
    assert means["Mean Energy RelRMSE"] != pytest.approx(800.0 / 3.0)


def test_each_system_uses_its_own_reference_spreads():
    """Distinct reference spreads require independent system normalisation."""
    systems, reference, predicted, counts, forces, predicted_forces = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    energy_stds, force_stds = [], []
    for rank, system in enumerate(SYSTEM_ORDER, start=1):
        # Two one-atom frames: energy std = rank; force-component std =
        # (6 - rank) * sqrt(2/3). Different means also distinguish pooled data.
        energy_pair = np.array([10 * rank - rank, 10 * rank + rank], dtype=float)
        amplitude = 6 - rank
        force = np.array([[-amplitude, 0.0, amplitude]]) + 3 * rank
        systems.extend([system, system])
        reference.extend(energy_pair)
        predicted.extend(energy_pair + 2.0)
        counts.extend([1, 1])
        forces.extend([force.copy(), force.copy()])
        predicted_forces.extend([force + 0.5, force + 0.5])
        energy_stds.append(float(rank))
        force_stds.append(amplitude * np.sqrt(2 / 3))

    metrics, means = calculate_am26_metrics(
        systems, reference, predicted, counts, forces, predicted_forces
    )
    # The constant component errors give RMSEs of 2 eV/atom and 0.5 eV/angstrom.
    expected_energy = 200 / np.array(energy_stds)
    expected_force = 50 / np.array(force_stds)
    pooled_energy = 200 / np.std(reference, ddof=0)
    pooled_force = 50 / np.std(np.concatenate(forces).ravel(), ddof=0)
    for index, system in enumerate(SYSTEM_ORDER):
        values = metrics[system]
        assert values["energy_std"] == pytest.approx(energy_stds[index])
        assert values["force_std"] == pytest.approx(force_stds[index])
        assert values["Energy RelRMSE"] == pytest.approx(expected_energy[index])
        assert values["Force RelRMSE"] == pytest.approx(expected_force[index])
        assert values["Energy RelRMSE"] != pytest.approx(pooled_energy)
        assert values["Force RelRMSE"] != pytest.approx(pooled_force)
    assert means["Mean Energy RelRMSE"] == pytest.approx(np.mean(expected_energy))
    assert means["Mean Force RelRMSE"] == pytest.approx(np.mean(expected_force))
    assert means["Mean Energy RelRMSE"] != pytest.approx(pooled_energy)
    assert means["Mean Force RelRMSE"] != pytest.approx(pooled_force)


@pytest.mark.parametrize("property_name", ["Energy", "Force"])
def test_nan_system_propagates_to_mean(property_name):
    """Use ordinary mean: one NaN cannot become a four-system score."""
    metrics = {
        system: {"Energy RelRMSE": 20.0, "Force RelRMSE": 30.0}
        for system in reversed(SYSTEM_ORDER)
    }
    metrics["silica"][f"{property_name} RelRMSE"] = np.nan
    means = mean_system_metrics(metrics)
    assert np.isnan(means[f"Mean {property_name} RelRMSE"])
    other = "Force" if property_name == "Energy" else "Energy"
    assert means[f"Mean {other} RelRMSE"] == pytest.approx(
        30.0 if other == "Force" else 20.0
    )


def test_missing_system_cannot_be_averaged():
    """Reject an incomplete mapping instead of producing a subset mean."""
    metrics = {
        system: {"Energy RelRMSE": 0.0, "Force RelRMSE": 0.0}
        for system in SYSTEM_ORDER[:-1]
    }
    with pytest.raises(ValueError, match="five named systems"):
        mean_system_metrics(metrics)


@pytest.mark.parametrize("property_name", ["Energy", "Force"])
def test_nan_prediction_propagates_through_grouping(unequal_structures, property_name):
    """A single component/frame failure survives the full five-system pipeline."""
    systems = [system for system in SYSTEM_ORDER for _ in range(2)]
    reference = unequal_structures["reference_energies"] * 5
    predicted = unequal_structures["predicted_energies"] * 5
    counts = unequal_structures["atom_counts"] * 5
    forces = unequal_structures["reference_forces"] * 5
    predicted_forces = [
        force.copy() for force in unequal_structures["predicted_forces"] * 5
    ]
    if property_name == "Energy":
        predicted[5] = np.nan  # Second silica frame.
    else:
        predicted_forces[5][2, 1] = np.nan
    metrics, means = calculate_am26_metrics(
        systems, reference, predicted, counts, forces, predicted_forces
    )
    assert np.isnan(metrics["silica"][f"{property_name} RelRMSE"])
    assert np.isnan(means[f"Mean {property_name} RelRMSE"])
    assert all(
        np.isfinite(metrics[system][f"{property_name} RelRMSE"])
        for system in SYSTEM_ORDER
        if system != "silica"
    )


def test_force_shape_must_match_atom_count(unequal_structures):
    """Reject missing components instead of silently shortening the dataset."""
    unequal_structures["predicted_forces"][1] = np.zeros((2, 3))
    with pytest.raises(ValueError, match="shape"):
        calculate_system_metrics(**unequal_structures)


@pytest.mark.parametrize("reference_std", [0.0, np.nan, np.inf])
def test_invalid_normaliser_is_rejected(reference_std):
    """An undefined reference normalisation cannot produce a valid score."""
    with pytest.raises(ValueError, match="standard deviation"):
        relrmse_from_rmse(1.0, reference_std)


def test_fixed_rmse_conversion():
    """Normalise fixed absolute RMSEs in matching units, without inference."""
    assert relrmse_from_rmse(0.03, 0.06) == pytest.approx(50.0)
    assert np.isnan(relrmse_from_rmse(np.nan, 0.06))
