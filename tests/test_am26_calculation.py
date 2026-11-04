"""Synthetic AM26 calculation tests without optional MLIP backends or disk outputs."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.calcs.amorphous_materials.AM26 import calc_AM26 as driver
from ml_peg.models.mock import MockCalculator, MockErrorCalculator


@pytest.fixture
def references():
    """Create three small DFT-like frames with distinct sizes and metadata."""
    frames = []
    for index, symbols in enumerate(("C", "Si2", "O3")):
        atoms = Atoms(
            symbols,
            positions=np.arange(3 * (index + 1)).reshape(-1, 3) / 4 + 0.125,
            cell=[8 + index, 9, 10],
            pbc=[True, bool(index % 2), True],
        )
        energy = -4.0 * len(atoms) + 0.25
        forces = np.arange(len(atoms) * 3).reshape(-1, 3) / 8
        atoms.info.update(
            label=f"frame-{index}",
            system=("carbon", "silicon", "silica")[index],
            energy_per_at=energy / len(atoms),
            virial=np.eye(3) * (index + 1),
            cell_origin=np.zeros(3),
        )
        atoms.calc = SinglePointCalculator(atoms, energy=energy, forces=forces)
        frames.append(atoms)
    return frames


class SelectiveMock(MockCalculator):
    """Mock that clears its cache and can fail one property on the first frame."""

    def __init__(self, failure=None, invalid=None):
        """Choose an exception or invalid output; record calls across copies."""
        super().__init__()
        self.failure = failure
        self.invalid = invalid
        self.calls = []
        self.instances = []

    def calculate(self, atoms=None, properties=None, system_changes=None, **kwargs):
        """Return deliberately unshifted constants or the requested failure."""
        prop = properties[0]
        self.calls.append((atoms.info["label"], prop, deepcopy(atoms.info)))
        self.instances.append(self)
        # Emulate a backend whose force call discards its cached energy.
        self.results = {}
        if atoms.info["label"] == "frame-0" and prop == self.failure:
            raise RuntimeError("synthetic\nproperty failure")
        super().calculate(atoms, properties, system_changes, **kwargs)
        if prop == "energy":
            self.results[prop] = 17.25
        elif prop == "forces":
            self.results[prop] = np.full((len(atoms), 3), 1.25)
        if atoms.info["label"] == "frame-0":
            if prop == self.invalid:
                if prop == "energy":
                    self.results[prop] = np.inf
                else:
                    self.results[prop][0, 0] = np.nan
            elif prop == "forces" and self.invalid == "forces-shape":
                self.results[prop] = np.zeros((len(atoms), 2))


def assert_reference_preserved(output, original, index):
    """Check geometry, metadata, references, and index without scientific analysis."""
    np.testing.assert_array_equal(output.numbers, original.numbers)
    np.testing.assert_array_equal(output.positions, original.positions)
    np.testing.assert_array_equal(output.cell, original.cell)
    np.testing.assert_array_equal(output.pbc, original.pbc)
    for key, value in original.info.items():
        np.testing.assert_array_equal(output.info[key], value)
    assert output.info["input_index"] == index
    assert output.info["ref_energy"] == original.get_potential_energy()
    assert output.info["ref_energy_per_atom"] == original.info["energy_per_at"]
    np.testing.assert_array_equal(output.arrays["ref_forces"], original.get_forces())
    assert "charge" not in output.info
    assert "spin" not in output.info


def round_trip(outputs):
    """Use real ASE extxyz serialization, entirely in memory."""
    stream = StringIO()
    write(stream, outputs, format="extxyz")
    stream.seek(0)
    return read(stream, index=":", format="extxyz")


def test_mock_outputs_preserve_all_frames_and_references(references):
    """Store mock predictions separately and leave source calculators untouched."""
    original_calculators = [atoms.calc for atoms in references]
    outputs = driver.evaluate_structures(references, MockCalculator(), "mock")
    assert len(outputs) == len(references)
    for index, (output, original) in enumerate(zip(outputs, references, strict=True)):
        assert_reference_preserved(output, original, index)
        assert original.calc is original_calculators[index]
        assert "ref_energy" not in original.info
        assert "ref_forces" not in original.arrays
        assert output.get_potential_energy() == 0.0
        np.testing.assert_array_equal(output.get_forces(), np.zeros((len(output), 3)))
        assert not output.info["energy_failed"]
        assert not output.info["forces_failed"]
        assert set(output.calc.results) == {"energy", "forces"}


def test_reference_arrays_and_metadata_are_independent(references):
    """Copies of array-valued reference metadata cannot mutate source frames."""
    output = driver.preserve_reference(references[0], 0)
    assert output.calc is None
    output.arrays["ref_forces"][0, 0] = 123
    output.info["virial"][0, 0] = 456
    assert references[0].get_forces()[0, 0] == 0
    assert references[0].info["virial"][0, 0] == 1


def test_raw_predictions_survive_cache_clearing_without_stress(references):
    """Keep raw constants even when the force call discards cached energy."""
    calculator = SelectiveMock()
    outputs = driver.evaluate_structures(references, calculator, "constant-mock")
    assert [(label, prop) for label, prop, _ in calculator.calls] == [
        (f"frame-{i}", prop) for i in range(3) for prop in ("energy", "forces")
    ]
    assert len({id(calc) for calc in calculator.instances}) == len(references)
    assert all(calc is not calculator for calc in calculator.instances)
    assert calculator.results == {}
    for _, _, info in calculator.calls:
        assert "charge" not in info
        assert "spin" not in info
    for output in outputs:
        assert output.get_potential_energy() == 17.25
        np.testing.assert_array_equal(
            output.get_forces(), np.full((len(output), 3), 1.25)
        )


def test_stale_model_cache_is_not_shared(references):
    """Reset copied calculators without changing the original calculator's cache."""
    calculator = SelectiveMock()
    calculator.atoms = references[0].copy()
    calculator.results = {"energy": -999.0, "forces": np.full((1, 3), -999.0)}
    outputs = driver.evaluate_structures(references, calculator, "stale-cache")
    assert calculator.results["energy"] == -999.0
    np.testing.assert_array_equal(calculator.results["forces"], np.full((1, 3), -999.0))
    assert len(calculator.calls) == 2 * len(references)
    for output in outputs:
        assert output.get_potential_energy() == 17.25
        np.testing.assert_array_equal(
            output.get_forces(), np.full((len(output), 3), 1.25)
        )


@pytest.mark.parametrize("property_name", ["energy", "forces"])
def test_independent_property_exception_and_continuation(references, property_name):
    """Preserve the other property and evaluate later frames after one failure."""
    calculator = SelectiveMock(failure=property_name)
    with pytest.warns(
        UserWarning, match=f"selective: carbon/frame-0 .*{property_name}"
    ):
        outputs = driver.evaluate_structures(references, calculator, "selective")
    assert len(outputs) == len(references)
    first = outputs[0]
    assert first.info[f"{property_name}_failed"]
    assert "\n" not in first.info[f"{property_name}_failure"]
    if property_name == "energy":
        assert np.isnan(first.get_potential_energy())
        np.testing.assert_array_equal(first.get_forces(), np.full((1, 3), 1.25))
        assert not first.info["forces_failed"]
    else:
        assert first.get_potential_energy() == 17.25
        assert first.get_forces().shape == (1, 3)
        assert np.isnan(first.get_forces()).all()
        assert not first.info["energy_failed"]
    for output in outputs[1:]:
        assert output.get_potential_energy() == 17.25
        assert np.isfinite(output.get_forces()).all()
        assert not output.info["energy_failed"]
        assert not output.info["forces_failed"]
    assert len(calculator.calls) == 2 * len(references)
    restored = round_trip(outputs)
    assert restored[0].info[f"{property_name}_failed"]
    for index, (output, original) in enumerate(zip(restored, references, strict=True)):
        assert_reference_preserved(output, original, index)
    if property_name == "energy":
        assert np.isnan(restored[0].get_potential_energy())
    else:
        assert np.isnan(restored[0].get_forces()).all()


def test_mock_error_keeps_every_failed_frame(references):
    """The standard always-failing calculator yields two failures per frame."""
    with pytest.warns(UserWarning) as warnings:
        outputs = driver.evaluate_structures(
            references, MockErrorCalculator(), "mock-error"
        )
    failures = [
        warning
        for warning in warnings
        if issubclass(warning.category, UserWarning)
        and str(warning.message).startswith("mock-error:")
    ]
    assert len(failures) == 2 * len(references)
    for index, (output, original) in enumerate(
        zip(round_trip(outputs), references, strict=True)
    ):
        assert_reference_preserved(output, original, index)
        assert np.isnan(output.get_potential_energy())
        assert np.isnan(output.get_forces()).all()
        assert output.info["energy_failed"]
        assert output.info["forces_failed"]


@pytest.mark.parametrize("invalid", ["energy", "forces", "forces-shape"])
def test_invalid_predictions_are_property_failures(references, invalid):
    """Treat non-finite energies/forces and malformed arrays as failed outputs."""
    with pytest.warns(UserWarning, match="invalid: carbon/frame-0"):
        outputs = driver.evaluate_structures(
            references, SelectiveMock(invalid=invalid), "invalid"
        )
    first = outputs[0]
    assert len(outputs) == len(references)
    if invalid == "energy":
        assert first.info["energy_failed"]
        assert np.isnan(first.get_potential_energy())
        assert np.isfinite(first.get_forces()).all()
    else:
        assert first.info["forces_failed"]
        assert first.get_forces().shape == (1, 3)
        assert np.isnan(first.get_forces()).all()
        assert first.get_potential_energy() == 17.25


def test_successful_extxyz_round_trip(references):
    """Standard ASE energy/forces and explicit DFT fields remain distinguishable."""
    outputs = driver.evaluate_structures(references, SelectiveMock(), "round-trip")
    for index, (output, original) in enumerate(
        zip(round_trip(outputs), references, strict=True)
    ):
        assert_reference_preserved(output, original, index)
        assert output.get_potential_energy() == 17.25
        np.testing.assert_array_equal(
            output.get_forces(), np.full((len(output), 3), 1.25)
        )
        assert set(output.calc.results) == {"energy", "forces"}


@pytest.fixture
def isolated_driver(monkeypatch, references):
    """Exercise driver wiring while keeping all output serialization in memory."""
    directory = MagicMock(spec=Path)
    destination = Path("synthetic-model") / "am26_results.extxyz"
    directory.__truediv__.return_value = destination
    root = MagicMock(spec=Path)
    root.__truediv__.return_value = directory
    monkeypatch.setattr(driver, "OUT_PATH", root)
    monkeypatch.setattr(driver, "load_reference_structures", lambda: references)
    serialized = StringIO()
    writer = Mock(
        side_effect=lambda path, frames, **kwargs: write(serialized, frames, **kwargs)
    )
    monkeypatch.setattr(driver, "write", writer)
    return SimpleNamespace(
        root=root,
        directory=directory,
        destination=destination,
        writer=writer,
        serialized=serialized,
    )


def test_driver_requests_high_precision_and_standard_output(
    isolated_driver, references
):
    """Use the standard wrapper factory and model-specific extxyz output path."""
    model = SimpleNamespace(get_calculator=Mock(return_value=MockCalculator()))
    driver.test_am26(("synthetic-model", model))
    model.get_calculator.assert_called_once_with(precision="high")
    isolated_driver.root.__truediv__.assert_called_once_with("synthetic-model")
    isolated_driver.directory.mkdir.assert_called_once_with(parents=True, exist_ok=True)
    assert isolated_driver.writer.call_args.args[0] == isolated_driver.destination
    assert isolated_driver.writer.call_args.kwargs == {"format": "extxyz"}
    isolated_driver.serialized.seek(0)
    frames = read(isolated_driver.serialized, index=":", format="extxyz")
    assert len(frames) == len(references)


def test_initialisation_failure_creates_no_output(isolated_driver):
    """A failed model factory raises normally without a fabricated result file."""
    model = SimpleNamespace(
        get_calculator=Mock(side_effect=RuntimeError("initialisation failed"))
    )
    with pytest.raises(RuntimeError, match="initialisation failed"):
        driver.test_am26(("synthetic-model", model))
    isolated_driver.directory.mkdir.assert_not_called()
    isolated_driver.writer.assert_not_called()


def test_wrong_dataset_hash_rejected_before_read(monkeypatch):
    """Reject another dataset before loading structures or constructing a model."""
    reader = Mock()
    monkeypatch.setattr(
        driver, "get_reference_path", Mock(side_effect=ValueError("SHA-256 mismatch"))
    )
    monkeypatch.setattr(driver, "read", reader)
    with pytest.raises(ValueError, match="SHA-256"):
        driver.load_reference_structures()
    reader.assert_not_called()


def test_incomplete_dataset_is_rejected(monkeypatch, references):
    """The driver cannot silently run a subset of the canonical dataset."""
    path = Mock()
    monkeypatch.setattr(driver, "get_reference_path", Mock(return_value=path))
    monkeypatch.setattr(driver, "read", Mock(return_value=references))
    with pytest.raises(ValueError, match="934 frames"):
        driver.load_reference_structures()


def test_duplicate_input_labels_are_rejected(monkeypatch, references):
    """Ambiguous input identities must fail before running any model."""
    path = Mock()
    references[1].info["label"] = references[0].info["label"]
    monkeypatch.setattr(driver, "get_reference_path", Mock(return_value=path))
    monkeypatch.setattr(
        driver, "REFERENCE_COUNTS", Counter(a.info["system"] for a in references)
    )
    monkeypatch.setattr(driver, "read", Mock(return_value=references))
    with pytest.raises(ValueError, match="unique"):
        driver.load_reference_structures()
