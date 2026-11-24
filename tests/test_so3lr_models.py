"""Tests for original SO3LR registration and JAX device/precision routing.

Prepared with AI assistance; human review is still required before submission.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import copy, deepcopy
import sys
from types import ModuleType, SimpleNamespace

from ase.build import molecule
from ase.calculators.calculator import Calculator, PropertyNotImplementedError
import numpy as np
import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import SO3LRCalc


@pytest.fixture
def backend(monkeypatch):
    """Provide a recording JAX/SO3LR pair without optional model dependencies."""
    state = SimpleNamespace(current=None, x64=False, calls=[], evaluations=[], gpu=True)
    jax = ModuleType("jax")

    def update(key, value):
        assert key == "jax_enable_x64"
        state.x64 = value

    def devices(platform=None):
        if platform == "gpu" and not state.gpu:
            raise RuntimeError("No GPU backend")
        return ["cpu:0"] if platform == "cpu" else ["gpu:0", "gpu:1"]

    @contextmanager
    def default_device(device):
        previous = state.current
        state.current = device
        try:
            yield
        finally:
            state.current = previous

    jax.config = SimpleNamespace(update=update)
    jax.devices = devices
    jax.default_device = default_device
    so3lr = ModuleType("so3lr")

    class Backend(Calculator):
        implemented_properties = ["energy", "forces", "stress"]

        def __init__(self, precision, stress):
            super().__init__()
            self.precision = precision
            self.dtype = np.dtype(precision)
            self.stress = stress
            self.neighbors = self.spatial_partitioning = None

        def calculate(self, atoms, properties, system_changes):
            super().calculate(atoms, properties, system_changes)
            assert not self.stress or atoms.pbc.all()
            if self.spatial_partitioning is None:
                assert self.neighbors is None
                self.neighbors = SimpleNamespace(reference_positions=atoms.positions)
                self.spatial_partitioning = object()
            else:
                assert self.neighbors.reference_positions.shape == atoms.positions.shape
            state.evaluations.append((state.current, self))
            self.results = {"energy": float((atoms.numbers**2).sum())}
            if self.stress:
                self.results["stress"] = np.zeros(6)

    def calculator(
        model=None,
        precision=None,
        calculate_stress=False,
        lr_cutoff=12.0,
        dispersion_energy_cutoff_lr_damping=2.0,
    ):
        state.calls.append(
            {"model": model, "precision": precision, "device": state.current}
        )

        return Backend(precision, calculate_stress)

    so3lr.So3lrCalculator = calculator
    monkeypatch.setitem(sys.modules, "jax", jax)
    monkeypatch.setitem(sys.modules, "so3lr", so3lr)
    return state


@pytest.mark.parametrize(
    ("precision", "dtype"), [("low", "float32"), ("high", "float64")]
)
def test_so3lr_precision_and_explicit_legacy_model(backend, precision, dtype):
    """Never select the upstream Tauri default when loading the v1 entry."""
    model = SO3LRCalc(device="cpu", kwargs={"model": "so3lr"})
    model.get_calculator(precision=precision)
    assert backend.calls == [{"model": "so3lr", "precision": dtype, "device": "cpu:0"}]
    assert backend.x64 is (precision == "high")
    assert model.kwargs == {"model": "so3lr"}


@pytest.mark.parametrize(
    ("device", "expected"),
    [
        ("auto", "gpu:0"),
        (None, "gpu:0"),
        ("cpu", "cpu:0"),
        ("cuda", "gpu:0"),
        ("cuda:1", "gpu:1"),
    ],
)
def test_so3lr_device_used_for_construction_and_later_ase_calls(
    backend, device, expected
):
    """Subsequent evaluations retain placement after the constructor context exits."""
    calc = SO3LRCalc(device=device, kwargs={"model": "so3lr"}).get_calculator(
        precision="low"
    )
    assert backend.calls[-1]["device"] == expected
    assert backend.current is None
    atoms = molecule("H2O")
    atoms.calc = calc
    for shift in (0, 0.02):
        atoms.positions[0, 0] += shift
        assert np.isfinite(atoms.get_potential_energy())
        assert backend.evaluations[-1][0] == expected
        assert backend.current is None


def test_so3lr_switches_stress_mode_on_one_calculator(backend):
    """Use the periodic stress kernel only for truly periodic structures."""
    calc = load_models("so3lr-v1")["so3lr-v1"].get_calculator(precision="high")
    atoms = molecule("H2O")
    atoms.calc = calc
    isolated_energy = atoms.get_potential_energy()
    atoms.cell = [30, 30, 30]
    assert atoms.get_potential_energy() == isolated_energy
    with pytest.raises(PropertyNotImplementedError):
        atoms.get_stress()
    atoms.pbc = True
    assert atoms.get_stress().shape == (6,)
    atoms.pbc = False
    assert atoms.get_potential_energy() == isolated_energy
    assert "stress" not in calc.results
    assert len(backend.calls) == 2


def test_so3lr_rebuilds_glp_cache_when_atom_count_changes(backend):
    """Keep a compatible neighbour state for same-sized molecules."""
    calc = SO3LRCalc().get_calculator(precision="low")
    partitions = []
    for formula in ("H2O", "SH2", "CH4", "H2O"):
        atoms = molecule(formula)
        atoms.calc = calc
        assert atoms.get_potential_energy() == (atoms.numbers**2).sum()
        partitions.append(backend.evaluations[-1][1].spatial_partitioning)
    assert partitions[0] is partitions[1]
    assert len(set(partitions)) == 3
    assert len(backend.calls) == 1


@pytest.mark.parametrize("clone", [copy, deepcopy])
def test_so3lr_copied_calculator_updates_its_own_state(backend, clone):
    """A copy of an evaluated calculator has independent ASE and neighbour state."""
    original = molecule("H2O")
    original.calc = SO3LRCalc().get_calculator(precision="high")
    energy = original.get_potential_energy()
    original_backend = backend.evaluations[-1][1]
    neighbours = original_backend.neighbors
    other = molecule("CH4")
    other.calc = clone(original.calc)
    assert other.get_potential_energy() != energy
    assert backend.evaluations[-1][1] is not original_backend
    assert original_backend.neighbors is neighbours
    assert original.calc.results["energy"] == energy
    assert len(backend.calls) == 1


def test_so3lr_explicit_unavailable_gpu_does_not_fall_back_to_cpu(backend):
    """An explicit GPU request cannot silently become a CPU calculation."""
    backend.gpu = False
    with pytest.raises(RuntimeError, match="unavailable"):
        SO3LRCalc(device="cuda").get_calculator(precision="low")
    assert not backend.calls


@pytest.mark.parametrize("device", ["mps", "cuda:invalid", "cuda:-1", "cpu:99"])
def test_so3lr_invalid_or_missing_device(backend, device):
    """Reject unsupported devices and out-of-range device indices."""
    with pytest.raises((ValueError, RuntimeError)):
        SO3LRCalc(device=device).get_calculator(precision="low")


def test_so3lr_dtype_override_and_low_precision_preserve_global_x64(backend):
    """An override wins, and float32 construction does not turn global x64 off."""
    SO3LRCalc(default_dtype="float64").get_calculator(precision="low")
    assert backend.calls[-1]["precision"] == "float64"
    SO3LRCalc().get_calculator(precision="low")
    assert backend.calls[-1]["precision"] == "float32"
    assert backend.x64


@pytest.mark.parametrize("key", ["device", "default_dtype", "dtype", "precision"])
def test_so3lr_configured_kwargs_cannot_override_reserved_arguments(backend, key):
    """Use registry device/overwrite_dtype and the benchmark's precision request."""
    with pytest.raises(ValueError, match="set by ML-PEG"):
        SO3LRCalc(kwargs={key: "wrong"}).get_calculator(precision="high")


def test_so3lr_rejects_unknown_constructor_options(backend):
    """A misspelled cutoff must not silently select a backend default."""
    with pytest.raises(ValueError, match="would be ignored"):
        SO3LRCalc(kwargs={"lr_cuttof": 8}).get_calculator(precision="low")


@pytest.mark.parametrize("dtype", ["float16", "mixed"])
def test_so3lr_invalid_dtype(backend, dtype):
    """Only the two precision modes mapped by ML-PEG are supported here."""
    with pytest.raises(ValueError, match="float32 or float64"):
        SO3LRCalc(default_dtype=dtype).get_calculator(precision="high")


def test_so3lr_invalid_precision(backend):
    """Check benchmark precision before attempting to load the backend."""
    with pytest.raises(ValueError, match="precision"):
        SO3LRCalc().get_calculator(precision="mixed")


def test_so3lr_default_registry_and_no_additional_dispersion():
    """The public entry selects legacy weights and already contains dispersion."""
    model = load_models("so3lr-v1")["so3lr-v1"]
    assert isinstance(model, SO3LRCalc)
    assert model.kwargs["model"] == "so3lr"
    assert model.kwargs["calculate_stress"]
    assert model.trained_on_dispersion
    sentinel = object()
    assert model.add_d3_calculator(sentinel) is sentinel
