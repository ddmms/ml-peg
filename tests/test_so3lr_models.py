"""Tests for original SO3LR registration and JAX device/precision routing.

Prepared with AI assistance; human review is still required before submission.
"""

from __future__ import annotations

from contextlib import contextmanager
import sys
from types import ModuleType, SimpleNamespace

import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import SO3LRCalc


@pytest.fixture
def backend(monkeypatch):
    """Provide a recording JAX/SO3LR pair without optional model dependencies."""
    state = SimpleNamespace(current=None, x64=False, calls=[], gpu=True)
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

        def calculate(atoms=None, properties=None, system_changes=None):
            return state.current, atoms

        return SimpleNamespace(calculate=calculate)

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
    for atoms in ["original geometry", "changed geometry and cell"]:
        assert calc.calculate(atoms=atoms) == (expected, atoms)
        assert backend.current is None


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
