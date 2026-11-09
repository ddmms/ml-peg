"""Tests for AIMNet2 model registration and calculator construction."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

from ase.build import molecule
import numpy as np
import pytest
import yaml

from ml_peg.models.get_models import load_models
from ml_peg.models.models import AIMNetCalc

AIMNET_MODELS = ("aimnet2", "aimnet2-2025")
AIMNET_ELEMENTS = set("H B C N O F Si P S Cl As Se Br I".split())

# Reference predictions at the pinned aimnet revision for ``molecule(formula)``
# with ``info["charge"] = charge``. Units: eV and eV/Å.
REFERENCE = {
    ("aimnet2", "H2O", 0): (
        -2081.040875,
        [
            [0.0, 0.0, -0.684212],
            [0.0, -0.348737, 0.342106],
            [0.0, 0.348737, 0.342106],
        ],
    ),
    ("aimnet2", "OH", -1): (
        -2063.335538,
        [[0.0, 0.0, -0.522833], [0.0, 0.0, 0.522833]],
    ),
    ("aimnet2-2025", "H2O", 0): (
        -2078.904087,
        [
            [0.0, 0.0, -0.292685],
            [0.0, -0.231222, 0.146343],
            [0.0, 0.231221, 0.146343],
        ],
    ),
    ("aimnet2-2025", "OH", -1): (
        -2061.295099,
        [[0.0, 0.0, -0.363737], [0.0, 0.0, 0.363737]],
    ),
}


class _FakeAIMNet2Calculator:
    """Minimal stand-in for ``AIMNet2Calculator``."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeAIMNet2ASE:
    """Minimal stand-in for ``AIMNet2ASE``."""

    def __init__(self, base_calc):
        self.base_calc = base_calc


@pytest.fixture
def fake_aimnet(monkeypatch):
    """Provide a fake ``aimnet.calculators`` module."""
    aimnet_module = ModuleType("aimnet")
    calculators_module = ModuleType("aimnet.calculators")
    calculators_module.AIMNet2Calculator = _FakeAIMNet2Calculator
    calculators_module.AIMNet2ASE = _FakeAIMNet2ASE
    aimnet_module.calculators = calculators_module
    monkeypatch.setitem(sys.modules, "aimnet", aimnet_module)
    monkeypatch.setitem(sys.modules, "aimnet.calculators", calculators_module)


def test_aimnet_registry_metadata():
    """AIMNet2 entries record their native precision, D3 and element coverage."""
    root = Path(__file__).parents[1]
    registry = yaml.safe_load(
        (root / "ml_peg" / "models" / "models.yml").read_text(encoding="utf8")
    )
    coverage = yaml.safe_load(
        (root / "ml_peg" / "app" / "data" / "element_coverage.json").read_text(
            encoding="utf8"
        )
    )["datasets"]

    assert registry["aimnet2"]["datasets"] == ["AIMNet2"]
    assert registry["aimnet2-2025"]["datasets"] == ["AIMNet2", "AIMNet2-2025"]
    for name in AIMNET_MODELS:
        config = registry[name]
        assert config["class_name"] == "AIMNet2ASE"
        assert config["kwargs"]["model"] == name
        assert config["trained_on_dispersion"]
        assert "overwrite_dtype" not in config
        assert "additional_supported_elements" not in config
        supported = set()
        for dataset in config["datasets"]:
            supported.update(coverage[dataset]["supported"])
        assert supported == AIMNET_ELEMENTS


def test_aimnet_models_use_dedicated_wrapper(fake_aimnet):
    """All registered AIMNet2 models route through the dedicated wrapper."""
    models = load_models(AIMNET_MODELS)

    assert tuple(models) == AIMNET_MODELS
    assert all(isinstance(model, AIMNetCalc) for model in models.values())
    assert all(model.available for model in models.values())


@pytest.mark.parametrize("precision", ["low", "high"])
def test_aimnet_passes_model_and_device(fake_aimnet, precision):
    """The wrapper builds an ``AIMNet2Calculator`` and wraps it for ASE."""
    model = AIMNetCalc(device="cpu", kwargs={"model": "aimnet2"})

    calculator = model.get_calculator(precision=precision)

    assert isinstance(calculator, _FakeAIMNet2ASE)
    assert calculator.base_calc.kwargs == {"model": "aimnet2", "device": "cpu"}


def test_aimnet_rejects_device_in_kwargs(fake_aimnet):
    """The device is only set by ``device``, not by configured or passed kwargs."""
    model = AIMNetCalc(device="cpu", kwargs={"model": "aimnet2", "device": "cuda"})
    with pytest.raises(ValueError, match="device is set by ML-PEG"):
        model.get_calculator(precision="low")

    model = AIMNetCalc(device="cpu", kwargs={"model": "aimnet2"})
    with pytest.raises(ValueError, match="device is set by ML-PEG"):
        model.get_calculator(precision="low", device="cuda")


def test_aimnet_rejects_unknown_kwargs():
    """Misspelled keyword arguments are rejected rather than silently ignored."""
    pytest.importorskip("aimnet")
    model = AIMNetCalc(device="cpu", kwargs={"modle": "aimnet2"})
    with pytest.raises(ValueError, match="modle would be ignored"):
        model.get_calculator(precision="low")


def test_aimnet_rejects_unknown_precision(fake_aimnet):
    """The AIMNet2 wrapper rejects unknown ML-PEG precision choices."""
    model = AIMNetCalc(kwargs={"model": "aimnet2"})

    with pytest.raises(ValueError, match="Invalid precision"):
        model.get_calculator(precision="medium")


@pytest.mark.parametrize("model_name", AIMNET_MODELS)
def test_registered_aimnet_model_evaluates(model_name):
    """Run a real checkpoint on a neutral molecule and a charged ion."""
    pytest.importorskip("aimnet")
    model = load_models((model_name,))[model_name]
    calculator = model.get_calculator(precision="low")

    water = molecule("H2O")
    water.calc = calculator
    neutral_energy = water.get_potential_energy()
    assert np.isfinite(neutral_energy)
    assert np.isfinite(water.get_forces()).all()

    hydroxide = molecule("OH")
    hydroxide.info["charge"] = -1
    hydroxide.calc = calculator
    assert np.isfinite(hydroxide.get_potential_energy())
    assert np.isfinite(hydroxide.get_forces()).all()
    assert hydroxide.calc.charge == -1

    # Now test on neutral molecule again and check that the charge is reset to 0
    water.calc = calculator
    assert water.get_potential_energy() == pytest.approx(neutral_energy, abs=1e-5)
    assert water.calc.charge == 0


@pytest.mark.parametrize(("model_name", "formula", "charge"), REFERENCE)
def test_aimnet_reference_energy_and_forces(model_name, formula, charge):
    """The pinned checkpoints reproduce reference energies and forces."""
    pytest.importorskip("aimnet")
    model = load_models((model_name,))[model_name]
    energy, forces = REFERENCE[(model_name, formula, charge)]

    atoms = molecule(formula)
    atoms.info["charge"] = charge
    atoms.calc = model.get_calculator(precision="low")

    assert atoms.get_potential_energy() == pytest.approx(energy, abs=1e-4)
    np.testing.assert_allclose(atoms.get_forces(), forces, atol=1e-4)


def test_aimnet_reused_calculator_is_history_independent():
    """A sparse evaluation must not truncate neighbours for a later dense one."""
    pytest.importorskip("aimnet")
    model = load_models(("aimnet2",))["aimnet2"]
    calculator = model.get_calculator(precision="low")

    dense = molecule("CH4", cell=[3.5, 3.5, 3.5], pbc=True)
    sparse = molecule("CH4", vacuum=10.0, pbc=True)

    energies = []
    for atoms in (dense, sparse, dense):
        atoms = atoms.copy()
        atoms.calc = calculator
        energies.append(atoms.get_potential_energy())

    assert energies[2] == pytest.approx(energies[0], abs=1e-4)
