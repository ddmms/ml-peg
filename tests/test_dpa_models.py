"""Tests for DPA model registration and calculator construction."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from types import ModuleType

from ase.build import bulk
from ase.data import chemical_symbols
import numpy as np
import pytest
import yaml

from ml_peg.models.get_models import load_models
from ml_peg.models.models import DpaCalc

DPA_MODELS = (
    "dpa-3p3-1M-omat",
    "dpa-4-nano-omat",
    "dpa-4-neo-omat",
    "dpa-4-plus-omat",
)
DPA3_DATASETS = ["OpenLAM-v1", "OMAT", "MPtrj", "OC20", "OC22", "ODAC23", "SPICE2"]
RUN_DPA_MODEL_TESTS = os.environ.get("ML_PEG_RUN_DPA_MODEL_TESTS") == "1"


class _FakeDP:
    """Minimal stand-in for the DeePMD ASE calculator."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs


@pytest.fixture
def fake_deepmd(monkeypatch):
    """Provide a fake ``deepmd.calculator`` module."""
    deepmd_module = ModuleType("deepmd")
    calculator_module = ModuleType("deepmd.calculator")
    calculator_module.DP = _FakeDP
    deepmd_module.calculator = calculator_module
    monkeypatch.setitem(sys.modules, "deepmd", deepmd_module)
    monkeypatch.setitem(sys.modules, "deepmd.calculator", calculator_module)


def test_dpa_registry_metadata():
    """DPA entries record their native precision, domains, and D3 behaviour."""
    registry_path = Path(__file__).parents[1] / "ml_peg" / "models" / "models.yml"
    registry = yaml.safe_load(registry_path.read_text(encoding="utf8"))

    assert all("overwrite_dtype" not in registry[name] for name in DPA_MODELS)
    assert all(registry[name]["class_name"] == "DP" for name in DPA_MODELS)
    assert registry["dpa-3p3-1M-omat"]["kwargs"]["head"] == "Omat24"
    assert registry["dpa-3p3-1M-omat"]["datasets"] == DPA3_DATASETS

    omat_models = DPA_MODELS[1:]
    assert all(registry[name]["datasets"] == ["OMAT"] for name in omat_models)
    assert all(not registry[name]["trained_on_dispersion"] for name in DPA_MODELS)


def test_dpa_registry_records_compatible_elements():
    """Every DPA checkpoint's probed H--Og type map is represented in the app."""
    root = Path(__file__).parents[1]
    registry = yaml.safe_load(
        (root / "ml_peg" / "models" / "models.yml").read_text(encoding="utf8")
    )
    coverage = yaml.safe_load(
        (root / "ml_peg" / "app" / "data" / "element_coverage.json").read_text(
            encoding="utf8"
        )
    )["datasets"]
    expected = set(chemical_symbols[1:])

    for name in DPA_MODELS:
        config = registry[name]
        assert set(config["datasets"]) <= set(coverage)
        supported = set(config.get("additional_supported_elements", []))
        for dataset in config["datasets"]:
            supported.update(coverage[dataset]["supported"])
        assert supported == expected


def test_openlam_coverage_matches_constituent_union():
    """OpenLAM-v1 coverage contains the verified constituent element union."""
    root = Path(__file__).parents[1]
    coverage = yaml.safe_load(
        (root / "ml_peg" / "app" / "data" / "element_coverage.json").read_text(
            encoding="utf8"
        )
    )["datasets"]
    constituents = ("OMAT", "MPtrj", "OC20", "OC22", "ODAC23", "SPICE2")
    expected = set().union(*(coverage[name]["supported"] for name in constituents))
    openlam = coverage["OpenLAM-v1"]

    assert set(openlam["supported"]) == expected
    assert openlam["number"] == len(expected) == 89


def test_dpa_models_use_dedicated_wrapper(fake_deepmd):
    """All registered DPA models route through the dedicated wrapper."""
    models = load_models(DPA_MODELS)

    assert tuple(models) == DPA_MODELS
    assert all(isinstance(model, DpaCalc) for model in models.values())
    assert all(model.available for model in models.values())


@pytest.mark.parametrize("precision", ["low", "high"])
def test_dpa_native_precision_does_not_pass_unsupported_options(fake_deepmd, precision):
    """The DPA wrapper leaves fixed checkpoint precision and device to DeePMD."""
    model = load_models(("dpa-3p3-1M-omat",))["dpa-3p3-1M-omat"]

    calculator = model.get_calculator(precision=precision)

    assert calculator.kwargs == {"model": "DPA-3.3-1M", "head": "Omat24"}


def test_dpa_rejects_unknown_precision(fake_deepmd):
    """The DPA wrapper rejects unknown ML-PEG precision choices."""
    model = DpaCalc(kwargs={"model": "checkpoint.pt"})

    with pytest.raises(ValueError, match="Unknown precision"):
        model.get_calculator(precision="medium")


@pytest.mark.skipif(
    not RUN_DPA_MODEL_TESTS,
    reason="set ML_PEG_RUN_DPA_MODEL_TESTS=1 to run pretrained DPA checks",
)
@pytest.mark.parametrize("model_name", DPA_MODELS)
def test_registered_dpa_model_evaluates_in_native_float32(model_name):
    """Run a real checkpoint and inspect its compute dtype and finite outputs."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("deepmd")
    model = load_models((model_name,))[model_name]
    calculator = model.get_calculator(precision="high")

    atoms = bulk("Si", cubic=True)
    atoms.calc = calculator

    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()
    assert np.isfinite(atoms.get_stress()).all()

    deep_eval = calculator.dp.deep_eval
    backend = getattr(deep_eval, "_backend", deep_eval)
    parameter_dtypes = {
        parameter.dtype
        for parameter in backend.dp.parameters()
        if parameter.is_floating_point()
    }
    assert parameter_dtypes == {torch.float32}
