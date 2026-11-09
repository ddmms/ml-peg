"""Tests for NequIP model registration and calculator construction."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from ase import Atoms
from ase.build import bulk
from ase.data import chemical_symbols
import numpy as np
import pytest
import yaml

from ml_peg.models import models
from ml_peg.models.get_models import load_models

NEQUIP_MODELS = (
    "nequip-oam-s-0p2",
    "nequip-oam-m-0p2",
    "nequip-oam-l-0p2",
    "nequip-oam-xl-0p2",
    "allegro-oam-s-0p2",
    "allegro-oam-m-0p2",
    "allegro-oam-l-0p2",
    "nequip-omat24-s-0p2",
    "nequip-omat24-m-0p2",
    "nequip-omat24-l-0p2",
    "nequip-omat24-xl-0p2",
    "allegro-omat24-s-0p2",
    "allegro-omat24-m-0p2",
    "allegro-omat24-l-0p2",
)
OAM_ELEMENTS = set(chemical_symbols[1:84] + chemical_symbols[89:95])  # also OMat24's
ROOT = Path(__file__).parents[1]
# OAM-S energies of the rattled NaCl cell used in the real model test below
NACL_REFERENCE_ENERGIES = {
    "nequip-oam-s-0p2": -216.15304486661688,
    "allegro-oam-s-0p2": -216.1002435376874,
}


def _registry() -> dict:
    """Load the default model registry."""
    path = ROOT / "ml_peg" / "models" / "models.yml"
    return yaml.safe_load(path.read_text(encoding="utf8"))


def test_nequip_registry_metadata():
    """NequIP entries point at the Zenodo packages and record their D3 behaviour."""
    registry = _registry()

    for name in NEQUIP_MODELS:
        config = registry[name]
        assert config["class_name"] == "NequIPCalculator"
        datasets = ["OMAT"] if "-omat24-" in name else ["MPtrj", "sAlex", "OMAT"]
        assert config["datasets"] == datasets
        assert config["device"] == "auto"
        assert config["trained_on_dispersion"] is False
        assert "overwrite_dtype" not in config
        assert config["kwargs"]["model"].startswith(
            "https://zenodo.org/api/records/23472045/"
        )


def test_nequip_registry_records_package_elements():
    """The registry datasets cover exactly the 89 atom types of the packages."""
    registry = _registry()
    coverage = json.loads(
        (ROOT / "ml_peg" / "app" / "data" / "element_coverage.json").read_text(
            encoding="utf8"
        )
    )["datasets"]

    for name in NEQUIP_MODELS:
        config = registry[name]
        supported = set(config.get("additional_supported_elements", []))
        for dataset in config["datasets"]:
            supported.update(coverage[dataset]["supported"])
        assert supported == OAM_ELEMENTS


def test_compile_caches_by_package_and_thread_count(monkeypatch, tmp_path):
    """Packages compile once, with scalar C++ and a thread count recorded in the key."""
    import torch

    package = tmp_path / "model.nequip.zip"
    package.write_bytes(b"package")
    runs = []

    def run(command, **kwargs):
        runs.append((command, kwargs["env"]))
        Path(command[4]).write_bytes(b"compiled")

    nequip = ModuleType("nequip")
    nequip.__version__ = "0.0.0"
    monkeypatch.setitem(sys.modules, "nequip", nequip)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("ATEN_CPU_CAPABILITY", raising=False)
    monkeypatch.setattr(subprocess, "run", run)

    path = models._compile_nequip_package(str(package), "cpu")
    assert models._compile_nequip_package(str(package), "cpu") == path
    assert len(runs) == 1
    command, env = runs[0]
    assert command[3] == str(package)
    assert " ".join(command[5:]) == "--mode aotinductor --device cpu --target ase"
    assert env["ATEN_CPU_CAPABILITY"] == "default"
    assert env["OMP_NUM_THREADS"] == str(torch.get_num_threads())
    assert Path(path).read_bytes() == b"compiled"
    files = [file.name for file in Path(path).parent.iterdir()]
    assert [name for name in files if not name.endswith(".lock")] == [Path(path).name]

    threads = torch.get_num_threads() + 1
    monkeypatch.setattr(torch, "get_num_threads", lambda: threads)
    assert models._compile_nequip_package(str(package), "cpu") != path
    assert runs[1][1]["OMP_NUM_THREADS"] == str(threads)


def test_nequip_compile_variants(monkeypatch):
    """GPU kernels are tried first on CUDA, when installed, then plain AOTInductor."""
    import torch

    nequip = {"enable_OpenEquivariance", "enable_CuEquivariance"}
    allegro = {"enable_CuEquivarianceContracter", "enable_TritonContracter"}
    plain = ("aotinductor", None)
    installed = {"openequivariance", "cuequivariance_ops_torch", "triton"}
    monkeypatch.setattr(models, "find_spec", lambda name: name in installed)
    monkeypatch.setattr(torch, "__version__", "2.8.0+cu128")

    assert models._nequip_compile_variants(nequip, "cpu") == [plain]
    assert models._nequip_compile_variants(nequip, "cuda") == [
        ("torchscript", "enable_OpenEquivariance"),
        plain,
    ]
    assert models._nequip_compile_variants(allegro, "cuda") == [
        ("aotinductor", "enable_CuEquivarianceContracter"),
        ("aotinductor", "enable_TritonContracter"),
        plain,
    ]
    monkeypatch.setattr(torch, "__version__", "2.10.0")
    assert models._nequip_compile_variants(nequip, "cuda")[0] == (
        "aotinductor",
        "enable_OpenEquivariance",
    )
    # e.g. ROCm, without cuEquivariance:
    installed.discard("cuequivariance_ops_torch")
    assert models._nequip_compile_variants(allegro, "cuda") == [
        ("aotinductor", "enable_TritonContracter"),
        plain,
    ]
    installed.clear()
    assert models._nequip_compile_variants(nequip | allegro, "cuda") == [plain]


def test_nequip_falls_back_to_eager_model(monkeypatch):
    """Variants that fail to import or compile are skipped with warnings."""
    pytest.importorskip("nequip")

    def fail(model, device, mode, modifier):
        raise subprocess.CalledProcessError(1, "nequip-compile", stderr="no compiler")

    variants = [("aotinductor", "enable_OpenEquivariance"), ("aotinductor", None)]
    monkeypatch.setitem(sys.modules, "openequivariance", None)
    monkeypatch.setattr(models, "_nequip_compile_variants", lambda *args: variants)
    monkeypatch.setattr(models, "_compile_nequip_package", fail)
    models._load_nequip_calculator.cache_clear()
    model = load_models(("nequip-oam-s-0p2",))["nequip-oam-s-0p2"]

    with pytest.warns(UserWarning) as warnings:
        model.get_calculator(precision="high")
    models._load_nequip_calculator.cache_clear()
    messages = [str(w.message) for w in warnings if "NequIP model" in str(w.message)]
    assert "openequivariance" in messages[0]
    assert "no compiler" in messages[1]
    assert messages[2].startswith("Skipping the eager NequIP model")
    assert messages[-1].startswith("Using the eager")


def test_eager_nequip_model_uses_gpu_kernels(monkeypatch):
    """Eager models use GPU kernels, as compiled models do."""
    pytest.importorskip("nequip")
    import nequip.model.modify_utils

    applied = []

    def modify(model, modifiers):
        applied.extend(modifiers)
        return model

    variants = [("torchscript", "enable_OpenEquivariance"), ("aotinductor", None)]
    monkeypatch.setitem(sys.modules, "openequivariance", ModuleType("oeq"))
    monkeypatch.setattr(models, "_nequip_compile_variants", lambda *args: variants)
    monkeypatch.setattr(nequip.model.modify_utils, "modify", modify)
    models._load_nequip_calculator.cache_clear()
    model = load_models(("nequip-oam-s-0p2",))["nequip-oam-s-0p2"]

    dataclasses.replace(model, compile=False).get_calculator(precision="high")
    models._load_nequip_calculator.cache_clear()
    assert applied == [{"modifier": "enable_OpenEquivariance"}]


# Failed compilation falls back to the eager model with this warning
@pytest.mark.filterwarnings("error:Using the eager")
@pytest.mark.parametrize("name", NACL_REFERENCE_ENERGIES)
def test_oam_s_compiled_matches_eager(name):
    """Compiled OAM-S packages match their eager models, including tiny structures."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("nequip")
    from nequip.utils.dtype import torch_default_dtype

    default_dtype = torch.get_default_dtype()
    models._load_nequip_calculator.cache_clear()
    model = load_models((name,))[name]
    compiled = model.get_calculator(precision="high")
    eager = dataclasses.replace(model, compile=False).get_calculator(precision="high")

    # loaded once per process, with calculators sharing the model:
    assert model.get_calculator(precision="high").model is compiled.model
    assert torch.get_default_dtype() == default_dtype
    assert eager.model.metadata["model_dtype"] == "float32"
    assert set(eager.model.metadata["type_names"].split()) == OAM_ELEMENTS

    cell = bulk("NaCl", "rocksalt", a=5.64, cubic=True) * (2, 2, 2)
    cell.rattle(0.05, seed=1)
    structures = (
        cell,
        bulk("Fe", "bcc", a=2.87),
        Atoms("Na"),
        # No edges beyond the cutoff, then two edges within it:
        Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 10]]),
        Atoms("NaCl", positions=[[0, 0, 0], [0, 0, 2.4]]),
    )

    def evaluate(calc, atoms, dtype=torch.float32):
        """Evaluate a fresh copy of ``atoms``, bypassing ASE's results cache."""
        calc.reset()
        atoms = atoms.copy()
        atoms.calc = calc
        with torch_default_dtype(dtype):
            return atoms.get_potential_energy(), atoms.get_forces()

    assert evaluate(eager, cell)[0] == pytest.approx(
        NACL_REFERENCE_ENERGIES[name], abs=1e-3
    )
    for atoms in structures:
        energy, forces = evaluate(eager, atoms, torch.float64)
        for calc in (compiled, copy.deepcopy(compiled)):
            compiled_energy, compiled_forces = evaluate(calc, atoms)
            assert compiled_forces.dtype == np.float64
            assert compiled_energy == pytest.approx(energy, abs=1e-4)
            np.testing.assert_allclose(compiled_forces, forces, atol=1e-4)

    cell.calc = compiled
    stress = cell.get_stress()
    cell.calc = eager
    np.testing.assert_allclose(stress, cell.get_stress(), atol=1e-5)
