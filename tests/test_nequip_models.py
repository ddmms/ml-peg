"""Tests for NequIP model registration and calculator construction."""

from __future__ import annotations

import copy
import dataclasses
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

from ase import Atoms
from ase.build import bulk
from ase.data import chemical_symbols
import numpy as np
import pytest
import yaml

from ml_peg.models import models
from ml_peg.models.get_models import load_models

ROOT = Path(__file__).parents[1]
REGISTRY = yaml.safe_load((ROOT / "ml_peg" / "models" / "models.yml").read_text("utf8"))
NEQUIP_MODELS = [
    name
    for name, config in REGISTRY.items()
    if config["class_name"] == "NequIPCalculator"
]
OAM_ELEMENTS = set(chemical_symbols[1:84] + chemical_symbols[89:95])  # also OMat24's
# OAM-S energies of the rattled NaCl cell used in the real model test below:
NACL_REFERENCE_ENERGIES = {
    "nequip-oam-s-0p2": -216.15304486661688,
    "allegro-oam-s-0p2": -216.1002435376874,
}
OEQ = "enable_OpenEquivariance"


@pytest.fixture(autouse=True)
def clear_calculator_cache():
    """Stop calculators loaded in one test, some with mocks, being reused in others."""
    yield
    models._load_nequip_calculator.cache_clear()


def test_nequip_registry_records_package_elements():
    """The 14 entries' datasets cover exactly the 89 atom types of the packages."""
    path = ROOT / "ml_peg" / "app" / "data" / "element_coverage.json"
    coverage = json.loads(path.read_text("utf8"))["datasets"]

    assert len(NEQUIP_MODELS) == 14
    for name in NEQUIP_MODELS:
        datasets = REGISTRY[name]["datasets"]
        supported = set().union(*(coverage[key]["supported"] for key in datasets))
        assert supported == OAM_ELEMENTS


def test_compile_caches_by_package_and_thread_count(monkeypatch, tmp_path):
    """Packages compile once, with scalar C++ and a thread count recorded in the key."""
    torch = pytest.importorskip("torch")

    package = tmp_path / "model.nequip.zip"
    package.write_bytes(b"package")
    runs = []

    def run(command, **kwargs):
        runs.append((command, kwargs["env"]))
        Path(command[4]).write_bytes(b"compiled")

    monkeypatch.setitem(sys.modules, "nequip", SimpleNamespace(__version__="0.0.0"))
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


def test_nequip_kernels(monkeypatch):
    """Installed GPU kernels are listed in order of preference, on CUDA only."""
    pytest.importorskip("torch")
    # NequIP packages also offer enable_CuEquivariance, unused as (currently) slower
    # than no kernels
    nequip = {OEQ, "enable_CuEquivariance"}
    allegro = ["enable_CuEquivarianceContracter", "enable_TritonContracter"]
    installed = {"openequivariance", "cuequivariance_ops_torch", "triton"}
    monkeypatch.setattr(models, "find_spec", lambda name: name in installed)

    assert models._nequip_kernels(nequip | set(allegro), "cpu") == []
    assert models._nequip_kernels(nequip, "cuda") == [OEQ]
    assert models._nequip_kernels(set(allegro), "cuda") == allegro
    installed.discard("cuequivariance_ops_torch")  # e.g. ROCm
    assert models._nequip_kernels(set(allegro), "cuda") == allegro[1:]


def test_nequip_falls_back_to_eager_model(monkeypatch):
    """Kernels that fail to import and models that fail to compile are skipped."""
    pytest.importorskip("nequip")

    def fail(*args):
        raise subprocess.CalledProcessError(1, "nequip-compile", stderr="no compiler")

    monkeypatch.setitem(sys.modules, "openequivariance", None)
    monkeypatch.setattr(models, "_nequip_kernels", lambda *args: [OEQ])
    monkeypatch.setattr(models, "_compile_nequip_package", fail)
    model = load_models(("nequip-oam-s-0p2",))["nequip-oam-s-0p2"]

    with pytest.warns(UserWarning) as warnings:
        model.get_calculator(precision="high")
    messages = [str(w.message) for w in warnings if "NequIP model" in str(w.message)]
    assert len(messages) == 4
    assert "openequivariance" in messages[0]
    assert "no compiler" in messages[1]
    assert messages[2].startswith("Skipping the eager NequIP model")
    assert messages[3].startswith("Using the eager")


def test_eager_nequip_model_uses_gpu_kernels(monkeypatch):
    """Eager models use GPU kernels, in eval mode, as compiled models do."""
    pytest.importorskip("nequip")
    import nequip.model.modify_utils
    import torch

    applied = []

    def modify(model, modifiers):
        applied.extend(modifiers)
        model.swapped = torch.nn.Linear(1, 1)  # new modules start in training mode
        return model

    monkeypatch.setitem(sys.modules, "openequivariance", SimpleNamespace())
    monkeypatch.setattr(models, "_nequip_kernels", lambda *args: [OEQ])
    monkeypatch.setattr(nequip.model.modify_utils, "modify", modify)
    model = load_models(("nequip-oam-s-0p2",))["nequip-oam-s-0p2"]

    calc = dataclasses.replace(model, compile=False).get_calculator(precision="high")
    assert applied == [{"modifier": OEQ}]
    assert not calc.model.swapped.training


def test_nequip_registry_options(tmp_path):
    """``compile: false`` loads the eager model, and ``overwrite_dtype`` only warns."""
    pytest.importorskip("nequip")
    name = "nequip-oam-s-0p2"
    kwargs = REGISTRY[name]["kwargs"] | {"compile": False}
    config = REGISTRY[name] | {"kwargs": kwargs, "overwrite_dtype": "float64"}
    path = tmp_path / "models.yml"
    path.write_text(yaml.safe_dump({name: config}))
    model = load_models((name,), filepath=path)[name]

    assert not model.compile
    with pytest.warns(UserWarning, match="overwrite_dtype 'float64' has no effect"):
        model.get_calculator(precision="high")


# Failed compilation falls back to the eager model with this warning
@pytest.mark.filterwarnings("error:Using the eager")
@pytest.mark.parametrize("name", NACL_REFERENCE_ENERGIES)
def test_oam_s_compiled_matches_eager(name):
    """Compiled OAM-S packages match their eager models, including tiny structures."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("nequip")
    from nequip.utils.dtype import torch_default_dtype

    default_dtype = torch.get_default_dtype()
    model = load_models((name,))[name]
    compiled = model.get_calculator(precision="high")
    eager = dataclasses.replace(model, compile=False).get_calculator(precision="high")

    # loaded once per process, with calculators sharing the model:
    assert model.get_calculator(precision="high").model is compiled.model
    assert torch.get_default_dtype() == default_dtype
    assert eager.model.metadata["model_dtype"] == "float32"
    assert not any(module.training for module in eager.model.modules())
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
