"""Shared fixtures for the GPUMD runner tests (no GPU needed)."""

from __future__ import annotations

import json
from pathlib import Path
import stat
import sys

import ase.io
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
import numpy as np
import pytest

POC_DIR = Path(__file__).resolve().parents[1]
DATA = Path(__file__).parent / "data"
sys.path.insert(0, str(POC_DIR / "runner"))

import mlpeg_gpumd  # noqa: E402

SIO2_FILE = DATA / "amorphous-silica-structure.xyz"
# NEP89 restricted to H, O and Si with calorine's Model.keep_species.
NEP_SUBSET = DATA / "nep89-H-O-Si.txt"


@pytest.fixture
def sio2_atoms():
    """
    Amorphous SiO2 (768 atoms) with 300 K Maxwell-Boltzmann momenta, seed 0.

    Returns
    -------
    ase.Atoms
        Structure with masses and momenta.
    """
    atoms = ase.io.read(SIO2_FILE)
    MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(0))
    Stationary(atoms)
    return atoms


@pytest.fixture
def fake_engine(monkeypatch, tmp_path):
    """
    Point the runner at the fake gpumd and the H/O/Si subset of NEP89.

    Parameters
    ----------
    monkeypatch
        Pytest monkeypatch fixture.
    tmp_path
        Pytest temporary directory.

    Returns
    -------
    pytest.MonkeyPatch
        The monkeypatch, so tests can set ``FAKE_GPUMD_MODE``.
    """
    wrapper = tmp_path / "gpumd"
    wrapper.write_text(
        f"#!{sys.executable}\n"
        "import runpy, sys\n"
        f"sys.path.insert(0, {str(Path(__file__).parent)!r})\n"
        f"runpy.run_path({str(Path(__file__).parent / 'fake_gpumd.py')!r}, "
        "run_name='__main__')\n"
    )
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setattr(mlpeg_gpumd, "GPUMD_BIN", wrapper)
    monkeypatch.setattr(mlpeg_gpumd, "NEP_PATH", NEP_SUBSET)
    monkeypatch.setattr(mlpeg_gpumd, "BUILD_INFO", tmp_path / "missing.json")
    tmp = tmp_path / "container-tmp"
    tmp.mkdir()
    monkeypatch.setattr(mlpeg_gpumd, "TMP_DIR", tmp)
    monkeypatch.setenv("FAKE_GPUMD_MODE", "ok")
    return monkeypatch


@pytest.fixture
def campaign(tmp_path, sio2_atoms):
    """
    Stage a campaign input directory with one SiO2 size and one NaCl size.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    sio2_atoms
        Structure fixture.

    Returns
    -------
    pathlib.Path
        Path to the staged ``spec.json``.
    """
    in_dir = tmp_path / "in"
    (in_dir / "structures").mkdir(parents=True)
    spec = json.loads((POC_DIR / "template_spec.json").read_text())
    rel = "structures/sio2_quartz_768.extxyz"
    ase.io.write(in_dir / rel, sio2_atoms, format="extxyz")
    spec["systems"][0]["sizes"] = [
        {
            "n_atoms": 768,
            "supercell": [1, 1, 1],
            "file": rel,
            "sha256": mlpeg_gpumd.sha256_file(in_dir / rel),
            "validate": True,
        },
        {
            "n_atoms": 769,
            "supercell": [1, 1, 1],
            "file": "structures/sio2_quartz_769.extxyz",
            "sha256": "<set by ML-PEG>",
            "validate": False,
        },
    ]
    nacl = sio2_atoms.copy()
    nacl.set_chemical_symbols(["Na" if s == "Si" else "Cl" for s in nacl.symbols])
    nacl_rel = "structures/nacl_rocksalt_1000.extxyz"
    ase.io.write(in_dir / nacl_rel, nacl, format="extxyz")
    spec["systems"][1]["sizes"][0].update(n_atoms=768, file=nacl_rel)
    (in_dir / "spec.json").write_text(json.dumps(spec, indent=2))
    return in_dir / "spec.json"
