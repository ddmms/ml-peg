"""Tests for TACE model registration and calculator construction."""

from __future__ import annotations

import os
import sys
from types import ModuleType

import pytest

from ml_peg.models.get_models import load_models
from ml_peg.models.models import TaceCalc, _patch_os_sched_getaffinity


@pytest.mark.parametrize(
    ("model_name", "checkpoint"),
    [("tace-omat24-7m", "TACE-OMat24-7M"), ("tace-oam-7m", "TACE-OAM-7M")],
)
def test_tace_models_use_tace_wrapper(model_name: str, checkpoint: str) -> None:
    """Load TACE foundation models through the TACE wrapper."""
    model = load_models((model_name,), run_mock=False)[model_name]

    assert isinstance(model, TaceCalc)
    assert model.kwargs == {"model": checkpoint}
    assert model.trained_on_dispersion is False


@pytest.mark.parametrize(
    ("precision", "dtype"), [("low", "float32"), ("high", "float64")]
)
def test_tace_calculator_resolves_checkpoint_and_dtype(
    monkeypatch: pytest.MonkeyPatch, precision: str, dtype: str
) -> None:
    """Map ML-PEG precision to TACE's dtype and pass a checkpoint path."""

    class FakeTACEAseCalc:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    tace = ModuleType("tace")
    foundations = ModuleType("tace.foundations")
    foundations.tace_foundations = {"TACE-OAM-7M": "/cache/TACE-OAM-7M.pt"}
    interface = ModuleType("tace.interface")
    interface_ase = ModuleType("tace.interface.ase")
    interface_ase.TACEAseCalc = FakeTACEAseCalc
    for name, module in {
        "tace": tace,
        "tace.foundations": foundations,
        "tace.interface": interface,
        "tace.interface.ase": interface_ase,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    model = TaceCalc(device="cpu", kwargs={"model": "TACE-OAM-7M"})
    calc = model.get_calculator(precision=precision)

    assert calc.kwargs == {
        "model": "/cache/TACE-OAM-7M.pt",
        "dtype": dtype,
        "device": "cpu",
    }
    assert model.kwargs == {"model": "TACE-OAM-7M"}


def test_sched_getaffinity_fallback_on_platforms_without_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Provide the Linux-only call TACE's eqx kernels need at import time."""
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 4)

    _patch_os_sched_getaffinity()

    assert os.sched_getaffinity(0) == {0, 1, 2, 3}
