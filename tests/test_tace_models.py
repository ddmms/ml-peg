"""Tests for TACE model registration and calculator construction."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from ml_peg.models.models import TaceCalc


@pytest.mark.parametrize(
    ("precision", "dtype"), [("low", "float32"), ("high", "float64")]
)
@pytest.mark.parametrize("checkpoint", ["TACE-OAM-7M", "/tmp/local-model.pt"])
def test_tace_calculator_passes_model_and_dtype(
    monkeypatch: pytest.MonkeyPatch, precision: str, dtype: str, checkpoint: str
) -> None:
    """Map ML-PEG precision to TACE's dtype and preserve model names or paths."""

    class FakeTACEAseCalc:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    tace = ModuleType("tace")
    interface = ModuleType("tace.interface")
    interface_ase = ModuleType("tace.interface.ase")
    interface_ase.TACEAseCalc = FakeTACEAseCalc
    for name, module in {
        "tace": tace,
        "tace.interface": interface,
        "tace.interface.ase": interface_ase,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    model = TaceCalc(device="cpu", kwargs={"model": checkpoint})
    calc = model.get_calculator(precision=precision)

    assert calc.kwargs == {
        "model": checkpoint,
        "dtype": dtype,
        "device": "cpu",
    }
    assert model.kwargs == {"model": checkpoint}
