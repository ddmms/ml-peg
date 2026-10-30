"""Tests for UMA (FairChem) inference settings and UMA Nitro."""

from __future__ import annotations

import pytest

from ml_peg.models.get_models import load_models

NITRO_MODELS = (
    "uma-s-1p2p1-omol",
    "uma-s-1p2p1-omat",
    "uma-s-1p2p1-ft-d4rev",
    "uma-s-1p2p1-ft-r2scan",
)


class _FakeFAIRChemCalculator:
    """Minimal stand-in for the fairchem ASE calculator."""

    def __init__(self, predict_unit, task_name=None):
        self.predict_unit = predict_unit
        self.task_name = task_name


@pytest.fixture
def fake_fairchem(monkeypatch):
    """
    Replace checkpoint loading and the ASE calculator with fakes.

    Returns
    -------
    dict
        Keyword arguments passed to the last ``get_predict_unit`` call.
    """
    fairchem_core = pytest.importorskip("fairchem.core")
    calls = {}

    def get_predict_unit(model_name, device=None, overrides=None, **kwargs):
        calls.update(model_name=model_name, device=device, **kwargs)
        return object()

    monkeypatch.setattr(
        fairchem_core.pretrained_mlip, "get_predict_unit", get_predict_unit
    )
    monkeypatch.setattr(fairchem_core, "FAIRChemCalculator", _FakeFAIRChemCalculator)
    return calls


def _preset(name, dtype):
    """
    Get a fairchem inference preset at a given precision.

    Parameters
    ----------
    name
        Name of the fairchem inference preset.
    dtype
        Base precision dtype.

    Returns
    -------
    InferenceSettings
        The preset with ``base_precision_dtype`` set.
    """
    from dataclasses import replace

    from fairchem.core.units.mlip_unit.api.inference import (
        NAME_TO_INFERENCE_SETTING,
    )

    return replace(NAME_TO_INFERENCE_SETTING[name], base_precision_dtype=dtype)


@pytest.mark.parametrize(
    ("precision", "dtype"), [("low", "float32"), ("high", "float64")]
)
def test_existing_uma_models_use_batch_settings(fake_fairchem, precision, dtype):
    """Models without inference_settings keep the fairchem<2.25 default."""
    calc = load_models(("uma-s-1p2-omol",))["uma-s-1p2-omol"].get_calculator(
        precision=precision
    )

    assert calc.task_name == "omol"
    assert fake_fairchem["inference_settings"] == _preset("batch", dtype)


@pytest.mark.parametrize("name", NITRO_MODELS)
def test_nitro_models_use_nitro_in_float32(fake_fairchem, name):
    """UMA-S-1.2.1 models request UMA Nitro for float32 calculations."""
    load_models((name,))[name].get_calculator(precision="low")

    settings = fake_fairchem["inference_settings"]
    assert settings.nitro
    assert settings == _preset("nitro", "float32")


@pytest.mark.parametrize("name", NITRO_MODELS)
def test_nitro_models_use_batch_settings_in_float64(fake_fairchem, name):
    """Nitro is float32 only, so float64 uses the standard settings instead."""
    load_models((name,))[name].get_calculator(precision="high")

    assert fake_fairchem["inference_settings"] == _preset("batch", "float64")
