"""Test Al3Ti calculation, analysis, and app failure handling."""

from __future__ import annotations

from importlib import import_module
from inspect import unwrap
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from ml_peg.analysis.alloy_metallurgy.al3ti_polymorphs import (
    analyse_al3ti_polymorphs as al3ti_analysis,
)
from ml_peg.app.alloy_metallurgy.al3ti_polymorphs import (
    app_al3ti_polymorphs as al3ti_app,
)


@pytest.fixture
def benchmark():
    """Import the calculation module without loading configured models."""
    with patch("ml_peg.models.get_models.load_models", return_value={}):
        return import_module(
            "ml_peg.calcs.alloy_metallurgy.al3ti_polymorphs.calc_al3ti_polymorphs"
        )


def test_old_outputs_removed_before_calculator_setup(benchmark, monkeypatch, tmp_path):
    """A calculator setup failure must not leave previous results visible."""

    class FailingModel:
        def get_calculator(self, *, precision):
            raise RuntimeError("calculator setup failed")

    monkeypatch.setattr(benchmark, "OUT_PATH", tmp_path)
    output_dir = tmp_path / "model"
    output_dir.mkdir()
    for name in benchmark.STRUCTURE_NAMES:
        (output_dir / f"{name}.xyz").write_text("stale")

    with pytest.raises(RuntimeError, match="calculator setup failed"):
        benchmark.test_al3ti_polymorphs(("model", FailingModel()), tmp_path)

    assert not list(output_dir.glob("*.xyz"))


def test_failed_write_does_not_publish_partial_output(benchmark, monkeypatch, tmp_path):
    """A write error leaves neither a final nor temporary XYZ file."""

    def fail_after_partial_write(path: Path, *_args, **_kwargs):
        path.write_text("partial")
        raise OSError("write failed")

    monkeypatch.setattr(benchmark, "write", fail_after_partial_write)

    with pytest.warns(UserWarning, match="Error writing structure D022"):
        assert not benchmark.write_structure("D022", object(), tmp_path)

    assert not (tmp_path / "D022.xyz").exists()
    assert not (tmp_path / "D022.xyz.tmp").exists()


def test_structure_urls_use_clicked_model_and_polymorph(monkeypatch, tmp_path):
    """Plot labels select only structures available for their own model."""
    monkeypatch.setattr(al3ti_app, "DATA_PATH", tmp_path)
    monkeypatch.setattr(al3ti_app, "MODELS", ["model-a", "model-b"])
    for model, name in (("model-a", "D022"), ("model-b", "D023")):
        model_dir = tmp_path / model
        model_dir.mkdir()
        (model_dir / f"{name}.xyz").touch()

    figure = {"data": [{"name": "model-b"}, {"name": "model-a"}]}
    urls = al3ti_app.structure_urls_by_point(figure)

    assert figure["data"][0]["customdata"] == [
        "model-b:D022",
        "model-b:D023",
        "model-b:L12",
    ]
    assert urls == {
        "model-b:D023": "/assets/alloy_metallurgy/al3ti_polymorphs/model-b/D023.xyz",
        "model-a:D022": "/assets/alloy_metallurgy/al3ti_polymorphs/model-a/D022.xyz",
    }


@pytest.mark.parametrize("failed_structure", ["Al_fcc", "Ti_hcp", "D023"])
def test_energy_failure_keeps_other_models_scoreable(
    monkeypatch, tmp_path, failed_structure
):
    """An unreadable energy leaves NaN and does not stop later models."""

    class EnergyAtoms:
        def __init__(self, energy: float | None = -4.0):
            self.energy = energy

        def __len__(self):
            return 4

        def get_potential_energy(self):
            if self.energy is None:
                raise RuntimeError("energy unavailable")
            return self.energy

    monkeypatch.setattr(al3ti_analysis, "MODELS", ["bad", "good"])
    monkeypatch.setattr(al3ti_analysis, "CALC_PATH", tmp_path)
    for model in ("bad", "good"):
        (tmp_path / model).mkdir()

    def load_reference(structure, model_dir):
        if model_dir.name == "bad" and structure == failed_structure:
            return EnergyAtoms(None)
        return EnergyAtoms()

    monkeypatch.setattr(al3ti_analysis, "load_structure", load_reference)
    relaxed = {
        model: [EnergyAtoms(-5.0) for _ in al3ti_analysis.POLYMORPHS]
        for model in ("bad", "good")
    }
    if failed_structure == "D023":
        relaxed["bad"][1] = EnergyAtoms(None)
    references = {
        name: {"formation_energy_0k_ev_atom": {"value": -0.25}}
        for name in al3ti_analysis.POLYMORPHS
    }

    with pytest.warns(UserWarning, match=f"model bad, structure {failed_structure}"):
        results = unwrap(al3ti_analysis.formation_energies)(references, relaxed)

    assert np.isnan(results["bad"][1])
    assert results["good"] == pytest.approx([-0.25] * len(al3ti_analysis.POLYMORPHS))
    if failed_structure in ("Al_fcc", "Ti_hcp"):
        assert np.isnan(results["bad"]).all()
    else:
        assert np.isfinite([results["bad"][0], results["bad"][2]]).all()
