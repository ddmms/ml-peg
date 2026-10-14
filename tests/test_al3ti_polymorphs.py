"""Test Al3Ti calculation output and app structure selection."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import pytest

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
