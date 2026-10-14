"""Test Al3Ti calculation output handling."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from unittest.mock import patch

import pytest


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
