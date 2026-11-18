"""Regression tests for the MLIP Audit geometry benchmark."""

from __future__ import annotations

from types import SimpleNamespace

from ase.calculators.lj import LennardJones
from ase.io import read
import numpy as np
import pytest

pytest.importorskip("mlipaudit")
from ml_peg.analysis.molecular.reference_geometry_stability import (
    analyse_reference_geometry_stability as analysis,
)
from ml_peg.calcs.molecular.reference_geometry_stability import (
    calc_reference_geometry_stability as calc,
)


def test_transferred_model_output_matches_upstream(tmp_path, monkeypatch):
    """Score transferred calculation outputs against the staged reference subsets."""
    import json
    import shutil

    cache = tmp_path / "cache"
    dataset = cache / analysis.BENCHMARK
    dataset.mkdir(parents=True)
    molecule = {
        "atom_symbols": ["Ar", "Ar"],
        "coordinates": [[0, 0, 0], [1.4, 0, 0]],
        "charge": 0,
        "smiles": "[Ar].[Ar]",
    }
    for name in (
        analysis.rgs.OPENFF_NEUTRAL_FILENAME,
        analysis.rgs.OPENFF_CHARGED_FILENAME,
    ):
        (dataset / name).write_text(json.dumps({"mol_0": molecule}))

    outputs = tmp_path / "outputs"
    monkeypatch.setattr(calc, "OUT_PATH", outputs)
    monkeypatch.setattr(calc, "download_s3_data", lambda **kwargs: cache)
    model = SimpleNamespace(
        get_calculator=lambda **kwargs: LennardJones(),
        add_d3_calculator=lambda calculator: calculator,
    )
    calc.test_reference_geometry_stability(("test-model", model))

    transferred = tmp_path / "transferred"
    shutil.copytree(outputs / "test-model", transferred / "test-model")
    corrupt = transferred / "corrupt" / analysis.BENCHMARK
    corrupt.mkdir(parents=True)
    (corrupt / "model_output.zip").write_bytes(b"truncated")
    monkeypatch.setattr(analysis, "CALC_PATH", transferred)
    monkeypatch.setattr(analysis, "MODELS", ["corrupt", "test-model"])
    with pytest.raises(
        pytest.skip.Exception, match=analysis.rgs.OPENFF_NEUTRAL_FILENAME
    ):
        analysis.reference_benchmark.__wrapped__()
    shutil.copytree(outputs / analysis.BENCHMARK, transferred / analysis.BENCHMARK)
    monkeypatch.setattr(analysis, "OUT_PATH", tmp_path / "app")
    reference = analysis.reference_benchmark.__wrapped__()
    with pytest.warns(UserWarning, match="Cannot score corrupt"):
        values = analysis.analyze_results.__wrapped__(reference)
    assert values["corrupt"] is None
    monkeypatch.setattr(
        analysis.rgs.ReferenceGeometryStabilityBenchmark,
        "skip_if_elements_missing",
        False,
    )
    upstream = analysis.rgs.ReferenceGeometryStabilityBenchmark(
        force_field=LennardJones(), data_input_dir=cache, run_mode="standard"
    )
    upstream.run_model()
    expected = upstream.analyze()
    assert analysis.get_avg_rmsd.__wrapped__(values)["test-model"] == pytest.approx(
        expected.avg_rmsd
    )
    assert analysis.get_score.__wrapped__(values)["test-model"] == pytest.approx(
        expected.score
    )

    plot = json.loads((analysis.OUT_PATH / "figure_molecule_rmsd.json").read_text())
    (trace,) = plot["data"]
    assert trace["name"] == "test-model"
    assert trace["x"] == ["neutral / mol_0", "charged / mol_0"]
    for subset, asset, value in zip(
        analysis.rgs.DATASET_PREFIXES, trace["customdata"], trace["y"], strict=True
    ):
        frames = read(analysis.OUT_PATH / "structures" / f"{asset}.xyz", index=":")
        assert len(frames) == 2
        np.testing.assert_allclose(frames[0].positions.mean(axis=0), 0, atol=1e-8)
        np.testing.assert_allclose(frames[1].positions.mean(axis=0), 0, atol=1e-8)
        output = getattr(reference.model_output, subset)[0]
        for frame, raw in zip(
            frames,
            (molecule["coordinates"], output.simulation_state.positions[-1]),
            strict=True,
        ):
            assert frame.get_distance(0, 1) == pytest.approx(
                np.linalg.norm(np.subtract(raw[0], raw[1]))
            )
        assert value == getattr(expected, subset).rmsd_values[0]
