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
        analysis.reference_datasets.__wrapped__()
    shutil.copytree(outputs / analysis.BENCHMARK, transferred / analysis.BENCHMARK)
    monkeypatch.setattr(analysis, "OUT_PATH", tmp_path / "app")
    reference = analysis.reference_datasets.__wrapped__()
    with pytest.warns(UserWarning, match="Cannot score corrupt"):
        values = analysis.analyze_results.__wrapped__(reference)
    assert values["corrupt"] is None
    upstream = analysis.MlPegReferenceGeometryStabilityBenchmark(
        force_field=LennardJones(), data_input_dir=cache, run_mode="standard"
    )
    upstream.run_model()
    expected = upstream.analyze()
    # The saved frame must be the one the minimization converged to.
    state = upstream.model_output.openff_neutral[0].simulation_state
    assert np.linalg.norm(np.diff(state.positions[-1], axis=0)) == pytest.approx(
        2 ** (1 / 6), abs=1e-3
    )
    assert np.linalg.norm(state.forces[-1], axis=1).max() < 0.01
    table = analysis.metrics.__wrapped__.__wrapped__(None, values)
    for metric in ("Avg RMSD", "Avg RMSD (stable only)"):
        assert table[metric]["test-model"] == pytest.approx(expected.avg_rmsd)
        assert np.isnan(table[metric]["corrupt"])
        assert analysis.DEFAULT_WEIGHTS[metric] == 0
    for metric in ("Geometry Score", "Geometry Score (MLIP Audit)"):
        assert table[metric]["test-model"] == pytest.approx(expected.score)
        assert np.isnan(table[metric]["corrupt"])
    assert analysis.DEFAULT_WEIGHTS["Geometry Score"] == 1
    assert analysis.DEFAULT_WEIGHTS["Geometry Score (MLIP Audit)"] == 0

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
        output = getattr(upstream.model_output, subset)[0]
        for frame, raw in zip(
            frames,
            (molecule["coordinates"], output.simulation_state.positions[-1]),
            strict=True,
        ):
            assert frame.get_distance(0, 1) == pytest.approx(
                np.linalg.norm(np.subtract(raw[0], raw[1]))
            )
        assert value == getattr(expected, subset).rmsd_values[0]

    # A missing output and a recorded failure both invalidate the weighted score.
    reference["openff_neutral"]["mol_1"] = reference["openff_neutral"]["mol_0"]
    for recorded_failure in (False, True):
        if recorded_failure:
            failed = upstream.model_output.openff_neutral[0].model_copy(
                update={"molecule_name": "mol_1", "failed": True}
            )
            upstream.model_output.openff_neutral.append(failed)
        partial = upstream.analyze()
        row, _ = analysis.analyze_model("partial", upstream.model_output, reference)
        assert np.isnan(row["Avg RMSD"])
        assert row["Avg RMSD (stable only)"] == pytest.approx(partial.avg_rmsd)
        assert np.isnan(row["Geometry Score"])
        assert row["Geometry Score (MLIP Audit)"] == pytest.approx(partial.score)
        if recorded_failure:
            assert partial.score < expected.score
        else:
            assert partial.score == pytest.approx(expected.score)
