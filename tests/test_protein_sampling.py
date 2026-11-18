"""Regression tests for MLIP Audit sampling scores and archive failures."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("mlipaudit")
from mlipaudit.benchmarks.sampling.sampling import SamplingModelOutput
from mlipaudit.io import write_model_output_to_disk

from ml_peg.analysis.biomolecules.protein_sampling import (
    analyse_protein_sampling as sampling,
)
from ml_peg.analysis.utils.utils import calc_table_scores


def test_sampling_score_and_coverage():
    """The default table score follows upstream for partial and failed runs."""
    partial = SimpleNamespace(
        score=0.625,
        systems=[SimpleNamespace(failed=failed) for failed in (False, True, False)],
    )
    failed = SimpleNamespace(score=0.0, systems=[SimpleNamespace(failed=True)] * 3)
    results = {"partial": partial, "failed": failed, "corrupt": None}
    metrics = sampling.metrics.__wrapped__.__wrapped__(results)
    rows = [
        {"MLIP": model, **{key: values[model] for key, values in metrics.items()}}
        for model in results
    ]
    scores = calc_table_scores(
        rows, thresholds=sampling.DEFAULT_THRESHOLDS, weights=sampling.DEFAULT_WEIGHTS
    )
    assert scores[0]["Score"] == partial.score
    assert scores[0]["Stable Proteins"] == 2
    assert scores[1]["Score"] == 0
    assert scores[1]["Stable Proteins"] == 0
    assert scores[2]["Score"] == "NaN"
    assert np.isnan(scores[2]["Stable Proteins"])


def test_corrupt_archive_does_not_hide_later_models(tmp_path, monkeypatch):
    """Continue to the next model after a ZIP fails to load."""
    monkeypatch.setattr(sampling, "CALC_PATH", tmp_path)
    monkeypatch.setattr(sampling, "MODELS", ["corrupt", "valid"])
    corrupt = tmp_path / "corrupt" / sampling.BENCHMARK
    corrupt.mkdir(parents=True)
    (corrupt / "model_output.zip").write_bytes(b"truncated")
    with pytest.raises(pytest.skip.Exception):
        sampling.dataset_dir.__wrapped__()

    directory = tmp_path / sampling.BENCHMARK_DATA_DIR
    directory.mkdir()
    (directory / "backbone_reference_data.json").write_text(
        '{"general":{"phi":[0],"psi":[0]}}'
    )
    (directory / "sidechain_reference_data.json").write_text('{"SER":{"chi1":[0]}}')
    failed_run = SamplingModelOutput(
        structure_names=sampling.STRUCTURE_NAMES,
        simulation_states=[None] * len(sampling.STRUCTURE_NAMES),
    )
    write_model_output_to_disk(sampling.BENCHMARK, failed_run, tmp_path / "valid")
    with pytest.warns(UserWarning, match="Cannot score corrupt"):
        results = sampling.analyze_results.__wrapped__(directory)
    assert results["corrupt"] is None
    assert results["valid"].score == 0
    assert all(system.failed for system in results["valid"].systems)
