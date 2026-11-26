"""Tests for the small-cluster force benchmarks."""

from __future__ import annotations

import json
from unittest.mock import Mock

from ase import Atoms
from ase.calculators.calculator import Calculator
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils import analyse_cluster_forces as analysis
from ml_peg.app.utils.cluster_forces import FrameGroups
from ml_peg.app.utils.load import rebuild_table
from ml_peg.calcs.utils import cluster_forces as calc

MODEL = "mace-mp-0a"


def _cluster(symbols: str, offset: float = 0.0) -> Atoms:
    """
    Build a small cluster with both sets of reference labels.

    Parameters
    ----------
    symbols
        Chemical symbols of the cluster.
    offset
        Value added to the reference forces.

    Returns
    -------
    Atoms
        Cluster with ``mad2_*`` and ``omol25_*`` labels.
    """
    n_atoms = len(Atoms(symbols))
    positions = [[2.2 * i, 0.3 * (i % 2), 0.0] for i in range(n_atoms)]
    atoms = Atoms(symbols, positions=positions)
    atoms.info["config_type"] = f"{n_atoms}mer"
    atoms.info["mad2_energy"] = -10.0
    atoms.info["omol25_energy"] = -20.0
    atoms.arrays["mad2_forces"] = np.full((n_atoms, 3), 1.0 + offset)
    atoms.arrays["omol25_forces"] = np.full((n_atoms, 3), 2.0 + offset)
    return atoms


@pytest.fixture
def clusters() -> list[Atoms]:
    """
    Provide clusters, including one without a converged r2SCAN reference.

    Returns
    -------
    list[Atoms]
        Input clusters.
    """
    frames = [_cluster("H3"), _cluster("BeH2"), _cluster("HLi2")]
    frames[2].arrays["mad2_forces"][:] = np.nan
    frames[2].info["mad2_energy"] = np.nan
    return frames


@pytest.mark.parametrize(("reference", "add_d3"), [("r2scan", False), ("wb97mv", True)])
def test_run_cluster_forces(tmp_path, monkeypatch, clusters, reference, add_d3):
    """Keep one reference, store failures as NaN, and add D3 only when requested."""
    monkeypatch.setattr(calc, "load_clusters", lambda: clusters)

    calculator = Mock(spec=Calculator)
    calculator.get_forces.side_effect = [
        np.full((3, 3), 0.5),
        RuntimeError("Unsupported elements"),
        np.full((3, 3), 0.5),
    ]
    calculator.get_potential_energy.return_value = -1.0
    model = Mock()
    model.get_calculator.return_value = calculator
    model.add_d3_calculator.return_value = calculator

    with pytest.warns(UserWarning) as record:
        calc.run_cluster_forces(
            ("model", model), reference=reference, out_path=tmp_path, add_d3=add_d3
        )
    messages = [str(warning.message) for warning in record]
    assert any("Unsupported elements" in message for message in messages)
    assert any("1 of 3 cluster calculations failed" in message for message in messages)

    model.get_calculator.assert_called_once_with(precision="high")
    assert model.add_d3_calculator.called == add_d3

    results = read(tmp_path / "model" / calc.OUTPUT_FILENAME, index=":")
    assert len(results) == len(clusters)
    ref_key = calc.REFERENCE_KEYS[reference]["forces"]
    for index, (original, result) in enumerate(zip(clusters, results, strict=True)):
        np.testing.assert_array_equal(result.positions, original.positions)
        np.testing.assert_array_equal(
            result.arrays[calc.REF_FORCES_KEY], original.arrays[ref_key]
        )
        assert not {"mad2_forces", "omol25_forces"} & set(result.arrays)
        assert result.info["charge"] == 0
        assert result.info["spin"] == (2, 1, 2)[index]
        assert result.info["structure_index"] == index
        assert result.info["calculation_failed"] == (index == 1)
    assert np.isnan(results[1].arrays[calc.PRED_FORCES_KEY]).all()
    assert "Unsupported elements" in results[1].info["calculation_error"]
    np.testing.assert_array_equal(results[0].arrays[calc.PRED_FORCES_KEY], 0.5)


def _write_outputs(path, clusters, pred_offset=0.5, failed=()):
    """
    Write calculation outputs for one model.

    Parameters
    ----------
    path
        Output file.
    clusters
        Input clusters.
    pred_offset
        Difference between predicted and reference forces.
    failed
        Indices of clusters whose prediction failed.
    """
    outputs = []
    for index, atoms in enumerate(clusters):
        out = Atoms(atoms.numbers, positions=atoms.positions)
        out.info["structure_index"] = index
        out.arrays[calc.REF_FORCES_KEY] = atoms.arrays["mad2_forces"]
        pred = np.nan_to_num(atoms.arrays["mad2_forces"]) + pred_offset
        out.arrays[calc.PRED_FORCES_KEY] = np.nan * pred if index in failed else pred
        outputs.append(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    write(path, outputs)


def test_analysis(tmp_path, monkeypatch, clusters):
    """Exclude unconverged references, and report NaN when a prediction fails."""
    clusters += [_cluster("H4"), _cluster("Li4", offset=1.0)]
    calc_path = tmp_path / "calcs"
    out_path = tmp_path / "app"
    _write_outputs(calc_path / "mock" / calc.OUTPUT_FILENAME, clusters, 0.0)
    _write_outputs(calc_path / MODEL / calc.OUTPUT_FILENAME, clusters, 0.5, (1,))
    monkeypatch.setattr(analysis, "MODELS", ["mock", MODEL])

    reference_structs, by_size = analysis.load_reference(calc_path)
    force_data = analysis.get_force_data(calc_path, [MODEL], reference_structs, by_size)
    three = force_data[MODEL][3]
    assert three.n_clusters == 2  # unconverged reference excluded
    assert three.n_failed == 1
    assert np.isnan(three.mae)
    assert force_data[MODEL][4].mae == pytest.approx(0.5)
    assert set(force_data[MODEL]) == {3, 4}

    thresholds = {
        analysis.metric_name(size): {"good": 0.1, "bad": 1.0, "unit": "eV/Å"}
        for size in analysis.CLUSTER_SIZES
    }
    analysis.analyse_cluster_forces(
        title="Test",
        calc_path=calc_path,
        out_path=out_path,
        thresholds=thresholds,
        metric_tooltips={},
        weights={},
    )

    table = rebuild_table(out_path / "cluster_forces_metrics_table.json", id="test")
    row = next(row for row in table.data if row["id"] == MODEL)
    assert row[analysis.metric_name(4)] == pytest.approx(0.5)
    assert row.get(analysis.metric_name(3)) in (None, "NaN")
    assert (out_path / analysis.figure_filename(3)).exists()
    # Clusters with unconverged references are left out of the shared structures
    shared = read(out_path / analysis.structs_filename(3), index=":")
    assert [atoms.info["structure_index"] for atoms in shared] == [0, 1]
    groups = json.loads(
        (out_path / MODEL / analysis.frame_groups_filename(3)).read_text()
    )
    assert groups
    assert all(0 <= frame < len(shared) for group in groups for frame in group)
    info = json.loads((out_path / "info.json").read_text())
    assert info["elements"][:3] == [["H"], ["Be", "H"], ["H", "Li"]]


def test_unusable_outputs_give_nan(tmp_path, clusters):
    """Score a truncated output as failed, without stopping other models."""
    calc_path = tmp_path / "calcs"
    _write_outputs(calc_path / "mock" / calc.OUTPUT_FILENAME, clusters, 0.0)
    _write_outputs(calc_path / MODEL / calc.OUTPUT_FILENAME, clusters, 0.5)
    _write_outputs(calc_path / "truncated" / calc.OUTPUT_FILENAME, clusters[:1], 0.0)
    reference_structs, by_size = analysis.load_reference(calc_path)

    with pytest.warns(UserWarning, match="truncated: could not use"):
        force_data = analysis.get_force_data(
            calc_path, [MODEL, "truncated"], reference_structs, by_size
        )

    assert force_data[MODEL][3].mae == pytest.approx(0.5)
    truncated = force_data["truncated"][3]
    assert np.isnan(truncated.mae)
    assert truncated.n_failed == truncated.n_clusters == 2


def test_infinite_prediction_gives_nan():
    """Report NaN rather than raising when a model returns infinite forces."""
    data = analysis.SizeResults(
        ref=np.zeros(3),
        pred=np.array([0.0, np.inf, 1.0]),
        frames=np.zeros(3, dtype=int),
        labels=["", "", ""],
        n_clusters=1,
        n_failed=1,
    )

    assert np.isnan(data.mae)
    assert data.finite.tolist() == [True, False, True]


def test_frame_groups_load_compactly(tmp_path):
    """Return the same frames as the JSON file, one point at a time."""
    groups = [[3, 0, 1], [5], [2, 4]]
    path = tmp_path / "groups.json"
    path.write_text(json.dumps(groups))

    frame_groups = FrameGroups(path)

    assert len(frame_groups) == len(groups)
    assert [frame_groups[i] for i in range(len(groups))] == groups
    assert all(isinstance(frame, int) for frame in frame_groups[0])
    with pytest.raises(IndexError):
        frame_groups[len(groups)]


def test_frame_groups_are_capped():
    """Show each point's own cluster first, without duplicates."""
    n_clusters = 3 * analysis.MAX_FRAMES_PER_POINT
    frames = np.repeat(np.arange(n_clusters), 9)
    data = analysis.SizeResults(
        ref=np.zeros(len(frames)),
        pred=np.zeros(len(frames)),
        frames=frames,
        labels=[""] * len(frames),
        n_clusters=n_clusters,
        n_failed=0,
    )

    groups, cell_clusters = analysis.get_frame_groups(data)

    # All components share one density cell
    assert set(cell_clusters) == {n_clusters}
    sampled, _, _ = analysis.sample_density_grid(data.ref, data.pred)
    assert len(groups) == len(sampled)
    for own, group in zip(sampled, groups, strict=True):
        assert group[0] == frames[own]
        assert len(group) == len(set(group)) == analysis.MAX_FRAMES_PER_POINT
