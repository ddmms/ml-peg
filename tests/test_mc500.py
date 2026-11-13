"""Tests for the MC500 molecular-crystal benchmark."""

from __future__ import annotations

from ase import Atoms
import numpy as np
import pytest

from ml_peg.analysis.molecular_crystal.MC500.analyse_MC500 import (
    rms_cartesian_displacement,
)
from ml_peg.calcs.molecular_crystal.MC500 import cif_utils


def test_rmscd_identical_structures() -> None:
    """Identical structures have zero RMSCD."""
    atoms = Atoms(
        "CH",
        scaled_positions=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        cell=[5.0, 6.0, 7.0],
    )
    atoms.pbc = True

    rmscd, rmscd_no_h, displacements = rms_cartesian_displacement(atoms, atoms)

    assert rmscd == pytest.approx(0.0)
    assert rmscd_no_h == pytest.approx(0.0)
    assert displacements == pytest.approx([0.0, 0.0])


def test_rmscd_averages_reference_and_relaxed_cells() -> None:
    """The Cartesian displacement is averaged over the two cells."""
    reference = Atoms("C", scaled_positions=[[0.1, 0.0, 0.0]], cell=[2.0] * 3)
    relaxed = Atoms("C", scaled_positions=[[0.2, 0.0, 0.0]], cell=[4.0] * 3)
    reference.pbc = True
    relaxed.pbc = True

    rmscd, rmscd_no_h, displacements = rms_cartesian_displacement(reference, relaxed)

    assert rmscd == pytest.approx(0.3)
    assert rmscd_no_h == pytest.approx(0.3)
    assert displacements == pytest.approx([0.3])


def test_rmscd_rejects_different_atom_order() -> None:
    """RMSCD requires matching atoms in matching order."""
    reference = Atoms("CH", positions=np.zeros((2, 3)), cell=[5.0] * 3)
    relaxed = Atoms("HC", positions=np.zeros((2, 3)), cell=[5.0] * 3)

    with pytest.raises(ValueError, match="different atom order"):
        rms_cartesian_displacement(reference, relaxed)


def test_read_mc500_cif_matches_original_reader(monkeypatch) -> None:
    """The default MC500 reader uses ASE and retains the original CIF labels."""
    atoms = Atoms(
        "CC",
        scaled_positions=[[0.0, 0.2, 0.3], [0.5, 0.6, 0.7]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
        info={
            "_atom_site_label": ["C1"],
            "_atom_site_occupancy": [1.0],
        },
    )
    atoms.new_array("spacegroup_kinds", np.array([0, 0]))
    read_kwargs = {}

    def fake_read(filename, **kwargs):
        read_kwargs.update(kwargs)
        return atoms.copy()

    monkeypatch.setattr(cif_utils, "read", fake_read)
    result = cif_utils.read_mc500_cif("structure.cif")

    assert read_kwargs["reader"] == "ase"
    assert read_kwargs["store_tags"] is True
    assert result.arrays[cif_utils.CIF_LABEL_ARRAY].tolist() == ["C1", "C1"]
    assert not np.isclose(
        result.get_scaled_positions(wrap=False),
        0.0,
        atol=cif_utils.BOUNDARY_TOLERANCE,
        rtol=0.0,
    ).any()


def test_read_mc500_cif_pycodcif_is_retained(monkeypatch) -> None:
    """The alternative reader selects the optional pycodcif backend."""
    atoms = Atoms(
        "C",
        scaled_positions=[[0.1, 0.2, 0.3]],
        cell=[5.0, 5.0, 5.0],
        pbc=True,
        info={
            "_atom_site_label": ["C1"],
            "_atom_site_occupancy": [1.0],
        },
    )
    atoms.new_array("spacegroup_kinds", np.array([0]))
    read_kwargs = {}

    def fake_read(filename, **kwargs):
        read_kwargs.update(kwargs)
        return atoms.copy()

    monkeypatch.setattr(cif_utils, "read", fake_read)
    cif_utils.read_mc500_cif_pycodcif("structure.cif")

    assert read_kwargs["reader"] == "pycodcif"


def test_read_mc500_cif_accepts_implicit_full_occupancy(monkeypatch) -> None:
    """Missing occupancy tags use the CIF default of full occupancy."""
    atoms = Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=[5.0] * 3, pbc=True)
    atoms.info = {"_atom_site_label": ["C1"]}
    atoms.new_array("spacegroup_kinds", np.array([0]))
    monkeypatch.setattr(cif_utils, "read", lambda *args, **kwargs: atoms)

    result = cif_utils.read_mc500_cif("structure.cif")

    assert len(result) == 1


def test_read_mc500_cif_rejects_partial_occupancy(monkeypatch) -> None:
    """The MC500 reader rejects disordered sites with partial occupancy."""
    atoms = Atoms("C", positions=[[0.0, 0.0, 0.0]], cell=[5.0] * 3, pbc=True)
    atoms.info = {
        "_atom_site_label": ["C1"],
        "_atom_site_occupancy": [0.5],
    }
    atoms.new_array("spacegroup_kinds", np.array([0]))
    monkeypatch.setattr(cif_utils, "read", lambda *args, **kwargs: atoms)

    with pytest.raises(ValueError, match="partial occupancies"):
        cif_utils.read_mc500_cif("structure.cif")


@pytest.mark.parametrize(
    "failure_mode", ["missing_row", "unconverged", "failed", "missing_xyz"]
)
def test_analysis_preserves_reference_order_and_failed_results(
    tmp_path, monkeypatch, failure_mode
) -> None:
    """Failures keep their slot and cannot contribute a passing RMSCD score."""
    import csv
    import inspect
    import json

    from ase.io import write

    from ml_peg.analysis.molecular_crystal.MC500 import analyse_MC500 as analysis

    calc_path = tmp_path / "calc"
    app_path = tmp_path / "app"
    atoms = Atoms("C", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    for model in ("mock", "example"):
        (calc_path / model / "structures").mkdir(parents=True)
        for structure_id in ("A", "B"):
            write(
                calc_path / model / "structures" / f"{structure_id}.xyz", [atoms, atoms]
            )

    rows = [
        {
            "structure_id": structure_id,
            "refcode": structure_id,
            "converged": "True",
            "steps": "1",
            "max_force": "0",
            "failure": "",
        }
        for structure_id in ("B", "A")
    ]
    if failure_mode == "missing_row":
        rows.pop()
    elif failure_mode == "unconverged":
        rows[1]["converged"] = "False"
    elif failure_mode == "failed":
        rows[1]["failure"] = "RuntimeError: failed after optimization"
    else:
        (calc_path / "example" / "structures" / "A.xyz").unlink()
    with (calc_path / "example" / "results.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    stale_app = app_path / "example" / "structures" / "A.xyz"
    stale_app.parent.mkdir(parents=True)
    write(stale_app, atoms)
    monkeypatch.setattr(analysis, "CALC_PATH", calc_path)
    monkeypatch.setattr(analysis, "OUT_PATH", app_path)
    monkeypatch.setattr(analysis, "MODELS", ["example"])
    results = analysis.mc500_results.__wrapped__()
    result = results["example"]
    assert result["structure_ids"] == ["A", "B"]
    assert np.isnan(result["rmscd"][0])
    assert result["rmscd"][1] == pytest.approx(0)
    assert result["converged"] == [False, True]
    assert result["total"] == 2
    assert not stale_app.exists()
    analysis.rmscd_figures.__wrapped__(results)
    figure = json.loads((app_path / "figure_rmscd_example.json").read_text())
    info = json.loads((app_path / "info.json").read_text())
    assert info["filenames"] == ["A", "B"]
    for trace in figure["data"]:
        assert trace["y"] == [None, 0.0]
        assert [row[0] for row in trace["customdata"]] == info["filenames"]
    scores = inspect.unwrap(analysis.metrics)(results, None)
    assert np.isnan(scores["Mean RMSCD excluding H"]["example"])
    assert scores["Structures within threshold"]["example"] == 50
    assert scores["Convergence"]["example"] == 50


def test_failed_rerun_removes_old_trajectory(tmp_path, monkeypatch) -> None:
    """A failed rerun cannot retain an earlier successful geometry."""
    from types import SimpleNamespace

    from ml_peg.calcs.molecular_crystal.MC500 import calc_MC500 as calc

    (tmp_path / "MC500").mkdir()
    (tmp_path / "MC500" / "A.cif").touch()
    output = tmp_path / "outputs"
    trajectory = output / "example" / "structures" / "A.xyz"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text("stale geometry")
    summary = output / "example" / "results.csv"
    summary.write_text("stale summary")
    monkeypatch.setattr(calc, "OUT_PATH", output)
    monkeypatch.setattr(calc, "download_s3_data", lambda **kwargs: tmp_path)
    monkeypatch.setattr(calc, "read_mc500_cif", lambda path: Atoms("C"))
    monkeypatch.setattr(calc, "get_spacegroup_number", lambda atoms: 1)

    def fail(*args):
        assert not summary.exists()
        raise RuntimeError("optimization failed")

    monkeypatch.setattr(calc, "relax_crystal", fail)
    model = SimpleNamespace(
        get_calculator=lambda **kwargs: None,
        add_d3_calculator=lambda calculator: calculator,
    )
    with pytest.warns(UserWarning, match="optimization failed"):
        calc.test_mc500_relaxation(("example", model))
    assert not trajectory.exists()
    assert "RuntimeError: optimization failed" in summary.read_text()
