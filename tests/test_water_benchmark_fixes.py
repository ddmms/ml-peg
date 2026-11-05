"""Regression checks for water benchmark failure handling and interactions."""

from __future__ import annotations

from ase import Atoms
from ase.calculators.emt import EMT
from ase.io import read, write
import numpy as np
import pytest

from ml_peg.analysis.utils import aml_md_analysis as aml
from ml_peg.analysis.utils import md_water_analysis as md
from ml_peg.analysis.utils.dipoles import get_z_dipoles_frames
from ml_peg.analysis.utils.water_trajectory import cached_frames, validated_frames
from ml_peg.app.utils.water_comparison import render_subplot_component
from ml_peg.calcs.utils.water_md import prepare_restart, run_water_md


def make_frame(step=0):
    """Build a periodic frame with velocities and run metadata."""
    atoms = Atoms("Cu", positions=[[0, 0, 0]], cell=[4, 4, 4], pbc=True)
    atoms.set_velocities([[0.01, 0.02, 0.03]])
    atoms.info.update(step=step, charge=0, spin=1)
    return atoms


def test_validity_and_cache(tmp_path):
    """Reject invalid runs and invalidate cached data when the source changes."""
    path = tmp_path / "md-traj.extxyz"
    assert validated_frames(path, 2) is None
    write(path, [make_frame(0), make_frame(4)])
    frames = validated_frames(path, 2)
    assert len(frames) == 2
    assert cached_frames(path) is frames
    with pytest.warns(UserWarning, match="Expected"):
        assert validated_frames(path, 3) is None
    marker = tmp_path / "md-failed"
    marker.touch()
    assert validated_frames(path, 2) is None
    marker.unlink()
    write(path, make_frame(8), append=True)
    assert len(validated_frames(path, 3)) == 3
    path.write_text("1\ntruncated\n")
    with pytest.warns(UserWarning, match="Invalid"):
        assert validated_frames(path, 1) is None


def test_signed_score_and_precomputed_plot():
    """Signed cancellation is bounded, and the app displays analysis-generated data."""
    ref = {"dipole": (np.array([0, 1]), np.array([-1.0, 1.0]))}
    pred = {"dipole": (np.array([0, 1]), np.array([1.0, -1.0]))}
    error = aml.compute_all_errors_vacf(ref, pred)["dipole"][2]
    assert aml.error_score_percentage(error) == 0
    zero = {"dipole": (np.array([0, 1]), np.zeros(2))}
    assert aml.compute_all_errors_vacf(zero, zero)["dipole"][2] == 0
    bundle = md.build_bar_data(
        {"m": [0]},
        {"ref": ref, "m": pred},
        ["m"],
        metric_key="dipole",
        metric_label="Dipole",
        ylabel="Dipole",
        xlabel="z",
        errors_fn=aml.compute_all_errors_vacf,
        ref_key=lambda _: "ref",
    )
    point = bundle["models"]["m"]["metrics"]["dipole"][0]
    result = render_subplot_component(
        {"points": [{"customdata": ["dipole", point["data"]]}]}
    )
    assert len(result.children.figure["data"]) == 3


def test_periodic_water_neighbours():
    """Neighbour lists recover O-H bonds across the periodic boundary."""
    atoms = Atoms(
        "OH2",
        positions=[[0.1, 0, 1], [3.2, 0, 1], [0.1, 0.9, 1]],
        cell=[4, 4, 4],
        pbc=True,
    )
    totals, points = get_z_dipoles_frames([atoms], surface_idx_1=0, surface_idx_2=1)
    assert len(points) == 1
    assert np.all(np.isfinite(totals))


def test_restart_keeps_calculator_and_trims(tmp_path):
    """Resume the checkpoint, not later frames, with original calculator and masses."""
    atoms = make_frame()
    atoms.calc = EMT()
    atoms.set_masses([70.0])
    write(tmp_path / "md-res-1000.extxyz", make_frame(1000))
    write(
        tmp_path / "md-traj.extxyz",
        [make_frame(500), make_frame(1000), make_frame(1100)],
    )
    (tmp_path / "md-stats.dat").write_text("# Step values\n500 1\n1000 2\n1100 3\n")
    restored, offset = prepare_restart(atoms, tmp_path, 500, 2000)
    assert offset == 1000
    assert restored.calc is atoms.calc
    assert restored.get_masses()[0] == 70
    assert [a.info["step"] for a in read(tmp_path / "md-traj.extxyz", ":")] == [
        500,
        1000,
    ]
    assert (tmp_path / "md-stats.dat").read_text().endswith("1000 2\n")


def test_short_real_md_clears_marker(tmp_path):
    """Exercise Janus with a tiny fresh run and an existing stale failure marker."""
    atoms = make_frame().repeat((2, 1, 1))
    atoms.calc = EMT()
    (tmp_path / "md-failed").touch()
    run_water_md(
        atoms,
        tmp_path,
        equil_steps=2,
        run_steps=4,
        timestep=0.5,
        temperature=100,
        friction=0.05,
        traj_every=2,
    )
    assert not (tmp_path / "md-failed").exists()
    frames = read(tmp_path / "md-traj.extxyz", ":")
    assert [a.info["step"] for a in frames] == [2, 4, 6]


def test_real_checkpoint_resume(tmp_path):
    """Resumed Janus runs preserve the timeline without duplicate frames."""
    atoms = make_frame().repeat((2, 1, 1))
    atoms.calc = EMT()
    kwargs = {
        "equil_steps": 100,
        "timestep": 0.5,
        "temperature": 100,
        "friction": 0.05,
        "traj_every": 100,
    }
    run_water_md(atoms, tmp_path, run_steps=1000, **kwargs)
    assert not (tmp_path / "md-failed").exists()
    # Extend a run with output beyond the last checkpoint. Resume must trim that
    # tail first, then append from step 1000 through the new final step.
    run_water_md(atoms, tmp_path, run_steps=1100, **kwargs)
    assert not (tmp_path / "md-failed").exists()
    frames = read(tmp_path / "md-traj.extxyz", ":")
    assert [a.info["step"] for a in frames] == list(range(100, 1201, 100))


def test_failed_run_keeps_marker(tmp_path, monkeypatch):
    """Construction failures stay invalid even before dynamics begin."""
    from ml_peg.calcs.utils import water_md

    def fail(**kwargs):
        raise ValueError("calculator failure")

    monkeypatch.setattr(water_md, "NVT", fail)
    with pytest.warns(UserWarning, match="calculator failure"):
        run_water_md(
            make_frame(),
            tmp_path,
            equil_steps=2,
            run_steps=4,
            timestep=0.5,
            temperature=100,
            friction=0.05,
            traj_every=2,
        )
    assert (tmp_path / "md-failed").exists()


def test_cached_arrays_preserve_units_and_masses(tmp_path):
    """Cache converted data once and recover velocities using the saved masses."""
    from ml_peg.analysis.utils.water_trajectory import trajectory_arrays

    atoms = make_frame()
    atoms.set_masses([70.0])
    atoms.set_positions([[1, 2, 3]])
    atoms.set_velocities([[0.01, 0.02, 0.03]])
    path = tmp_path / "md-traj.extxyz"
    write(path, [atoms, atoms])
    positions, velocities = trajectory_arrays(path)
    assert isinstance(positions, np.memmap)
    assert trajectory_arrays(path)[0] is positions
    np.testing.assert_allclose(positions[0], [[0.1, 0.2, 0.3]])
    np.testing.assert_allclose(velocities[0], [[0.001, 0.002, 0.003]])


def test_density_profile_accepts_validated_frames():
    """The density computation accepts the trajectory returned by validation."""
    from ml_peg.analysis.surfaces.copper_water_interface.density_profile import (
        compute_density_profiles,
    )

    atoms = Atoms("CuOH", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]])
    profiles = compute_density_profiles([atoms], surface_idx_1=0, surface_idx_2=1)
    assert profiles["O"][1].sum() == pytest.approx(1)
    assert profiles["H"][1].sum() == pytest.approx(1)


@pytest.mark.parametrize("metric", ["rdf", "vdos", "vacf", "density"])
def test_cell_to_bar_preserves_click_payload(tmp_path, metric):
    """Correct hover labels without changing scores or comparison click payloads."""
    import json

    from ml_peg.analysis.utils.decorators import cell_to_bar

    path = tmp_path / "bar.json"
    payload = {"figure": {"data": []}, "reference": [1, 2]}

    @cell_to_bar(filename=path, x_label="Species", y_label="Score")
    def build():
        return {
            "models": {
                "model": {
                    "metrics": {
                        metric: [{"label": "O-H", "value": 87.5, "data": payload}]
                    }
                }
            }
        }

    build()
    result = json.loads(path.read_text())
    trace = result["models"]["model"]["figures"][metric]["data"][0]
    assert trace["x"] == ["O-H"]
    assert trace["y"] == [87.5]
    assert trace["customdata"] == [["O-H", payload]]
    assert trace["hovertemplate"] == "<b>Label: </b>%{x}<br><b>Score: </b>%{y}<br>"


def test_water_slab_dipole_normalisation():
    """Keep the slab benchmark's charge, sign and area normalisation unchanged."""
    from ml_peg.analysis.utils.dipoles import get_z_dipoles

    atoms = Atoms(
        "OH2",
        positions=[[1, 1, 5], [1.8, 1, 5.5], [1, 1.8, 5.5]],
        cell=[4, 5, 12],
        pbc=True,
    )
    translated = atoms.copy()
    translated.translate([0, 0, 3])
    np.testing.assert_allclose(
        get_z_dipoles([atoms, translated], q=0.6), [0.015, 0.015]
    )


def test_molecular_dipole_profile_with_periodic_bond():
    """Check signed bond vectors, area and surface-relative oxygen height."""
    atoms = Atoms(
        "CuOH2",
        positions=[[2, 2, 2], [0.1, 1, 5], [7.2, 1, 5.2], [0.1, 1.8, 5.5]],
        cell=[8, 8, 12],
        pbc=True,
    )
    total, profile = get_z_dipoles_frames(
        [atoms], q=0.6, surface_idx_1=0, surface_idx_2=1
    )
    expected = 0.3 * (0.2 + 0.5) / 64
    np.testing.assert_allclose(total, [expected])
    np.testing.assert_allclose(profile, [[3, expected]])
