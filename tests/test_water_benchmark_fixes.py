"""Regression checks for water benchmark failure handling and interactions."""

from __future__ import annotations

from ase import Atoms
import numpy as np
import pytest

from ml_peg.analysis.utils import aml_md_analysis as aml
from ml_peg.analysis.utils import md_water_analysis as md
from ml_peg.analysis.utils.dipoles import get_z_dipoles_frames


def test_signed_score_and_comparison_data():
    """Signed cancellation is bounded and comparison data retains the curves."""
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
    assert point["data"]["ref"] == [-1.0, 1.0]
    assert point["data"]["pred"] == [1.0, -1.0]
    assert "figure" not in point["data"]


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


@pytest.mark.parametrize(
    "module_name",
    [
        "ml_peg.app.aqueous_solutions.bulk_water.app_bulk_water",
        "ml_peg.app.aqueous_solutions.ice.app_ice",
        "ml_peg.app.surfaces.copper_water_interface.app_copper_water_interface",
    ],
)
def test_restored_comparison_renderers(module_name):
    """Each restored callback renders the curve payload without a shared helper."""
    from importlib import import_module

    module = import_module(module_name)
    data = {
        "x_values": [0, 1],
        "ref": [1, 2],
        "pred": [1, 1.5],
        "error": [0, 0.5],
        "xlabel": "Distance (nm)",
        "ylabel": "g(r)",
    }
    result = module.render_subplot_component(
        {"points": [{"customdata": ["O-H", data]}]}
    )
    assert result.children[1].src.startswith("data:image/png;base64,")
